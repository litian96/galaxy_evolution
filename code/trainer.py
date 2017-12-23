 from __future__ import print_function

import os
import StringIO
import scipy.misc
import numpy as np
from glob import glob
from tqdm import trange
from itertools import chain
from collections import deque
from PIL import Image

from models import *
from utils import save_image

def next(loader):
    return loader.next()[0].data.numpy()

def to_nhwc(image, data_formaty    if data_forma;t == 'NCHW':
        new_image = nchw_to_nhwc(image)
    else:
        new_image = image
    return new_image

def to_nchw_numpy(image):
    if image.shape[3] in [1, 3]:
        new_image = image.transpose([0, 3, 1, 2])
    else:
        new_image = image
    return new_image

def norm_img(image, data_format=None):
    image = image/127.5 - 1.
    if data_format:
        image = to_nhwc(image, data_format)
    return image

def denorm_img(norm, data_format):
    return tf.clip_by_value(to_nhwc((norm + 1)*127.5, data_format), 0, 255)

def slerp(val, low, high):
    """Code from https://github.com/soumith/dcgan.torch/issues/14"""
    omega = np.arccos(np.clip(np.dot(low/np.linalg.norm(low), high/np.linalg.norm(high)), -1, 1))
    so = np.sin(omega)
    if so == 0:
        return (1.0-val) * low + val * high # L'Hopital's rule/LERP
    return np.sin((1.0-val)*omega) / so * low + np.sin(val*omega) / so * high

class Trainer(object):
    def __init__(self, config, data_loader):
        self.config = config
        self.data_loader = data_loader
        self.dataset = config.dataset

        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.optimizer = config.optimizer
        self.batch_size = config.batch_size

        self.step = tf.Variable(0, name='step', trainable=False)

        self.ae_lr = tf.Variable(config.ae_lr, name='ae_lr')
        self.ae_lr_update = tf.assign(self.ae_lr, tf.maximum(self.ae_lr * 0.5, config.lr_lower_boundary), name='ae_lr_update')
     
        self.d_lr = tf.Variable(config.d_lr, name='d_lr')
        self.d_lr_update = tf.assign(self.d_lr, tf.maximum(self.d_lr * 0.5, config.lr_lower_boundary), name='d_lr_update')

        self.gamma = config.gamma
        self.lambda_k = config.lambda_k
        #self.weight = tf.Variable(config.weight, name='weight')
        #self.weight_update = tf.assign(self.weight, tf.minimum(self.weight * 2,  0.025), name='weight_update')

        self.weight_ = tf.Variable(config.weight_, name="weight_")
        #self.weight_update1 = tf.assign(self.weight_, tf.constant(0, dtype=tf.float32), name='weight_update1')
        self.weight_update = tf.assign(self.weight_, tf.constant(8.0), name='weight_update')

        self.z_num = config.z_num
        self.conv_hidden_num = config.conv_hidden_num
        self.input_scale_size = config.input_scale_size

        self.model_dir = config.model_dir
        self.load_path = config.load_path

        self.use_gpu = config.use_gpu
        self.data_format = config.data_format

        #_, height, width, self.channel = get_conv_shape(self.data_loader, self.data_format)
        height = 64
        width = 64
        self.channel = 1
       

        self.repeat_num = int(np.log2(height)) - 2

        self.start_step = 0
        self.log_step = config.log_step
        self.max_step = config.max_step
        self.save_step = config.save_step
        self.lr_update_step = config.lr_update_step

        self.is_train = config.is_train
        self.build_model()

        self.saver = tf.train.Saver()
        self.summary_writer = tf.summary.FileWriter(self.model_dir)

        sv = tf.train.Supervisor(logdir=self.model_dir,
                                is_chief=True,
                                saver=self.saver,
                                summary_op=None,
                                summary_writer=self.summary_writer,
                                save_model_secs=180,
                                global_step=self.step,
                                ready_for_local_init_op=None)

        gpu_options = tf.GPUOptions(allow_growth=True)
        sess_config = tf.ConfigProto(allow_soft_placement=True,
                                    gpu_options=gpu_options)

        self.sess = sv.prepare_or_wait_for_session(config=sess_config)

        if not self.is_train:
            # dirty way to bypass graph finilization error
            g = tf.get_default_graph()
            g._finalized = False

            self.build_test_model()

    def train(self):

        for step in trange(self.start_step, self.max_step):
            x_fixed, _ = self.get_image_from_loader()
            save_image(x_fixed, '{}/x_fixed.png'.format(self.model_dir))

            

            if  step < 1500: 
                #self.sess.run(self.weight_update1)
                
                self.sess.run(self.ae_optim)
                w = self.sess.run(self.weight_)
                AE_loss = self.sess.run(self.ae_loss)  
                if step % 16 == 0:            
                    print("# step: "+str(step)+", ae_loss: "+str(AE_loss)+", weight="+str(w)+"\n")

            elif step < 1800:
                self.sess.run(self.d_optim)
                D_loss, re = self.sess.run([self.d_loss, self.label_reshaped])   
                if step % 16 == 0:             
                    print("only D: step: "+str(step)+", d_loss:"+ str(D_loss)+"\n")

         
            else:
                self.sess.run([self.weight_update])
                w = self.sess.run(self.weight_)
                self.sess.run(self.ae_optim)
                AE_loss = self.sess.run(self.ae_loss)
                self.sess.run(self.d_optim)
                D_loss, D_loss2, re = self.sess.run([self.d_loss, self.d_loss2, self.label_reshaped])
                

                if step % 16 == 0:
                    print("only D: step: "+str(step)+", d_loss:"+ str(D_loss)+", ae_loss:"+str(AE_loss)+", weight="+str(w)+"\n")
                    print("D loss of every image is: ")
                    print(D_loss2)



            if step % self.lr_update_step == self.lr_update_step - 1:
                self.sess.run([self.ae_lr_update, self.d_lr_update])

            if step % 100 == 0:
             self.autoencode(x_fixed, _, self.model_dir, idx=step)
            #if step % 100 == 0:
            #    self.sess.run([self.weight_update])
            

    def build_model(self):

        self.x, self.label = self.data_loader 
        print(self.label.get_shape())
        self.label1 = tf.reshape(self.label,[self.batch_size,1])
        self.label2 = tf.concat([tf.constant(1, shape=[self.batch_size,1], dtype=tf.float32) - self.label1, self.label1], 1)
        print(self.label2.get_shape())    

        x = norm_img(self.x)
        d_out, self.D_z, self.AE_var = AE(x, self.label, self.batch_size, self.channel, self.z_num, self.repeat_num,self.conv_hidden_num, self.data_format)        

        AE_x = d_out
        self.AE_x = denorm_img(AE_x, self.data_format)
        print(self.AE_x.get_shape())
        
        self.out_label, self.D_var = Discriminator(self.D_z, self.batch_size)

        self.label_reshaped = tf.reshape(self.out_label,[-1])

        if self.optimizer == 'adam':
            optimizer = tf.train.AdamOptimizer
        else:
            raise Exception("[!] Caution! Paper didn't use {} opimizer other than Adam".format(config.optimizer))

        ae_optimizer = optimizer(self.ae_lr)
        d_optimizer = optimizer(self.d_lr)

        ##...........................Define the loss function here.May be changed................................##
 
        #self.ae_loss = 4 * tf.reduce_mean(tf.abs(AE_x - x)) - 0.04 * tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.out_label, labels = self.label2))
        self.ae_loss = tf.reduce_sum(tf.abs(AE_x-x)) + self.weight_* tf.reduce_sum(tf.reduce_sum(self.label2 * tf.log(self.out_label+1e-8), 1))
        self.ae_optim = ae_optimizer.minimize(self.ae_loss, var_list=self.AE_var)

        #self.d_loss = tf.reduce_mean(tf.abs(self.label-self.label_reshaped))
        #self.d_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.out_label, labels = self.label2))
        #self.d_loss = tf.reduce_mean()   #[16,1] [16,2]
        self.d_loss = -tf.reduce_sum(tf.reduce_sum(self.label2 * tf.log(self.out_label+1e-8), 1))
        self.d_loss2 = tf.reduce_sum(self.label2 * tf.log(self.out_label+1e-8), 1)
        self.d_optim = d_optimizer.minimize(self.d_loss, var_list=self.D_var)


    def build_test_model(self):
        with tf.variable_scope("test") as vs:
            # Extra ops for interpolation
            z_optimizer = tf.train.AdamOptimizer(0.0001)
            self.z_r = tf.get_variable("z_r", [self.batch_size, self.z_num], tf.float32)

        test_variables = tf.contrib.framework.get_variables(vs)
        self.sess.run(tf.variables_initializer(test_variables))


    def autoencode(self, inputs, label, path, idx=None):
        items = {'real': inputs}
        for key, img in items.items():
            if img is None:
                continue
            if img.shape[3] in [1, 3]:
                img = img.transpose([0, 3, 1, 2])

            x_path = os.path.join(path, '{}_D_real.png'.format(idx))
            x = self.sess.run(self.AE_x, {self.x: img, self.label: label})
            save_image(x, x_path)
            print("[*] Samples saved: {}".format(x_path))

    def encode(self, inputs):
        if inputs.shape[3] in [1, 3]:
            inputs = inputs.transpose([0, 3, 1, 2])
        return self.sess.run(self.D_z, {self.x: inputs})

    def decode(self, z, label):
        return self.sess.run(self.AE_x, {self.D_z: z, self.label: label})

    def interpolate_D(self, real1_batch, real2_batch, label1, label2, step=0, root_path="."):

        real1_encode = self.encode(real1_batch)
        real2_encode = self.encode(real2_batch)
        decodes = []
        for idx, ratio in enumerate(np.linspace(0, 1, 10)):
            z = np.stack([slerp(ratio, r1, r2) for r1, r2 in zip(real1_encode, real2_encode)])
            z_decode = self.decode(z, label1)
            decodes.append(z_decode)

        decodes = np.stack(decodes).transpose([1, 0, 2, 3, 4])
        for idx, img in enumerate(decodes):
            img = np.concatenate([[real1_batch[idx]], img, [real2_batch[idx]]], 0)
            save_image(img, os.path.join(root_path, 'test{}_interp_D_{}.png'.format(step, idx)), nrow=10 + 2)

    def change_attributes(self, real_batch, root_path='.'):

        real_encode = self.encode(real_batch)
        decodes = []
        imgs = []
        # test batch size
     
        decodes=self.decode(real_encode,[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1])
        
        save_image(decodes, os.path.join(root_path, "test_changed_attri.png"))

    def test(self):
        root_path = "./"#self.model_dir

        for step in range(3):
            real1_batch, label1_batch = self.get_image_from_loader()
            #real2_batch, label2_batch = self.get_image_from_loader()
            save_image(real1_batch, os.path.join(root_path, 'test{}_real1.png'.format(step)))
            #save_image(real2_batch, os.path.join(root_path, 'test{}_real2.png'.format(step)))

            self.autoencode(real1_batch, label1_batch, self.model_dir, idx=os.path.join(root_path, "test{}_real1".format(step)))
           
            self.change_attributes(real1_batch, self.model_dir)
            


    def get_image_from_loader(self):
        tmp, label = self.data_loader
        x = tmp.eval(session=self.sess)
        label2 = label.eval(session=self.sess)
        if self.data_format == 'NCHW':
            x = x.transpose([0, 2, 3, 1])
        return x, label2
