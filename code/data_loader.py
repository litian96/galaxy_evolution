import os
from PIL import Image
from glob import glob
import tensorflow as tf

def get_loader(root, batch_size, scale_size, data_format, split=None, is_grayscale=False, seed=None):
    tf_decode = tf.image.decode_jpeg
    shape = [64, 64, 3]
    records=[]
    fin = open("data/galaxy_64_bdtr/train.txt",'r')
    for line in fin:
        records.append(line.strip("\n"))
    filename_queue = tf.train.string_input_producer(records, shuffle=False, seed=seed)
    filename, label = tf.decode_csv(filename_queue.dequeue(), [[""], [""]], " ") # make the filename and the label tensor
    file_contents = tf.read_file(filename) # read all the trainig samples
    
    image = tf_decode(file_contents, channels=3) # this is one image from the queue

    if is_grayscale:
        image = tf.image.rgb_to_grayscale(image)
    image.set_shape(shape)

    min_after_dequeue = 150000
    capacity = min_after_dequeue + 3 * batch_size

    queue, batch_label = tf.train.shuffle_batch(
        [image, label], batch_size=batch_size,
        num_threads=4, capacity=capacity,
        min_after_dequeue=min_after_dequeue, name='synthetic_inputs')

    queue = tf.image.resize_nearest_neighbor(queue, [scale_size, scale_size])

    if data_format == 'NCHW':
        queue = tf.transpose(queue, [0, 3, 1, 2])
    elif data_format == 'NHWC':
        pass
    else:
        raise Exception("[!] Unkown data_format: {}".format(data_format))
    #print(queue)

    return tf.to_float(queue), tf.string_to_number(batch_label)


def get_loader_test(root, batch_size, scale_size, data_format, split=None, is_grayscale=False, seed=None):
    tf_decode = tf.image.decode_jpeg
    shape = [64, 64, 3]
    records=[]
    fin = open("data/galaxy_64_bdtr/test.txt",'r')
    for line in fin:
        records.append(line.strip("\n"))
    filename_queue = tf.train.string_input_producer(records, shuffle=False, seed=seed)
    filename, label = tf.decode_csv(filename_queue.dequeue(), [[""], [""]], " ") # make the filename and the label tensor
    file_contents = tf.read_file(filename) # read all the trainig samples
    
    image = tf_decode(file_contents, channels=3) # this is one image from the queue

    if is_grayscale:
        image = tf.image.rgb_to_grayscale(image)
    image.set_shape(shape)

    min_after_dequeue = 150000
    capacity = min_after_dequeue + 3 * batch_size

    queue, batch_label = tf.train.shuffle_batch(
        [image, label], batch_size=batch_size,
        num_threads=4, capacity=capacity,
        min_after_dequeue=min_after_dequeue, name='synthetic_inputs')

    queue = tf.image.resize_nearest_neighbor(queue, [scale_size, scale_size])

    if data_format == 'NCHW':
        queue = tf.transpose(queue, [0, 3, 1, 2])
    elif data_format == 'NHWC':
        pass
    else:
        raise Exception("[!] Unkown data_format: {}".format(data_format))
    #print(queue)

    return tf.to_float(queue), tf.string_to_number(batch_label)

def get_loader_generated(root, batch_size, scale_size, data_format, split=None, is_grayscale=False, seed=None):
    tf_decode = tf.image.decode_jpeg
    shape = [64, 64, 3]
    records=[]
    fin = open("data/galaxy_64_bdtr/generated2.txt",'r')
    for line in fin:
        records.append(line.strip("\n"))
    filename_queue = tf.train.string_input_producer(records, shuffle=False, seed=seed)
    filename, label = tf.decode_csv(filename_queue.dequeue(), [[""], [""]], " ") # make the filename and the label tensor
    file_contents = tf.read_file(filename) # read all the trainig samples
    
    image = tf_decode(file_contents, channels=3) # this is one image from the queue

    if is_grayscale:
        image = tf.image.rgb_to_grayscale(image)
    image.set_shape(shape)

    min_after_dequeue = 150000
    capacity = min_after_dequeue + 3 * batch_size

    queue, batch_label = tf.train.shuffle_batch(
        [image, label], batch_size=batch_size,
        num_threads=4, capacity=capacity,
        min_after_dequeue=min_after_dequeue, name='synthetic_inputs')

    queue = tf.image.resize_nearest_neighbor(queue, [scale_size, scale_size])

    if data_format == 'NCHW':
        queue = tf.transpose(queue, [0, 3, 1, 2])
    elif data_format == 'NHWC':
        pass
    else:
        raise Exception("[!] Unkown data_format: {}".format(data_format))
    #print(queue)

    return tf.to_float(queue), tf.string_to_number(batch_label)
