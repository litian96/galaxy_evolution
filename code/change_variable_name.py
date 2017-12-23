import sys, getopt
import tensorflow as tf




def rename(checkpoint_dir):
    
    with tf.Session() as sess:

        for var_name, _ in tf.contrib.framework.list_variables(checkpoint_dir):
            # Load the variable
            var = tf.contrib.framework.load_variable(checkpoint_dir, var_name)

            new_name = var_name
##
            if var_name == 'R/Variable':
                new_name = 'R/w1'
            if var_name == 'R/Variable_1':
                new_name = 'R/b1'
            if var_name == 'R/Variable_2':
                new_name = 'R/w2'
            if var_name == 'R/Variable_3':
                new_name = 'R/b2'

            if var_name == 'R/Variable_4':
                new_name = 'R/fc_w1'
            if var_name == 'R/Variable_5':
                new_name = 'R/fc_b1'
            if var_name == 'R/Variable_6':
                new_name = 'R/fc_w2'
            if var_name == 'R/Variable_7':
                new_name = 'R/fc_b2'

            if var_name == 'R/Variable/Adam':
                new_name = 'R/w1/Adam'
            if var_name == 'R/Variable_1/Adam':
                new_name = 'R/b1/Adam'
            if var_name == 'R/Variable_2/Adam':
                new_name = 'R/w2/Adam'
            if var_name == 'R/Variable_3/Adam':
                new_name = 'R/b2/Adam'

            if var_name == 'R/Variable_4/Adam':
                new_name = 'R/fc_w1/Adam'
            if var_name == 'R/Variable_5/Adam':
                new_name = 'R/fc_b1/Adam'
            if var_name == 'R/Variable_6/Adam':
                new_name = 'R/fc_w2/Adam'
            if var_name == 'R/Variable_7/Adam':
                new_name = 'R/fc_b2/Adam'

            if var_name == 'R/Variable/Adam_1':
                new_name = 'R/w1/Adam_1'
            if var_name == 'R/Variable_1/Adam_1':
                new_name = 'R/b1/Adam_1'
            if var_name == 'R/Variable_2/Adam_1':
                new_name = 'R/w2/Adam_1'
            if var_name == 'R/Variable_3/Adam_1':
                new_name = 'R/b2/Adam_1'

            if var_name == 'R/Variable_4/Adam_1':
                new_name = 'R/fc_w1/Adam_1'
            if var_name == 'R/Variable_5/Adam_1':
                new_name = 'R/fc_b1/Adam_1'
            if var_name == 'R/Variable_6/Adam_1':
                new_name = 'R/fc_w2/Adam_1'
            if var_name == 'R/Variable_7/Adam_1':
                new_name = 'R/fc_b2/Adam_1'
#
            var = tf.Variable(var, name=new_name)
            #print(var_name)


        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        saver.save(sess, './3')


def main():
    checkpoint_dir = None

    rename('2')


if __name__ == '__main__':
    main()