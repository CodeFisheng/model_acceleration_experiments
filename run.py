import tensorflow as tf
import numpy as np
import argparse
import os
from tensorflow.examples.tutorials.mnist import input_data

ROOT_PATH = os.path.dirname(os.path.abspath(__file__))

def float32_variable_storage_getter(getter, name, shape=None, dtype=None,
                                    initializer=None, regularizer=None,
                                    trainable=True,
                                    *args, **kwargs):
    """Custom variable getter that forces trainable variables to be stored in
    float32 precision and then casts them to the training precision.
    """
    storage_dtype = tf.float32 if trainable else dtype
    variable = getter(name, shape, dtype=storage_dtype,
                      initializer=initializer, regularizer=regularizer,
                      trainable=trainable,
                      *args, **kwargs)
    if trainable and dtype != tf.float32:
        variable = tf.cast(variable, dtype)
    return variable

def gradients_display_list(loss, display_gradients_variables):
    ret = []
    for grad in display_gradients_variables:
        ret.append(tf.gradients(loss, grad))
    return ret

def gradients_with_loss_scaling(loss, variables, loss_scale):
    """Gradient calculation with loss scaling to improve numerical stability
    when training with float16.
    """
    return [grad / loss_scale
            for grad in tf.gradients(loss * loss_scale, variables)]

def mnist_model(args):
    inputs = tf.placeholder(args.dtype, shape=[None, 784])#(args.batch_size, 784))
    target = tf.placeholder(tf.float32, shape=[None, 10])#(args.batch_size, 10))
    keep_prob = tf.placeholder(args.dtype, shape=())

    def conv2d(x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def max_pool_2x2(x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1], padding='SAME')

    x_image = tf.reshape(inputs, [-1, 28, 28, 1])

    W_conv1 = tf.get_variable(name='Conv_Layer_1', shape=(5, 5, 1, 32), dtype=args.dtype, 
                              initializer=tf.truncated_normal_initializer(stddev=0.1))
    b_conv1 = tf.get_variable(name='bias_for_Conv_Layer_1', shape=(32),
                              initializer=tf.random_normal_initializer(stddev=0.1),
                              dtype=args.dtype)
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    W_conv2 = tf.get_variable(name='Conv_Layer_2', shape=(5, 5, 32, 64),
                              dtype=args.dtype,
                              initializer=tf.truncated_normal_initializer(stddev=0.1))
    b_conv2 = tf.get_variable(name='bias_for_Conv_Layer_2', shape=(64),
                              initializer=tf.random_normal_initializer(stddev=0.1),
                              dtype=args.dtype)
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    W_fc1 = tf.get_variable(name='Fully_Connected_Layer_1', shape=(7*7*64, 1024), 
                            initializer=tf.truncated_normal_initializer(stddev=0.1),
                            dtype=args.dtype)
    b_fc1 = tf.get_variable(name='bias_for_Fully_Connected_Layer_1', shape=(1024),
                            initializer=tf.random_normal_initializer(stddev=0.1),
                            dtype=args.dtype)
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    h_fc1_drop = tf.nn.dropout(h_fc1, rate=1-keep_prob)
    W_fc2 = tf.get_variable('W_fc2', shape=(1024, 10),
                            initializer=tf.truncated_normal_initializer(stddev=0.1),
                            dtype=args.dtype)
    b_fc2 = tf.get_variable('b_fc2', shape=(10),
                            initializer=tf.zeros_initializer(),
                            dtype=args.dtype)
    output = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits_v2(logits=tf.cast(output, tf.float32),
                                                   labels=target))

    tf.add_to_collection('variables_display', x_image)

    tf.add_to_collection('variables_display', W_conv1)
    tf.add_to_collection('variables_display', b_conv1)
    tf.add_to_collection('variables_display', h_conv1)
    tf.add_to_collection('variables_display', h_pool1)

    tf.add_to_collection('variables_display', W_conv2)
    tf.add_to_collection('variables_display', b_conv2)
    tf.add_to_collection('variables_display', h_conv2)
    tf.add_to_collection('variables_display', h_pool2)

    tf.add_to_collection('variables_display', W_fc1)
    tf.add_to_collection('variables_display', b_fc1)
    tf.add_to_collection('variables_display', h_pool2_flat)
    tf.add_to_collection('variables_display', h_fc1)

    tf.add_to_collection('variables_display', h_fc1_drop)
    tf.add_to_collection('variables_display', W_fc2)
    tf.add_to_collection('variables_display', b_fc2)

    tf.add_to_collection('variables_display', output)


    return inputs, output, target, loss, keep_prob



def softmax_model(args):
    """A simple softmax model."""
    inputs = tf.placeholder(args.dtype, shape=(args.batch_size, args.input_size))
    weights = tf.get_variable('weights', (args.input_size, args.output_size), args.dtype,
                              initializer=tf.random_normal_initializer())
    biases = tf.get_variable('biases', args.output_size, args.dtype,
                              initializer=tf.random_normal_initializer())
    output = tf.matmul(inputs, weights) + biases
    target = tf.placeholder(tf.float32, shape=(args.batch_size, args.output_size))
    # Note: The softmax should be computed in float32 precision
    loss = tf.losses.softmax_cross_entropy(
        target, tf.cast(output, tf.float32))

    tf.add_to_collection('variables_display', weights)
    tf.add_to_collection('variables_display', biases)
    tf.add_to_collection('variables_display', output)

    return inputs, target, loss


def compile(args):
    global inputs, target, keep_prob, loss, output, display_gradients_variables, \
        display_grads, training_step_op, init_op, accuracy
    # Create training graph
    with tf.device('/cpu:0'), \
         tf.variable_scope(
             # Note: This forces trainable variables to be stored as float32
             'fp32_storage', custom_getter=float32_variable_storage_getter):
        if args.model == 'softmax_model':
            inputs, target, loss = softmax_model(args)
        elif args.model == 'mnist_model':
            inputs, output, target, loss, keep_prob = mnist_model(args)
        else:
            print('Unimplemented')
            return False
        loss_scale_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        display_gradients_variables = tf.get_collection('variables_display')
        # Note: Loss scaling can improve numerical stability for fp16 training
        train_grads = gradients_with_loss_scaling(loss, loss_scale_variables, loss_scale)
        display_grads = gradients_display_list(loss, display_gradients_variables)

        optimizer = tf.train.MomentumOptimizer(learning_rate, momentum)
        training_step_op = optimizer.apply_gradients(zip(train_grads,
                                                         loss_scale_variables))
        if args.model == 'mnist_model':
            output2 = tf.nn.softmax(output)
            correct_prediction = tf.equal(tf.argmax(output2, 1), tf.argmax(target, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        init_op = tf.global_variables_initializer()
        return True


def train(args):
    # Run training
    sess = tf.Session()
    sess.run(init_op)
    if args.model == 'softmax_model':
        np_data = np.random.normal(size=(args.batch_size, args.input_size)).astype(
            np.float16)
        np_target = np.zeros((args.batch_size, args.output_size), dtype=np.float32)
        np_target[:, 0] = 1
    if args.model == 'mnist_model':
        mnist = input_data.read_data_sets(train_dir="{}/dataset".format(ROOT_PATH), one_hot=True)  # y labels are oh-encoded

    print('Start Training --- Step Loss Display')
    for step in range(args.epoch):
        if args.model == 'mnist_model':
            print('Enter mnist training')
            batch = mnist.train.next_batch(args.batch_size)
            loss_, _ = sess.run([loss, training_step_op],
                                feed_dict={inputs: batch[0], target: batch[1],
                                           keep_prob: args.keep_prob})
            print('step: %4i, loss = %6f' % (step + 1, loss_))

            if args.display_gradients_level == 1:
                for grad_var in display_gradients_variables:
                    print('id = ', grad_var.name)
                    print('dtype = ', grad_var.dtype)
            elif args.display_gradients_level == 2:
                gradient_list = sess.run(display_grads,
                                         feed_dict={inputs: batch[0], target: batch[1],
                                                    keep_prob: args.keep_prob})
                for k in range(len(gradient_list)):
                    print('id = ', display_gradients_variables[k].name)
                    print('dtype = ', display_gradients_variables[k].dtype)
                    print('gradients to loss = ', gradient_list[k])
                    print('\n')

        elif args.model == 'softmax_model':
            loss_, _ = sess.run([loss, training_step_op],
                              feed_dict={inputs: np_data, target: np_target})
            print('step: %4i, loss = %6f' % (step + 1, loss_))

            if args.display_gradients_level == 1:
                for grad_var in display_gradients_variables:
                    print('id = ', grad_var.name)
                    print('dtype = ', grad_var.dtype)
            elif args.display_gradients_level == 2:
                gradient_list = sess.run(display_grads, feed_dict={inputs: np_data,
                                                            target: np_target})
                for k in range(len(gradient_list)):
                    print('id = ', display_gradients_variables[k].name)
                    print('dtype = ', display_gradients_variables[k].dtype)
                    print('gradients to loss = ', gradient_list[k])
                    print('\n')
    # test
    if args.model == 'mnist_model':
        X = mnist.test.images.reshape(10, 1000, 784)
        Y = mnist.test.labels.reshape(10, 1000, 10)
        test_accuracy = np.mean([sess.run(accuracy, feed_dict={inputs: X[i], target: Y[i],
                                keep_prob: 1.0}) for i in range(10)])
        print('Validation Accuracy =%f' % test_accuracy)


if __name__ == '__main__':
    parser = argparse.ArgumentParser();
    parser.add_argument('--dtype', default='tf.float32',
                        help='training precision: 1-tf.float16, 2-tf.float16')
    parser.add_argument('--batch_size', type=int, default=64, help='training batch size')
    parser.add_argument('--epoch', type=int, default=1000, help='training stopping epoch')
    parser.add_argument('--keep_prob', type=float, default=0.5, help='dropout_rate')
    parser.add_argument('--input_size', type=int, default=224, help='input size')
    parser.add_argument('--output_size', type=int, default=10, help='output(class) size')
    parser.add_argument('--model', default='mnist_model', help='simple(softmax) model,'
                                                           'mnist model'
                                                           'resnet(resnet-50) model'
                                                           'SSD model')
    parser.add_argument('--display_gradients_level', type=int, default=0,
                        help='show gradients for debug')
    args = parser.parse_args()

    learning_rate = 0.01
    momentum = 0.9
    loss_scale = 1
    if args.dtype == 'tf.float16':
        args.dtype = tf.float16
    elif args.dtype == 'tf.float32':
        args.dtype = tf.float32
    tf.set_random_seed(1234)
    np.random.seed(4321)

    if compile(args):
       train(args)
       print('termination')
