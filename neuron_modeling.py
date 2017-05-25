import tensorflow as tf
import tensor_summary as ts

flags = tf.app.flags
FLAGS = flags.FLAGS


def placeholders_init():
    # placeHolder
    x = tf.placeholder(tf.float32, [None, FLAGS.image_size], name="X")
    y = tf.placeholder(tf.float32, [None, FLAGS.label_number], name="Y")
    y_label = tf.placeholder(tf.float32, [None], name="Y_label")

    ph_set = {"X": x, "Y": y, "Y_label": y_label}

    return ph_set


# for NN
def layer_perceptron(X, input_shape, layer_width, layer_name=None, activation=tf.sigmoid):
    with tf.name_scope("weights"):
        W = tf.Variable(tf.random_normal(input_shape + layer_width))
        ts.variable_summaries(W)
    with tf.name_scope("bias"):
        bias = tf.Variable(tf.random_normal(layer_width))
        ts.variable_summaries(bias)
    return activation(tf.matmul(X, W) + bias)


# TODO write&test CNN code #9
# for CNN
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


# [conv->relu->pooling] --> conv->relu --> affine->relu --> affine->softmax
def layer_convolution(x, input_shape, layer_width, name=None, activation=tf.nn.relu):
    # weights & bias
    with tf.name_scope("conv_weights"):
        W = tf.Variable(tf.random_normal(input_shape + layer_width))
    with tf.name_scope("conv_bias"):
        bias = tf.Variable(tf.random_normal(layer_width))

    # convolution
    with tf.name_scope("conv_conv2d"):
        h = conv2d(x, W) + bias

    # activation function(relu)
    with tf.name_scope("conv_relu"):
        h_conv = activation(h)

    # pooling
    with tf.name_scope("conv_pooling"):
        h_pool = max_pool_2x2(h_conv)

    return h_pool
