import tensorflow as tf
import tensor_summary as ts
import ML_Flags


def layer_perceptron(X, input_shape, layer_width, layer_name=None):
    with tf.name_scope("weights"):
        W = tf.Variable(tf.random_normal(input_shape + layer_width))
        ts.variable_summaries(W)
    with tf.name_scope("bias"):
        bias = tf.Variable(tf.random_normal(layer_width))
        ts.variable_summaries(bias)
    return tf.sigmoid(tf.matmul(X, W) + bias)


def placeholders_init(mf):

    # placeHolder
    x = tf.placeholder(tf.float32, [None, mf.IMAGE_SIZE], name="X")
    y = tf.placeholder(tf.float32, [None, mf.LABEL_NUMBER], name="Y")
    y_label = tf.placeholder(tf.float32, [None], name="Y_label")

    ph_set = {"X": x, "Y": y, "Y_label": y_label}

    return ph_set