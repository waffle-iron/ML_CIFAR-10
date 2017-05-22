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
    x = tf.placeholder(tf.float32, [None, mf.IMAGE_SIZE], name="x")
    y = tf.placeholder(tf.float32, [None, mf.LABEL_NUMBER], name="y")
    y_label = tf.placeholder(tf.float32, [None], name="y_label")

    ph_set = {"x": x, "y": y, "y_label": y_label}

    return ph_set