import tensorflow as tf
from input_data import *

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float("learning_rage", 0.01)
flags.DEFINE_integer("max_train_step", 1000)
BATCH_SIZE = 100
PRINT_STEP_SIZE = 10

# TODO
class Batch:
    def __init__(self, data=None):
        self.X = []
        self.Y_index = []
        self.Y_list = []
        self.iter_index = 0

        if data != None:
            generate_train_batch_data()
            pass

        pass

    def Y_index_To_Y_list(self, Y_index, Y_max):
        Y_list = []
        for index in range(len(Y_index)):
            l = [0 for _ in range(Y_max + 1)]
            l[index] = 1
            Y_list += [l]

        return Y_list

    def nextBatch(size):
        data = {}

        return data

    pass


def nextBatch(size, ):
    return


def testBatch():
    return


# TODO need test
def NN_model():
    image_size = 32 * 32 * 3
    label_size = 10

    # X = [batch_size, imagesize]
    # Y = [batch_size, label_size]
    X = tf.placeholder(tf.float32, [None, image_size], name="X")
    Y_label_list = tf.placeholder(tf.float32, [None, label_size], name="Y_label_list")

    def Perceptron_layer(X, input_shape, layer_width):
        W = tf.Variable(tf.random_normal(input_shape + layer_width))
        bias = tf.Variable(tf.random_normal(input_shape))
        layer_output = tf.sigmoid(tf.matmul(W, X) + bias, )
        return layer_output

    layer1 = Perceptron_layer(X, [image_size], [100])
    layer2 = Perceptron_layer(layer1, [100], [100])
    h = Perceptron_layer(layer2, [100], [label_size])

    cost = tf.reduce_mean((h - Y_label_list) ** 2, name="cost")
    train_op = tf.train.GradientDescentOptimizer(FLAGS.learning_rage).minimize(cost)

    predicted_output = tf.arg_max(h, 1, name="predicted_output")
    train_output = tf.arg_max(Y_label_list, 1, name="train_output")
    acc = tf.reduce_mean(tf.cast(tf.equal(predicted_output, train_output), tf.float32), name="acc")

    init_op = tf.initialize_all_variables()
    with tf.Session() as sess:
        sess.run(init_op)

        # train step
        for i in range(FLAGS.MAX_TRAIN_STEP):
            data = nextBatch(BATCH_SIZE)
            sess.run(train_op, feed_dict=data)

            if i % PRINT_STEP_SIZE == 0:
                _acc, _cost = sess.run([acc, cost], feed_dict=data)
                print(i, "acc :", _acc, "cost :", cost)

                # test step
                # total_acc = 0
                # test_size = 1
                # for i in range(test_size / BATCH_SIZE):
                #     data = testBatch(BATCH_SIZE)
                #     _acc, _cost = sess.run([acc, cost], feed_dict=data)
                #     print(i, "acc :", _acc, "cost :", cost)

    return



if __name__ == '__main__':
    pass
