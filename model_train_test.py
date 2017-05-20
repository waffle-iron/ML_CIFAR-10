import tensorflow as tf
import datetime
import os

import Batch

# TODO file must split
# TODO write more comment please

# TODO this code does not work need fix
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float("learning_rage", 0.01, "default learning rate")
flags.DEFINE_integer("max_train_step", 1000, "default max train step")

# TODO refactoring constant
IMAGE_SIZE = 32 * 32 * 3
LABEL_NUMBER = 10
BATCH_SIZE = 1000
# TODO watch me this way is better
MAX_TRAIN_STEP = 1000*1000
MAX_TEST_STEP = 500
PRINT_LOG_STEP_SIZE = 100
SUMMARY_STEP_SIZE = 1000
CHECK_POINT_STEP_SIZE = 1000

# linux
# DIR_CHECKPOINT_TRAIN_SAVE = './checkpoint/train/save'
# windows
# DIR_CHECKPOINT_TRAIN_SAVE = '.\\checkpoint\\train\\save'
DIR_CHECKPOINT_TRAIN_SAVE = os.path.join(".", "checkpoint", "train", "save")

# linux
# DIR_CHECKPOINT_TEST_SAVE = './checkpoint/test/save'
# windows
# DIR_CHECKPOINT_TEST_SAVE = '.\\checkpoint\\test\\save'
DIR_CHECKPOINT_TEST_SAVE = os.path.join(".", "checkpoint", "test", "save")


def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


# TODO add argv for modeling function ex) layer width, layer number
def model_NN():
    # placeHolder
    X = tf.placeholder(tf.float32, [None, IMAGE_SIZE], name="X")
    Y = tf.placeholder(tf.float32, [None, LABEL_NUMBER], name="Y")
    Y_label = tf.placeholder(tf.float32, [None], name="Y_label")

    # perceptron_layer
    def perceptron_layer(X, input_shape, layer_width, layer_name=None):
        with tf.name_scope(layer_name):
            with tf.name_scope("weights"):
                W = tf.Variable(tf.random_normal(input_shape + layer_width))
                variable_summaries(W)
            with tf.name_scope("bias"):
                bias = tf.Variable(tf.random_normal(layer_width))
                variable_summaries(bias)
            with tf.name_scope("sigmoid_w_mul_x_plus_b"):
                preactivate = tf.sigmoid(tf.matmul(X, W) + bias)
                # tf.summary.histogram(preactivate)
                variable_summaries(preactivate)
            return preactivate

    # NN layer
    layer1 = perceptron_layer(X, [IMAGE_SIZE], [1000], "layer_1")
    layer2 = perceptron_layer(layer1, [1000], [1000], "layer_2")
    h = perceptron_layer(layer2, [1000], [LABEL_NUMBER], "layer_3")

    # cost function
    with tf.name_scope("cost_function"):
        # TODO LOOK ME logistic regression does not work, for now use square error method
        # cost function square error method
        cost = tf.reduce_mean((h - Y) ** 2, name="cost")
        # logistic regression
        # cost = -tf.reduce_mean(Y * tf.log(h) + (1 - Y) * tf.log(1 - h), name="cost")
        variable_summaries(cost)

    # train op
    learning_rate = 0.01
    with tf.name_scope("train_op"):
        train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
        # if train_op is None:
        #     pass
        # tf.summary.histogram(train_op)

    # predicted label and batch batch_acc
    predicted_label = tf.cast(tf.arg_max(h, 1, name="predicted_label"), tf.float32)
    with tf.name_scope("NN_batch_acc"):
        batch_acc = tf.reduce_mean(tf.cast(tf.equal(predicted_label, Y_label), tf.float32),
                                   name="batch_acc")
        tf.summary.scalar("accuracy", batch_acc)
        batch_hit_count = tf.reduce_sum(tf.cast(tf.equal(predicted_label, Y_label), tf.float32),
                                        name="batch_hit_count")
        tf.summary.scalar("hit_count", batch_hit_count)

    # merge summary
    summary = tf.summary.merge_all()

    # init op
    init_op = tf.global_variables_initializer()

    # save tensor
    tensor_set = {"X": X,
                  "Y": Y,
                  "Y_label": Y_label,
                  "layer1": layer1,
                  "layer2": layer2,
                  "h": h,
                  "cost": cost,
                  "train_op": train_op,
                  "predicted_label": predicted_label,
                  "batch_acc": batch_acc,
                  "batch_hit_count ": batch_hit_count,
                  "init_op": init_op,
                  "summary": summary,
                  }

    return tensor_set


# TODO add argv for modeling function ex) layer width, layer number
def model_NN_softmax():
    # placeHolder
    X = tf.placeholder(tf.float32, [None, IMAGE_SIZE], name="X")
    Y = tf.placeholder(tf.float32, [None, LABEL_NUMBER], name="Y")
    Y_label = tf.placeholder(tf.float32, [None], name="Y_label")

    # perceptron_layer
    def perceptron_layer(X, input_shape, layer_width, layer_name=None):
        with tf.name_scope("weights"):
            W = tf.Variable(tf.random_normal(input_shape + layer_width))
            variable_summaries(W)
        with tf.name_scope("bias"):
            bias = tf.Variable(tf.random_normal(layer_width))
            variable_summaries(bias)
        return tf.sigmoid(tf.matmul(X, W) + bias)

    # NN layer
    layer1 = perceptron_layer(X, [IMAGE_SIZE], [1000], "softmax_L1")
    layer2 = perceptron_layer(layer1, [1000], [1000], "softmax_L2")
    layer3 = perceptron_layer(layer2, [1000], [LABEL_NUMBER], "softmax_L3")

    # softmax layer
    with tf.name_scope("softmax_func"):
        W_softmax = tf.Variable(tf.zeros([10, 10]), name="W_softmax")
        h = tf.nn.softmax(tf.matmul(layer3, W_softmax), name="h")
        variable_summaries(h)

    # cross entropy function
    with tf.name_scope("cross_entropy"):
        cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(h), reduction_indices=1))
        variable_summaries(cost)

    # train op
    learning_rate = 0.01
    train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

    # predicted label and batch batch_acc
    predicted_label = tf.cast(tf.arg_max(h, 1, name="predicted_label"), tf.float32)
    with tf.name_scope("softmax_batch_acc"):
        with tf.name_scope("accuracy"):
            batch_acc = tf.reduce_mean(tf.cast(tf.equal(predicted_label, Y_label), tf.float32),
                                       name="batch_acc")
            tf.summary.scalar("accuracy", batch_acc)
        with tf.name_scope("batch_hit_count"):
            batch_hit_count = tf.reduce_sum(tf.cast(tf.equal(predicted_label, Y_label), tf.float32),
                                            name="batch_hit_count")
            tf.summary.scalar("hit_count", batch_hit_count)

    # merge summary
    summary = tf.summary.merge_all()

    # init op
    init_op = tf.global_variables_initializer()

    # save tensor
    tensor_set = {"X": X,
                  "Y": Y,
                  "Y_label": Y_label,
                  "layer1": layer1,
                  "layer2": layer2,
                  "layer3": layer3,
                  "W_softmax ": W_softmax,
                  "h": h,
                  "cost": cost,
                  "train_op": train_op,
                  "predicted_label": predicted_label,
                  "batch_acc": batch_acc,
                  "batch_hit_count ": batch_hit_count,
                  "init_op": init_op,
                  "summary": summary,
                  }
    return tensor_set


# TODO split function train_model and test_model
def train_and_model(model):
    # TODO LOOK ME this make show all tensor belong cpu or gpu
    # with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess, tf.device("/CPU:0"):
    with tf.Session() as sess:

        # checkpoint val
        saver = tf.train.Saver()

        # tensorboard
        train_writer = tf.summary.FileWriter("./train_log", sess.graph)
        test_writer = tf.summary.FileWriter("./test_log")

        # train step
        print("Train Start...")
        train_batch_config = Batch.Config()
        train_batch = Batch.Batch(train_batch_config)
        sess.run(model["init_op"])

        for step in range(MAX_TRAIN_STEP + 1):
            key_list = [Batch.INPUT_DATA, Batch.OUTPUT_LABEL, Batch.OUTPUT_DATA]

            data = train_batch.next_batch(BATCH_SIZE, key_list)

            feed_dict = {model["X"]: data[Batch.INPUT_DATA],
                         model["Y"]: data[Batch.OUTPUT_DATA],
                         model["Y_label"]: data[Batch.OUTPUT_LABEL]}
            sess.run(model["train_op"], feed_dict)

            # print log
            if step % PRINT_LOG_STEP_SIZE == 0:
                summary_train, _acc, _cost = sess.run([model["summary"], model["batch_acc"], model["cost"]],
                                                      feed_dict=feed_dict)
                print(datetime.datetime.utcnow(), "train step: %d" % step
                      , "batch_acc:", _acc, "cost:", _cost)

            if step % CHECK_POINT_STEP_SIZE == 0:
                # checkpoint
                saver.save(sess, DIR_CHECKPOINT_TRAIN_SAVE, global_step=step)

            if step % SUMMARY_STEP_SIZE:
                # summary tensorboard
                train_writer.add_summary(summary=summary_train, global_step=step)

        # test step
        print("Test Start...")
        test_batch_config = Batch.Config()
        test_batch = Batch.Batch(test_batch_config)

        total_acc = 0.
        for step in range(MAX_TEST_STEP + 1):
            key_list = [Batch.INPUT_DATA, Batch.OUTPUT_LABEL, Batch.OUTPUT_DATA]

            data = test_batch.next_batch(BATCH_SIZE, key_list)

            feed_dict = {model["X"]: data[Batch.INPUT_DATA],
                         model["Y"]: data[Batch.OUTPUT_DATA],
                         model["Y_label"]: data[Batch.OUTPUT_LABEL]}
            # print("input:", data[Batch.INPUT_DATA])
            # print("output:", data[Batch.OUTPUT_DATA])
            # print("label:", data[Batch.OUTPUT_LABEL])

            summary_test, _acc = sess.run([model["summary"], model["batch_acc"]], feed_dict=feed_dict)
            print(datetime.datetime.utcnow(), "test step: %d" % step
                  , "batch_acc: ", _acc)
            total_acc += _acc

            if step % PRINT_LOG_STEP_SIZE == 0:
                summary_test, _acc = sess.run([model["summary"], model["batch_acc"]], feed_dict=feed_dict)
                # print(datetime.datetime.utcnow(), "test step: %d" % step
                #       , "batch_acc: ", _acc)

            # if step % SUMMARY_STEP_SIZE == 0:
            #     test_writer.add_summary(summary=summary_test, global_step=step)
            test_writer.add_summary(summary=summary_test, global_step=step)

            # if step % CHECK_POINT_STEP_SIZE == 0:
            #     saver.save(sess, DIR_CHECKPOINT_TEST_SAVE, global_step=step)


        print("test complet: total acc =", total_acc / (MAX_TEST_STEP + 1))
    return


if __name__ == '__main__':
    # print("Neural Networks")
    # train_and_model(model_NN())
    # print("NN softmax")
    # train_and_model(model_NN_softmax())
    pass
