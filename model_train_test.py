import tensorflow as tf
import datetime

import Batch
from ML_Flags import MlFlags
import tensor_summary as ts
import neural_networks as nn

# TODO file must split
# TODO write more comment please

# TODO this code does not work need fix
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float("learning_rage", 0.01, "default learning rate")
flags.DEFINE_integer("max_train_step", 1000, "default max train step")


# TODO add argv for modeling function ex) layer width, layer number
def model_NN():

    ph_set = nn.placeholders_init(mf)

    # NN layer
    layer1 = nn.layer_perceptron(ph_set["x"], [mf.IMAGE_SIZE], [mf.PERCEPTRON_INPUT_SHAPE_SIZE], "layer_1")
    layer2 = nn.layer_perceptron(layer1, [mf.PERCEPTRON_INPUT_SHAPE_SIZE], [mf.PERCEPTRON_OUTPUT_SHAPE_SIZE], "layer_2")
    h = nn.layer_perceptron(layer2, [mf.PERCEPTRON_INPUT_SHAPE_SIZE], [mf.LABEL_NUMBER], "layer_3")

    # cost function
    with tf.name_scope("cost_function"):
        # TODO LOOK ME logistic regression does not work, for now use square error method
        # cost function square error method
        cost = tf.reduce_mean((h - ph_set["y"]) ** 2, name="cost")
        # logistic regression
        # cost = -tf.reduce_mean(Y * tf.log(h) + (1 - Y) * tf.log(1 - h), name="cost")
        ts.variable_summaries(cost)

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
        batch_acc = tf.reduce_mean(tf.cast(tf.equal(predicted_label, ph_set["y_label"]), tf.float32),
                                   name="batch_acc")
        tf.summary.scalar("accuracy", batch_acc)
        batch_hit_count = tf.reduce_sum(tf.cast(tf.equal(predicted_label, ph_set["y_label"]), tf.float32),
                                        name="batch_hit_count")
        tf.summary.scalar("hit_count", batch_hit_count)

    # merge summary
    summary = tf.summary.merge_all()

    # init op
    init_op = tf.global_variables_initializer()

    # save tensor
    tensor_set = {"X": ph_set["x"],
                  "Y": ph_set["y"],
                  "Y_label": ph_set["y_label"],
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
    X = tf.placeholder(tf.float32, [None, mf.IMAGE_SIZE], name="X")
    Y = tf.placeholder(tf.float32, [None, mf.LABEL_NUMBER], name="Y")
    Y_label = tf.placeholder(tf.float32, [None], name="Y_label")

    # NN layer
    layer1 = nn.layer_perceptron(X, [mf.IMAGE_SIZE], [mf.PERCEPTRON_OUTPUT_SHAPE_SIZE], "softmax_L1")
    layer2 = nn.layer_perceptron(layer1, [mf.PERCEPTRON_INPUT_SHAPE_SIZE], [mf.PERCEPTRON_OUTPUT_SHAPE_SIZE], "softmax_L2")
    layer3 = nn.layer_perceptron(layer2, [mf.PERCEPTRON_INPUT_SHAPE_SIZE], [mf.LABEL_NUMBER], "softmax_L3")

    # softmax layer
    with tf.name_scope("softmax_func"):
        W_softmax = tf.Variable(tf.zeros([10, 10]), name="W_softmax")
        h = tf.nn.softmax(tf.matmul(layer3, W_softmax), name="h")
        ts.variable_summaries(h)

    # cross entropy function
    with tf.name_scope("cross_entropy"):
        cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(h), reduction_indices=1))
        ts.variable_summaries(cost)

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
        train_writer = tf.summary.FileWriter(mf.DIR_TENSORBOARD_TRAIN_SAVE, sess.graph)
        test_writer = tf.summary.FileWriter(mf.DIR_TENSORBOARD_TRAIN_SAVE)

        # train step
        print("Train Start...")
        train_batch_config = Batch.Config()
        train_batch = Batch.Batch(train_batch_config)
        sess.run(model["init_op"])

        for step in range(mf.MAX_TRAIN_STEP + 1):
            key_list = [Batch.INPUT_DATA, Batch.OUTPUT_LABEL, Batch.OUTPUT_DATA]

            data = train_batch.next_batch(mf.BATCH_SIZE, key_list)

            feed_dict = {model["X"]: data[Batch.INPUT_DATA],
                         model["Y"]: data[Batch.OUTPUT_DATA],
                         model["Y_label"]: data[Batch.OUTPUT_LABEL]}
            sess.run(model["train_op"], feed_dict)

            # print log
            if step % mf.PRINT_LOG_STEP_SIZE == 0:
                summary_train, _acc, _cost = sess.run([model["summary"], model["batch_acc"], model["cost"]],
                                                      feed_dict=feed_dict)
                print(datetime.datetime.utcnow(), "train step: %d" % step
                      , "batch_acc:", _acc, "cost:", _cost)

            # checkpoint
            if step % mf.CHECK_POINT_STEP_SIZE == 0:
                saver.save(sess, mf.DIR_CHECKPOINT_TRAIN_SAVE, global_step=step)

            # summary tensorboard
            if step % mf.SUMMARY_STEP_SIZE:
                train_writer.add_summary(summary=summary_train, global_step=step)

        # test step
        print("Test Start...")
        test_batch_config = Batch.Config()
        test_batch = Batch.Batch(test_batch_config)

        total_acc = 0.
        for step in range(mf.MAX_TEST_STEP + 1):
            key_list = [Batch.INPUT_DATA, Batch.OUTPUT_LABEL, Batch.OUTPUT_DATA]

            data = test_batch.next_batch(mf.BATCH_SIZE, key_list)

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

            if step % mf.PRINT_LOG_STEP_SIZE == 0:
                summary_test, _acc = sess.run([model["summary"], model["batch_acc"]], feed_dict=feed_dict)
                # print(datetime.datetime.utcnow(), "test step: %d" % step
                #       , "batch_acc: ", _acc)

            test_writer.add_summary(summary=summary_test, global_step=step)

        print("test complete: total acc =", total_acc / (mf.MAX_TEST_STEP + 1))
    return


if __name__ == '__main__':
    mf = MlFlags()

    print("Neural Networks")
    train_and_model(model_NN())
    # print("NN softmax")
    # train_and_model(model_NN_softmax())
    pass
