import tensorflow as tf
import datetime

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
MAX_TRAIN_STEP = 100000
BATCH_SIZE = 1000
PRINT_STEP_SIZE = 100


# TODO add argv for modeling function ex) layer width, layer number
def model_NN():
    # placeHolder
    X = tf.placeholder(tf.float32, [None, IMAGE_SIZE], name="X")
    Y = tf.placeholder(tf.float32, [None, LABEL_NUMBER], name="Y")
    Y_label = tf.placeholder(tf.float32, [None], name="Y_label")

    # perceptron_layer
    def perceptron_layer(X, input_shape, layer_width):
        W = tf.Variable(tf.random_normal(input_shape + layer_width))
        bias = tf.Variable(tf.random_normal(layer_width))
        return tf.sigmoid(tf.matmul(X, W) + bias)

    # NN layer
    layer1 = perceptron_layer(X, [IMAGE_SIZE], [1000])
    layer2 = perceptron_layer(layer1, [1000], [1000])
    h = perceptron_layer(layer2, [1000], [LABEL_NUMBER])

    # cost function square error method
    # cost = tf.reduce_mean((h - Y) ** 2, name="cost")

    # logistic regression
    cost = -tf.reduce_mean(Y * tf.log(h) + (1 - Y) * tf.log(1 - h), name="cost")

    # train op
    train_op = tf.train.GradientDescentOptimizer(0.1).minimize(cost)

    # predicted label and batch batch_acc
    predicted_label = tf.cast(tf.arg_max(h, 1, name="predicted_label"), tf.float32)
    batch_acc = tf.reduce_mean(tf.cast(tf.equal(predicted_label, Y_label), tf.float32),
                               name="batch_acc")
    batch_hit_count = tf.reduce_sum(tf.cast(tf.equal(predicted_label, Y_label), tf.float32),
                                    name="batch_hit_count")

    init_op = tf.initialize_all_variables()

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
                  }

    return tensor_set


# TODO add argv for modeling function ex) layer width, layer number
def model_NN_softmax():
    # placeHolder
    X = tf.placeholder(tf.float32, [None, IMAGE_SIZE], name="X")
    Y = tf.placeholder(tf.float32, [None, LABEL_NUMBER], name="Y")
    Y_label = tf.placeholder(tf.float32, [None], name="Y_label")

    # perceptron_layer
    def perceptron_layer(X, input_shape, layer_width):
        W = tf.Variable(tf.random_normal(input_shape + layer_width))
        bias = tf.Variable(tf.random_normal(layer_width))
        return tf.sigmoid(tf.matmul(X, W) + bias)

    # NN layer
    layer1 = perceptron_layer(X, [IMAGE_SIZE], [1000])
    layer2 = perceptron_layer(layer1, [1000], [1000])
    layer3 = perceptron_layer(layer2, [1000], [LABEL_NUMBER])

    # softmax layer
    W_softmax = tf.Variable(tf.zeros([10, 10]), name="W_softmax")
    h = tf.nn.softmax(tf.matmul(layer3, W_softmax), name="h")

    # cross entropy function
    cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(h), reduction_indices=1))

    # train op
    learning_rate = 0.01
    train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

    # predicted label and batch batch_acc
    predicted_label = tf.cast(tf.arg_max(h, 1, name="predicted_label"), tf.float32)
    batch_acc = tf.reduce_mean(tf.cast(tf.equal(predicted_label, Y_label), tf.float32),
                               name="batch_acc")
    batch_hit_count = tf.reduce_sum(tf.cast(tf.equal(predicted_label, Y_label), tf.float32),
                                    name="batch_hit_count")

    init_op = tf.initialize_all_variables()

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
                  "init_op": init_op
                  }
    return tensor_set


# TODO add check point function
# TODO add tensorboard
# TODO split function train_model and test_model
def train_and_test_model(model):
    with tf.Session() as sess, tf.device("/GPU:0"):
        train_batch_config = Batch.Config()
        train_batch = Batch.Batch(train_batch_config)
        sess.run(model["init_op"])

        # train step
        for step in range(MAX_TRAIN_STEP):
            key_list = [Batch.INPUT_DATA, Batch.OUTPUT_LABEL, Batch.OUTPUT_DATA]

            data = train_batch.next_batch(BATCH_SIZE, key_list)

            feed_dict = {model["X"]: data[Batch.INPUT_DATA],
                         model["Y"]: data[Batch.OUTPUT_DATA],
                         model["Y_label"]: data[Batch.OUTPUT_LABEL]}
            sess.run(model["train_op"], feed_dict)

            if step % PRINT_STEP_SIZE == 0:
                _acc, _cost = sess.run([model["batch_acc"], model["cost"]], feed_dict=feed_dict)
                print(datetime.datetime.utcnow(), "train step :%d" % step
                      , "batch_acc :", _acc, "cost :", _cost)


                # label = sess.run(predicted_label, feed_dict=feed_dict)
                # print("predict :", _out)
                # print("Y_data :", data[b'labels'])
                # print()

                # # test step
                # # for now use train data
                # train_batch.reset_batch_index()
                #
                # total_acc = 0
                # count = 0
                # for step in range(int(test_size / BATCH_SIZE)):
                #     data = train_batch.next_batch(BATCH_SIZE)
                #
                #     feed_dict = {X: data[b'data'], Y: Y_data}
                #
                #     _acc, _cost = sess.run([batch_acc, cost], feed_dict=feed_dict)
                #     print("test batch_acc :", _acc, "cost :", _cost)
                #     total_acc += _acc
                #     count += 1
                # print(total_acc, count)
                # print("total batch_acc :", total_acc / count)
    return


if __name__ == '__main__':
    train_and_test_model(model_NN())
    train_and_test_model(model_NN_softmax())
    pass
