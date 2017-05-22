# ML_CIFAR-10 flags
import os
import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

TEST = "test"
TRAIN = "train"
TENSOR_BOARD = "tensor_board"
CHECKPOINT = "checkpoint"
SAVE = "save"


class MlFlags:

    def __init__(self):
        self.IMAGE_SIZE = 32 * 32 * 3
        self.LABEL_NUMBER = 10
        self.BATCH_SIZE = 1000

        self.PRINT_LOG_STEP_SIZE = 100
        self.SUMMARY_STEP_SIZE = 1000
        self.CHECK_POINT_STEP_SIZE = 1000

        self.PERCEPTRON_INPUT_SHAPE_SIZE = 1000
        self.PERCEPTRON_OUTPUT_SHAPE_SIZE = 1000

        flags.DEFINE_float("learning_rate", 0.01, """learning rate""")
        flags.DEFINE_integer("max_train_step", 1000 * 1000, """max train step""")
        flags.DEFINE_integer("max_test_step", 500, """max test step""")

        # dir flags
        # linux -> SAVE_DIR = './save/dir/type'
        # windows -> SAVE_DIR = '.\\save\\dir\\type'
        # Checkpoint dirs
        self.DIR_CHECKPOINT_TRAIN_SAVE = os.path.join(".", SAVE, CHECKPOINT, TRAIN)
        self.DIR_CHECKPOINT_TEST_SAVE = os.path.join(".", SAVE, CHECKPOINT, TEST)
        # Tensorboard dirs
        self.DIR_TENSORBOARD_TRAIN_SAVE = os.path.join(".", SAVE, TENSOR_BOARD, TRAIN)
        self.DIR_TENSORBOARD_TEST_SAVE = os.path.join(".", SAVE, TENSOR_BOARD, TEST)

        # dirs exist check & make dirs
        if not os.path.exists(self.DIR_CHECKPOINT_TRAIN_SAVE):
            os.makedirs(self.DIR_CHECKPOINT_TRAIN_SAVE)

        if not os.path.exists(self.DIR_CHECKPOINT_TEST_SAVE):
            os.makedirs(self.DIR_CHECKPOINT_TEST_SAVE)

        if not os.path.exists(self.DIR_TENSORBOARD_TRAIN_SAVE):
            os.makedirs(self.DIR_TENSORBOARD_TRAIN_SAVE)

        if not os.path.exists(self.DIR_TENSORBOARD_TEST_SAVE):
            os.makedirs(self.DIR_TENSORBOARD_TEST_SAVE)

