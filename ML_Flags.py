# ML_CIFAR-10 flags
from os import path
import os

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

        self.MAX_TRAIN_STEP = 1000 * 1000
        self.MAX_TEST_STEP = 500

        self.PRINT_LOG_STEP_SIZE = 100
        self.SUMMARY_STEP_SIZE = 1000
        self.CHECK_POINT_STEP_SIZE = 1000

        self.PERCEPTRON_INPUT_SHAPE_SIZE = 1000
        self.PERCEPTRON_OUTPUT_SHAPE_SIZE = 1000

        self.LEARNING_RATE = 0.01

        # dir flags
        # linux -> SAVE_DIR = './save/dir/type'
        # windows -> SAVE_DIR = '.\\save\\dir\\type'
        # Checkpoint dirs
        self.DIR_CHECKPOINT_TRAIN_SAVE = path.join(".", SAVE, CHECKPOINT, TRAIN)
        self.DIR_CHECKPOINT_TEST_SAVE = path.join(".", SAVE, CHECKPOINT, TEST)
        # Tensorboard dirs
        self.DIR_TENSORBOARD_TRAIN_SAVE = path.join(".", SAVE, TENSOR_BOARD, TRAIN)
        self.DIR_TENSORBOARD_TEST_SAVE = path.join(".", SAVE, TENSOR_BOARD, TEST)

        # dirs exist check & make dirs
        if not path.exists(self.DIR_CHECKPOINT_TRAIN_SAVE):
            os.makedirs(self.DIR_CHECKPOINT_TRAIN_SAVE)

        if not path.exists(self.DIR_CHECKPOINT_TEST_SAVE):
            os.makedirs(self.DIR_CHECKPOINT_TEST_SAVE)

        if not path.exists(self.DIR_TENSORBOARD_TRAIN_SAVE):
            os.makedirs(self.DIR_TENSORBOARD_TRAIN_SAVE)

        if not path.exists(self.DIR_TENSORBOARD_TEST_SAVE):
            os.makedirs(self.DIR_TENSORBOARD_TEST_SAVE)

