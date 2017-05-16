import pickle
import numpy as np
import os

DEFAULT_DIR_LIST = "dir_list"
DEFAULT_NAME_KEY_LIST = "key_list"
DEFAULT_BATCH_FILE_NUMBER = 5
DEFAULT_DATA_BATCH_FILE_FORMAT = "data_batch_%d"
DEFAULT_BATCH_FOLDER = ".\\cifar-10-batches-py"

BATCH_FILE_LABEL = b'batch_label'
INPUT_DATA = b'data'
INPUT_FILE_NAME = b'filenames'
OUTPUT_LABEL = b'labels'
OUTPUT_DATA = "output_list"


# TODO need refactoring
class Config:
    KEY_LIST = DEFAULT_NAME_KEY_LIST
    DIR_LIST = DEFAULT_DIR_LIST
    FOLDER_NAME = DEFAULT_BATCH_FOLDER
    BATCH_FILE_NAME_FORMAT = DEFAULT_DATA_BATCH_FILE_FORMAT
    BATCH_FILE_NUMBER = DEFAULT_BATCH_FILE_NUMBER

    def __init__(self):
        # init config
        self.config = dict()

        # init dir_list
        self.config[DEFAULT_DIR_LIST] = []
        for i in range(1, self.BATCH_FILE_NUMBER + 1):
            self.config[DEFAULT_DIR_LIST] \
                += [os.path.join(self.FOLDER_NAME, self.BATCH_FILE_NAME_FORMAT % i)]

        self.config[self.KEY_LIST] = [
            BATCH_FILE_LABEL,
            INPUT_DATA,
            INPUT_FILE_NAME,
            OUTPUT_LABEL,
            OUTPUT_DATA,
        ]


class Batch:
    def __init__(self, config):
        self.config = config.config
        self.batch = {}
        self.load_batch()
        self.batch_index = 0
        self.batch_size = len(self.batch[INPUT_DATA])
        self.__generate_y_data()
        pass

    @staticmethod
    def unpickle(file):
        import pickle
        with open(file, 'rb') as fo:
            unpickled = pickle.load(fo, encoding='bytes')
        return unpickled

    # TODO need refactoring
    def __append(self, a, b, key):
        if key == BATCH_FILE_LABEL:
            return a + b
        elif key == INPUT_DATA:
            return np.concatenate((a, b))
        elif key == INPUT_FILE_NAME:
            return a + b
        elif key == OUTPUT_LABEL:
            return a + b

    def __assign(self, a, key):
        if key == BATCH_FILE_LABEL:
            return [a]
        elif key == INPUT_DATA:
            return a
        elif key == INPUT_FILE_NAME:
            return a
        elif key == OUTPUT_LABEL:
            return a

    def load_batch(self):
        for dir_ in self.config["dir_list"]:
            data = self.unpickle(dir_)
            for key in data:
                if key in self.batch:
                    self.batch[key] = self.__append(self.batch[key], data[key], key)
                else:
                    self.batch[key] = self.__assign(data[key], key)
        pass

    def next_batch(self, size, key_list=None):
        if key_list is None:
            key_list = self.config[Config.KEY_LIST]

        batch = {}
        for key in key_list:
            part = self.batch[key][self.batch_index:self.batch_index + size]
            while len(part) < size:
                part += self.batch[key]
            batch[key] = part[:size]

        self.batch_index = (self.batch_index + size) % self.batch_size
        return batch

    def __generate_y_data(self):
        self.batch[OUTPUT_DATA] = []
        for idx in self.batch[OUTPUT_LABEL]:
            temp = [0 for _ in range(10)]
            temp[idx] = 1
            self.batch[OUTPUT_DATA] += [temp]
        pass

    def reset_batch_index(self):
        self.batch_index = 0


def test_train_Batch():
    batch_config = Config()
    # print(batch_config.config["dir_list"])

    b = Batch(batch_config)

    size = 3
    for _ in range(1):
        key_list = [b'data', b'labels', "output_list"]
        batch = b.next_batch(size, key_list)
        for key in batch:
            print(key)
            for i in batch[key]:
                print(i)
                print()
    return

# TODO implement test_batch
# TODO test test_batch
def test_test_Batch():
    return

if __name__ == '__main__':
    test_train_Batch()
    pass
