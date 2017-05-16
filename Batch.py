import pickle
import numpy as np
import os


# TODO

# config = {"dir_list": [], "data_name_list": []}
# batch_label = batch[b'batch_label']
# data = batch[b'data']
# file_names = batch[b'filenames']
# labels = batch[b'labels']


class Config:
    def __init__(self):
        self.config = dict()
        self.FOLDER_NAME = ".\\cifar-10-batches-py"
        self.BATCH_FILE_NAME_FORMAT = "data_batch_%d"

        self.config["dir_list"] = []
        for i in range(1, 5 + 1):
            self.config["dir_list"] \
                += [os.path.join(self.FOLDER_NAME, self.BATCH_FILE_NAME_FORMAT % i)]

        pass


class Batch:
    def __init__(self, config):
        self.config = config.config
        self.batch = {}
        self.load_batch()
        self.batch_index = 0
        self.batch_size = len(self.batch[b'data'])

        pass

    @staticmethod
    def unpickle(file):
        import pickle
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict

    def __append(self, a, b, key):
        if key == b'batch_label':
            return a + b
        elif key == b'data':
            return np.concatenate((a, b))
        elif key == b'filenames':
            return a + b
        elif key == b'labels':
            return a + b

    # TODO
    def load_batch(self):
        print(self.config["dir_list"])
        for dir_ in self.config["dir_list"]:
            data = self.unpickle(dir_)
            for key in data:
                if key in self.batch:
                    self.batch[key] = self.__append(self.batch[key], data[key], key)
                else:
                    self.batch[key] = data[key]
        pass

    # TODO
    def next_batch(self, size, key_list=None):
        if key_list is None:
            key_list = self.config.default_key_list

        batch = {}
        for key in key_list:
            batch[key] = self.__next_batch(key, size)

        self.batch_index = (self.batch_index + size) % self.batch_size
        return batch

    def __next_batch(self, key, size):
        print(key)
        print(self.batch[key])

        ret = self.batch[key][self.batch_index:self.batch_index + size]
        while len(ret) < size:
            ret += self.batch[key]
        return ret[:size]


pass

if __name__ == '__main__':
    batch_config = Config()
    print(batch_config.config["dir_list"])

    b = Batch(batch_config)

    size = 3
    for _ in range(10):
        key_list = [b'data', b'labels']
        batch = b.next_batch(size, key_list)
        for key in batch:
            print(key)
            for i in batch[key]:
                print(i)
