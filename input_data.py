import os
from PIL import Image
from progressbar import Percentage, Bar, SimpleProgress, ProgressBar

BATCH_FILE_NAME_FORMAT = "data_batch_%d"
FOLDER_NAME = "cifar-10-batches-py"
BATCHES_META_FILE_NAME = "batches.meta"
TEST_BATCH_FILE_NAME = "test_batch"

# progressBar widgets def
widgets = ['Running: ', Percentage(), ' ',
           Bar(marker='#', left='[', right=']'),
           ' ', SimpleProgress()]


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def show_image(raw_data, img):
    r, g, b = raw_data[:1024], raw_data[1024:1024 * 2], raw_data[1024 * 2:1024 * 3]
    rgb_list = list(zip(r, g, b))
    for x in range(32):
        for y in range(32):
            img.putpixel((x, y), rgb_list[y * 32 + x])
    img.show()


def open_meta_file():
    print("Open meta file...")
    meta = unpickle(os.path.join(FOLDER_NAME, BATCHES_META_FILE_NAME))
    label_names = meta[b'label_names']

    for i in meta:
        print("labels = %s" % label_names)

    print()
    return meta


def open_train_batch_files():
    print("Open train batch files...")

    data_file = []
    for batch_number in range(1, 6):
        # print("\nbatch:", batch_number)
        batch = unpickle(os.path.join(FOLDER_NAME, BATCH_FILE_NAME_FORMAT % batch_number))

        # batch data inputs
        batch_label = batch[b'batch_label']
        data = batch[b'data']
        file_names = batch[b'filenames']
        labels = batch[b'labels']

        # image gen
        img_mode = 'RGB'
        img_size = (32, 32)
        img = Image.new(img_mode, img_size)

        p_bar = ProgressBar(widgets=widgets, maxval=len(file_names)).start()
        for i in range(len(file_names)):
            # print("number: %4d, name: %s, label: %s" % (i, file_names[i], labels[i]))
            # show_image(data[i], img)
            p_bar.update(i)

        data_file.append([batch_label, data, labels, file_names])
        p_bar.finish()

    print()
    return data_file


# test batch
def open_image_batch():
    print("Open test batch file...")
    test_batch = unpickle(os.path.join(FOLDER_NAME, TEST_BATCH_FILE_NAME))

    batch_label = test_batch[b'batch_label']
    data = test_batch[b'data']
    file_names = test_batch[b'filenames']
    labels = test_batch[b'labels']

    test_data = [batch_label, data, file_names, labels]

    p_bar = ProgressBar(widgets=widgets, maxval=len(file_names)).start()
    for i in range(len(file_names)):
        p_bar.update(i)
        # print("number: %4d, name: %s, label: %s" % (i, file_names[i], labels[i]))
    p_bar.finish()

    print()
    return test_data


# config = {"dir_list": [], "data_name_list": []}


class Batch:
    def __init__(self, config):
        self.config = config
        data = self.load_data()
        self.batch = self.load_batch(data)
        self.batch_index = 0
        pass

    def load_data(self, config=None):
        if config is None:
            dir_list = self.config["dir_list"]
        else:
            dir_list = config["dir_list"]

        data = []
        for dir_ in dir_list:
            data += [unpickle(dir_)]
        return data

    # TODO
    def load_batch(self, data):
        batch = {}
        for data_name in self.config["data_name_list"]:

        return batch

    # TODO
    def next_batch(self, size, symbolList=None):
        if symbolList is None:
            elementList = self.config["elementList"]
        else:
            elementList = symbolList

        batch = {}
        for el in elementList:
            batch[el] = self.get_element(el, size)
        return batch

    # TODO
    def get_element(self, name, size):
        elements = []
        return elements

    pass


if __name__ == '__main__':
    # open_meta_file()
    # open_train_batch_files()
    # open_image_batch()
    generate_train_batch_data()
