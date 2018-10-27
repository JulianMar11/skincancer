import os
from os.path import join
import cv2
import numpy as np
import models.preprocessing as pp
import pickle


class AbstractData(object):
    def __init__(self, directory, input_shape):
        self.input_shape = input_shape
        self.directory = directory

    def load(self):
        pass

    def random_batch(self, batch_size, mode="train"):
        pass

    def size(self):
        pass

    def size_train(self):
        pass

    def size_test(self):
        pass


class CancerData(AbstractData):
    def __init__(self, directory, input_shape):
        super(CancerData, self).__init__(directory, input_shape)
        self.pointers = {}
        self.rows = []
        self.classes = 3
        self.classnames = ["Melanoma", "Nevus", "Seborrheic Keartosis"]

    def load_with_label(self, pic_id, path, label):
        print(path)
        for root, dirs, files in os.walk(path):
            for file in files:
                print(file)
                if file.endswith(".png") or file.endswith(".PNG") or file.endswith(".jpg") or file.endswith(".jpeg") or file.endswith(".bmp"):
                    pic_id += 1
                    if np.random.randint(0, 5) == 2:
                        set = "test"
                    else:
                        set = "train"
                    file_path = join(path, file)
                    print(file_path)
                    print("LABEL", label)
                    pic = {"id": pic_id, "set": set, "label": label, "path": file_path}
                    self.rows.append(pic)
        return pic_id

    def load(self):
        print("LOADING")
        pic_id = 0
        path_nevus = os.path.join(self.directory, "nevus")
        path_melanoma = os.path.join(self.directory, "melanoma")
        path_seborrheic_keratosis = os.path.join(self.directory, "seborrheic_keratosis")

        pic_id = self.load_with_label(pic_id, path_nevus, 0)
        pic_id = self.load_with_label(pic_id, path_seborrheic_keratosis, 1)
        _ = self.load_with_label(pic_id, path_melanoma, 2)
        print("Loaded Data")
        print(self.size())
        print(len(self.rows))
        for i in range(0, 5):
            print(self.rows[np.random.randint(0, len(self.rows) - 1)])

    def get_classes(self):
        return self.classes

    def set_classnames(self, list):
        self.classnames = list

    def get_classnames(self):
        return self.classnames

    def getimage(self, path):
        img = cv2.imread(path)
        final = pp.preprocess(img, size=(self.input_shape[0], self.input_shape[1]))
        return final

    def random_batch(self, batch_size, mode="train"):
        x = []
        y = []
        subset = [pic for pic in self.rows if pic["set"] == mode]

        while len(x) < batch_size:
            label = np.random.randint(0, self.classes)
            class_pics = [pic for pic in subset if pic["label"] == label]
            x_path = class_pics[np.random.randint(0, len(class_pics))]['path']
            x.append(self.getimage(x_path))
            y.append(label)

        y = np.array(y).astype(np.int32)
        x = np.array(x)
        return x, y

    def size_train(self):
        return len([pic for pic in self.rows if pic["set"] == "train"])

    def size_test(self):
        return len([pic for pic in self.rows if pic["set"] == "test"])

    def size(self):
        train = [pic for pic in self.rows if pic["set"] == "train"]
        test = [pic for pic in self.rows if pic["set"] == "test"]
        sizes = {}
        for i in range(0, self.classes):
            subset_train = [pic for pic in train if pic["label"] == i]
            subset_test = [pic for pic in test if pic["label"] == i]
            string_train = "train_" + str(i)
            string_test = "test_" + str(i)
            length_train = len(subset_train)
            length_test = len(subset_test)
            sizes[string_train] = length_train
            sizes[string_test] = length_test
        return sizes


class Cifar10Data(AbstractData):
    def __init__(self, directory, input_shape):
        super(Cifar10Data, self).__init__(directory, input_shape)
        self.pointers = {}
        self.rows = []
        self.classes = 10
        self.classnames = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

    def get_cifar_batch(self, batch_id):
        with open(self.directory + '/data_batch_' + str(batch_id), mode='rb') as file:
            # note the encoding type is 'latin1'
            batch = pickle.load(file, encoding='latin1')
        features = batch['data'].reshape((len(batch['data']), 3, 32, 32)).transpose(0, 2, 3, 1)
        labels = batch['labels']
        return features, labels

    def get_cifar_batch_test(self):
        with open(self.directory + '/test_batch', mode='rb') as file:
            # note the encoding type is 'latin1'
            batch = pickle.load(file, encoding='latin1')
        features = batch['data'].reshape((len(batch['data']), 3, 32, 32)).transpose(0, 2, 3, 1)
        labels = batch['labels']
        return features, labels

    def load(self):
        print("LOADING")

    def random_batch(self, batch_size, mode="train", split=0.5):
        x = []
        y = []
        if mode == "train":
            features, labels = self.get_cifar_batch(np.random.randint(1, 6))
        else:
            features, labels = self.get_cifar_batch_test()
        counter = np.random.randint(0, len(features))
        while len(x) < batch_size:
            img = features[counter]
            img = cv2.resize(img, (self.input_shape[0], self.input_shape[1]))
            x.append(img)
            y.append(labels[counter])
            counter += 1
            if counter == len(features):
                if mode == "train":
                    features, labels = self.get_cifar_batch(np.random.randint(1, 6))
                counter = 0
        y = np.array(y).astype(np.int32)
        y = y.flatten()
        x = np.array(x)
        return x, y

    def get_classes(self):
        return self.classes

    def get_classnames(self):
        return self.classnames

    def size(self):
        mydict = {}
        for i in range(0,7):
            _, labels = self.get_cifar_batch(np.random.randint(1, 6))
            _, labels_test = self.get_cifar_batch_test()
            string_train = "train_" + str(i)
            string_test = "test_" + str(i)
            length_train = 5*len(labels)
            length_test = len(labels_test)
            mydict[string_train] = length_train
            mydict[string_test] = length_test
        return mydict

    def size_train(self):
        return 50000

    def size_test(self):
        return 10000

