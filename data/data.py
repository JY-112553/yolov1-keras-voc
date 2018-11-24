from keras.utils import Sequence
import math
import cv2 as cv
import numpy as np
import os


class SequenceData(Sequence):

    def __init__(self, path, target_size, batch_size=1, shuffle=True):
        self.images_path = os.path.join(path, 'images')
        self.labels_path = os.path.join(path, 'labels')
        self.target_size = target_size
        self.image_size = target_size[0:2]
        self.batch_size = batch_size
        self.image_filenames = os.listdir(self.images_path)
        self.label_filenames = os.listdir(self.labels_path)
        self.indexes = np.arange(len(self.image_filenames))
        self.shuffle = shuffle

    def __len__(self):
        # 计算每一个epoch的迭代次数
        num_imgs = len(self.image_filenames)
        return math.ceil(num_imgs / float(self.batch_size))

    def __getitem__(self, idx):
        # 生成batch_size个索引
        batch_indexs = self.indexes[idx * self.batch_size:(idx + 1) * self.batch_size]
        # 根据索引获取datas集合中的数据
        batch_X = [self.image_filenames[k] for k in batch_indexs]
        batch_y = [self.label_filenames[k] for k in batch_indexs]
        # 生成数据
        X, y = self.data_generation(batch_X, batch_y)
        return X, y

    def on_epoch_end(self):
        # 在每一次epoch结束是否需要进行一次随机，重新随机一下index
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def read_image(self, x):
        try:
            imagepath = os.path.join(self.images_path, x)
            image = cv.imread(imagepath)
            image = cv.cvtColor(image, cv.COLOR_BGR2RGB)  # opencv读取通道顺序为BGR，所以要转换
            image = cv.resize(image, self.image_size)
            image = image / 255.
        except IOError:
            print('no image')
        else:
            return image

    def read_label(self, x):
        try:
            labelpath = os.path.join(self.labels_path, x)
            f = open(labelpath, 'r')
            label = f.readlines()
            f.close()

            label_matrix = np.zeros([7, 7, 25])
            for l in label:
                l = l.strip().split()
                l = np.array(l, dtype=np.float32)
                cls = int(l[0])
                loc = l[1:3] * 7
                loc_i = int(loc[1])
                loc_j = int(loc[0])
                x = loc[0] - loc_j
                y = loc[1] - loc_i
                w = l[3]
                h = l[4]
                if label_matrix[loc_i, loc_j, 24] == 0:
                    label_matrix[loc_i, loc_j, cls] = 1
                    label_matrix[loc_i, loc_j, 20:24] = [x, y, w, h]
                    label_matrix[loc_i, loc_j, 24] = 1  # response
        except IOError:
            print('no label')
        else:
            return label_matrix

    def data_generation(self, batch_image_filenames, batch_label_filenames):
        images = []
        labels = []

        for image_filename in batch_image_filenames:
            image = self.read_image(image_filename)
            images.append(image)

        for label_filename in batch_label_filenames:
            label = self.read_label(label_filename)
            labels.append(label)

        X = np.array(images)
        y = np.array(labels)

        return X, y
