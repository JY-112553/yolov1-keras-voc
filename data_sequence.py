from keras.utils import Sequence
import math
import cv2 as cv
import numpy as np
import os


class SequenceData(Sequence):

    def __init__(self, model, dir, target_size, batch_size, shuffle=True):
        self.model = model
        self.datasets = []
        if self.model is 'train':
            with open(os.path.join(dir, '2007_train.txt'), 'r') as f:
                self.datasets = self.datasets + f.readlines()
        elif self.model is 'val':
            with open(os.path.join(dir, '2007_val.txt'), 'r') as f:
                self.datasets = self.datasets + f.readlines()
        self.image_size = target_size[0:2]
        self.batch_size = batch_size
        self.indexes = np.arange(len(self.datasets))
        self.shuffle = shuffle

    def __len__(self):
        # 计算每一个epoch的迭代次数
        num_imgs = len(self.datasets)
        return math.ceil(num_imgs / float(self.batch_size))

    def __getitem__(self, idx):
        # 生成batch_size个索引
        batch_indexs = self.indexes[idx * self.batch_size:(idx + 1) * self.batch_size]
        # 根据索引获取datas集合中的数据
        batch = [self.datasets[k] for k in batch_indexs]
        # 生成数据
        X, y = self.data_generation(batch)
        return X, y

    def on_epoch_end(self):
        # 在每一次epoch结束是否需要进行一次随机，重新随机一下index
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def read(self, dataset):
        dataset = dataset.strip().split()
        image_path = dataset[0]
        label = dataset[1:]

        image = cv.imread(image_path)
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)  # opencv读取通道顺序为BGR，所以要转换
        image_h, image_w = image.shape[0:2]
        image = cv.resize(image, self.image_size)
        image = image / 255.

        label_matrix = np.zeros([7, 7, 25])
        for l in label:
            l = l.split(',')
            l = np.array(l, dtype=np.int)
            xmin = l[0]
            ymin = l[1]
            xmax = l[2]
            ymax = l[3]
            cls = l[4]
            x = (xmin + xmax) / 2 / image_w
            y = (ymin + ymax) / 2 / image_h
            w = (xmax - xmin) / image_w
            h = (ymax - ymin) / image_h
            loc = [7 * x, 7 * y]
            loc_i = int(loc[1])
            loc_j = int(loc[0])
            y = loc[1] - loc_i
            x = loc[0] - loc_j

            if label_matrix[loc_i, loc_j, 24] == 0:
                label_matrix[loc_i, loc_j, cls] = 1
                label_matrix[loc_i, loc_j, 20:24] = [x, y, w, h]
                label_matrix[loc_i, loc_j, 24] = 1  # response

        return image, label_matrix

    def data_generation(self, batch_datasets):
        images = []
        labels = []

        for dataset in batch_datasets:
            image, label = self.read(dataset)
            images.append(image)
            labels.append(label)

        X = np.array(images)
        y = np.array(labels)

        return X, y
