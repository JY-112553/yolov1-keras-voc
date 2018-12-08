import argparse
import os
import numpy as np
import cv2 as cv
import xml.etree.ElementTree as ET
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser(description='Build Datasets.')
parser.add_argument('dir', default='..', help='Datasets dir.')

classes_num = {'aeroplane': 0, 'bicycle': 1, 'bird': 2, 'boat': 3, 'bottle': 4, 'bus': 5,
               'car': 6, 'cat': 7, 'chair': 8, 'cow': 9, 'diningtable': 10, 'dog': 11,
               'horse': 12, 'motorbike': 13, 'person': 14, 'pottedplant': 15, 'sheep': 16,
               'sofa': 17, 'train': 18, 'tvmonitor': 19}


def label_process(xmin, ymin, xmax, ymax, width, height):
    x = (xmin + xmax) / 2 / width
    y = (ymin + ymax) / 2 / height
    w = (xmax - xmin) / width
    h = (ymax - ymin) / height

    return x, y, w, h


def parse_xml(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    labels = []
    width, height, obj_num, xmin, ymin, xmax, ymax = 0, 0, 0, 0, 0, 0, 0
    for item in root:
        if item.tag == 'size':
            for i in item:
                if i.tag == 'width':
                    width = int(i.text)
                elif i.tag == 'height':
                    height = int(i.text)
    for item in root:
        if item.tag == 'object':
            for i in item:
                if i.tag == 'name':
                    obj_name = i.text
                    obj_num = int(classes_num[obj_name])
                elif i.tag == 'bndbox':
                    for ii in i:
                        if ii.tag == 'xmin':
                            xmin = float(ii.text)
                        elif ii.tag == 'ymin':
                            ymin = float(ii.text)
                        elif ii.tag == 'xmax':
                            xmax = float(ii.text)
                        elif ii.tag == 'ymax':
                            ymax = float(ii.text)
            try:
                x, y, w, h = label_process(
                    xmin, ymin, xmax, ymax, width, height)
                labels.append([obj_num, x, y, w, h])
            except Exception as e:
                print(e)

    return np.array(labels)


def build(x, y, images_path, labels_path):
    N = x.shape[0]
    for n in range(N):
        image = x[n]
        label = y[n]
        image_path = os.path.join(images_path, str(n + 1).zfill(5))
        cv.imwrite(image_path + '.jpg', image)
        label_path = os.path.join(labels_path, str(n + 1).zfill(5))
        with open(label_path + '.txt', 'w') as f:
            for ll in label:
                for l in range(len(ll)):
                    f.write(str(ll[l]) + ' ')
                    if l == len(ll) - 1:
                        f.write('\n')


def load_data(path):
    x_path = os.path.join(path, 'VOC2012/JPEGImages')
    y_path = os.path.join(path, 'VOC2012/Annotations')

    x = [cv.imread(os.path.join(x_path, p)) for p in os.listdir(x_path)]
    x = np.array(x)

    y = [parse_xml(os.path.join(y_path, p)) for p in os.listdir(y_path)]
    y = np.array(y)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

    return x_train, y_train, x_test, y_test


def _main(args):
    dir = os.path.expanduser(args.dir)
    datasets_dir = os.path.join(dir, 'DataSets')
    if not os.path.isdir(datasets_dir):
        os.makedirs(datasets_dir)
    train_datasets_dir = os.path.join(datasets_dir, 'train/')
    val_datasets_dir = os.path.join(datasets_dir, 'val/')
    images_path = 'images/'
    labels_path = 'labels/'
    train_images_path = os.path.join(train_datasets_dir, images_path)
    train_labels_path = os.path.join(train_datasets_dir, labels_path)
    val_images_path = os.path.join(val_datasets_dir, images_path)
    val_labels_path = os.path.join(val_datasets_dir, labels_path)
    if not os.path.isdir(train_images_path):
        os.makedirs(train_images_path)
    if not os.path.isdir(train_labels_path):
        os.makedirs(train_labels_path)
    if not os.path.isdir(val_images_path):
        os.makedirs(val_images_path)
    if not os.path.isdir(val_labels_path):
        os.makedirs(val_labels_path)

    source_path = os.path.join(dir, 'VOCdevkit')
    x_train, y_train, x_val, y_val = load_data(source_path)

    build(x_train, y_train, train_images_path, train_labels_path)
    build(x_val, y_val, val_images_path, val_labels_path)


if __name__ == '__main__':
    _main(parser.parse_args())
    # _main(parser.parse_args(['D:/Datasets/VOC']))
