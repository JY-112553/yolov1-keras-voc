import argparse
import os
import cv2 as cv
import numpy as np
from models.model_tiny_yolov1 import model_tiny_yolov1
from keras.engine import Input
from keras.models import Model

parser = argparse.ArgumentParser(description='Use Tiny-Yolov1 To Detect Picture.')
parser.add_argument('weights_path', help='Path to model weights.')
parser.add_argument('image_path', help='Path to detect image.')

classes_name = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
                'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
                'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
                'tvmonitor']


class Tiny_Yolov1(object):

    def __init__(self, weights_path, input_path):
        self.weights_path = weights_path
        self.input_path = input_path
        self.classes_name = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
                             'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
                             'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
                             'tvmonitor']

    def predict(self):
        image = cv.imread(self.input_path)
        input_shape = (1, 448, 448, 3)
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        image = cv.resize(image, input_shape[1:3])
        image = np.reshape(image, input_shape)
        image = image / 255.
        inputs = Input(input_shape[1:4])
        outputs = model_tiny_yolov1(inputs)
        model = Model(inputs=inputs, outputs=outputs)
        model.load_weights(self.weights_path, by_name=True)
        y = model.predict(image, batch_size=1)

        return y


def yolo_head(feats):
    # Dynamic implementation of conv dims for fully convolutional model.
    conv_dims = np.shape(feats)[0:2]  # assuming channels last
    # In YOLO the height index is the inner most iteration.
    conv_height_index = np.arange(0, stop=conv_dims[0])
    conv_width_index = np.arange(0, stop=conv_dims[1])
    conv_height_index = np.tile(conv_height_index, [conv_dims[1]])

    # TODO: Repeat_elements and tf.split doesn't support dynamic splits.
    conv_width_index = np.tile(np.expand_dims(conv_width_index, 0), [conv_dims[0], 1])
    conv_width_index = np.reshape(np.transpose(conv_width_index), [conv_dims[0] * conv_dims[1]])
    conv_index = np.transpose(np.stack([conv_height_index, conv_width_index]))
    conv_index = np.reshape(conv_index, [conv_dims[0], conv_dims[1], 1, 2])

    conv_dims = np.reshape(conv_dims, [1, 1, 1, 2])

    box_xy = (feats[..., :2] + conv_index) / conv_dims * 448
    box_wh = feats[..., 2:4] * 448

    return box_xy, box_wh


def xywh2minmax(xy, wh):
    xy_min = xy - wh / 2
    xy_max = xy + wh / 2

    return xy_min, xy_max


def iou(pred_mins, pred_maxes, true_mins, true_maxes):
    intersect_mins = np.maximum(pred_mins, true_mins)
    intersect_maxes = np.minimum(pred_maxes, true_maxes)
    intersect_wh = np.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]

    pred_wh = pred_maxes - pred_mins
    true_wh = true_maxes - true_mins
    pred_areas = pred_wh[..., 0] * pred_wh[..., 1]
    true_areas = true_wh[..., 0] * true_wh[..., 1]

    union_areas = pred_areas + true_areas - intersect_areas
    iou_scores = intersect_areas / union_areas

    return iou_scores


def _main(args):
    weights_path = os.path.expanduser(args.weights_path)
    image_path = os.path.expanduser(args.image_path)

    tyv1 = Tiny_Yolov1(weights_path, image_path)
    prediction = tyv1.predict()

    predict_class = prediction[..., :20]  # 1 * 7 * 7 * 20
    predict_trust = prediction[..., 20:22]  # 1 * 7 * 7 * 2
    predict_box = prediction[..., 22:]  # 1 * 7 * 7 * 8

    predict_class = np.reshape(predict_class, [7, 7, 1, 20])
    predict_trust = np.reshape(predict_trust, [7, 7, 2, 1])
    predict_box = np.reshape(predict_box, [7, 7, 2, 4])

    predict_scores = predict_class * predict_trust  # 7 * 7 * 2 * 20

    box_classes = np.argmax(predict_scores, axis=-1)  # 7 * 7 * 2
    box_class_scores = np.max(predict_scores, axis=-1)  # 7 * 7 * 2
    best_box_class_scores = np.max(box_class_scores, axis=-1, keepdims=True)  # 7 * 7 * 1

    box_mask = box_class_scores >= best_box_class_scores  # ? * 7 * 7 * 2

    filter_mask = box_class_scores >= 0.6  # 7 * 7 * 2
    filter_mask *= box_mask  # 7 * 7 * 2

    filter_mask = np.expand_dims(filter_mask, axis=-1)  # 7 * 7 * 2 * 1

    predict_scores *= filter_mask  # 7 * 7 * 2 * 20
    predict_box *= filter_mask  # 7 * 7 * 2 * 4

    box_classes = np.expand_dims(box_classes, axis=-1)
    box_classes *= filter_mask  # 7 * 7 * 2 * 1

    box_xy, box_wh = yolo_head(predict_box)  # 7 * 7 * 2 * 2
    box_xy_min, box_xy_max = xywh2minmax(box_xy, box_wh)  # 7 * 7 * 2 * 2

    predict_trust *= filter_mask  # 7 * 7 * 2 * 1
    nms_mask = np.zeros_like(filter_mask)  # 7 * 7 * 2 * 1
    predict_trust_max = np.max(predict_trust)  # 找到置信度最高的框
    max_i = max_j = max_k = 0
    while predict_trust_max > 0:
        for i in range(nms_mask.shape[0]):
            for j in range(nms_mask.shape[1]):
                for k in range(nms_mask.shape[2]):
                    if predict_trust[i, j, k, 0] == predict_trust_max:
                        nms_mask[i, j, k, 0] = 1
                        filter_mask[i, j, k, 0] = 0
                        max_i = i
                        max_j = j
                        max_k = k
        for i in range(nms_mask.shape[0]):
            for j in range(nms_mask.shape[1]):
                for k in range(nms_mask.shape[2]):
                    if filter_mask[i, j, k, 0] == 1:
                        iou_score = iou(box_xy_min[max_i, max_j, max_k, :],
                                        box_xy_max[max_i, max_j, max_k, :],
                                        box_xy_min[i, j, k, :],
                                        box_xy_max[i, j, k, :])
                        if iou_score > 0.2:
                            filter_mask[i, j, k, 0] = 0
        predict_trust *= filter_mask  # 7 * 7 * 2 * 1
        predict_trust_max = np.max(predict_trust)  # 找到置信度最高的框

    box_xy_min *= nms_mask
    box_xy_max *= nms_mask

    image = cv.imread(image_path)
    origin_shape = image.shape[0:2]
    image = cv.resize(image, (448, 448))
    detect_shape = filter_mask.shape

    for i in range(detect_shape[0]):
        for j in range(detect_shape[1]):
            for k in range(detect_shape[2]):
                if nms_mask[i, j, k, 0]:
                    cv.rectangle(image, (int(box_xy_min[i, j, k, 0]), int(box_xy_min[i, j, k, 1])),
                                 (int(box_xy_max[i, j, k, 0]), int(box_xy_max[i, j, k, 1])),
                                 (0, 0, 255))
                    cv.putText(image, classes_name[box_classes[i, j, k, 0]],
                               (int(box_xy_min[i, j, k, 0]), int(box_xy_min[i, j, k, 1])),
                               1, 1, (0, 0, 255))

    image = cv.resize(image, (origin_shape[1], origin_shape[0]))
    cv.imshow('image', image)
    cv.waitKey(0)


if __name__ == '__main__':
    _main(parser.parse_args())
    # _main(parser.parse_args(['my-tiny-yolov1.hdf5', 'C:/Users/JY/Desktop/test.jpg']))
