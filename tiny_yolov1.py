import cv2 as cv
import numpy as np
from models.model_tiny_yolov1 import model_tiny_yolov1
from keras.engine import Input
from keras.models import Model


class Tiny_Yolov1(object):

    def __init__(self, weight_path, input_path):
        self.weight_path = weight_path
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
        model.load_weights(self.weight_path, by_name=True)
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


classes_name = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
                'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
                'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
                'tvmonitor']

weight_path = 'weights_cifar10_vgg16.hdf5'
image_path = 'D:\\Datasets\\VOC\\Data2\\test\\images\\00006.jpg'

tyv1 = Tiny_Yolov1(weight_path, image_path)
prediction = tyv1.predict()

predict_class = prediction[..., :20]  # 1 * 7 * 7 * 20
predict_trust = prediction[..., 20:22]  # 1 * 7 * 7 * 2
predict_box = prediction[..., 22:]  # 1 * 7 * 7 * 8

predict_class = np.reshape(predict_class, [7, 7, 1, 20])
predict_trust = np.reshape(predict_trust, [7, 7, 2, 1])
predict_box = np.reshape(predict_box, [7, 7, 2, 4])

predict_scores = predict_class * predict_trust  # 7 * 7 * 2 * 20

# predict_scores = np.reshape(predict_scores, [7 * 7 * 2, 20])
# predict_box = np.reshape(predict_box, [7 * 7 * 2, 4])

box_classes = np.argmax(predict_scores, axis=-1)  # 7 * 7 * 2
box_class_scores = np.max(predict_scores, axis=-1)  # 7 * 7 * 2

filter_mask = box_class_scores >= 0.6  # 7 * 7 * 2
filter_mask = np.expand_dims(filter_mask, axis=-1)  # 7 * 7 * 2 * 1

predict_scores *= filter_mask  # 7 * 7 * 2 * 20
predict_box *= filter_mask  # 7 * 7 * 2 * 4

box_classes = np.expand_dims(box_classes, axis=-1)
box_classes *= filter_mask  # 7 * 7 * 2 * 1

box_xy, box_wh = yolo_head(predict_box)
box_xy_min, box_xy_max = xywh2minmax(box_xy, box_wh)

image = cv.imread(image_path)
origin_shape = image.shape[0:2]
image = cv.resize(image, (448, 448))
detect_shape = filter_mask.shape

for i in range(detect_shape[0]):
    for j in range(detect_shape[1]):
        for k in range(detect_shape[2]):
            if filter_mask[i, j, k, 0]:
                cv.rectangle(image, (int(box_xy_min[i, j, k, 0]), int(box_xy_min[i, j, k, 1])),
                             (int(box_xy_max[i, j, k, 0]), int(box_xy_max[i, j, k, 1])),
                             (0, 0, 255))
                cv.putText(image, classes_name[box_classes[i, j, k, 0]],
                           (int(box_xy_min[i, j, k, 0]), int(box_xy_min[i, j, k, 1])),
                           1, 1, (0, 0, 255))

image = cv.resize(image, (origin_shape[1], origin_shape[0]))
cv.imshow('image', image)
cv.waitKey(0)
