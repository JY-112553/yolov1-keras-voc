# yolov1-keras-voc

Keras implementation of YOLOv1 (Tensorflow backend) inspired by [allanzelener/YAD2K](https://github.com/allanzelener/YAD2K) and [qqwweee/keras-yolo3](https://github.com/qqwweee/keras-yolo3).


## Download Yolo weights
Download tiny-yolov1 weights from [here](https://pjreddie.com/darknet/yolov1/)


## Convert the model
Convert the Darknet model to a Keras model
```
python convert.py yolov1-tiny.cfg tiny-yolov1.weights tiny-yolov1.hdf5
```

## Train your datasets
use data/build datasets.py to build datasets
use train.py to train by your datasets
