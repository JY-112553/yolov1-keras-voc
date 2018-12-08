# yolov1-keras-voc

Keras implementation of YOLOv1 (Tensorflow backend) inspired by [allanzelener/YAD2K](https://github.com/allanzelener/YAD2K) and [qqwweee/keras-yolo3](https://github.com/qqwweee/keras-yolo3).


## Download Yolo weights
Download tiny-yolov1 weights from [here](https://pjreddie.com/darknet/yolov1/).
```
wget http://pjreddie.com/media/files/yolov1/tiny-yolov1.weights
```


## Convert the model
Convert the Darknet model to a Keras model.
```
python convert.py cfg/yolov1-tiny.cfg tiny-yolov1.weights tiny-yolov1.hdf5
```

## Train your datasets
I use pascal-v0c2012 to train, you can download the datasets from [here](http://host.robots.ox.ac.uk:8080/).
```
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
tar xvf VOCtrainval_11-May-2012.tar
```

Build datasets.
```
python data/build datasets.py
```

Train by own datasets.
```
python train.py 10 32 Datasets
```

## Detect image
Imagepath is the path to detect.
```
python tiny_yolov1.py tiny-yolov1.hdf5 imagepath
```
