import argparse
from keras.engine import Input
from keras.models import Model
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
import os
from models.model_tiny_yolov1 import model_tiny_yolov1
from data import data
from yolo.yolo import yolo_loss
from callback import callback

parser = argparse.ArgumentParser(description='Train NetWork.')
parser.add_argument('epochs', help='Num of epochs.')
parser.add_argument('batch_size', help='Num of batch size.')
parser.add_argument('datasets_path', help='Path to datasets.')
parser.add_argument('output_path', help='Path to output Keras models file.')


def _main(args):
    epochs = int(os.path.expanduser(args.epochs))
    batch_size = int(os.path.expanduser(args.batch_size))

    input_shape = (448, 448, 3)
    inputs = Input(input_shape)
    yolo_outputs = model_tiny_yolov1(inputs)

    model = Model(inputs=inputs, outputs=yolo_outputs)
    model.compile(loss=yolo_loss, optimizer='adam')

    save_dir = 'checkpoints'
    weights_path = os.path.join(save_dir, 'weights.hdf5')
    checkpoint = ModelCheckpoint(
        weights_path, save_weights_only=True, period=1)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    if len(os.listdir(save_dir)) == 0:
        model.load_weights(
            'tiny-yolov1.hdf5', by_name=True)
    else:
        model.load_weights(weights_path, by_name=True)

    epoch_file_path = 'checkpoints/epoch.txt'
    try:
        with open(epoch_file_path, 'r') as f:
            now_epoch = int(f.read())
        epochs -= now_epoch
    except IOError:
        print('no train history')

    myCallback = callback.MyCallback()

    early_stopping = EarlyStopping(
        monitor='val_loss', min_delta=0, patience=15, verbose=1, mode='auto')

    # log_dir = 'logs'
    # tbCallBack = TensorBoard(log_dir=log_dir,  # log 目录
    #                          histogram_freq=0,  # 按照何等频率（epoch）来计算直方图，0为不计算
    #                          batch_size=batch_size,  # 用多大量的数据计算直方图
    #                          write_graph=True,  # 是否存储网络结构图
    #                          write_grads=True,  # 是否可视化梯度直方图
    #                          write_images=True,  # 是否可视化参数
    #                          embeddings_freq=0,
    #                          embeddings_layer_names=None,
    #                          embeddings_metadata=None)
    # if not os.path.isdir(log_dir):
    #     os.makedirs(log_dir)

    datasets_path = os.path.expanduser(args.datasets_path)
    datasets_train_path = os.path.join(datasets_path, 'train')
    datasets_val_path = os.path.join(datasets_path, 'val')

    # 数据生成器
    train_generator = data.SequenceData(
        datasets_train_path, input_shape, batch_size=batch_size)
    validation_generator = data.SequenceData(
        datasets_val_path, input_shape, batch_size=batch_size)

    model.fit_generator(
        train_generator,
        steps_per_epoch=len(train_generator),
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=len(validation_generator),
        use_multiprocessing=False,
        workers=4,
        callbacks=[checkpoint, myCallback, early_stopping]
    )

    output_path = os.path.expanduser(args.output_path)
    model.save_weights(os.path.join(output_path, 'my-tiny-yolov13.hdf5'))


if __name__ == '__main__':
    _main(parser.parse_args())
    # _main(parser.parse_args(['10', '32', 'D:/Datasets/VOC/DataSets', '']))
