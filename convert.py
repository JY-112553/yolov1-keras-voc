"""
Reads Darknet config and weights and creates Keras models with TF backend.
Currently only supports layers in Yolov1-tiny config.
"""

import argparse
import configparser
import io
import os
from collections import defaultdict

import numpy as np
from keras import backend as K
from keras.layers import Input, Conv2D, MaxPooling2D, \
    Dense, Flatten, Dropout, Reshape, LeakyReLU, ReLU

from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.regularizers import l2


parser = argparse.ArgumentParser(description='Darknet Yolov1-tiny To Keras Converter.')
parser.add_argument('config_path', help='Path to Darknet cfg file.')
parser.add_argument('weights_path', help='Path to Darknet weights file.')
parser.add_argument('output_path', help='Path to output Keras models file.')


def unique_config_sections(config_file):
    """Convert all config sections to have unique names.

    Adds unique suffixes to config sections for compability with configparser.
    """
    section_counters = defaultdict(int)
    output_stream = io.StringIO()
    with open(config_file) as fin:
        for line in fin:
            if line.startswith('['):
                section = line.strip().strip('[]')
                _section = section + '_' + str(section_counters[section])
                section_counters[section] += 1
                line = line.replace(section, _section)
            output_stream.write(line)
    output_stream.seek(0)
    return output_stream


def _main(args):
    config_path = os.path.expanduser(args.config_path)
    weights_path = os.path.expanduser(args.weights_path)
    output_path = os.path.expanduser(args.output_path)
    assert config_path.endswith('.cfg'), \
        '{} is not a .cfg file'.format(config_path)
    assert weights_path.endswith('.weights'), \
        '{} is not a .weights file'.format(weights_path)
    assert output_path.endswith('.hdf5'), \
        'output path {} is not a .hdf5 file'.format(output_path)

    # Load weights and config.
    print('Loading weights.')
    weights_file = open(weights_path, 'rb')
    weights_header = np.ndarray(
        shape=(4, ), dtype='int32', buffer=weights_file.read(16))
    print('Weights Header: ', weights_header)

    print('Parsing Darknet config.')
    unique_config_file = unique_config_sections(config_path)
    cfg_parser = configparser.ConfigParser()
    cfg_parser.read_file(unique_config_file)

    print('Creating Keras models.')

    try:
        image_height = int(cfg_parser['crop_0']['crop_height'])
        image_width = int(cfg_parser['crop_0']['crop_width'])
    except KeyError:
        image_height = int(cfg_parser['net_0']['height'])
        image_width = int(cfg_parser['net_0']['width'])

    prev_layer = Input(shape=(image_height, image_width, 3))
    all_layers = [prev_layer]

    weight_decay = float(cfg_parser['net_0']['decay']
                         ) if 'net_0' in cfg_parser.sections() else 5e-4
    count = 0
    fc_flag = False
    for section in cfg_parser.sections():
        print('Parsing section {}'.format(section))
        if section.startswith('convolutional'):
            filters = int(cfg_parser[section]['filters'])
            size = int(cfg_parser[section]['size'])
            stride = int(cfg_parser[section]['stride'])
            pad = int(cfg_parser[section]['pad'])
            activation = cfg_parser[section]['activation']
            batch_normalize = 'batch_normalize' in cfg_parser[section]

            # padding='same' is equivalent to Darknet pad=1
            padding = 'same' if pad == 1 else 'valid'

            # Setting weights.
            # Darknet serializes convolutional weights as:
            # [bias/beta, [gamma, mean, variance], conv_weights]
            prev_layer_shape = K.int_shape(prev_layer)

            # TODO: This assumes channel last dim_ordering.
            weights_shape = (size, size, prev_layer_shape[-1], filters)
            darknet_w_shape = (filters, weights_shape[2], size, size)
            weights_size = np.product(weights_shape)

            print('conv2d', 'bn'
                  if batch_normalize else '  ', activation, weights_shape)

            conv_bias = np.ndarray(
                shape=(filters, ),
                dtype='float32',
                buffer=weights_file.read(filters * 4))
            count += filters

            bn_weight_list = []
            if batch_normalize:
                bn_weights = np.ndarray(
                    shape=(3, filters),
                    dtype='float32',
                    buffer=weights_file.read(filters * 12))
                count += 3 * filters

                # TODO: Keras BatchNormalization mistakenly refers to var
                # as std.
                bn_weight_list = [
                    bn_weights[0],  # scale gamma
                    conv_bias,  # shift beta
                    bn_weights[1],  # running mean
                    bn_weights[2]  # running var
                ]

            conv_weights = np.ndarray(
                shape=darknet_w_shape,
                dtype='float32',
                buffer=weights_file.read(weights_size * 4))
            count += weights_size

            # DarkNet conv_weights are serialized Caffe-style:
            # (out_dim, in_dim, height, width)
            # We would like to set these to Tensorflow order:
            # (height, width, in_dim, out_dim)
            # TODO: Add check for Theano dim ordering.
            conv_weights = np.transpose(conv_weights, [2, 3, 1, 0])
            conv_weights = [conv_weights] if batch_normalize \
                else [conv_weights, conv_bias]

            # Handle activation.
            act_fn = None
            if activation == 'leaky':
                pass  # Add advanced activation later.
            elif activation == 'relu':
                pass
            elif activation != 'linear':
                raise ValueError(
                    'Unknown activation function `{}` in section {}'.format(
                        activation, section))

            # Create Conv2D layer
            conv_layer = Conv2D(
                filters, (size, size),
                strides=(stride, stride),
                kernel_regularizer=l2(weight_decay),
                use_bias=not batch_normalize,
                weights=conv_weights,
                activation=act_fn,
                padding=padding,
                name=format(section))(prev_layer)

            if batch_normalize:
                conv_layer = BatchNormalization(
                    weights=bn_weight_list,
                    name='bn' + format(section))(conv_layer)

            prev_layer = conv_layer

            if activation == 'linear':
                all_layers.append(prev_layer)
            elif activation == 'leaky':
                act_layer = LeakyReLU(alpha=0.1)(prev_layer)
                prev_layer = act_layer
                all_layers.append(act_layer)
            elif activation == 'relu':
                act_layer = ReLU()(prev_layer)
                prev_layer = act_layer
                all_layers.append(act_layer)

        elif section.startswith('maxpool'):
            size = int(cfg_parser[section]['size'])
            stride = int(cfg_parser[section]['stride'])
            all_layers.append(
                MaxPooling2D(
                    padding='same',
                    pool_size=(size, size),
                    strides=(stride, stride))(prev_layer))
            prev_layer = all_layers[-1]

        elif section.startswith('connected'):
            output_size = int(cfg_parser[section]['output'])
            activation = cfg_parser[section]['activation']

            prev_layer_shape = K.int_shape(prev_layer)

            # TODO: This assumes channel last dim_ordering.
            weights_shape = (np.prod(prev_layer_shape[1:]), output_size)
            darknet_w_shape = (output_size, weights_shape[0])
            weights_size = np.product(weights_shape)

            print('full-connected', activation, weights_shape)

            fc_bias = np.ndarray(
                shape=(output_size,),
                dtype='float32',
                buffer=weights_file.read(output_size * 4))
            count += output_size

            fc_weights = np.ndarray(
                shape=darknet_w_shape,
                dtype='float32',
                buffer=weights_file.read(weights_size * 4))
            count += weights_size

            # DarkNet fc_weights are serialized Caffe-style:
            # (out_dim, in_dim)
            # We would like to set these to Tensorflow order:
            # (in_dim, out_dim)
            # TODO: Add check for Theano dim ordering.
            fc_weights = np.transpose(fc_weights, [1, 0])
            fc_weights = [fc_weights, fc_bias]

            # Handle activation.
            act_fn = None
            if activation == 'leaky':
                pass  # Add advanced activation later.
            elif activation == 'relu':
                pass
            elif activation != 'linear':
                raise ValueError(
                    'Unknown activation function `{}` in section {}'.format(
                        activation, section))

            if not fc_flag:
                prev_layer = Flatten()(prev_layer)
                fc_flag = True

            # Create Full-Connect layer
            fc_layer = Dense(
                output_size,
                kernel_regularizer=l2(weight_decay),
                weights=fc_weights,
                activation=act_fn,
                name=format(section))(prev_layer)

            prev_layer = fc_layer

            if activation == 'linear':
                all_layers.append(prev_layer)
            elif activation == 'leaky':
                act_layer = LeakyReLU(alpha=0.1)(prev_layer)
                prev_layer = act_layer
                all_layers.append(act_layer)
            elif activation == 'relu':
                act_layer = ReLU()(prev_layer)
                prev_layer = act_layer
                all_layers.append(act_layer)

        elif section.startswith('dropout'):
            probability = float(cfg_parser[section]['probability'])
            dropout_layer = Dropout(probability)(prev_layer)
            prev_layer = dropout_layer
            all_layers.append(prev_layer)

        elif section.startswith('detection'):
            classes = int(cfg_parser[section]['classes'])
            coords = int(cfg_parser[section]['coords'])
            rescore = int(cfg_parser[section]['rescore'])
            side = int(cfg_parser[section]['side'])
            num = int(cfg_parser[section]['num'])
            reshape_layer = Reshape(
                (side, side, classes + num * (coords + rescore))
            )(prev_layer)
            prev_layer = reshape_layer
            all_layers.append(prev_layer)

        elif (section.startswith('net') or
              section.startswith('crop') or
              section.startswith('detection') or
              section.startswith('softmax')):
            pass  # Configs not currently handled during models definition.

        else:
            raise ValueError(
                'Unsupported section header type: {}'.format(section))

    # Create and save models.
    model = Model(inputs=all_layers[0], outputs=all_layers[-1])
    print(model.summary())

    model.save_weights('{}'.format(output_path))
    print('Saved Keras models to {}'.format(output_path))
    # Check to see if all weights have been read.
    remaining_weights = len(weights_file.read()) / 4
    weights_file.close()
    print('Read {} of {} from Darknet weights.'.format(count, count +
                                                       remaining_weights))
    if remaining_weights > 0:
        print('Warning: {} unused weights'.format(remaining_weights))


if __name__ == '__main__':
    _main(parser.parse_args())
    # _main(parser.parse_args(['cfg/yolov1-tiny.cfg', 'weights/tiny-yolov1.weights', 'weights/tiny-yolov1.hdf5']))
