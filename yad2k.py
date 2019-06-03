"""
Reads Darknet19 config & weights and creates Keras model with Tensorflow backend.
Currently only supports layers in Darknet19 config.
"""
# Libraries
import argparse
import configparser
import io
import os
from collections import defaultdict
import numpy as np
from keras import backend as K
from keras.layers import Conv2D, GlobalAveragePooling2D, Input, Lambda,MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.merge import concatenate
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.regularizers import l2
# Utilities
from yad2k.models.keras_yolo import space_to_depth_x2,space_to_depth_x2_output_shape
from config import config

def unique_config_sections(config_file):
    """
    Convert all config sections to have unique names.
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

def _main():
    config_path = os.path.expanduser(config['model_cfg'])
    weights_path = os.path.expanduser(config['model_weights'])
    output_path = os.path.expanduser(config['keras_model_path'])
    assert config_path.endswith('.cfg'), '{} is not a .cfg file'.format(config_path)
    assert weights_path.endswith('.weights'), '{} is not a .weights file'.format(weights_path)
    assert output_path.endswith('.hdf5'), 'output path {} is not a .hdf5 file'.format(output_path)
    print('Config file at {}'.format(config_path))
    print('Weights file at {}'.format(weights_path))
    output_root = os.path.splitext(output_path)[0]
    # Load weights and config.
    print('Loading weights.')
    weights_file = open(weights_path, 'rb')
    weights_header = np.ndarray(shape=(4, ), dtype='int32', buffer=weights_file.read(16))
    print('Weights Header: ', weights_header)
    print('Parsing Darknet config file.')
    unique_config_file = unique_config_sections(config_path)
    cfg_parser = configparser.ConfigParser()
    cfg_parser.read_file(unique_config_file)
    print('Creating Keras model.')
    image_height = int(cfg_parser['net_0']['height'])
    image_width = int(cfg_parser['net_0']['width'])
    # Input layer which takes input of shape (image_height, image_width, 3)
    prev_layer = Input(shape=(image_height, image_width, 3))
    # All layers list
    all_layers = [prev_layer]
    # Weight decay, default = 0.0005
    weight_decay = float(cfg_parser['net_0']['decay']) if 'net_0' in cfg_parser.sections() else 5e-4
    # Counter to count number of weights loaded
    count = 0
    # For each section in cfg_parser.sections()
    for section in cfg_parser.sections():
        print('Parsing section {}'.format(section))
        # If section is `convolutional`
        if section.startswith('convolutional'):
            # Number of kernels/filters
            filters = int(cfg_parser[section]['filters'])
            # Kernel/filter size
            size = int(cfg_parser[section]['size'])
            # Stride
            stride = int(cfg_parser[section]['stride'])
            # Padding
            pad = int(cfg_parser[section]['pad'])
            # Activation
            activation = cfg_parser[section]['activation']
            # Batch normailzation
            batch_normalize = 'batch_normalize' in cfg_parser[section]
            # padding='same' is equivalent to Darknet pad=1
            padding = 'same' if pad == 1 else 'valid'
            # Setting weights.
            # Darknet serializes convolutional weights as: [bias/beta, [gamma, mean, variance], conv_weights]
            # Shape of layer previous to this convolutional layer
            prev_layer_shape = K.int_shape(prev_layer)
            weights_shape = (size, size, prev_layer_shape[-1], filters)
            darknet_w_shape = (filters, weights_shape[2], size, size)
            weights_size = np.product(weights_shape)
            # If batch normailzation is enabled, print it else leave it as normal conv2d
            print('conv2d', 'bn' if batch_normalize else '  ', activation, weights_shape)
            conv_bias = np.ndarray(
                shape=(filters, ),
                dtype='float32',
                buffer=weights_file.read(filters * 4))
            # Increment weights counter
            count += filters
            # If batch normailzation is enabled
            if batch_normalize:
                bn_weights = np.ndarray(
                    shape=(3, filters),
                    dtype='float32',
                    buffer=weights_file.read(filters * 12))
                # Increment weights counter
                count += 3 * filters
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
            # Increment weights counter
            count += weights_size
            # DarkNet conv_weights are serialized Caffe-style: (out_dim, in_dim, height, width)
            # We would like to set these to Tensorflow order: (height, width, in_dim, out_dim)
            conv_weights = np.transpose(conv_weights, [2, 3, 1, 0])
            conv_weights = [conv_weights] if batch_normalize else [ conv_weights, conv_bias ]
            # Handle activation.
            act_fn = None
            if activation not in ['linear','leaky']:
                raise ValueError('Unknown activation function `{}` in section {}'.format(activation, section))
            # Create Conv2D layer
            conv_layer = (Conv2D(
                filters, (size, size),
                strides=(stride, stride),
                kernel_regularizer=l2(weight_decay),
                use_bias=not batch_normalize,
                weights=conv_weights,
                activation=act_fn,
                padding=padding))(prev_layer)
            # Create batch normalization layer if enabled
            if batch_normalize:
                conv_layer = (BatchNormalization(weights=bn_weight_list))(conv_layer)
            prev_layer = conv_layer
            # If linear activation is provided
            if activation == 'linear':
                # Append activation layer to `all_layers`
                all_layers.append(prev_layer)
            # If leaky ReLU activation is provided
            elif activation == 'leaky':
                act_layer = LeakyReLU(alpha=0.1)(prev_layer)
                prev_layer = act_layer
                # Append activation layer to `all_layers`
                all_layers.append(act_layer)
        # If section is `maxpool`
        elif section.startswith('maxpool'):
            # Kernel/filter size
            size = int(cfg_parser[section]['size'])
            # Stride
            stride = int(cfg_parser[section]['stride'])
            # Append MaxPooling2D layer to `all_layers`
            all_layers.append(MaxPooling2D(
                    padding='same',
                    pool_size=(size, size),
                    strides=(stride, stride))(prev_layer))
            # Set `prev_layer` to last layer from `all_layers`
            prev_layer = all_layers[-1]
        # If section is `avgpool`
        elif section.startswith('avgpool'):
            if cfg_parser.items(section) != []:
                raise ValueError('{} with params unsupported.'.format(section))
            # Append GlobalAveragePooling2D layer to `all_layers`
            all_layers.append(GlobalAveragePooling2D()(prev_layer))
            # Set `prev_layer` to last layer from all layers
            prev_layer = all_layers[-1]
        # If section is `route` (Concatenation)
        elif section.startswith('route'):
            # IDs of all required layers
            ids = [int(i) for i in cfg_parser[section]['layers'].split(',')]
            # Extract all required layers
            layers = [all_layers[i] for i in ids]
            # If more than 1 layers exists
            if len(layers) > 1:
                # Concatenate all required layers
                print('Concatenating route layers:', layers)
                concatenate_layer = concatenate(layers)
                # Append to all layers
                all_layers.append(concatenate_layer)
                prev_layer = concatenate_layer
            # only one layer to route
            else:
                skip_layer = layers[0]
                # Append to all layers
                all_layers.append(skip_layer)
                prev_layer = skip_layer
        # If section is `reorg` (Lambda)
        elif section.startswith('reorg'):
            block_size = int(cfg_parser[section]['stride'])
            assert block_size == 2, 'Only reorg with stride 2 supported.'
            # Append Lambda layer to `all_layers`
            all_layers.append(Lambda(
                    space_to_depth_x2,
                    output_shape=space_to_depth_x2_output_shape,
                    name='space_to_depth_x2')(prev_layer))
            # Set `prev_layer` to last layer from `all_layers`
            prev_layer = all_layers[-1]
        # If section is `region`
        elif section.startswith('region'):
            with open('{}_anchors.txt'.format(output_root), 'w') as f:
                print(cfg_parser[section]['anchors'], file=f)
        # If section is net_0
        elif section.startswith('net_0'):
            continue
        # If section is not recognized
        else:
            raise ValueError('Unsupported section header type: {}'.format(section))
    # Create model
    model = Model(inputs=all_layers[0], outputs=all_layers[-1])
    # Model summary
    print(model.summary())
    # Save model
    model.save('{}'.format(output_path))
    print('Saved Keras model to {}'.format(output_path))
    # Check to see if all weights have been read.
    remaining_weights = len(weights_file.read()) / 4
    weights_file.close()
    print('Read {} of {} from Darknet weights.'.format(count, count +remaining_weights))
    if remaining_weights > 0:
        print('Warning: {} unused weights'.format(remaining_weights))

if __name__ == '__main__':
    _main()