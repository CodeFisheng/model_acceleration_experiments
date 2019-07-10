import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_estimator as est
import os
import matplotlib.pyplot as plt
import argparse
import numpy as np
import time
import tensorflow as tf
from tensorflow.python import keras
from tensorflow.python.keras import layers, regularizers
from tensorflow.python.keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from tensorflow.python.keras.models import Model, load_model
from tensorflow.python.keras.preprocessing import image
from tensorflow.python.keras.utils import layer_utils
from tensorflow.python.keras.utils.data_utils import get_file
from tensorflow.python.keras.applications.imagenet_utils import preprocess_input
from tensorflow.python.keras.datasets import cifar10
from tensorflow.python.keras.utils.vis_utils import model_to_dot
from tensorflow.python.keras.utils import plot_model
from tensorflow.python.keras.initializers import glorot_uniform
from matplotlib.pyplot import imshow
import keras.backend as K
from tensorflow.python.keras import models
from tensorflow.python.keras import layers
from tensorflow.python.keras.applications import ResNet50, MobileNet
_IS_TRAINING = False
_SIZE=224
_A = 16
_B = 16
_C = 32
_D = 64
_DATASET_NAME = ''
_NUM_CLASSES = 1000
global _TRAIN_DATASET, _EVAL_DATASET
# step 2 find tfrecord file
def dataset_init(data_dir):
    if tf.gfile.Exists(data_dir):
        # return a list of (filepath + filename)s that match the path pattern
        _TRAIN_DATASET = tf.gfile.Glob(data_dir+"train.tfrecords")
        _EVAL_DATASET = tf.gfile.Glob(data_dir+"eval.tfrecords")
        # check if find file, raise errors
        if not len(_TRAIN_DATASET):
            raise Exception("[Train Error]: unable to find train*.tfrecord file")
        if not len(_EVAL_DATASET):
            raise Exception("[Eval Error]: unable to find test*.tfrecord file")
        return _TRAIN_DATASET, _EVAL_DATASET
    else:
        raise Exception("[Train Error]: unable to find input directory!")

def data_parser(record):
    # how data organised: dict, key to tf.feature
    if _DATASET_NAME == 'imagenet':
        keys_to_features = {
            "image": tf.FixedLenFeature((), tf.string, default_value=""),
            "label": tf.FixedLenFeature((), tf.int64,
                                        default_value=tf.zeros([], dtype=tf.int64)),
            "image_height": tf.FixedLenFeature((), tf.int64,
                                        default_value=tf.zeros([], dtype=tf.int64)),
            "image_width": tf.FixedLenFeature((), tf.int64,
                                        default_value=tf.zeros([], dtype=tf.int64)),
        }
    elif _DATASET_NAME == 'cifar':
        keys_to_features = {
            "image": tf.FixedLenFeature((), tf.string, default_value=""),
            "label": tf.FixedLenFeature((), tf.int64,
                                        default_value=tf.zeros([], dtype=tf.int64)),
        }
    # parsed result on one sample
    parsed = tf.parse_single_example(record, keys_to_features)

    # Perform additional preprocessing on the parsed data.
    if _DATASET_NAME == 'imagenet':
        height = tf.cast(parsed["image_height"], tf.int32)
        width = tf.cast(parsed["image_width"], tf.int32)
    elif _DATASET_NAME == 'cifar':
        height = _SIZE
        width = _SIZE
    label = tf.cast(parsed["label"], tf.int32)
    image = tf.decode_raw(parsed['image'], tf.uint8)
    image = tf.reshape(image, [height, width, 3])
    image = tf.image.resize_image_with_crop_or_pad(image, _SIZE, _SIZE)
    image = tf.reshape(image, [_SIZE, _SIZE, 3])
    if _IS_TRAINING == True:
        image = tf.image.random_flip_left_right(image)

    image = tf.cast(image, tf.float32)
    image = image/255.
    label = tf.one_hot(label, _NUM_CLASSES)

    return {'input_1': image}, label

def input_fn(data_path, batch_size, is_training=True):
    # step 3 read tfrecord
    dataset = tf.data.TFRecordDataset(data_path)
    if is_training == True:
        _IS_TRAINING = True
    else:
        _IS_TRAINING = False
    dataset = dataset.map(data_parser)
    dataset = dataset.repeat(10)
    dataset = dataset.batch(batch_size)
    dataset = dataset.shuffle(100)
    iterator = dataset.make_one_shot_iterator()
    features, label = iterator.get_next()
    return features, label


def residual_block(X, f, filters, stage, block):
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    F1, F2, F3 = filters

    X_shortcut = X

    X = BatchNormalization(axis=3, epsilon=1e-5, name=bn_name_base + '2a')(X)
    X = Activation('relu')(X)
    X = Conv2D(filters=F1, kernel_size=(1, 1), strides=(1, 1), padding='valid',
               kernel_regularizer=regularizers.l2(0.0001),
               name=conv_name_base + '2a', kernel_initializer=glorot_uniform(seed=0))(X)

    X = BatchNormalization(axis=3, epsilon=1e-5, name=bn_name_base + '2b')(X)
    X = Activation('relu')(X)
    X = Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding='same',
               kernel_regularizer=regularizers.l2(0.0001),
               name=conv_name_base + '2b', kernel_initializer=glorot_uniform(seed=0))(X)

    X = BatchNormalization(axis=3, epsilon=1e-5, name=bn_name_base + '2c')(X)
    X = Activation('relu')(X)
    X = Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding='valid',
               kernel_regularizer=regularizers.l2(0.0001),
               name=conv_name_base + '2c', kernel_initializer=glorot_uniform(seed=0))(X)

    X = Add()([X, X_shortcut])
    return X


def bottleneck_block(X, f, filters, stage, block, s=2):
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    F1, F2, F3 = filters


    X = BatchNormalization(axis=3, epsilon=1e-5, name=bn_name_base + '2a')(X)
    X = Activation('relu')(X)
    X_shortcut = X
    X = Conv2D(F1, (1, 1), strides=(s, s), name=conv_name_base + '2a',
               kernel_regularizer=regularizers.l2(0.0001),
               kernel_initializer=glorot_uniform(seed=0))(X)

    X = BatchNormalization(axis=3, epsilon=1e-5, name=bn_name_base + '2b')(X)
    X = Activation('relu')(X)
    X = Conv2D(F2, (f, f), strides=(1, 1), name=conv_name_base + '2b', padding='same',
               kernel_regularizer=regularizers.l2(0.0001),
               kernel_initializer=glorot_uniform(seed=0))(X)

    X = BatchNormalization(axis=3, epsilon=1e-5, name=bn_name_base + '2c')(X)
    X = Activation('relu')(X)
    X = Conv2D(F3, (1, 1), strides=(1, 1), name=conv_name_base + '2c', padding='valid',
               kernel_regularizer=regularizers.l2(0.0001),
               kernel_initializer=glorot_uniform(seed=0))(X)

    X_shortcut = Conv2D(F3, (1, 1), strides=(s, s), name=conv_name_base + '1',
                        kernel_regularizer=regularizers.l2(0.0001),
                        padding='valid', kernel_initializer=glorot_uniform(seed=0))(
        X_shortcut)

    X = Add()([X, X_shortcut])

    return X

def func(inp):
    def my_func(x):
        return np.abs(x)
    result_tensor = tf.py_func(my_func, [inp], tf.float32)
    result_tensor.set_shape(inp.get_shape())
    return result_tensor

def ResNet50(input_shape=(_SIZE, _SIZE, 3), classes=_NUM_CLASSES):
    X_input = Input(input_shape)

    X = ZeroPadding2D((3, 3))(X_input)

    X = Conv2D(_A, (7, 7), strides=(2, 2), name='conv1',
               kernel_regularizer=regularizers.l2(0.0001),
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, epsilon=1e-5, name='bn_conv1')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((3, 3), strides=(2, 2))(X)

    X = bottleneck_block(X, f=3, filters=[_A, _A, 4*_A], stage=2, block='a', s=1)
    X = residual_block(X, 3, [_A, _A, 4*_A], stage=2, block='b')
    X = residual_block(X, 3, [_A, _A, 4*_A], stage=2, block='c')

    X = bottleneck_block(X, f=3, filters=[_B, _B, 4*_B], stage=3, block='a', s=2)
    X = residual_block(X, f=3, filters=[_B, _B, 4*_B], stage=3, block='b')
    X = residual_block(X, f=3, filters=[_B, _B, 4*_B], stage=3, block='c')
    X = residual_block(X, f=3, filters=[_B, _B, 4*_B], stage=3, block='d')

    X = bottleneck_block(X, f=3, filters=[_C, _C, 4*_C], stage=4, block='a', s=2)
    X = residual_block(X, f=3, filters=[_C, _C, 4*_C], stage=4, block='b')
    X = residual_block(X, f=3, filters=[_C, _C, 4*_C], stage=4, block='c')
    X = residual_block(X, f=3, filters=[_C, _C, 4*_C], stage=4, block='d')
    X = residual_block(X, f=3, filters=[_C, _C, 4*_C], stage=4, block='e')
    X = residual_block(X, f=3, filters=[_C, _C, 4*_C], stage=4, block='f')

    X = bottleneck_block(X, f=3, filters=[_D, _D, 4*_D], stage=5, block='a', s=2)
    X = residual_block(X, f=3, filters=[_D, _D, 4*_D], stage=5, block='b')
    X = residual_block(X, f=3, filters=[_D, _D, 4*_D], stage=5, block='c')

    # X = tf.keras.layers.AveragePooling2D(pool_size=(7, 7), strides=1)(X)
    X = tf.keras.layers.Flatten()(X)
    X = Dense(classes, activation='softmax', name='fc' + str(classes),
              kernel_regularizer=regularizers.l2(0.0001),
              kernel_initializer=glorot_uniform(seed=0))(X)

    model = Model(inputs=X_input, outputs=X, name='ResNet50')

    return model


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='resnet', help='resnet|ssd')
    parser.add_argument('--batch_size', type=int, default=16, help='training batch size')
    parser.add_argument('--dataset', default='imagenet', help='cifar|imagenet')
    parser.add_argument('--datasetsize', default='dog2cat', help='full|dog2cat')
    parser.add_argument('--scale', default='demo', help='demo|real')
    parser.add_argument('--epoch', type=int, default=20000, help='training stopping '
                                                                 'epoch')
    parser.add_argument('--api', default='keras', help='api module type')
    parser.add_argument('--training_once', action='store_true', default=False, help='api '
                                                                                    'module type')
    args = parser.parse_args()

    if args.dataset == 'cifar':
        data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                'datasets/cifar/')
        model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 'models/cifar/')
        _DATASET_NAME = args.dataset
        _NUM_CLASSES = 10
        _SIZE = 32
    elif args.dataset == 'imagenet':
        data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                'datasets/imagenet/')
        model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 'models/imagenet/')
        _DATASET_NAME = args.dataset

        if args.datasetsize == 'full':
            _NUM_CLASSES = 1000
        elif args.datasetsize == 'dog2cat':
            _NUM_CLASSES = 2

        _SIZE = 224
    else:
        raise NotImplementedError('Unimplemented: Dataset')

    if args.scale == 'demo':
        _A = 16
        _B = 16
        _C = 32
        _D = 64
    elif args.scale == 'real':
        _A = 64
        _B = 128
        _C = 256
        _D = 512
    else:
        raise NotImplementedError('Unimplemented: demo scale')

    train_data, eval_data = dataset_init(data_dir)

    def resnet_model():
        model = ResNet50(input_shape=(_SIZE, _SIZE, 3), classes=_NUM_CLASSES)
        optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)
        model.compile(optimizer=optimizer, loss='categorical_crossentropy',
                      metrics=['accuracy'])
        print(model.summary())
        # config = tf.estimator.RunConfig(model_dir=model_dir)
        estimator = tf.keras.estimator.model_to_estimator(keras_model=model)
        return estimator

    classifier = resnet_model()

    train_spec = tf.estimator.TrainSpec(input_fn=lambda: input_fn(data_path=train_data,
                                                                  batch_size=args.batch_size,
                                                                  is_training=True),
                                        max_steps=args.steps)
    eval_spec = tf.estimator.EvalSpec(input_fn=lambda: input_fn(data_path=eval_data,
                                                                batch_size=16,
                                                                is_training=False),
                                      steps=200,
                                      throttle_secs=900)
    tf.estimator.train_and_evaluate(estimator=classifier, train_spec=train_spec, 
                                    eval_spec=eval_spec)

