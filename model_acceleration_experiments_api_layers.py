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
from tensorflow.python.keras import layers
from tensorflow.python.keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from tensorflow.python.keras.models import Model, load_model
from tensorflow.python.keras.initializers import glorot_uniform
_IS_TRAINING = False
_SIZE=224
_A = 16
_B = 16
_C = 32
_D = 64
_DATASET_NAME = ''
_NUM_CLASSES = 2
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
    label = tf.one_hot(label, _NUM_CLASSES)
    image = tf.decode_raw(parsed['image'], tf.uint8)
    image = tf.reshape(image, [height, width, 3])
    image = tf.image.resize_image_with_crop_or_pad(image, _SIZE, _SIZE)
    image = tf.reshape(image, [_SIZE, _SIZE, 3])
    if _IS_TRAINING == True:
        image = tf.image.random_flip_left_right(image)

    image = tf.cast(image, tf.float32)
    image = image/255.

    return {'input_1':image}, label

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
    # to use strategy.distribute, use return dataset rather than iterator elements
    # another change, move one_hot to data parser
    #
    #iterator = dataset.make_one_shot_iterator()
    #label, image = iterator.get_next()
    #label = tf.one_hot(label, _NUM_CLASSES)
    #return {'input_1': image}, label
    return dataset


def bottleneck_top_v2(X, f, filters, stage, block):
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    F1, F2, F3 = filters
    # use identical shortpath
    X_shortcut = X

    # tf.layers will be deprecated into tf.keras.layers
    X = tf.layers.batch_normalization(inputs=X, axis=3, name=bn_name_base + '2a')
    X = tf.nn.relu(features=X, name='relu')
    X = tf.layers.conv2d(inputs=X, filters=F1, kernel_size=(1, 1), strides=(1, 1),
                         padding='valid',
               name=conv_name_base + '2a', kernel_initializer=glorot_uniform(seed=0))

    X = tf.layers.batch_normalization(inputs=X, axis=3, name=bn_name_base + '2b')
    X = tf.nn.relu(features=X, name='relu')
    X = tf.layers.conv2d(inputs=X, filters=F2, kernel_size=(f, f), strides=(1, 1),
                         padding='same',
                         name=conv_name_base + '2b', kernel_initializer=glorot_uniform(seed=0))

    X = tf.layers.batch_normalization(inputs=X, axis=3, name=bn_name_base + '2c')
    X = tf.nn.relu(features=X, name='relu')
    X = tf.layers.conv2d(inputs=X, filters=F3, kernel_size=(1, 1), strides=(1, 1),
                         padding='valid',
                         name=conv_name_base + '2c', kernel_initializer=glorot_uniform(seed=0))
    X = X + X_shortcut
    return X

#     this is class call method way to invode conv2d
#     X = tf.layers.Conv2D(filters=F1, kernel_size=(1, 1), strides=(s, s),
#                          name=conv_name_base+'2a',
#                kernel_initializer=glorot_uniform(seed=0))(X)
# v2 resnet, uses BN, RELU, CONV sequence,
# v1 uses Conv, BN, RELU sequence, post activation
def bottleneck_base_v2(X, f, filters, stage, block, s=2):
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    [F1, F2, F3] = filters


    X = tf.layers.batch_normalization(inputs=X, axis=3, name=bn_name_base + '2a')
    X = tf.nn.relu(features=X, name='relu')
    # see http://git.enflame.cn/sw/benchmark/blob/master/tf_cnn_benchmarks/models/resnet_model.py page 160-180
    # if in first block, need pre activation among all, but in the middle, use identical shortpath
    preact = X
    X = tf.layers.conv2d(inputs=X, filters=F1, kernel_size=(1, 1), strides=(s, s),
                         name=conv_name_base+'2a',
                         kernel_initializer=glorot_uniform(seed=0))

    X = tf.layers.batch_normalization(inputs=X, axis=3, name=bn_name_base + '2b')
    X = tf.nn.relu(features=X, name='relu')
    X = tf.layers.conv2d(inputs=X, filters=F2, kernel_size=(f, f), strides=(1, 1),
                         name=conv_name_base+'2b', padding='same',
                         kernel_initializer=glorot_uniform(seed=0))

    X = tf.layers.batch_normalization(inputs=X, axis=3, name=bn_name_base + '2c')
    X = tf.nn.relu(features=X, name='relu')
    X = tf.layers.conv2d(inputs=X, filters=F3, kernel_size=(1, 1), strides=(1, 1),
                  name=conv_name_base + '2c', padding='valid',
               kernel_initializer=glorot_uniform(seed=0))

    X_shortcut = tf.layers.conv2d(inputs=preact, filters=F3, kernel_size=(1, 1),
                                                                       strides=(s, s),
                        name=conv_name_base + '1',
                        padding='valid', kernel_initializer=glorot_uniform(seed=0))

    X = X + X_shortcut

    return X

# debug, use print(tensor.shape/get_shape(return shape)
# print(tensor.shape.as_list() dims and its size
# print(tensor.shape.ndims, number of dimensions or tf.rank(input)
def ResNet50(input_tensor, classes=_NUM_CLASSES):
    # X_input = Input(input_shape)
    # X_input = tf.feature_column.input_layer(features, 'input_1')
    X_input = input_tensor
    X = ZeroPadding2D((3, 3))(X_input)
    print('anchor')
    print(X.shape)
    X = tf.layers.conv2d(inputs=X, filters=_A, kernel_size=(7, 7), strides=(2, 2),
                         name='conv1',
               kernel_initializer=glorot_uniform(seed=0))
    X = tf.layers.batch_normalization(inputs=X, axis=3, name='bn_conv1')
    X = tf.nn.relu(features=X, name='relu')
    X = tf.layers.MaxPooling2D((3, 3), strides=(2, 2))(X)

    X = bottleneck_base_v2(X, f=3, filters=[_A, _A, 4*_A], stage=2, block='a', s=1)
    X = bottleneck_top_v2(X, 3, [_A, _A, 4*_A], stage=2, block='b')
    X = bottleneck_top_v2(X, 3, [_A, _A, 4*_A], stage=2, block='c')

    X = bottleneck_base_v2(X, f=3, filters=[_B, _B, 4*_B], stage=3, block='a', s=2)
    X = bottleneck_top_v2(X, f=3, filters=[_B, _B, 4*_B], stage=3, block='b')
    X = bottleneck_top_v2(X, f=3, filters=[_B, _B, 4*_B], stage=3, block='c')
    X = bottleneck_top_v2(X, f=3, filters=[_B, _B, 4*_B], stage=3, block='d')

    X = bottleneck_base_v2(X, f=3, filters=[_C, _C, 4*_C], stage=4, block='a', s=2)
    X = bottleneck_top_v2(X, f=3, filters=[_C, _C, 4*_C], stage=4, block='b')
    X = bottleneck_top_v2(X, f=3, filters=[_C, _C, 4*_C], stage=4, block='c')
    X = bottleneck_top_v2(X, f=3, filters=[_C, _C, 4*_C], stage=4, block='d')
    X = bottleneck_top_v2(X, f=3, filters=[_C, _C, 4*_C], stage=4, block='e')
    X = bottleneck_top_v2(X, f=3, filters=[_C, _C, 4*_C], stage=4, block='f')

    X = bottleneck_base_v2(X, f=3, filters=[_D, _D, 4*_D], stage=5, block='a', s=2)
    X = bottleneck_top_v2(X, f=3, filters=[_D, _D, 4*_D], stage=5, block='b')
    X = bottleneck_top_v2(X, f=3, filters=[_D, _D, 4*_D], stage=5, block='c')

    X = tf.layers.average_pooling2d(inputs=X, pool_size=(7, 7), strides=1)
    X = tf.layers.flatten(inputs=X)
    X = tf.layers.dense(inputs=X, units=classes, activation='softmax', name='fc' + str(
        classes), kernel_initializer=glorot_uniform(seed=0))

    return X


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='resnet', help='resnet|ssd')
    parser.add_argument('--batch_size', type=int, default=32, help='training batch size')
    parser.add_argument('--dataset', default='imagenet', help='cifar|imagenet')
    parser.add_argument('--scale', default='demo', help='demo|real')
    parser.add_argument('--epoch', type=int, default=50000, help='training stopping '
                                                                 'epoch')
    parser.add_argument('--api', default='keras', help='api module type')
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

    def model_fn(features, labels, mode, params):
        input_tensor = features['input_1']
        logits = ResNet50(input_tensor,
                          classes=_NUM_CLASSES)
        arm = tf.argmax(logits, 1)
        loss = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=logits)
        # loss = tf.losses.sparse_softmax_cross_entropy(
        #     labels=arm, logits=logits)
        predictions = tf.one_hot(arm, _NUM_CLASSES)
        accuracy = tf.metrics.accuracy(labels=labels,
                                       predictions=predictions)
        metrics = {'accuracy': accuracy}
        tf.summary.scalar('accuracy', accuracy[1])
        if mode == tf.estimator.ModeKeys.EVAL:
            return tf.estimator.EstimatorSpec(
                mode, loss=loss, eval_metric_ops=metrics)

        assert mode==tf.estimator.ModeKeys.TRAIN
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode,
                                          loss=loss,
                                          train_op=train_op)

    train_config = tf.estimator.RunConfig()
    strategy = tf.contrib.distribute.OneDeviceStrategy("device:GPU:0")
    new_config = train_config.replace(train_distribute=strategy,
                                      model_dir=model_dir,
                                      save_checkpoints_steps=1000,
                                      keep_checkpoint_max=5)
    params = tf.contrib.training.HParams(
        learning_rate=0.001,
        train_steps=args.epoch,
        min_eval_frequency=660
    )
    classifier = tf.estimator.Estimator(model_fn=model_fn, config=new_config, params=params)

    train_spec = tf.estimator.TrainSpec(input_fn=lambda: input_fn(data_path=train_data,
                                                                  batch_size=args.batch_size,
                                                                  is_training=True),
                                        max_steps=args.epoch)
    eval_spec = tf.estimator.EvalSpec(input_fn=lambda: input_fn(data_path=eval_data,
                                                                batch_size=args.batch_size,
                                                                is_training=False),
                                      steps=660,
                                      throttle_secs=900)
    tf.estimator.train_and_evaluate(estimator=classifier, train_spec=train_spec,
                                    eval_spec=eval_spec)
