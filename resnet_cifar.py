import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_estimator as est
import os
import matplotlib.pyplot as plt

import numpy as np
import time
import tensorflow as tf
from tensorflow.python import keras
from tensorflow.python.keras import layers
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
from tensorflow.python.keras import models
from tensorflow.python.keras import layers
# import cifar10_model
IS_TRAINING = False
_A = 16
_B = 16
_C = 32
_D = 64
# step 1 find data directory
# define directory path
# data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'datasets/imagenet/')
data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'datasets/cifar/')
model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models/cifar/')

# step 2 find tfrecord file
if tf.gfile.Exists(data_dir):
    # return a list of (filepath + filename)s that match the path pattern
    train_data_paths = tf.gfile.Glob(data_dir+"train.tfrecords")
    test_data_paths = tf.gfile.Glob(data_dir+"eval.tfrecords")

    # check if find file, raise errors
    if not len(train_data_paths):
        raise Exception("[Train Error]: unable to find train*.tfrecord file")
    if not len(test_data_paths):
        raise Exception("[Eval Error]: unable to find test*.tfrecord file")
else:
    raise Exception("[Train Error]: unable to find input directory!")

def data_parser(record):
    # how data organised: dict, key to tf.feature
    keys_to_features = {
        "image": tf.FixedLenFeature((), tf.string, default_value=""),
        "label": tf.FixedLenFeature((), tf.int64,
                                    default_value=tf.zeros([], dtype=tf.int64)),
        # "image_height": tf.FixedLenFeature((), tf.int64,
        #                             default_value=tf.zeros([], dtype=tf.int64)),
        # "image_width": tf.FixedLenFeature((), tf.int64,
        #                             default_value=tf.zeros([], dtype=tf.int64)),
    }
    # parsed result on one sample
    parsed = tf.parse_single_example(record, keys_to_features)

    # Perform additional preprocessing on the parsed data.
    # height = tf.cast(parsed["image_height"], tf.int32)
    # width = tf.cast(parsed["image_width"], tf.int32)
    label = tf.cast(parsed["label"], tf.int32)
    # image = tf.image.decode_jpeg(parsed["image"], channels=3)
    image = tf.decode_raw(parsed['image'], tf.uint8)
    image = tf.reshape(image, [32, 32, 3])
    # image = tf.image.crop_and_resize(image, [0, 0, 224, 224], box_ind=0, crop_size=1)
    # image = tf.image.resize_image_with_crop_or_pad(image, 32, 32)
    if IS_TRAINING == True:
        # Pad 4 pixels on each dimension of feature map, done in mini-batch
        image = tf.image.resize_image_with_crop_or_pad(image, 40, 40)
        image = tf.random_crop(image, [32, 32, 3])
        image = tf.image.random_flip_left_right(image)

    image = tf.cast(image, tf.float32)
    image = image/255.

    return label, image

def input_fn(data_path, batch_size, is_training=True):
    # step 3 read tfrecord
    dataset = tf.data.TFRecordDataset(data_path)
    if is_training == True:
        IS_TRAINING = True
    else:
        IS_TRAINING = False
    # test_dataset = tf.data.TFRecordDataset(train_data_paths)
    # another api usage
    # dataset = tf.data.Dataset.from_tensor_slices(train_data_paths)
    # parse string into tf.tensor objects
    # train_dataset = train_dataset.map(lambda record: tf.parse_single_example(record))
    dataset = dataset.map(data_parser)
    # will be used to train 100 epochs
    dataset = dataset.repeat(10)
    # batch size = 128
    dataset = dataset.batch(batch_size)
    # set buffer size = 10000, shuffle the whole buffer
    dataset = dataset.shuffle(1000)

    iterator = dataset.make_one_shot_iterator()

    label, image = iterator.get_next()

    label = tf.one_hot(label, 10)
    # label = tf.reshape(label, (batch_size, 10) )
    # if we need float image rgb or int32, if we need -1 to 1, or 0 to 1
    return {'input_1': image}, label

def residual_block(x, f, filters, stage, block):
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    F1, F2, F3 = filters

    x_shortcut = x

    x = tf.layers.conv2d(x, )
    x = Conv2D(filters=F1, kernel_size=(1, 1), strides=(1, 1), padding='valid',
               name=conv_name_base + '2a', kernel_initializer=glorot_uniform(seed=0))
    x = BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
    X = Activation('relu')(X)

    X = Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding='same',
               name=conv_name_base + '2b', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    X = Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding='valid',
               name=conv_name_base + '2c', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2c')(X)

    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)

    return X

def xception(features, classes=10, is_training=True):
    """
    The Xception architecture written in tf.layers
    Args:
        features: input image tensor
        classes: number of classes to classify images into
        is_training: is training stage or not

    Returns:
        2-D logits prediction output after pooling and activation
    """
    x = tf.layers.conv2d(features, 32, (3, 3), strides=(2, 2), use_bias=False, name='block1_conv1')
    x = tf.layers.batch_normalization(x, training=is_training, name='block1_conv1_bn')
    x = tf.nn.relu(x, name='block1_conv1_act')
    x = tf.layers.conv2d(x, 64, (3, 3), use_bias=False, name='block1_conv2')
    x = tf.layers.batch_normalization(x, training=is_training, name='block1_conv2_bn')
    x = tf.nn.relu(x, name='block1_conv2_act')

    residual = tf.layers.conv2d(x, 128, (1, 1), strides=(2, 2), padding='same', use_bias=False)
    residual = tf.layers.batch_normalization(residual, training=is_training)

    x = tf.layers.separable_conv2d(x, 128, (3, 3), padding='same', use_bias=False, name='block2_sepconv1')
    x = tf.layers.batch_normalization(x, training=is_training, name='block2_sepconv1_bn')
    x = tf.nn.relu(x, name='block2_sepconv2_act')
    x = tf.layers.separable_conv2d(x, 128, (3, 3), padding='same', use_bias=False, name='block2_sepconv2')
    x = tf.layers.batch_normalization(x, training=is_training, name='block2_sepconv2_bn')

    x = tf.layers.max_pooling2d(x, (3, 3), strides=(2, 2), padding='same', name='block2_pool')
    x = tf.add(x, residual, name='block2_add')

    residual = tf.layers.conv2d(x, 256, (1, 1), strides=(2, 2), padding='same', use_bias=False)
    residual = tf.layers.batch_normalization(residual, training=is_training)

    x = tf.nn.relu(x, name='block3_sepconv1_act')
    x = tf.layers.separable_conv2d(x, 256, (3, 3), padding='same', use_bias=False, name='block3_sepconv1')
    x = tf.layers.batch_normalization(x, training=is_training, name='block3_sepconv1_bn')
    x = tf.nn.relu(x, name='block3_sepconv2_act')
    x = tf.layers.separable_conv2d(x, 256, (3, 3), padding='same', use_bias=False, name='block3_sepconv2')
    x = tf.layers.batch_normalization(x, training=is_training, name='block3_sepconv2_bn')

    x = tf.layers.max_pooling2d(x, (3, 3), strides=(2, 2), padding='same', name='block3_pool')
    x = tf.add(x, residual, name="block3_add")

    residual = tf.layers.conv2d(x, 728, (1, 1), strides=(2, 2), padding='same', use_bias=False)
    residual = tf.layers.batch_normalization(residual, training=is_training)

    x = tf.nn.relu(x, name='block4_sepconv1_act')
    x = tf.layers.separable_conv2d(x, 728, (3, 3), padding='same', use_bias=False, name='block4_sepconv1')
    x = tf.layers.batch_normalization(x, training=is_training, name='block4_sepconv1_bn')
    x = tf.nn.relu(x, name='block4_sepconv2_act')
    x = tf.layers.separable_conv2d(x, 728, (3, 3), padding='same', use_bias=False, name='block4_sepconv2')
    x = tf.layers.batch_normalization(x, training=is_training, name='block4_sepconv2_bn')

    x = tf.layers.max_pooling2d(x, (3, 3), strides=(2, 2), padding='same', name='block4_pool')
    x = tf.add(x, residual, name="block4_add")

    for i in range(8):
        residual = x
        prefix = 'block' + str(i + 5)

        x = tf.nn.relu(x, name=prefix + '_sepconv1_act')
        x = tf.layers.separable_conv2d(x, 728, (3, 3), padding='same', use_bias=False, name=prefix + '_sepconv1')
        x = tf.layers.batch_normalization(x, training=is_training, name=prefix + '_sepconv1_bn')
        x = tf.nn.relu(x, name=prefix + '_sepconv2_act')
        x = tf.layers.separable_conv2d(x, 728, (3, 3), padding='same', use_bias=False, name=prefix + '_sepconv2')
        x = tf.layers.batch_normalization(x, training=is_training, name=prefix + '_sepconv2_bn')
        x = tf.nn.relu(x, name=prefix + '_sepconv3_act')
        x = tf.layers.separable_conv2d(x, 728, (3, 3), padding='same', use_bias=False, name=prefix + '_sepconv3')
        x = tf.layers.batch_normalization(x, training=is_training, name=prefix + '_sepconv3_bn')

        x = tf.add(x, residual, name=prefix+"_add")

    residual = tf.layers.conv2d(x, 1024, (1, 1), strides=(2, 2), padding='same', use_bias=False)
    residual = tf.layers.batch_normalization(residual, training=is_training)

    x = tf.nn.relu(x, name='block13_sepconv1_act')
    x = tf.layers.separable_conv2d(x, 728, (3, 3), padding='same', use_bias=False, name='block13_sepconv1')
    x = tf.layers.batch_normalization(x, training=is_training, name='block13_sepconv1_bn')
    x = tf.nn.relu(x, name='block13_sepconv2_act')
    x = tf.layers.separable_conv2d(x, 1024, (3, 3), padding='same', use_bias=False, name='block13_sepconv2')
    x = tf.layers.batch_normalization(x, training=is_training, name='block13_sepconv2_bn')

    x = tf.layers.max_pooling2d(x, (3, 3), strides=(2, 2), padding='same', name='block13_pool')
    x = tf.add(x, residual, name="block13_add")

    x = tf.layers.separable_conv2d(x, 1536, (3, 3), padding='same', use_bias=False, name='block14_sepconv1')
    x = tf.layers.batch_normalization(x, training=is_training, name='block14_sepconv1_bn')
    x = tf.nn.relu(x, name='block14_sepconv1_act')

    x = tf.layers.separable_conv2d(x, 2048, (3, 3), padding='same', use_bias=False, name='block14_sepconv2')
    x = tf.layers.batch_normalization(x, training=is_training, name='block14_sepconv2_bn')
    x = tf.nn.relu(x, name='block14_sepconv2_act')

    x = tf.layers.average_pooling2d(x, (3, 3), (2, 2), name="global_average_pooling")
    x = tf.layers.conv2d(x, 2048, [1, 1], activation=None, name="block15_conv1")
    x = tf.layers.conv2d(x, classes, [1, 1], activation=None, name="block15_conv2")
    x = tf.squeeze(x, axis=[1, 2], name="logits")
    return x

def model_fn(features, labels, mode, params):
    """
    Model_fn for estimator model
    Args:
        features (Tensor): Input features to the model.
        labels (Tensor): Labels tensor for training and evaluation.
        mode (ModeKeys): Specifies if training, evaluation or prediction.
        params (HParams): hyper-parameters for estimator model
    Returns:
        (EstimatorSpec): Model to be run by Estimator.
    """
    if mode == tf.estimator.ModeKeys.TRAIN:
        is_training = True
    else:
        is_training = False
    input_tensor = features["input_1"]
    logits = xception(input_tensor, classes=10, is_training=is_training)
    probs = tf.nn.softmax(logits, name="output_score")
    predictions = tf.argmax(probs, axis=-1, name="output_label")
    onehot_labels = tf.one_hot(tf.cast(labels, tf.int32), 10)
    predictions_dict = {"score": probs,
                        "label": predictions}
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions_output = tf.estimator.export.PredictOutput(predictions_dict)
        return tf.estimator.EstimatorSpec(mode=mode,
                                          predictions=predictions_dict,
                                          export_outputs={
                                              tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: predictions_output
                                          })
    gamma = 1.5
    weights = tf.reduce_sum(tf.multiply(onehot_labels, tf.pow(1. - probs, gamma)), axis=-1)
    loss = tf.losses.softmax_cross_entropy(onehot_labels, logits, weights=weights)
    accuracy = tf.metrics.accuracy(labels=labels,
                                   predictions=predictions)
    if mode == tf.estimator.ModeKeys.TRAIN:
        lr = params.learning_rate
        optimizer = tf.train.RMSPropOptimizer(learning_rate=lr, decay=0.9)
        update_ops = tf.get_collections(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
          train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
        tensors_to_log = {'batch_accuracy': accuracy[1],
                          'logits': logits,
                          'label': labels}
        logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=1000)
        return tf.estimator.EstimatorSpec(mode=mode,
                                          loss=loss,
                                          train_op=train_op,
                                          training_hooks=[logging_hook])
    else:
        eval_metric_ops = {"accuracy": accuracy}
        return tf.estimator.EstimatorSpec(mode=mode,
                                          loss=loss,
                                          eval_metric_ops=eval_metric_ops)

def get_estimator_model(config=None, params=None):
    classifier = tf.estimator.Estimator(model_fn=model_fn,
                                        config=config,
                                        params=params)
    return classifier


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    # train()
    train_config = tf.estimator.RunConfig()
    new_config = train_config.replace(model_dir=model_dir,
                                      save_checkpoints_steps=500,
                                      keep_checkpoint_max=5)
    params = tf.contrib.training.HParams(
        learning_rate=0.01,
        train_steps=1000,
        min_eval_frequency=100
    )
    classifier = get_estimator_model(config=new_config, params=params)
    train_spec = tf.estimator.TrainSpec(input_fn=lambda: input_fn(data_path=train_data_paths,
                                                                  batch_size=16,
                                                                  is_training=True),
                                        max_steps=1000)

    eval_spec = tf.estimator.EvalSpec(input_fn=lambda: input_fn(data_path=test_data_paths,
                                                                batch_size=16),
                                      steps=100,
                                      throttle_secs=900)
    tf.estimator.train_and_evaluate(estimator=classifier,
                                    train_spec=train_spec,
                                    eval_spec=eval_spec)

