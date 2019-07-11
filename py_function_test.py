import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.python.framework import ops


def relu(inputs):
    # Define the op in python
    def _py_relu(x):
        print('forward')
        # quant
        input = x
        max_activation = tf.reduce_max(tf.abs(input))
        input = input * 127. / max_activation
        input = tf.clip_by_value(input, -126., 126.)
        input = tf.round(input)
        input = input * max_activation / 127.
        return input

    # Define the op's gradient in python
    def _py_relu_grad(x):
        print('backward')
        return np.float32(x > 0)

    @tf.custom_gradient
    def _relu(x):
        y = tf.py_function(_py_relu, [x], tf.float32)

        def _relu_grad(dy):
            return dy * tf.py_function(_py_relu_grad, [x], tf.float32)

        return y, _relu_grad

    return _relu(inputs)

# 计算解析梯度
# with tf.Session() as sess:
tf.logging.set_verbosity(tf.logging.INFO)
x = tf.random.normal([3], dtype=np.float32)
with tf.GradientTape() as tape:
    tape.watch(x)
    y = relu(x)
g = tape.gradient(y, x)
# 计算数值梯度
dx_n = 1e-6
dy_n = relu(x + dx_n) - relu(x)
g_n = dy_n / dx_n
print(tf.Session().run([x, y, g, g_n]))
