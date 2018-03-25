from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tarfile
import sys
import collections

from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import sonnet as snt
import tensorflow as tf
import numpy as np

from tensorflow.contrib.learn.python.learn.datasets import mnist as mnist_dataset
import seggenerator


_nn_initializers = {
    "w": tf.random_normal_initializer(mean=0, stddev=0.01),
    "b": tf.random_normal_initializer(mean=0, stddev=0.01),
}


def simple():
    """Simple problem: f(x) = x^2."""

    def build():
        """Builds loss graph."""
        x = tf.get_variable(
            "x",
            shape=[],
            dtype=tf.float32,
            initializer=tf.ones_initializer())
        return tf.square(x, name="x_squared")

    def convex_loss():
        v = tf.get_variable("v", shape=[1, 10], dtype=tf.float32,
                                initializer=tf.random_normal_initializer(stddev=0.01))

        # Non-trainable variables.
        target = tf.get_variable("target", shape=[1, 10], dtype=tf.float32,
                                     initializer=tf.random_normal_initializer(stddev=0.01), trainable=False)

        return tf.reduce_mean(tf.clip_by_value(tf.square(v - target), 0, 10))


    return collections.OrderedDict([('Opt_loss', build), ('Aux_loss', convex_loss)])



def simple_multi_optimizer(num_dims=2):
    """Multidimensional simple problem."""

    def get_coordinate(i):
        return tf.get_variable("x_{}".format(i),
                            shape=[],
                            dtype=tf.float32,
                            initializer=tf.ones_initializer())

    def build():
        coordinates = [get_coordinate(i) for i in xrange(num_dims)]
        x = tf.concat([tf.expand_dims(c, 0) for c in coordinates], 0)
        return tf.reduce_sum(tf.square(x, name="x_squared"))

    def convex_loss():
        v = tf.get_variable("v", shape=[1, 10], dtype=tf.float32,
                                initializer=tf.random_normal_initializer(stddev=0.01))

        # Non-trainable variables.
        target = tf.get_variable("target", shape=[1, 10], dtype=tf.float32,
                                     initializer=tf.random_normal_initializer(stddev=0.01), trainable=False)

        return tf.reduce_mean(tf.clip_by_value(tf.square(v - target), 0, 10))


    return collections.OrderedDict([('Opt_loss', build), ('Aux_loss', convex_loss)])


def quadratic(batch_size=128, num_dims=10, stddev=0.01, dtype=tf.float32):
    """Quadratic problem: f(x) = ||Wx - y||."""

    def build():
        """Builds loss graph."""

        # Trainable variable.
        x = tf.get_variable("x",
                            shape=[batch_size, num_dims],
                            dtype=dtype,
                            initializer=tf.random_normal_initializer(stddev=stddev))

        # Non-trainable variables.
        w = tf.get_variable("w",
                            shape=[batch_size, num_dims, num_dims],
                            dtype=dtype,
                            initializer=tf.random_uniform_initializer(),
                            trainable=False)
        y = tf.get_variable("y",
                            shape=[batch_size, num_dims],
                            dtype=dtype,
                            initializer=tf.random_uniform_initializer(),
                            trainable=False)

        product = tf.squeeze(tf.matmul(w, tf.expand_dims(x, -1)))
        return tf.reduce_mean(tf.reduce_sum((product - y) ** 2, 1))
    def convex_loss():
        v = tf.get_variable("v", shape=[1, 10], dtype=tf.float32,
                                initializer=tf.random_normal_initializer(stddev=0.01))

        # Non-trainable variables.
        target = tf.get_variable("target", shape=[1, 10], dtype=tf.float32,
                                     initializer=tf.random_normal_initializer(stddev=0.01), trainable=False)

        return tf.reduce_mean(tf.clip_by_value(tf.square(v - target), 0, 10))


    return collections.OrderedDict([('Opt_loss', build), ('Aux_loss', convex_loss)])



def ensemble(problems, weights=None):
    """Ensemble of problems.

    Args:
        problems: List of problems. Each problem is specified by a dict containing
            the keys 'name' and 'options'.
        weights: Optional list of weights for each problem.

    Returns:
        Sum of (weighted) losses.

    Raises:
        ValueError: If weights has an incorrect length.
    """
    if weights and len(weights) != len(problems):
        raise ValueError("len(weights) != len(problems)")

    build_fns = [getattr(sys.modules[__name__], p["name"])(**p["options"])
                for p in problems]

    def build():
        loss = 0
        for i, build_fn in enumerate(build_fns):
            with tf.variable_scope("problem_{}".format(i)):
                 loss_p = build_fn()
                 if weights:
                    loss_p *= weights[i]
                    loss += loss_p
        return loss

    def convex_loss():
        v = tf.get_variable("v",
                                 shape=[1, 10],
                                 dtype=tf.float32,
                                 initializer=tf.random_normal_initializer(stddev=0.01))

        # Non-trainable variables.
        target = tf.get_variable("target",
                                      shape=[1, 10],
                                      dtype=tf.float32,
                                      initializer=tf.random_normal_initializer(stddev=0.01), trainable=False)

        return tf.reduce_mean(tf.clip_by_value(tf.square(v - target), 0, 10))

    return collections.OrderedDict([('Opt_loss', build), ('Aux_loss', convex_loss)])


def _xent_loss(output, labels):
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=output,
                                                        labels=labels)
    return tf.reduce_mean(loss)


def mnist(layers,  # pylint: disable=invalid-name
          activation="relu",
          batch_size=128,
          mode="train"):
    """Mnist classification with a multi-layer perceptron."""

    if activation == "sigmoid":
        activation_op = tf.sigmoid
    elif activation == "relu":
        activation_op = tf.nn.relu
    else:
        raise ValueError("{} activation not supported".format(activation))

    # Data.
    proxy1 = 'http://proxy:8080'
    proxy2 = 'https://proxy:8080'
    os.environ['http_proxy'] = proxy1
    os.environ['HTTP_PROXY'] = proxy1
    os.environ['https_proxy'] = proxy2
    os.environ['HTTPS_PROXY'] = proxy2
    data = mnist_dataset.load_mnist()
    data = getattr(data, mode)
    images = tf.constant(data.images, dtype=tf.float32, name="MNIST_images")
    images = tf.reshape(images, [-1, 28, 28, 1])
    labels = tf.constant(data.labels, dtype=tf.int64, name="MNIST_labels")

    # Network.
    mlp = snt.nets.MLP(list(layers) + [10],
                     activation=activation_op,
                     initializers=_nn_initializers)
    network = snt.Sequential([snt.BatchFlatten(), mlp])

    def build():
        indices = tf.random_uniform([batch_size], 0, data.num_examples, tf.int64)
        batch_images = tf.gather(images, indices)
        batch_labels = tf.gather(labels, indices)
        output = network(batch_images)
        return _xent_loss(output, batch_labels)

    def convex_loss():
        v = tf.get_variable("v", shape=[1, 10], dtype=tf.float32,
                                initializer=tf.random_normal_initializer(stddev=0.01))

        # Non-trainable variables.
        target = tf.get_variable("target", shape=[1, 10], dtype=tf.float32,
                                     initializer=tf.random_normal_initializer(stddev=0.01), trainable=False)

        return tf.reduce_mean(tf.clip_by_value(tf.square(v - target), 0, 10))


    return collections.OrderedDict([('Opt_loss', build), ('Aux_loss', convex_loss)])


CIFAR10_URL = "http://www.cs.toronto.edu/~kriz"
CIFAR10_FILE = "cifar-10-binary.tar.gz"
CIFAR10_FOLDER = "cifar-10-batches-bin"
proxy1 = 'http://proxy:8080'
proxy2 = 'https://proxy:8080'
os.environ['http_proxy'] = proxy1
os.environ['HTTP_PROXY'] = proxy1
os.environ['https_proxy'] = proxy2
os.environ['HTTPS_PROXY'] = proxy2

def _maybe_download_cifar10(path):
    """Download and extract the tarball from Alex's website."""
    filepath = os.path.join(path, CIFAR10_FILE)
    tarfile.open(filepath, "r:gz").extractall(path)
    return


def cifar10(path,  # pylint: disable=invalid-name
            conv_channels=None,
            linear_layers=None,
            batch_norm=True,
            batch_size=128,
            mode="train"):
    """Cifar10 classification with a convolutional network."""
    import cifar10
    cifar10.data_path = "CIFAR-10-data/"
    cifar10.maybe_download_and_extract()

    images_train, cls_train, labels_train = cifar10.load_training_data()
    images = tf.constant(images_train, dtype=tf.float32, name="CIFAR_images")
    labels = tf.constant(cls_train, dtype=tf.int64, name="CIFAR_labels")

    # Network.
    def _conv_activation(x):  # pylint: disable=invalid-name
        return tf.nn.max_pool(tf.nn.relu(x),
                              ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1],
                              padding="SAME")

    conv = snt.nets.ConvNet2D(output_channels=conv_channels,
                              kernel_shapes=[5],
                              strides=[1],
                              paddings=[snt.SAME],
                              activation=_conv_activation,
                              activate_final=True,
                              initializers=_nn_initializers,
                              use_batch_norm=batch_norm,
                              batch_norm_config={'update_ops_collection': None})

    if batch_norm:
        conv1 = lambda x: conv(x, is_training=True)
        linear_activation = lambda x: tf.nn.relu(snt.BatchNorm(update_ops_collection = None)(x, is_training=True))
    else:
        conv1 = conv
        linear_activation = tf.nn.relu

    mlp = snt.nets.MLP(list(linear_layers) + [10],
                       activation=linear_activation,
                       initializers=_nn_initializers)
    network = snt.Sequential([conv1, snt.BatchFlatten(), mlp])

    def build():
        indices = tf.random_uniform([batch_size], 0, len(images_train)-1, tf.int64)
        image_batch = tf.gather(images, indices)
        label_batch = tf.gather(labels, indices)
        output = network(image_batch)
        return _xent_loss(output, label_batch)

    def convex_loss():
        v = tf.get_variable("v", shape=[1, 10], dtype=tf.float32,
                                initializer=tf.random_normal_initializer(stddev=0.01))

        # Non-trainable variables.
        target = tf.get_variable("target", shape=[1, 10], dtype=tf.float32,
                                     initializer=tf.random_normal_initializer(stddev=0.01), trainable=False)

        return tf.reduce_mean(tf.clip_by_value(tf.square(v - target), 0, 10))


    return collections.OrderedDict([('Opt_loss', build), ('Aux_loss', convex_loss)])

