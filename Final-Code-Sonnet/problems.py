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

train_path = os.getcwd() + '/train_linux_local.txt'
val_path = os.getcwd() + '/test_linux_local.txt'
cache_path = os.getcwd() + '/segmentation_cache/'
def segmentation(trainpath = train_path, valpath = val_path, cache_dir = cache_path, batch_size=5, mode = 'train'):
    trainning_cache_dir = os.path.join(cache_dir, "train")
    validation_cache_dir = os.path.join(cache_dir, "valid")
    if not os.path.exists(trainning_cache_dir):
        os.makedirs(trainning_cache_dir)
    if not os.path.exists(validation_cache_dir):
        os.makedirs(validation_cache_dir)
    if mode == 'train':
        training_generator_inst = seggenerator.SegmentationGenerator(data_path=trainpath,
                                                                 batch_size=batch_size,
                                                                 flip=True,
                                                                 cache_dir=trainning_cache_dir)
        generator = training_generator_inst.generate()
    else:
        validation_generator_inst = seggenerator.SegmentationGenerator(data_path=valpath,
                                                                       batch_size=batch_size,
                                                                       flip=False,
                                                                       cache_dir=validation_cache_dir)
        generator = validation_generator_inst.generate()

    h1 = snt.nets.ConvNet2D(output_channels=[64],
                              kernel_shapes=[3],
                              strides=[1],
                              paddings=[snt.SAME],
                              activation=tf.keras.activations.linear,
                              activate_final=True,
                              initializers=_nn_initializers,
                              use_batch_norm=False,
                              batch_norm_config={'update_ops_collection': None})
    h2 = snt.nets.ConvNet2D(output_channels=[128],
                              kernel_shapes=[3],
                              strides=[1],
                              paddings=[snt.SAME],
                              activation=tf.nn.relu,
                              activate_final=True,
                              initializers=_nn_initializers,
                              use_batch_norm=True,
                              batch_norm_config={'update_ops_collection': None})
    h_2 = lambda x: h2(x, is_training=True)
    h3 = snt.nets.ConvNet2D(output_channels=[128],
                              kernel_shapes=[3],
                              strides=[1],
                              paddings=[snt.SAME],
                              activation=tf.nn.relu,
                              activate_final=True,
                              initializers=_nn_initializers,
                              use_batch_norm=False,
                              batch_norm_config={'update_ops_collection': None})
    h4 = snt.nets.ConvNet2D(output_channels=[128],
                              kernel_shapes=[3],
                              strides=[1],
                              paddings=[snt.SAME],
                              activation=tf.nn.relu,
                              activate_final=True,
                              initializers=_nn_initializers,
                              use_batch_norm=True,
                              batch_norm_config={'update_ops_collection': None})
    h_4 = lambda x: h4(x, is_training=True)

    h5 = snt.nets.ConvNet2D(output_channels=[128],
                            kernel_shapes=[3],
                            strides=[1],
                            paddings=[snt.SAME],
                            activation=tf.nn.relu,
                            activate_final=True,
                            initializers=_nn_initializers,
                            use_batch_norm=True,
                            batch_norm_config={'update_ops_collection': None})
    h_5 = lambda x: h5(x, is_training=True)

    h6 = snt.nets.ConvNet2D(output_channels=[128],
                            kernel_shapes=[3],
                            strides=[1],
                            paddings=[snt.SAME],
                            activation=tf.nn.relu,
                            activate_final=True,
                            initializers=_nn_initializers,
                            use_batch_norm=True,
                            batch_norm_config={'update_ops_collection': None})
    h_6 = lambda x: h6(x, is_training=True)

    h7 = snt.nets.ConvNet2D(output_channels=[128],
                            kernel_shapes=[3],
                            strides=[1],
                            paddings=[snt.SAME],
                            activation=tf.nn.relu,
                            activate_final=True,
                            initializers=_nn_initializers,
                            use_batch_norm=True,
                            batch_norm_config={'update_ops_collection': None})
    h_7 = lambda x: h7(x, is_training=True)
    h8 = snt.nets.ConvNet2D(output_channels=[128],
                            kernel_shapes=[3],
                            strides=[1],
                            paddings=[snt.SAME],
                            activation=tf.nn.relu,
                            activate_final=True,
                            initializers=_nn_initializers,
                            use_batch_norm=True,
                            batch_norm_config={'update_ops_collection': None})
    h_8 = lambda x: h8(x, is_training=True)
    h9 = snt.nets.ConvNet2D(output_channels=[32],
                              kernel_shapes=[1],
                              strides=[1],
                              paddings=[snt.SAME],
                              activation=tf.keras.activations.linear,
                              activate_final=True,
                              initializers=_nn_initializers,
                              use_batch_norm=False,
                              batch_norm_config={'update_ops_collection': None})

    h10 = snt.nets.ConvNet2D(output_channels=[128],
                            kernel_shapes=[3],
                            strides=[1],
                            paddings=[snt.SAME],
                            activation=tf.nn.relu,
                            activate_final=True,
                            initializers=_nn_initializers,
                            use_batch_norm=True,
                            batch_norm_config={'update_ops_collection': None})
    h_10 = lambda x: h10(x, is_training=True)
    h11 = snt.nets.ConvNet2DTranspose(output_channels=[64],
                            output_shapes=[None],
                            kernel_shapes=[4],
                            strides=[2],
                            paddings=[snt.SAME],
                            activation=tf.keras.activations.linear,
                            activate_final=True,
                            initializers=_nn_initializers,
                            use_batch_norm=False,
                            batch_norm_config={'update_ops_collection': None})
    h12 = snt.nets.ConvNet2D(output_channels=[32],
                              kernel_shapes=[1],
                              strides=[1],
                              paddings=[snt.SAME],
                              activation=tf.keras.activations.linear,
                              activate_final=True,
                              initializers=_nn_initializers,
                              use_batch_norm=False,
                              batch_norm_config={'update_ops_collection': None})

    h13 = snt.nets.ConvNet2D(output_channels=[128],
                            kernel_shapes=[3],
                            strides=[1],
                            paddings=[snt.SAME],
                            activation=tf.nn.relu,
                            activate_final=True,
                            initializers=_nn_initializers,
                            use_batch_norm=True,
                            batch_norm_config={'update_ops_collection': None})
    h_13 = lambda x: h13(x, is_training=True)
    h14 = snt.nets.ConvNet2DTranspose(output_channels=[64],
                                      output_shapes=[None],
                                      kernel_shapes=[8],
                                      strides=[4],
                                      paddings=[snt.SAME],
                                      activation=tf.keras.activations.linear,
                                      activate_final=True,
                                      initializers=_nn_initializers,
                                      use_batch_norm=False,
                                      batch_norm_config={'update_ops_collection': None})
    h15 = snt.nets.ConvNet2D(output_channels=[32],
                             kernel_shapes=[1],
                             strides=[1],
                             paddings=[snt.SAME],
                             activation=tf.keras.activations.linear,
                             activate_final=True,
                             initializers=_nn_initializers,
                             use_batch_norm=False,
                             batch_norm_config={'update_ops_collection': None})

    h16 = snt.nets.ConvNet2D(output_channels=[128],
                             kernel_shapes=[3],
                             strides=[1],
                             paddings=[snt.SAME],
                             activation=tf.nn.relu,
                             activate_final=True,
                             initializers=_nn_initializers,
                             use_batch_norm=True,
                             batch_norm_config={'update_ops_collection': None})
    h_16 = lambda x: h16(x, is_training=True)
    h17 = snt.nets.ConvNet2DTranspose(output_channels=[64],
                                      output_shapes=[None],
                                      kernel_shapes=[16],
                                      strides=[8],
                                      paddings=[snt.SAME],
                                      activation=tf.keras.activations.linear,
                                      activate_final=True,
                                      initializers=_nn_initializers,
                                      use_batch_norm=False,
                                      batch_norm_config={'update_ops_collection': None})
    h18 = snt.nets.ConvNet2D(output_channels=[32],
                             kernel_shapes=[1],
                             strides=[1],
                             paddings=[snt.SAME],
                             activation=tf.keras.activations.linear,
                             activate_final=True,
                             initializers=_nn_initializers,
                             use_batch_norm=False,
                             batch_norm_config={'update_ops_collection': None})

    h19 = snt.nets.ConvNet2D(output_channels=[128],
                             kernel_shapes=[3],
                             strides=[1],
                             paddings=[snt.SAME],
                             activation=tf.nn.relu,
                             activate_final=True,
                             initializers=_nn_initializers,
                             use_batch_norm=True,
                             batch_norm_config={'update_ops_collection': None})
    h_19 = lambda x: h19(x, is_training=True)
    h20 = snt.nets.ConvNet2D(output_channels=[128],
                             kernel_shapes=[3],
                             strides=[1],
                             paddings=[snt.SAME],
                             activation=tf.nn.relu,
                             activate_final=True,
                             initializers=_nn_initializers,
                             use_batch_norm=True,
                             batch_norm_config={'update_ops_collection': None})
    h_20 = lambda x: h20(x, is_training=True)
    h21 = snt.nets.ConvNet2D(output_channels=[7],
                             kernel_shapes=[3],
                             strides=[1],
                             paddings=[snt.SAME],
                             activation=tf.keras.activations.linear,
                             activate_final=True,
                             initializers=_nn_initializers,
                             use_batch_norm=False,
                             batch_norm_config={'update_ops_collection': None})

    def build():
        image_batch, label_batch = next(generator)
        image_batch = tf.constant(image_batch, dtype=tf.float32)
        label_batch = tf.constant(label_batch, dtype=tf.float32)
        pool1 = snt.Sequential([h1, h_2, h3])(image_batch)
        o1 = tf.nn.max_pool(pool1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
        pool2 = h_4(o1)
        o2 = tf.nn.max_pool(pool2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
        pool3 = h_5(o2)
        o3 = tf.nn.max_pool(pool3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
        pool4 = h_6(o3)
        s0 = snt.Sequential([h_7, h_8, h9])(pool1)
        s1 = snt.Sequential([h_10, h11, h12])(pool2)
        s2 = snt.Sequential([h_13, h14, h15])(pool3)
        s3 = snt.Sequential([h_16, h17, h18])(pool4)
        o4 = tf.keras.layers.concatenate([s0, s1, s2, s3], axis=-1)
        output = snt.Sequential([h_19, h_20, h21])(o4)
        label_output = tf.reshape(output,(batch_size,-1, 7))
        return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=label_output,
                                                        labels=label_batch))
    def convex_loss():
        v = tf.get_variable("v", shape=[1, 10], dtype=tf.float32,
                            initializer=tf.random_normal_initializer(stddev=0.01))

        # Non-trainable variables.
        target = tf.get_variable("target", shape=[1, 10], dtype=tf.float32,
                                 initializer=tf.random_normal_initializer(stddev=0.01), trainable=False)

        return tf.reduce_mean(tf.clip_by_value(tf.square(v - target), 0, 10))


    return collections.OrderedDict([('Opt_loss', build), ('Aux_loss', convex_loss)])
