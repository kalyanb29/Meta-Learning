import collections
import os
import sys
import tarfile
import tensorflow as tf
import numpy as np
from keras.utils import np_utils

pathcifar = '/Users/kalyanb/PycharmProjects/Final-Code/Cifar/'
def simple():
    """Simple Problem"""
    with tf.name_scope('Optimizee_loss'):
        def compute_loss():
            with tf.variable_scope("Optimizee_var", reuse=tf.AUTO_REUSE):
                x = tf.get_variable("x",
                                    shape=[],
                                    dtype=tf.float32,
                                    initializer=tf.ones_initializer())
            return tf.square(x)

    with tf.name_scope('Convex_loss'):
        def convex_loss():
            with tf.variable_scope('conv_var', reuse=tf.AUTO_REUSE):
                v = tf.get_variable("v", shape=[1, 10], dtype=tf.float32, initializer=tf.random_normal_initializer(stddev=0.01))

            # Non-trainable variables.
                target = tf.get_variable("target", shape=[1, 10], dtype=tf.float32, initializer=tf.random_uniform_initializer(), trainable=False)

            return tf.reduce_mean(tf.clip_by_value(tf.square(v - target), 0, 10))
    return collections.OrderedDict([('Opt_loss', compute_loss), ('Aux_loss', convex_loss)])

def simple_multi(num_dims=2):
    """Multidimensional simple problem."""
    with tf.name_scope('Optimizee_loss'):
        def get_coordinate(i):
            with tf.variable_scope("Optimizee_var", reuse=tf.AUTO_REUSE):
                a = tf.get_variable("x_{}".format(i),
                                    shape=[],
                                    dtype=tf.float32,
                                    initializer=tf.ones_initializer())
            return a

        def compute_loss():
            coordinates = [get_coordinate(i) for i in range(num_dims)]
            x = tf.concat([tf.expand_dims(c, 0) for c in coordinates], 0)
            return tf.reduce_sum(tf.square(x))
    with tf.name_scope('Convex_loss'):
        def convex_loss():
            with tf.variable_scope('conv_var', reuse=tf.AUTO_REUSE):
                v = tf.get_variable("v", shape=[1, 10], dtype=tf.float32, initializer=tf.random_normal_initializer(stddev=0.01))

            # Non-trainable variables.
                target = tf.get_variable("target", shape=[1, 10], dtype=tf.float32, initializer=tf.random_uniform_initializer(), trainable=False)

            return tf.reduce_mean(tf.clip_by_value(tf.square(v - target), 0, 10))
    return collections.OrderedDict([('Opt_loss', compute_loss), ('Aux_loss', convex_loss)])

def quadratic(batch_size=128, num_dims=10):
    with tf.name_scope('Optimizee_loss'):

        def compute_loss():
            with tf.variable_scope("Optimizee_var",reuse=tf.AUTO_REUSE):
                x = tf.get_variable("x",
                                    shape=[batch_size, num_dims],
                                    dtype=tf.float32,
                                    initializer=tf.random_normal_initializer(stddev=0.01))

                # Non-trainable variables.
                w = tf.get_variable("w",
                                    shape=[batch_size, num_dims, num_dims],
                                    dtype=tf.float32,
                                    initializer=tf.random_uniform_initializer(),
                                    trainable=False)
                y = tf.get_variable("y",
                                    shape=[batch_size, num_dims],
                                    dtype=tf.float32,
                                    initializer=tf.random_uniform_initializer(),
                                    trainable=False)

            product = tf.squeeze(tf.matmul(w, tf.expand_dims(x, -1)))
            return tf.reduce_mean(tf.reduce_sum((product - y) ** 2, 1))
    with tf.name_scope('Convex_loss'):

        def convex_loss():
            with tf.variable_scope('conv_var', reuse=tf.AUTO_REUSE):
                v = tf.get_variable("v", shape=[1, 10], dtype=tf.float32, initializer=tf.random_normal_initializer(stddev=0.01))

            # Non-trainable variables.
                target = tf.get_variable("target", shape=[1, 10], dtype=tf.float32, initializer=tf.random_uniform_initializer(), trainable=False)

            return tf.reduce_mean(tf.clip_by_value(tf.square(v - target), 0, 10))
    return collections.OrderedDict([('Opt_loss', compute_loss), ('Aux_loss', convex_loss)])

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

    with tf.name_scope('Optimizee_loss'):
        def compute_loss():
            loss = 0
            for i, build_fn in enumerate(build_fns):
                with tf.variable_scope("problem_{}".format(i)):
                    loss_p = build_fn()
                if weights:
                    loss_p *= weights[i]
                    loss += loss_p
            return loss
    with tf.name_scope('Convex_loss'):

        def convex_loss():
            with tf.variable_scope('conv_var', reuse=tf.AUTO_REUSE):
                v = tf.get_variable("v", shape=[1, 10], dtype=tf.float32, initializer=tf.random_normal_initializer(stddev=0.01))

            # Non-trainable variables.
                target = tf.get_variable("target", shape=[1, 10], dtype=tf.float32, initializer=tf.random_uniform_initializer(), trainable=False)

            return tf.reduce_mean(tf.clip_by_value(tf.square(v - target), 0, 10))
    return collections.OrderedDict([('Opt_loss', compute_loss), ('Aux_loss', convex_loss)])


def _xent_loss(output, labels):
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=output,
                                                        labels=labels)
    return tf.reduce_mean(loss)


def mnist(layers = 20, activation="sigmoid", batch_size=20, mode = "train"):
    """Mnist classification with a multi-layer perceptron."""
    if activation == "sigmoid":
        activation_op = tf.sigmoid
    elif activation == "relu":
        activation_op = tf.nn.relu
    else:
        raise ValueError("{} activation not supported".format(activation))

    f = np.load('mnist.npz')
    X_train, y_train = f['x_train'], f['y_train']
    X_test, y_test = f['x_test'], f['y_test']
    f.close()
    num_pixels = X_train.shape[1] * X_train.shape[2]
    X_train = X_train.reshape(X_train.shape[0], num_pixels).astype('float32')
    X_test = X_test.reshape(X_test.shape[0], num_pixels).astype('float32')
    if mode == "train":
        X = X_train / 255
        y = np_utils.to_categorical(y_train)
    else:
        X = X_test / 255
        y = np_utils.to_categorical(y_test)
    images = tf.constant(X, dtype=tf.float32)
    labels = tf.constant(y, dtype=tf.int64)

    with tf.name_scope('Optimizee_loss'):
        def compute_loss():
            indices = tf.random_uniform([batch_size], 0, images.shape[0], tf.int64)
            batch_images = tf.gather(images, indices)
            batch_labels = tf.gather(labels, indices)
            with tf.variable_scope('MLP',reuse=tf.AUTO_REUSE):
                W_in = tf.get_variable("W_in",
                                        shape=[images.shape[1], layers],
                                        dtype=tf.float32,
                                        initializer=tf.random_normal_initializer(stddev=0.01))
                b_in = tf.get_variable("b_in",
                                        shape=[layers, ],
                                        dtype=tf.float32,
                                        initializer=tf.random_normal_initializer(stddev=0.01))
                W_out = tf.get_variable("W_out",
                                        shape = [layers, labels.shape[1]],
                                        dtype = tf.float32,
                                        initializer = tf.random_normal_initializer(stddev=0.01))
                b_out = tf.get_variable("b_out",
                                        shape=[labels.shape[1], ],
                                        dtype=tf.float32,
                                        initializer=tf.random_normal_initializer(stddev=0.01))
            layer_out = activation_op(tf.add(tf.matmul(batch_images, W_in), b_in))
            output = tf.add(tf.matmul(layer_out, W_out), b_out)
            return _xent_loss(output, batch_labels)

    with tf.name_scope('Convex_loss'):

        def convex_loss():
            with tf.variable_scope('conv_var', reuse=tf.AUTO_REUSE):
                v = tf.get_variable("v", shape=[1, 10], dtype=tf.float32, initializer=tf.random_normal_initializer(stddev=0.01))

            # Non-trainable variables.
                target = tf.get_variable("target", shape=[1, 10], dtype=tf.float32, initializer=tf.random_uniform_initializer(), trainable=False)

            return tf.reduce_mean(tf.clip_by_value(tf.square(v - target), 0, 10))
    return collections.OrderedDict([('Opt_loss', compute_loss), ('Aux_loss', convex_loss)])


CIFAR10_URL = "http://www.cs.toronto.edu/~kriz"
CIFAR10_FILE = "cifar-10-binary.tar.gz"
CIFAR10_FOLDER = "cifar-10-batches-bin"


def _open_cifar10(path):
    """Download and extract the tarball from Alex's website."""
    filepath = os.path.join(path, CIFAR10_FILE)
    tarfile.open(filepath, "r:gz").extractall(path)
    return


def cifar10(path = pathcifar, activation = "sigmoid", conv_channels=(16, 16, 16), linear_layers=32,
            batch_size=128, num_threads=4, min_queue_examples=1000, mode="train"):
    """Cifar10 classification with a convolutional network."""

  # Data.
    _open_cifar10(path)

    if activation == "sigmoid":
        activation_op = tf.sigmoid
    elif activation == "relu":
        activation_op = tf.nn.relu
    else:
        raise ValueError("{} activation not supported".format(activation))
  # Read images and labels from disk.
    if mode == "train":
        filenames = [os.path.join(path, CIFAR10_FOLDER, "data_batch_{}.bin".format(i)) for i in range(1, 6)]
    elif mode == "test":
        filenames = [os.path.join(path, "test_batch.bin")]
    else:
        raise ValueError("Mode {} not recognised".format(mode))

    depth = 3
    height = 32
    width = 32
    label_bytes = 1
    image_bytes = depth * height * width
    record_bytes = label_bytes + image_bytes
    reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
    _, record = reader.read(tf.train.string_input_producer(filenames))
    record_bytes = tf.decode_raw(record, tf.uint8)

    label = tf.cast(tf.slice(record_bytes, [0], [label_bytes]), tf.int32)
    raw_image = tf.slice(record_bytes, [label_bytes], [image_bytes])
    image = tf.cast(tf.reshape(raw_image, [depth, height, width]), tf.float32)
    # height x width x depth.
    image = tf.transpose(image, [1, 2, 0])
    image = tf.div(image, 255)

    queue = tf.RandomShuffleQueue(capacity=min_queue_examples + 3 * batch_size,
                                  min_after_dequeue=min_queue_examples,
                                  dtypes=[tf.float32, tf.int32],
                                  shapes=[image.get_shape(), label.get_shape()])
    enqueue_ops = [queue.enqueue([image, label]) for _ in range(num_threads)]
    tf.train.add_queue_runner(tf.train.QueueRunner(queue, enqueue_ops))

    with tf.name_scope('Optimizee_loss'):
        def compute_loss():
            image_batch, label_batch = queue.dequeue_many(batch_size)
            label_batch = tf.reshape(label_batch, [batch_size])
            output = image_batch
            with tf.variable_scope('ConvMLP', reuse=tf.AUTO_REUSE):
                conv1_w = tf.get_variable("conv1_w",
                                           shape = [5, 5, depth, conv_channels(1)],
                                           dtype=tf.float32,
                                           initializer=tf.random_normal_initializer(stddev=0.01))
                conv1_b = tf.get_variable("conv1_b",
                                           shape = [conv_channels(1), ],
                                           dtype=tf.float32,
                                           initializer=tf.random_normal_initializer(stddev=0.01))
                conv1_beta = tf.get_variable("conv1_beta",
                                              shape = [1, 1, 1, conv_channels(1)],
                                              dtype=tf.float32,
                                              initializer=tf.random_normal_initializer(stddev=0.01))
                conv2_w = tf.get_variable("conv2_w",
                                          shape=[5, 5, conv_channels(1), conv_channels(2)],
                                          dtype=tf.float32,
                                          initializer=tf.random_normal_initializer(stddev=0.01))
                conv2_b = tf.get_variable("conv2_b",
                                          shape=[conv_channels(2), ],
                                          dtype=tf.float32,
                                          initializer=tf.random_normal_initializer(stddev=0.01))
                conv2_beta = tf.get_variable("conv2_beta",
                                             shape=[1, 1, 1, conv_channels(2)],
                                             dtype=tf.float32,
                                             initializer=tf.random_normal_initializer(stddev=0.01))
                conv3_w = tf.get_variable("conv3_w",
                                          shape=[5, 5, conv_channels(2), conv_channels(3)],
                                          dtype=tf.float32,
                                          initializer=tf.random_normal_initializer(stddev=0.01))
                conv3_b = tf.get_variable("conv3_b",
                                          shape=[conv_channels(3), ],
                                          dtype=tf.float32,
                                          initializer=tf.random_normal_initializer(stddev=0.01))
                conv3_beta = tf.get_variable("conv3_beta",
                                             shape=[1, 1, 1, conv_channels(3)],
                                             dtype=tf.float32,
                                             initializer=tf.random_normal_initializer(stddev=0.01))
                output = tf.nn.convolution(output, conv1_w, padding='SAME', strides=[1, 1, 1, 1])
                output = tf.nn.relu(tf.nn.bias_add(output,conv1_b))
                output = tf.nn.max_pool(output, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
                output = tf.nn.lrn(output,)
                output = tf.layers.batch_normalization(output,trainable=True)
                output = tf.layers.flatten(output)
                W_in = tf.get_variable("W_in",
                                        shape=[output.shape[1],linear_layers],
                                        dtype=tf.float32,
                                        initializer=tf.random_normal_initializer(stddev=0.01))
                b_in = tf.get_variable("b_in",
                                        shape=[linear_layers, ],
                                        dtype=tf.float32,
                                        initializer=tf.random_normal_initializer(stddev=0.01))
                W_out = tf.get_variable("W_out",
                                        shape=[linear_layers, 10],
                                        dtype=tf.float32,
                                        initializer=tf.random_normal_initializer(stddev=0.01))
                b_out = tf.get_variable("b_out",
                                        shape=[10, ],
                                        dtype=tf.float32,
                                        initializer=tf.random_normal_initializer(stddev=0.01))
            layer_out = activation_op(tf.add(tf.matmul(output, W_in), b_in))
            output = tf.add(tf.matmul(layer_out, W_out), b_out)
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=output,
                                                           labels=label_batch)
            return tf.reduce_mean(loss)
    with tf.name_scope('Convex_loss'):

        def convex_loss():
            with tf.variable_scope('conv_var', reuse=tf.AUTO_REUSE):
                v = tf.get_variable("v", shape=[1, 10], dtype=tf.float32, initializer=tf.random_normal_initializer(stddev=0.01))

            # Non-trainable variables.
                target = tf.get_variable("target", shape=[1, 10], dtype=tf.float32, initializer=tf.random_uniform_initializer(), trainable=False)

            return tf.reduce_mean(tf.clip_by_value(tf.square(v - target), 0, 10))
    return collections.OrderedDict([('Opt_loss', compute_loss), ('Aux_loss', convex_loss)])