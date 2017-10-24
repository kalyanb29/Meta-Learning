import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from keras import backend as K
from keras.optimizers import Optimizer
from keras.layers import Input, Dense
from keras.models import Sequential, Model
from keras.utils import np_utils
# f = np.load('mnist.npz')
# X_train, y_train = f['x_train'], f['y_train']
# X_test, y_test = f['x_test'], f['y_test']
# f.close()
# num_pixels = X_train.shape[1] * X_train.shape[2]
# X_train = X_train.reshape(X_train.shape[0], num_pixels).astype('float32')
# X_test = X_test.reshape(X_test.shape[0], num_pixels).astype('float32')
# X_train = X_train / 255
# X_test = X_test / 255
# y_train = np_utils.to_categorical(y_train)
# y_test = np_utils.to_categorical(y_test)
# Testing
X_train = np.random.uniform(0,1,[224,10])
y_train = np.power(X_train[:,1:6],2)

seed = 7
np.random.seed(seed)
Optimizee_steps = 100
batch_size = 50
n_input = X_train.shape[1]
n_output = y_train.shape[1]
n_hidden1 = 7
batch_num = 3
Optimizer_steps = 20
lstmunit = 1
hidden_size = 1
unroll_nn = 20
lr = 0.001

X = np.zeros((batch_num, batch_size, n_input), np.float32)
Y = np.zeros((batch_num, batch_size, n_output), np.float32)
cp = 0
for ii in range(batch_num):
    X[ii] = X_train[cp: cp + batch_size]
    Y[ii] = y_train[cp + 1: cp + batch_size + 1]
    cp += 10

x = tf.placeholder(tf.float32,[batch_size, n_input])
y_ = tf.placeholder(tf.float32,[batch_size, n_output])

with tf.variable_scope('Optimizee'):
    y1 = Dense(n_hidden1,activation='sigmoid')(x)
    y2 = Dense(n_output,activation='sigmoid')(y1)
    loss = K.pow((y_ - y2),2)/batch_size
    grads = K.gradients(loss,tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Optimizee'))

with tf.variable_scope('Optimizer') as scope2:
    g_new_list = [[] for _ in range(len(grads))]
    softmax_w = tf.get_variable("softmax_w", shape=[hidden_size, 1], dtype=tf.float32)
    softmax_b = tf.get_variable("softmax_b", shape=[1], dtype=tf.float32)
    C_in = tf.get_variable('C', shape=[1,1], dtype=tf.float32, trainable=False)
    H_in = tf.get_variable('H', shape=[1,1], dtype=tf.float32, trainable=False)
    state = []
    for j in range(len(grads)):
        gradsj = tf.reshape(grads[j], [-1, 1])
        for i in range(gradsj.get_shape().as_list()[0]):
            state.append((tf.contrib.rnn.LSTMStateTuple(C_in,H_in),))

    for j in range(len(grads)):
        if j > 0: scope2.reuse_variables()
        cell = tf.contrib.rnn.MultiRNNCell(
            [tf.contrib.rnn.LSTMCell(hidden_size) for _ in range(lstmunit)])
        gradsj = tf.reshape(grads[j], [-1, 1])
        for i in range(gradsj.get_shape().as_list()[0]):
            grad_f_t = tf.slice(gradsj, begin=[i, 0], size=[1, 1])
            cell_out, state_out = cell(grad_f_t, state[i*j])
            g_new_i = tf.add(tf.matmul(cell_out, softmax_w), softmax_b)
            g_new_list[j].append(g_new_i)
        g_new_list[j] = tf.reshape(g_new_list[j], grads[j].shape)

    g_new = g_new_list
# Construct the while loop.
def cond(i):
    return i <100

def body(i):
    # Dequeue a single new example each iteration.
    x, y = q_data.dequeue()
    # Compute the loss and gradient update based on the current example.
    loss = (tf.add(tf.multiply(x, w), b) - y) ** 2
    train_op = optimizer.minimize(loss, global_step=gs)
    # Ensure that the update is applied before continuing.
    with tf.control_dependencies([train_op]):
        return i + 5

loop = tf.while_loop(cond, body, [tf.constant(0)])

data = [k * 1. for k in range(100)]

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for _ in range(1):
        # NOTE: Constructing the enqueue op ahead of time avoids adding
        # (potentially many) copies of `data` to the graph.
        sess.run(enqueue_data_op,
                 feed_dict={placeholder_x: data, placeholder_y: data})
    print (sess.run([gs, w, b]))  # Prints before-loop values.
    sess.run(loop)
    print (sess.run([gs, w, b]))  # Prints after-loop values.