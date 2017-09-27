import numpy as np
import tensorflow as tf
from keras.utils import np_utils
f = np.load('mnist.npz')
X_train, y_train = f['x_train'], f['y_train']
X_test, y_test = f['x_test'], f['y_test']
f.close()
num_pixels = X_train.shape[1] * X_train.shape[2]
X_train = X_train.reshape(X_train.shape[0], num_pixels).astype('float32')
X_test = X_test.reshape(X_test.shape[0], num_pixels).astype('float32')
X_train = X_train / 255
X_test = X_test / 255
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
seed = 7
np.random.seed(seed)
TRAINING_STEPS = 10
m = 128
# Graph Construction

x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

def update_grad_sgd(loss,var,lr = 0.001):
    grad = tf.gradients(loss,var)
    l = [x*-lr for x in grad]
    var += l[0]
    return var
def update_grad_rms(loss,var,state,lr = 0.001,decay = 0.99):
    grad = tf.gradients(loss,var)
    if state is None:
        state = tf.zeros(tf.shape(var),dtype = tf.float32)
    q = [tf.pow(x, 2) for x in grad]
    p = [x*(1-decay) for x in q]
    state = tf.scalar_mul(decay,state) + p[0]
    l = [x*-lr for x in grad]
    var += l[0] / (tf.sqrt(state)+1e-5)
    return var,state
def update_grad_rnn(loss,var,state,LAYERS = 2, STATE_SIZE = 20):
    cell = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.LSTMCell(STATE_SIZE) for _ in range(LAYERS)])
    cell = tf.contrib.rnn.InputProjectionWrapper(cell, STATE_SIZE)
    cell = tf.contrib.rnn.OutputProjectionWrapper(cell, 1)
    cell = tf.make_template('cell', cell)
    update = [None] * len(var)
    for i in range(len(var)):
        grad = tf.reshape(tf.gradients(loss,var[i])[0],[-1])
        gradient = tf.expand_dims(grad, axis=1)
        state_in = state[i]
        if state_in is None:
            state_in = [[tf.zeros([tf.shape(gradient)[0],STATE_SIZE],dtype = tf.float32)] * 2] * LAYERS
        update[i], state[i] = cell(gradient, state_in)
        var[i] += tf.reshape(tf.squeeze(update[i], axis=[1]),tf.shape(var[i]))
    # Squeeze to make it a single batch again.
    return var, state

def learn_sgd(x,y_,keep_prob,W_conv1,b_conv1,W_conv2,b_conv2,W_fc1,b_fc1,W_fc2,b_fc2,state_wconv1,state_bconv1,state_wconv2,state_bconv2,state_wfc1,state_bfc1,state_wfc2,state_bfc2):
    if W_conv1 is None:
        W_conv1 = weight_variable([5, 5, 1, 32])
        b_conv1 = bias_variable([32])
        W_conv2 = weight_variable([5, 5, 32, 64])
        b_conv2 = bias_variable([64])
        W_fc1 = weight_variable([7 * 7 * 64, 1024])
        b_fc1 = bias_variable([1024])
        keep_prob = tf.placeholder(tf.float32)
        W_fc2 = weight_variable([1024, 10])
        b_fc2 = bias_variable([10])
        state_wconv1 = None
        state_bconv1 = None
        state_wconv2 = None
        state_bconv2 = None
        state_wfc1 = None
        state_bfc1 = None
        state_wfc2 = None
        state_bfc2 = None
    x_image = tf.reshape(x, [-1, 28, 28, 1])
    h_conv1 = tf.nn.softmax(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)
    h_conv2 = tf.nn.softmax(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
    h_fc1 = tf.nn.softmax(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
    loss = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv)
    W_conv1 = update_grad_sgd(loss,W_conv1)
    b_conv1 = update_grad_sgd(loss,b_conv1)
    W_conv2 = update_grad_sgd(loss,W_conv2)
    b_conv2 = update_grad_sgd(loss,b_conv2)
    W_fc1 = update_grad_sgd(loss,W_fc1)
    b_fc1 = update_grad_sgd(loss,b_fc1)
    W_fc2 = update_grad_sgd(loss,W_fc2)
    b_fc2 = update_grad_sgd(loss,b_fc2)
    state_wconv1 = None
    state_bconv1 = None
    state_wconv2 = None
    state_bconv2 = None
    state_wfc1 = None
    state_bfc1 = None
    state_wfc2 = None
    state_bfc2 = None
    return loss,keep_prob,W_conv1,b_conv1,W_conv2,b_conv2,W_fc1,b_fc1,W_fc2,b_fc2,state_wconv1,state_bconv1,state_wconv2,state_bconv2,state_wfc1,state_bfc1,state_wfc2,state_bfc2

def learn_rms(x,y_,keep_prob,W_conv1,b_conv1,W_conv2,b_conv2,W_fc1,b_fc1,W_fc2,b_fc2,state_wconv1,state_bconv1,state_wconv2,state_bconv2,state_wfc1,state_bfc1,state_wfc2,state_bfc2):
    if W_conv1 is None:
        W_conv1 = weight_variable([5, 5, 1, 32])
        b_conv1 = bias_variable([32])
        W_conv2 = weight_variable([5, 5, 32, 64])
        b_conv2 = bias_variable([64])
        W_fc1 = weight_variable([7 * 7 * 64, 1024])
        b_fc1 = bias_variable([1024])
        keep_prob = tf.placeholder(tf.float32)
        W_fc2 = weight_variable([1024, 10])
        b_fc2 = bias_variable([10])
        state_wconv1 = None
        state_bconv1 = None
        state_wconv2 = None
        state_bconv2 = None
        state_wfc1 = None
        state_bfc1 = None
        state_wfc2 = None
        state_bfc2 = None
    x_image = tf.reshape(x, [-1, 28, 28, 1])
    h_conv1 = tf.nn.softmax(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)
    h_conv2 = tf.nn.softmax(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
    h_fc1 = tf.nn.softmax(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
    loss = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv)
    W_conv1, state_wconv1 = update_grad_rms(loss,W_conv1,state_wconv1)
    b_conv1, state_bconv1 = update_grad_rms(loss, b_conv1, state_bconv1)
    W_conv2, state_wconv2 = update_grad_rms(loss, W_conv2, state_wconv2)
    b_conv2, state_bconv2 = update_grad_rms(loss, b_conv2, state_bconv2)
    W_fc1, state_wfc1 = update_grad_rms(loss, W_fc1, state_wfc1)
    b_fc1, state_bfc1 = update_grad_rms(loss, b_fc1, state_bfc1)
    W_fc2, state_wfc2 = update_grad_rms(loss, W_fc2, state_wfc2)
    b_fc2, state_bfc2 = update_grad_rms(loss, b_fc2, state_bfc2)
    return loss,keep_prob,W_conv1,b_conv1,W_conv2,b_conv2,W_fc1,b_fc1,W_fc2,b_fc2,state_wconv1,state_bconv1,state_wconv2,state_bconv2,state_wfc1,state_bfc1,state_wfc2,state_bfc2

def learn_rnn(x,y_,keep_prob,W_conv1,b_conv1,W_conv2,b_conv2,W_fc1,b_fc1,W_fc2,b_fc2,state_wconv1,state_bconv1,state_wconv2,state_bconv2,state_wfc1,state_bfc1,state_wfc2,state_bfc2):
    if W_conv1 is None:
        W_conv1 = weight_variable([5, 5, 1, 32])
        b_conv1 = bias_variable([32])
        W_conv2 = weight_variable([5, 5, 32, 64])
        b_conv2 = bias_variable([64])
        W_fc1 = weight_variable([7 * 7 * 64, 1024])
        b_fc1 = bias_variable([1024])
        keep_prob = tf.placeholder(tf.float32)
        W_fc2 = weight_variable([1024, 10])
        b_fc2 = bias_variable([10])
        state_wconv1 = None
        state_bconv1 = None
        state_wconv2 = None
        state_bconv2 = None
        state_wfc1 = None
        state_bfc1 = None
        state_wfc2 = None
        state_bfc2 = None
    x_image = tf.reshape(x, [-1, 28, 28, 1])
    h_conv1 = tf.nn.softmax(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)
    h_conv2 = tf.nn.softmax(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
    h_fc1 = tf.nn.softmax(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
    loss = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv)
    var_all = [W_conv1,b_conv1,W_conv2,b_conv2,W_fc1,b_fc1,W_fc2,b_fc2]
    state_all = [state_wconv1,state_bconv1,state_wconv2,state_bconv2,state_wfc1,state_bfc1,state_wfc2,state_bfc2]
    var_all,state_all = update_grad_rnn(loss,var_all,state_all)
    W_conv1 = var_all[0]
    b_conv1 = var_all[1]
    W_conv2 = var_all[2]
    b_conv2 = var_all[3]
    W_fc1 = var_all[4]
    b_fc1 = var_all[5]
    W_fc2 = var_all[6]
    b_fc2 = var_all[7]
    state_wconv1 = state_all[0]
    state_bconv1 = state_all[1]
    state_wconv2 = state_all[2]
    state_bconv2 = state_all[3]
    state_wfc1 = state_all[4]
    state_bfc1 = state_all[5]
    state_wfc2 = state_all[6]
    state_bfc2 = state_all[7]
    # W_conv1, state_wconv1 = update_grad_rnn(loss,W_conv1,state_wconv1)
    # b_conv1, state_bconv1 = update_grad_rnn(loss, b_conv1, state_bconv1)
    # W_conv2, state_wconv2 = update_grad_rnn(loss, W_conv2, state_wconv2)
    # b_conv2, state_bconv2 = update_grad_rnn(loss, b_conv2, state_bconv2)
    # W_fc1, state_wfc1 = update_grad_rnn(loss, W_fc1, state_wfc1)
    # b_fc1, state_bfc1 = update_grad_rnn(loss, b_fc1, state_bfc1)
    # W_fc2, state_wfc2 = update_grad_rnn(loss, W_fc2, state_wfc2)
    # b_fc2, state_bfc2 = update_grad_rnn(loss, b_fc2, state_bfc2)
    return loss,keep_prob,W_conv1,b_conv1,W_conv2,b_conv2,W_fc1,b_fc1,W_fc2,b_fc2,state_wconv1,state_bconv1,state_wconv2,state_bconv2,state_wfc1,state_bfc1,state_wfc2,state_bfc2

def learn(optimizer):
   losses = []
   W_conv1 = weight_variable([5, 5, 1, 32])
   b_conv1 = bias_variable([32])
   W_conv2 = weight_variable([5, 5, 32, 64])
   b_conv2 = bias_variable([64])
   W_fc1 = weight_variable([7 * 7 * 64, 1024])
   b_fc1 = bias_variable([1024])
   keep_prob = 0.5
   W_fc2 = weight_variable([1024, 10])
   b_fc2 = bias_variable([10])
   state_wconv1 = None
   state_bconv1 = None
   state_wconv2 = None
   state_bconv2 = None
   state_wfc1 = None
   state_bfc1 = None
   state_wfc2 = None
   state_bfc2 = None
   for _ in range(TRAINING_STEPS):
        chosen_idx = np.random.choice(X_train.shape[0], replace=False, size=m)
        x_s = X_train[chosen_idx,]
        y_s = y_train[chosen_idx,]
        loss,keep_prob,W_conv1,b_conv1,W_conv2,b_conv2,W_fc1,b_fc1,W_fc2,b_fc2,state_wconv1,state_bconv1,state_wconv2,state_bconv2,state_wfc1,state_bfc1,state_wfc2,state_bfc2 = optimizer(x_s,y_s,keep_prob,W_conv1,b_conv1,W_conv2,b_conv2,W_fc1,b_fc1,W_fc2,b_fc2,state_wconv1,state_bconv1,state_wconv2,state_bconv2,state_wfc1,state_bfc1,state_wfc2,state_bfc2)
        losses.append(tf.reduce_mean(loss))
   return losses


#sgd_losses = learn(learn_sgd)
#rms_losses = learn(learn_rms)
rnn_losses = learn(learn_rnn)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

import matplotlib.pyplot as plt
import numpy as np

x = np.arange(TRAINING_STEPS)
for _ in range(1):
    #sgd_l, rms_l, rnn_l = sess.run([sgd_losses, rms_losses, rnn_losses])
    rnn_l = sess.run([rnn_losses])
    #p1, = plt.plot(x, sgd_l, label='SGD')
    #p2, = plt.plot(x, rms_l, label='RMS')
    p3, = plt.plot(x, rnn_l, label='RNN')
    plt.legend(handles=[p3])
    plt.title('Losses')
    plt.show()