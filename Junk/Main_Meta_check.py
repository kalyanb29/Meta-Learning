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
    # cell = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.LSTMCell(STATE_SIZE) for _ in range(LAYERS)])
    # cell = tf.contrib.rnn.InputProjectionWrapper(cell, STATE_SIZE)
    # cell = tf.contrib.rnn.OutputProjectionWrapper(cell, 1)
    # cell = tf.make_template('cell', cell)
    lstm_cell = []
    num_hidden = 20
    for i in range(LAYERS):
        lstm_cell.append(tf.nn.rnn_cell.BasicLSTMCell(num_hidden))
    for i in range(len(var)):
        grad = tf.gradients(loss,var[i])[0]
        gradient = grad
        if i in (1,3):
            gradient = tf.expand_dims(gradient, axis=1)
        cell = tf.nn.rnn_cell.MultiRNNCell(cells = lstm_cell)
        if state[i] is None:
            state[i] = cell.zero_state([grad.shape[0]], tf.float32)
        for k in range(STATE_SIZE):
            #tf.get_variable_scope().reuse_variables()
            update, state[i] = cell(gradient, state[i])
        updatef = update
        if i in (1,3):
            updatef = tf.squeeze(updatef,axis = 1)
        var[i] += updatef
    # Squeeze to make it a single batch again.
    return var, state

def learn_sgd(x,y_,W_conv1,b_conv1,W_conv2,b_conv2,state_wconv1,state_bconv1,state_wconv2,state_bconv2):
    if W_conv1 is None:
        W_conv1 = weight_variable([784,20])
        b_conv1 = bias_variable([20])
        W_conv2 = weight_variable([20,10])
        b_conv2 = bias_variable([10])
        state_wconv1 = None
        state_bconv1 = None
        state_wconv2 = None
        state_bconv2 = None
    h1 = tf.matmul(x,W_conv1) + b_conv1
    y_conv = tf.matmul(h1, W_conv2) + b_conv2
    loss = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv)
    W_conv1 = update_grad_sgd(loss,W_conv1)
    b_conv1 = update_grad_sgd(loss,b_conv1)
    W_conv2 = update_grad_sgd(loss,W_conv2)
    b_conv2 = update_grad_sgd(loss,b_conv2)
    state_wconv1 = None
    state_bconv1 = None
    state_wconv2 = None
    state_bconv2 = None
    return loss,W_conv1,b_conv1,W_conv2,b_conv2,state_wconv1,state_bconv1,state_wconv2,state_bconv2

def learn_rms(x,y_,W_conv1,b_conv1,W_conv2,b_conv2,state_wconv1,state_bconv1,state_wconv2,state_bconv2):
    if W_conv1 is None:
        W_conv1 = weight_variable([784,20])
        b_conv1 = bias_variable([20])
        W_conv2 = weight_variable([20,10])
        b_conv2 = bias_variable([10])
        state_wconv1 = None
        state_bconv1 = None
        state_wconv2 = None
        state_bconv2 = None
    h1 = tf.matmul(x,W_conv1) + b_conv1
    y_conv = tf.matmul(h1, W_conv2) + b_conv2
    loss = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv)
    W_conv1, state_wconv1 = update_grad_rms(loss,W_conv1,state_wconv1)
    b_conv1, state_bconv1 = update_grad_rms(loss, b_conv1, state_bconv1)
    W_conv2, state_wconv2 = update_grad_rms(loss, W_conv2, state_wconv2)
    b_conv2, state_bconv2 = update_grad_rms(loss, b_conv2, state_bconv2)
    return loss,W_conv1,b_conv1,W_conv2,b_conv2,state_wconv1,state_bconv1,state_wconv2,state_bconv2

def learn_rnn(x,y_,W_conv1,b_conv1,W_conv2,b_conv2,state_wconv1,state_bconv1,state_wconv2,state_bconv2):
    if W_conv1 is None:
        W_conv1 = weight_variable([784,20])
        b_conv1 = bias_variable([20])
        W_conv2 = weight_variable([20,10])
        b_conv2 = bias_variable([10])
        state_wconv1 = None
        state_bconv1 = None
        state_wconv2 = None
        state_bconv2 = None
    h1 = tf.matmul(x,W_conv1) + b_conv1
    y_conv = tf.matmul(h1, W_conv2) + b_conv2
    loss = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv)
    var_all = [W_conv1,b_conv1,W_conv2,b_conv2]
    state_all = [state_wconv1,state_bconv1,state_wconv2,state_bconv2]
    var_all,state_all = update_grad_rnn(loss,var_all,state_all)
    W_conv1 = var_all[0]
    b_conv1 = var_all[1]
    W_conv2 = var_all[2]
    b_conv2 = var_all[3]
    state_wconv1 = state_all[0]
    state_bconv1 = state_all[1]
    state_wconv2 = state_all[2]
    state_bconv2 = state_all[3]
    return loss,W_conv1,b_conv1,W_conv2,b_conv2,state_wconv1,state_bconv1,state_wconv2,state_bconv2

def learn(optimizer):
   losses = []
   W_conv1 = weight_variable([784, 20])
   b_conv1 = bias_variable([20])
   W_conv2 = weight_variable([20, 10])
   b_conv2 = bias_variable([10])
   state_wconv1 = None
   state_bconv1 = None
   state_wconv2 = None
   state_bconv2 = None
   for _ in range(TRAINING_STEPS):
        chosen_idx = np.random.choice(X_train.shape[0], replace=False, size=m)
        x_s = X_train[chosen_idx,]
        y_s = y_train[chosen_idx,]
        loss,W_conv1,b_conv1,W_conv2,b_conv2,state_wconv1,state_bconv1,state_wconv2,state_bconv2 = optimizer(x_s,y_s,W_conv1,b_conv1,W_conv2,b_conv2,state_wconv1,state_bconv1,state_wconv2,state_bconv2)
        losses.append(tf.reduce_mean(loss))
   return losses


sgd_losses = learn(learn_sgd)
rms_losses = learn(learn_rms)
rnn_losses = learn(learn_rnn)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

import matplotlib.pyplot as plt
import numpy as np

x = np.arange(TRAINING_STEPS)
for _ in range(1):
    sgd_l, rms_l, rnn_l = sess.run([sgd_losses, rms_losses, rnn_losses])
    #rnn_l = sess.run([rnn_losses])
    p1, = plt.plot(x, sgd_l, label='SGD')
    p2, = plt.plot(x, rms_l, label='RMS')
    p3, = plt.plot(x, rnn_l, label='RNN')
    plt.legend(handles=[p1,p2,p3])
    plt.title('Losses')
    plt.show()