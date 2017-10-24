import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from keras import backend as K
from keras.optimizers import Optimizer
from keras.layers import Input, Dense
from tensorflow.python.util import nest
import os
import collections, mock
import sonnet as snt
from timeit import default_timer as timer
from tensorflow.contrib.learn.python.learn import monitored_session as ms
logging  = tf.logging
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
X_train = np.random.uniform(0, 1, [224, 10])
y_train = np.power(X_train[:, 1:6], 2)

seed = 7
np.random.seed(seed)
n_epoch = 40
batch_size = 50
n_input = X_train.shape[1]
n_output = y_train.shape[1]
n_hidden1 = 7
batch_num = 20
num_layer = 2
hidden_size = 5
unroll_nn = 10
lr = 0.001
logs_path = '/Users/kalyanb/PycharmProjects/MetaLearning/MetaLog/'
save_path = "/Users/kalyanb/PycharmProjects/MetaLearning/MetaOpt/model.ckpt"


_nn_initializers = {
    "w": tf.random_normal_initializer(mean=0, stddev=0.01),
    "b": tf.random_normal_initializer(mean=0, stddev=0.01),
}

def _wrap_variable_creation(func, custom_getter):
  """Provides a custom getter for all variable creations."""
  original_get_variable = tf.get_variable
  def custom_get_variable(*args, **kwargs):
    if hasattr(kwargs, "custom_getter"):
      raise AttributeError("Custom getters are not supported for optimizee "
                           "variables.")
    return original_get_variable(*args, custom_getter=custom_getter, **kwargs)

  # Mock the get_variable method.
  with mock.patch("tensorflow.get_variable", custom_get_variable):
    return func()


def _get_variables(func):

  variables = []
  constants = []

  def custom_getter(getter, name, **kwargs):
    trainable = kwargs["trainable"]
    kwargs["trainable"] = False
    variable = getter(name, **kwargs)
    if trainable:
      variables.append(variable)
    else:
      constants.append(variable)
    return variable

  with tf.name_scope("unused_graph"):
    _wrap_variable_creation(func, custom_getter)

  return variables, constants


def _make_with_custom_variables(func, variables):

  variables = collections.deque(variables)

  def custom_getter(getter, name, **kwargs):
    if kwargs["trainable"]:
      return variables.popleft()
    else:
      kwargs["reuse"] = True
      return getter(name, **kwargs)

  return _wrap_variable_creation(func, custom_getter)

def problem():
    with tf.name_scope('Optimizee_loss'):
        indices = tf.random_uniform([batch_size], 0, X_train.shape[0], tf.int64)
        X = tf.gather(tf.constant(X_train,dtype=tf.float32),indices)#tf.constant(X_train[:indices], dtype=tf.float32)
        Y = tf.gather(tf.constant(y_train,dtype=tf.float32),indices)#tf.constant(y_train[:indices], dtype=tf.float32)
        mlp = snt.nets.MLP([n_hidden1]+[n_output],
                       activation=tf.sigmoid,
                       initializers=_nn_initializers)
        network = snt.Sequential([snt.BatchFlatten(), mlp])

        def compute_loss():
            y2 = network(X)
            loss = tf.reduce_mean(tf.pow((Y - y2), 2))
            return loss
    return compute_loss

def metaopti(prob):
    opt_var = _get_variables(prob)[0]
    shapes = [K.get_variable_shape(p) for p in opt_var]
    softmax_w = tf.get_variable("softmax_w", shape=[hidden_size, 1], dtype=tf.float32)
    softmax_b = tf.get_variable("softmax_b", shape=[1], dtype=tf.float32)
    with tf.name_scope('states'):
        state_c = [[] for _ in range(len(opt_var))]
        state_h = [[] for _ in range(len(opt_var))]
        for i in range(len(opt_var)):
            n_param = int(np.prod(shapes[i]))
            state_c[i] = [tf.Variable(tf.zeros([n_param, hidden_size]), dtype=tf.float32,name="c_in",trainable=False) for _ in range(num_layer)]
            state_h[i] = [tf.Variable(tf.zeros([n_param, hidden_size]), dtype=tf.float32,name="h_in",trainable=False) for _ in range(num_layer)]


    def update_state(fx,x,state_c,state_h):
        with tf.name_scope("gradients"):
            shapes = [K.get_variable_shape(p) for p in x]
            grads = K.gradients(fx, x)
            grads = [tf.stop_gradient(g) for g in grads]
        with tf.variable_scope('MetaNetwork'):
            cell_count = 0
            delta = [[] for _ in range(len(grads))]
            S_C_out = [[] for _ in range(len(opt_var))]
            S_H_out = [[] for _ in range(len(opt_var))]
            for i in range(len(grads)):
                g = grads[i]
                n_param = int(np.prod(shapes[i]))
                flat_g = tf.reshape(g, [-1, n_param])
                rnn_new_c = [[] for _ in range(num_layer)]
                rnn_new_h = [[] for _ in range(num_layer)]
            # Apply RNN cell for each parameter
                with tf.variable_scope("RNN"):
                    rnn_outputs = []
                    rnn_state_c = [[] for _ in range(num_layer)]
                    rnn_state_h = [[] for _ in range(num_layer)]
                    for ii in range(n_param):
                        state_in = [tf.contrib.rnn.LSTMStateTuple(state_c[i][j][ii:ii + 1], state_h[i][j][ii:ii + 1]) for j in range(num_layer)]
                        rnn_cell = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.LSTMCell(num_units=hidden_size, reuse=cell_count > 0) for _ in range(num_layer)])
                        cell_count += 1

                    # Individual update with individual state but global cell params
                        rnn_out_all, state_out = rnn_cell(flat_g[:, ii:ii + 1], state_in)
                        rnn_out = tf.add(tf.matmul(rnn_out_all, softmax_w), softmax_b)
                        rnn_outputs.append(rnn_out)
                        for j in range(num_layer):
                            rnn_state_c[j].append(state_out[j].c)
                            rnn_state_h[j].append(state_out[j].h)

                    # Form output as tensor
                rnn_outputs = tf.reshape(tf.stack(rnn_outputs, axis=1), g.get_shape())
                for j in range(num_layer):
                    rnn_new_c[j] = tf.reshape(tf.stack(rnn_state_c[j], axis=1), (n_param, hidden_size))
                    rnn_new_h[j] = tf.reshape(tf.stack(rnn_state_h[j], axis=1), (n_param, hidden_size))

            # Dense output from state
                delta[i] = rnn_outputs
                S_C_out[i] = rnn_new_c
                S_H_out[i] = rnn_new_h

        return delta, S_C_out,S_H_out

    def time_step(t,f_array,x,state_c,state_h):
        x_new = x
        with tf.name_scope('Unroll_loss_t'):
            fx = _make_with_custom_variables(prob,x)
            f_array = f_array.write(t,fx)
        with tf.name_scope('Unroll_delta_state_update'):
            delta,s_c_out,s_h_out = update_state(fx, x, state_c,state_h)
        with tf.name_scope('Unroll_Optimizee_update'):
            x_new = [x_n + d for x_n,d in zip(x_new,delta)]
        t_new = t + 1

        return t_new, f_array, x_new,s_c_out,s_h_out

    fx_array = tf.TensorArray(tf.float32, size=unroll_nn + 1,
                              clear_after_read=False)
    _, fx_array, x_final, S_C,S_H = tf.while_loop(
        cond=lambda t, *_: t < unroll_nn,
        body=time_step,
        loop_vars=(0, fx_array, opt_var, state_c,state_h),
        parallel_iterations=1,
        swap_memory=True,
        name="unroll")
    with tf.name_scope('Unroll_loss_period'):
        fx_final = _make_with_custom_variables(prob, x_final)
        fx_array = fx_array.write(unroll_nn, fx_final)

    with tf.name_scope('Metaloss'):
        loss_optimizer = tf.reduce_sum(fx_array.stack())

    with tf.name_scope('state_optimizee_var'):
        #variables = (nest.flatten(state_c) + nest.flatten(state_h) + opt_var)
        variables = (nest.flatten(state_c) + nest.flatten(state_h))

    with tf.name_scope('state_reset'):
        reset = [tf.variables_initializer(variables), fx_array.close()]

    with tf.name_scope('Optimizee_update'):
        update = (nest.flatten([tf.assign(r,v) for r,v in zip(opt_var,x_final)]) +
                  (nest.flatten([tf.assign(r,v) for r,v in zip(state_c[i],S_C[i]) for i in range(len(state_c))])) +
                  (nest.flatten([tf.assign(r, v) for r, v in zip(state_h[i], S_H[i]) for i in range(len(state_h))])))

    with tf.name_scope('MetaOpt'):
        optimizer = tf.train.AdamOptimizer(lr)

    with tf.name_scope('Meta_update'):
        step = optimizer.minimize(loss_optimizer)

    return step, loss_optimizer, update, reset, fx_final, x_final

def run_epoch(sess, e, cost_op, m_summary, ops, reset, num_unrolls, summary_writer):
  """Runs one optimization epoch."""
  start = timer()
  cost, summary = [sess.run([cost_op, m_summary] + ops)[j] for j in range(2)]
  summary_writer.add_summary(summary, e)
  e += 1
  return timer() - start, e, cost, summary_writer

def print_stats(header, total_error, total_time, n):
  """Prints experiment statistics."""
  print(header)
  print("Mean Final Error: {:.2f}".format(total_error / n))
  print("Mean epoch time: {:.2f} s".format(total_time / n))

prob = problem()
step, loss_opt, update, reset, cost_op, _ = metaopti(prob)

tf.summary.scalar("Optimizeeloss", cost_op)
tf.summary.scalar("metaloss",loss_opt)
merged_summary_op = tf.summary.merge_all()

with tf.Session() as sess:
    # Prevent accidental changes to the graph.
    sess.run(tf.global_variables_initializer())
    best_evaluation = float("inf")
    count = 0
    summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())
    for e in range(n_epoch):
            # Training.
        time, count, cost,summary_writer = run_epoch(sess, count, cost_op, merged_summary_op, [update, step], reset,
                               unroll_nn,summary_writer)

        print_stats("Epoch {}".format(e), cost, time)
        saver = tf.train.Saver()
        if save_path is not None and cost < best_evaluation:
            print("Saving meta-optimizer to {}".format(save_path))
            saver.save(sess, save_path,global_step=e)
            best_evaluation = cost

    os.system('tensorboard --logdir=/Users/kalyanb/PycharmProjects/MetaLearning/MetaLog')
    # os.system('-m webbrowser -t "http://bairstow:6006/#graphs"')