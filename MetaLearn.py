import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from pylab import savefig
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

seed = 7
np.random.seed(seed)
n_epoch = 40
num_steps = 100
evaluation_period = 10
evaluation_epochs = 20
log_period = 10
batch_size = 100
num_dims = 10
num_layer = 2
hidden_size = 20
unroll_nn = 20
lr = 0.001
logs_path = '/Users/kalyanb/PycharmProjects/MetaLearning/MetaLog/'
save_path = "/Users/kalyanb/PycharmProjects/MetaLearning/MetaOpt/model.ckpt"


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

        def compute_loss():
            x = tf.get_variable(
                "x",
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
    return compute_loss

def metaopti(prob):
    opt_var = _get_variables(prob)[0]
    shapes = [K.get_variable_shape(p) for p in opt_var]
    softmax_w = tf.get_variable("softmax_w",shape=[hidden_size, 1], dtype=tf.float32)
    softmax_b = tf.get_variable("softmax_b",shape=[1], dtype=tf.float32)
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
            grads, _ = tf.clip_by_global_norm(grads, 5.0)
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
        variables = (nest.flatten(state_c) + nest.flatten(state_h) + opt_var)

    with tf.name_scope('state_reset'):
        reset = [tf.variables_initializer(variables), fx_array.close()]

    with tf.name_scope('Optimizee_update'):
        update = (nest.flatten([tf.assign(r,v) for r,v in zip(opt_var,x_final)]) +
                  (nest.flatten([tf.assign(r,v) for r,v in zip(state_c[i],S_C[i]) for i in range(len(state_c))])) +
                  (nest.flatten([tf.assign(r, v) for r, v in zip(state_h[i], S_H[i]) for i in range(len(state_h))])))

    with tf.name_scope('MetaOpt'):
        optimizer = tf.train.AdamOptimizer(lr)

    with tf.name_scope('Meta_update'):
        gvs = optimizer.compute_gradients(loss_optimizer)
        grads, tvars = zip(*gvs)
        clipped_grads,_ = tf.clip_by_global_norm(grads, 5.0)
        step = optimizer.apply_gradients(zip(clipped_grads, tvars))
    return step, loss_optimizer, update, reset, fx_final, x_final

def run_epoch(sess, num_iter, cost_op, ops, reset):
  sess.run(reset)
  """Runs one optimization epoch."""
  for _ in range(num_iter):
     cost= sess.run([cost_op] + ops)[0]
  return cost

def print_stats(header, total_error_optimizee, total_time):
  """Prints experiment statistics."""
  print(header)
  print("Mean Final Error Optimizee: {:.2f}".format(total_error_optimizee))
  print("Mean epoch time: {:.2f} s".format(total_time))

prob = problem()
step, loss_opt, update, reset, cost_op, _ = metaopti(prob)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    best_evaluation = float("inf")
    count = 0
    start = timer()
    num_iter = num_steps // unroll_nn
    losstrain = []
    losseval = []
    #Training
    for e in range(n_epoch):
        cost= run_epoch(sess, num_iter, cost_op, [update, step], reset)
        losstrain.append(np.log10(cost))
        print_stats("Training Epoch {}".format(e), cost, timer() - start)
        saver = tf.train.Saver()

        if (e + 1) % evaluation_period == 0:
            for _ in range(evaluation_epochs):
                evalcost = run_epoch(sess, num_iter, cost_op, [update], reset)
                losseval.append(np.log10(evalcost))
            if save_path is not None and evalcost < best_evaluation:
                print("Saving meta-optimizer to {}".format(save_path))
                saver.save(sess, save_path,global_step=e)
                best_evaluation = evalcost
    slengths = np.arange(n_epoch)
    plt.figure(figsize=(8, 5))
    plt.plot(slengths, losstrain, 'r-', label='Training Loss')
    plt.xlabel('Sequence length')
    plt.ylabel('Training Loss')
    plt.legend()
    savefig('Training.png')
    plt.close()
    slengths = np.arange(len(losseval))
    plt.figure(figsize=(8, 5))
    plt.plot(slengths, losseval, 'b-', label='Validation Loss')
    plt.xlabel('Sequence length')
    plt.ylabel('Validation Loss')
    plt.legend()
    savefig('Validation.png')
    plt.close()
    #os.system('tensorboard --logdir=/Users/kalyanb/PycharmProjects/MetaLearning/MetaLog')
    # os.system('-m webbrowser -t "http://bairstow:6006/#graphs"')