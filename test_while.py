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
    X = tf.constant(X_train[:batch_size], dtype=tf.float32)
    Y = tf.constant(y_train[:batch_size], dtype=tf.float32)
    mlp = snt.nets.MLP([n_hidden1]+[n_output],
                       activation=tf.sigmoid,
                       initializers=_nn_initializers)
    network = snt.Sequential([snt.BatchFlatten(), mlp])

    def compute_loss():
        y2 = network(X)
        loss = tf.reduce_mean(tf.pow((Y - y2), 2))
        return loss
    return compute_loss

def metaopti(loss):
    opt_var = _get_variables(loss)[0]
    shapes = [K.get_variable_shape(p) for p in opt_var[0]]
    state = [[] for _ in range(len(opt_var[0]))]
    for i in range(len(opt_var[0])):
        n_param = int(np.prod(shapes[i]))
        state_c = K.zeros((n_param, hidden_size), name="c_in")
        state_h = K.zeros((n_param, hidden_size), name="h_in")
        state[i] = [[] for _ in range(n_param)]
        for ii in range(n_param):
            state[i][ii]= tf.contrib.rnn.LSTMStateTuple(state_c[ii:ii + 1], state_h[ii:ii + 1])

    def update_state(fx,x,state):
        shapes = [K.get_variable_shape(p) for p in x]
        grads = K.gradients(fx, x)
        grads = [tf.stop_gradient(g) for g in grads]
        cell_count = 0
        state_f = [[] for _ in range(len(grads))]
        delta = [[] for _ in range(len(grads))]
        for i in range(len(grads)):
            g = grads[i]
            n_param = int(np.prod(shapes[i]))
            flat_g = tf.reshape(g, [-1, n_param])

            # Apply RNN cell for each parameter
            with tf.variable_scope("ml_rnn"):
                rnn_outputs = []
                state_f[i] = [[] for _ in range(n_param)]
                for ii in range(n_param):
                    rnn_cell = tf.contrib.rnn.LSTMCell(num_units=hidden_size, reuse=cell_count > 0)
                    cell_count += 1

                    # Individual update with individual state but global cell params
                    rnn_out, state_f[i][ii] = rnn_cell(flat_g[:, ii:ii + 1], state[i][ii])
                    rnn_outputs.append(rnn_out)

                    # Form output as tensor
            rnn_outputs = tf.reshape(tf.stack(rnn_outputs, axis=1), g.get_shape())

            # Dense output from state
            delta[i] = rnn_outputs
        return delta, state_f

    def time_step(t,f_array,x,state):
        x_new = x
        fx = _make_with_custom_variables(loss,x)
        f_array = f_array.write(t,fx)
        delta,state_f = update_state(fx, x, state)
        x_new = [x_n + d for x_n,d in zip(x_new,delta)]
        t_new = t + 1

        return t_new, f_array, x_new,state_f

    fx_array = tf.TensorArray(tf.float32, size=unroll_nn + 1,
                              clear_after_read=False)
    _, fx_array, x_final, s_final = tf.while_loop(
        cond=lambda t, *_: t < unroll_nn,
        body=time_step,
        loop_vars=(0, fx_array, opt_var[0], state),
        parallel_iterations=1,
        swap_memory=True,
        name="unroll")

    fx_final = _make_with_custom_variables(loss, x_final)

    fx_array = fx_array.write(unroll_nn, fx_final)

    loss_optimizer = tf.reduce_sum(fx_array.stack(), name="loss")

    variables = (nest.flatten(state) + opt_var)
    reset = [tf.variables_initializer(variables), fx_array.close()]
    update = (nest.flatten([tf.assign(r,v) for r,v in zip(opt_var,x_final)]) + nest.flatten([tf.assign(r,v) for r,v in zip(state,s_final)]))
    optimizer = tf.train.AdamOptimizer(lr)
    step = optimizer.minimize(loss_optimizer)

    return step, loss_optimizer, update, reset, fx_final, x_final

def run_epoch(sess, cost_op, ops, reset, num_unrolls,**kwargs):
  """Runs one optimization epoch."""
  start = timer()
  sess.run(reset)
  for _ in range(num_unrolls):
    cost = sess.run([cost_op] + ops,feed_dict=kwargs)[0]
  return timer() - start, cost

loss = problem()
step, loss_opt, update, reset, cost_op, _ = metaopti(loss)

with ms.MonitoredSession() as sess:
    # Prevent accidental changes to the graph.
    tf.get_default_graph().finalize()

    best_evaluation = float("inf")
    total_time = 0
    total_cost = 0
    for e in range(Optimizee_steps):
        # Training.
        time, cost = run_epoch(sess, cost_op, [update, step], reset,
                               unroll_nn,{x:X,y_:Y})
        total_time += time
        total_cost += cost