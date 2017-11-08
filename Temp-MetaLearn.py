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
from timeit import default_timer as timer
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

n_epoch = 1000
num_steps = 100
evaluation_period = 10
evaluation_epochs = 20
batch_size = 128
num_dims = 10
num_layer = 2
hidden_size = 20
unroll_nn = 20
lr = 0.001
logs_path = '/Users/kalyanb/PycharmProjects/MetaLearning/MetaLog/'
save_path = '/Users/kalyanb/PycharmProjects/MetaLearning/MetaOpt/model.ckpt'
alpha = 0.1

def log_encode(x, p=10.0):
    xa = tf.log(tf.maximum(tf.abs(x), np.exp(-p))) / p
    xb = tf.clip_by_value(x * np.exp(p), -1, 1)
    return tf.stack([xa, xb], axis=1)

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
    with tf.name_scope('Convex_loss'):
        def convex_loss():
            v = tf.get_variable(
                "v",
                shape=[1, num_dims],
                dtype=tf.float32,
                initializer=tf.random_normal_initializer(stddev=0.01))

            # Non-trainable variables.
            target = tf.get_variable("target",
                                shape=[1, num_dims],
                                dtype=tf.float32,
                                initializer=tf.random_uniform_initializer(),
                                trainable=False)

            return tf.reduce_mean(tf.clip_by_value(tf.square(v - target), 0, 10))
    return collections.OrderedDict([('Opt_loss',compute_loss),('Aux_loss',convex_loss)])

def metaopti(dictloss):
    opt_var = [_get_variables(a)[0][0] for a in dictloss.values()]
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


    def update_state(losstot,x,state_c,state_h):
        with tf.name_scope("gradients"):
            shapes = [K.get_variable_shape(p) for p in x]
            grads = [K.gradients(losstot[a], x[a])[0] for a in range(len(losstot))]
            grads = [tf.stop_gradient(g) for g in grads]
        with tf.variable_scope('MetaNetwork'):
            cell_count = 0
            delta = [[] for _ in range(len(grads))]
            S_C_out = [[] for _ in range(len(opt_var))]
            S_H_out = [[] for _ in range(len(opt_var))]
            for i in range(len(grads)):
                g = grads[i]
                n_param = int(np.prod(shapes[i]))
                flat_g = tf.reshape(g, [n_param,-1])
                flat_g_mod = tf.reshape(log_encode(flat_g),[n_param,-1])
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

                        # Verify whether the variables are used
                        # for v in tf.global_variables():
                        #     print(v.name)
                    # Individual update with individual state but global cell params
                        rnn_out_all, state_out = rnn_cell(flat_g_mod[ii:ii + 1,:], state_in)
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

    def time_step(t,f_array,f_array_opt,x,state_c,state_h):
        losstot = [_make_with_custom_variables(a,[x[b]]) for a,b in zip(dictloss.values(),range(len(x)))]
        with tf.name_scope('Unroll_Optimizee_loss'):
            fx_opt = losstot[0]
            f_array_opt = f_array_opt.write(t,fx_opt)
        with tf.name_scope('Unroll_loss_t'):
            fx = sum(losstot[a] for a in range(len(losstot)))
            f_array = f_array.write(t,fx)
        with tf.name_scope('Unroll_delta_state_update'):
            delta,s_c_out,s_h_out = update_state(losstot, x, state_c,state_h)
        with tf.name_scope('Unroll_Optimizee_update'):
            x_new = [x_n + alpha*tf.tanh(d) for x_n,d in zip(x,delta)]
        t_new = t + 1

        return t_new, f_array, f_array_opt, x_new,s_c_out,s_h_out

    fx_array = tf.TensorArray(tf.float32, size=unroll_nn,
                              clear_after_read=False)
    fx_array_opt = tf.TensorArray(tf.float32, size=unroll_nn,
                              clear_after_read=False)
    _, fx_array, fx_array_opt, x_final, S_C,S_H = tf.while_loop(
        cond=lambda t, *_: t < unroll_nn-1,
        body=time_step,
        loop_vars=(0, fx_array, fx_array_opt, opt_var, state_c,state_h),
        parallel_iterations=1,
        swap_memory=True,
        name="unroll")
    finaltotloss = [_make_with_custom_variables(a, [x_final[b]]) for a,b in zip(dictloss.values(),range(len(x_final)))]
    with tf.name_scope('Unroll_loss_period'):
        fx_final = sum(finaltotloss[a] for a in range(len(finaltotloss)))
        fx_array = fx_array.write(unroll_nn-1, fx_final)

    with tf.name_scope('Final_Optimizee_loss'):
        fx_final_opt = finaltotloss[0]
        fx_array_opt = fx_array_opt.write(unroll_nn - 1, fx_final_opt)
        arrayf = fx_array_opt.stack()

    with tf.name_scope('Metaloss'):
        loss_optimizer = tf.reduce_sum(fx_array.stack())

    with tf.name_scope('MetaOpt'):
        optimizer = tf.train.AdamOptimizer(lr)

    with tf.name_scope('Meta_update'):
        step = optimizer.minimize(loss_optimizer)
        # gvs = optimizer.compute_gradients(loss_optimizer)
        # grads, tvars = zip(*gvs)
        # clipped_grads,_ = tf.clip_by_global_norm(grads, 5.0)
        # step = optimizer.apply_gradients(zip(grads, tvars))

    with tf.name_scope('state_optimizee_var'):
        variables = (nest.flatten(state_c) + nest.flatten(state_h) + opt_var)

    with tf.name_scope('state_reset'):
        reset = [tf.variables_initializer(variables), fx_array.close()]

    with tf.name_scope('Optimizee_update'):
        update = (nest.flatten([tf.assign(r,v) for r,v in zip(opt_var,x_final)]) +
                  (nest.flatten([tf.assign(r,v) for r,v in zip(state_c[i],S_C[i]) for i in range(len(state_c))])) +
                  (nest.flatten([tf.assign(r, v) for r, v in zip(state_h[i], S_H[i]) for i in range(len(state_h))])))
    return step, loss_optimizer, update, reset, fx_final, arrayf, x_final

def run_epoch(sess, num_iter, arraycost, cost_op, ops, reset):
  sess.run(reset)
  costepoch = []
  """Runs one optimization epoch."""
  for _ in range(num_iter):
     cost, loss= [sess.run([arraycost,cost_op] + ops)[j] for j in range(2)]
     costepoch.append(np.log10(cost))
  return np.reshape(costepoch,-1), loss

def print_stats(header, total_error_optimizee, total_time):
  """Prints experiment statistics."""
  print(header)
  print("Mean Final Error Optimizee: {:.2f}".format(total_error_optimizee))
  print("Mean epoch time: {:.2f} s".format(total_time))

dictloss = problem()
step, loss_opt, update, reset, cost_op, arraycost, _ = metaopti(dictloss)


with tf.Session() as sess:
    graph_writer = tf.summary.FileWriter(logs_path, sess.graph)
    sess.run(tf.global_variables_initializer())
    best_evaluation = float("inf")
    count = 0
    start = timer()
    num_iter = num_steps // unroll_nn
    losstrain = []
    losseval = []
    plotlosstrain = []
    plotlosseval = []
    #Training
    for e in range(n_epoch):
        cost, trainloss= run_epoch(sess, num_iter, arraycost, cost_op, [step,update], reset)
        print(cost)
        losstrain.append(cost)
        print_stats("Training Epoch {}".format(e), trainloss, timer() - start)
        saver = tf.train.Saver()

        if (e + 1) % evaluation_period == 0:
            for _ in range(evaluation_epochs):
                evalcost,evaloss = run_epoch(sess, num_iter, arraycost, cost_op, [update], reset)
                losseval.append(evalcost)
                if save_path is not None and evaloss < best_evaluation:
                    print("Saving meta-optimizer to {}".format(save_path))
                    saver.save(sess, save_path,global_step=0)
                    best_evaluation = evaloss
                    plotlosstrain.append(cost)
                    plotlosseval.append(evalcost)
    slengths = np.arange(num_steps)
    plt.figure(figsize=(8, 5))
    plt.plot(slengths, plotlosstrain[-1], 'r-', label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    plt.legend()
    savefig('Training.png')
    plt.close()
    plt.figure(figsize=(8, 5))
    plt.plot(slengths, plotlosseval[-1], 'b-', label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Validation Loss')
    plt.legend()
    savefig('Validation.png')
    plt.close()
    graph_writer.close()