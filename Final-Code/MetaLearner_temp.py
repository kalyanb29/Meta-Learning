import numpy as np
import tensorflow as tf
from keras import backend as K
from tensorflow.python.util import nest
import multiprocessing
from itertools import product

import util
import Preprocess

class MetaOpt(object):

    def __init__(self, **kwargs):
        """Creates a MetaOptimizer.

        Args:
          **kwargs: A set of keyword arguments mapping network identifiers (the
              keys) to parameters that will be passed to networks.Factory (see docs
              for more info).  These can be used to assign different optimizee
              parameters to different optimizers (see net_assignments in the
              meta_loss method).
        """

        if not kwargs:
            # Use a default coordinatewise network if nothing is given. this allows
            # for no network spec and no assignments.
            self._config = {'hidden_size': 20, 'num_layer': 2, 'unroll_nn': 20,'lr': 0.001}
        else:
            self._config = kwargs

    def metaoptimizer(self,dictloss):

        hidden_size = self._config['hidden_size']
        num_layer = self._config['num_layer']
        unroll_nn = self._config['unroll_nn']
        lr = self._config['lr']
        with tf.device('/device:GPU:0'):
            input_var = [util._get_variables(a)[0] for a in dictloss.values()]
            num_var = nest.flatten([0,[len(a) for a in input_var]])
            opt_var = nest.flatten(input_var)
            shapes = [K.get_variable_shape(p) for p in opt_var]
            with tf.variable_scope("softmax", reuse=tf.AUTO_REUSE):
                softmax_w = tf.get_variable("softmax_w", shape=[hidden_size, 1], dtype=tf.float32)
                softmax_b = tf.get_variable("softmax_b", shape=[1], dtype=tf.float32)
            with tf.name_scope('states'):
                state_c = [[] for _ in range(len(opt_var))]
                state_h = [[] for _ in range(len(opt_var))]
                for i in range(len(opt_var)):
                    n_param = int(np.prod(shapes[i]))
                    for ii in range(n_param):
                        state_c[i].append(tf.Variable(tf.zeros([num_layer, hidden_size]), dtype=tf.float32, name="c_in", trainable=False))
                        state_h[i].append(tf.Variable(tf.zeros([num_layer, hidden_size]), dtype=tf.float32, name="h_in", trainable=False))
        def rnn_cell(input_loop):
            grad = input_loop[0]
            state_c = input_loop[1]
            state_h = input_loop[2]
            cell_count = input_loop[4]
            state_in = [tf.contrib.rnn.LSTMStateTuple(state_c[j,], state_h[j,])
                        for j in range(num_layer)]
            rnn_cell = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.LSTMCell(num_units=hidden_size, reuse=cell_count > 0)
                                                    for _ in range(num_layer)])
            cell_count += 1
            # Individual update with individual state but global cell params
            rnn_out_all, state_out = rnn_cell(grad, state_in)
            rnn_out = tf.add(tf.matmul(rnn_out_all, softmax_w), softmax_b)
            return rnn_out,state_out,cell_count

        def update_state(losstot, x, state_c, state_h):
            with tf.name_scope("gradients"):
                shapes = [K.get_variable_shape(p) for p in x]
                grads = nest.flatten([K.gradients(losstot[a], x[num_var[b]:num_var[b] + num_var[b+1]])
                         for a, b in zip(range(len(losstot)),range(len(num_var)-1))])
                grads = [tf.stop_gradient(g) for g in grads]
            with tf.variable_scope('MetaNetwork'):
                cell_count = 0
                delta = [[] for _ in range(len(grads))]
                S_C_out = [[] for _ in range(len(opt_var))]
                S_H_out = [[] for _ in range(len(opt_var))]
                for i in range(len(grads)):
                    g = grads[i]
                    n_param = int(np.prod(shapes[i]))
                    flat_g = tf.reshape(g, [n_param, -1])
                    flat_g_mod = tf.reshape(Preprocess.log_encode(flat_g), [n_param, -1])
                    rnn_new_c = [[] for _ in range(num_layer)]
                    rnn_new_h = [[] for _ in range(num_layer)]
                # Apply RNN cell for each parameter
                    with tf.variable_scope("RNN"):
                        rnn_state_c = [[] for _ in range(num_layer)]
                        rnn_state_h = [[] for _ in range(num_layer)]
                        input_loop = []
                        for ii in range(n_param):
                            input_loop.append([flat_g_mod[ii:ii + 1, :],state_c[i][ii],state_h[i][ii],cell_count])
                        pool = multiprocessing.Pool()
                        rnn_outputs,state_out,cell_count = zip(*pool.map(rnn_cell,input_loop))
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

            return delta, S_C_out, S_H_out

        def time_step(t, f_array, f_array_opt, x, state_c, state_h):
            losstot = [util._make_with_custom_variables(a, x[num_var[b]:num_var[b] + num_var[b+1]])
                       for a, b in zip(dictloss.values(), range(len(num_var)-1))]
            with tf.name_scope('Unroll_Optimizee_loss'):
                fx_opt = losstot[0]
                f_array_opt = f_array_opt.write(t, fx_opt)
            with tf.name_scope('Unroll_loss_t'):
                fx = sum(losstot[a] for a in range(len(losstot)))
                f_array = f_array.write(t, fx)
            with tf.name_scope('Unroll_delta_state_update'):
                delta, s_c_out, s_h_out = update_state(losstot, x, state_c, state_h)
            with tf.name_scope('Unroll_Optimizee_update'):
                x_new = [x_n + Preprocess.update_tanh(d) for x_n, d in zip(x, delta)]
            t_new = t + 1

            return t_new, f_array, f_array_opt, x_new, s_c_out, s_h_out

        with tf.device('/device:GPU:0'):
            fx_array = tf.TensorArray(tf.float32, size=unroll_nn, clear_after_read=False)
            fx_array_opt = tf.TensorArray(tf.float32, size=unroll_nn, clear_after_read=False)
            _, fx_array, fx_array_opt, x_final, S_C, S_H = tf.while_loop(
                cond=lambda t, *_: t < unroll_nn-1,
                body=time_step,
                loop_vars=(0, fx_array, fx_array_opt, opt_var, state_c, state_h),
                parallel_iterations=1,
                swap_memory=True,
                name="unroll")
            finaltotloss = [util._make_with_custom_variables(a, x_final[num_var[b]:num_var[b] + num_var[b+1]])
                            for a, b in zip(dictloss.values(), range(len(num_var)-1))]
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

            with tf.name_scope('state_optimizee_var'):
                variables = (nest.flatten(state_c) + nest.flatten(state_h) + opt_var)

            with tf.name_scope('state_reset'):
                reset = [tf.variables_initializer(variables), fx_array.close(), fx_array_opt.close()]

            with tf.name_scope('Optimizee_update'):
                update = (nest.flatten([tf.assign(r, v) for r, v in zip(opt_var, x_final)]) +
                          (nest.flatten([tf.assign(r, v) for r, v in zip(nest.flatten(state_c), nest.flatten(S_C))])) +
                          (nest.flatten([tf.assign(r, v) for r, v in zip(nest.flatten(state_h), nest.flatten(S_H))])))
        return step, loss_optimizer, update, reset, fx_final, fx_final_opt, arrayf, x_final