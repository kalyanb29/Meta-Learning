from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os

import numpy as np
import sonnet as snt
import tensorflow as tf

from tensorflow.python.util import nest

import networks
import util


MetaLoss = collections.namedtuple("MetaLoss", "loss, update, reset, fx, farray, lr_opt, x")
MetaStep = collections.namedtuple("MetaStep", "step, loss, update, reset, fx, farray, lr_opt, x")


class MetaOptimizer(object):
    """Learning to learn (meta) optimizer.

    Optimizer which has an internal RNN which takes as input, at each iteration,
    the gradient of the function being minimized and returns a step direction.
    This optimizer can then itself be optimized to learn optimization on a set of
    tasks.
    """

    def __init__(self, **kwargs):
        """Creates a MetaOptimizer.

        Args:
          **kwargs: A set of keyword arguments mapping network identifiers (the
              keys) to parameters that will be passed to networks.Factory (see docs
              for more info).  These can be used to assign different optimizee
              parameters to different optimizers (see net_assignments in the
              meta_loss method).
        """
        self._nets = None

        if not kwargs:
            # Use a default coordinatewise network if nothing is given. this allows
            # for no network spec and no assignments.
            self._config = {
                "coordinatewise": {
                    "net": "CoordinateWiseDeepLSTM",
                    "net_options": {
                        "layers": (20, 20),
                        "preprocess_name": "LogAndSign",
                        "preprocess_options": {"k": 10},
                        "scale": 0.01,
                    }}}
        else:
            self._config = kwargs

    def save(self, sess, path=None):
        """Save meta-optimizer."""
        result = {}
        for k, net in self._nets.items():
            if path is None:
                filename = None
                key = k
            else:
                filename = os.path.join(path, "{}.l2l".format(k))
                key = filename
            net_vars = networks.save(net, sess, filename=filename)
            result[key] = net_vars
        return result

    def meta_loss(self,
                  make_loss,
                  len_unroll,
                  net_assignments=None,
                  second_derivatives=False):
        """Returns an operator computing the meta-loss.

        Args:
          make_loss: Callable which returns the optimizee loss; note that this
              should create its ops in the default graph.
          len_unroll: Number of steps to unroll.
          net_assignments: variable to optimizer mapping. If not None, it should be
              a list of (k, names) tuples, where k is a valid key in the kwargs
              passed at at construction time and names is a list of variable names.
          second_derivatives: Use second derivatives (default is false).

        Returns:
          namedtuple containing (loss, update, reset, fx, x)
        """

        # Construct an instance of the problem only to grab the variables. This
        # loss will never be evaluated.
        x = []
        constants = []
        for a in make_loss.values():
            item1, item2 = util._get_variables(a)
            x.append(item1)
            constants.append(item2)
        num_var = nest.flatten([0, [len(a) for a in x]])
        var_num = np.cumsum(num_var)
        x = nest.flatten(x)
        constants = nest.flatten(constants)
        print("Optimizee variables")
        print([op.name for op in x])
        print("Problem variables")
        print([op.name for op in constants])

        # Create the optimizer networks and find the subsets of variables to assign
        # to each optimizer.
        nets, net_keys, subsets = util._make_nets(x, self._config, net_assignments)

        # Store the networks so we can save them later.
        self._nets = nets

        # Create hidden state for each subset of variables.
        state = []
        with tf.name_scope("states"):
            for i, (subset, key) in enumerate(zip(subsets, net_keys)):
                net = nets[key]
                with tf.name_scope("state_{}".format(i)):
                    state.append(util._nested_variable(
                        [net.initial_state_for_inputs(x[j], dtype=tf.float32)
                         for j in subset],
                        name="state", trainable=False))

        def update(net, fx, x, state, subset):
            """Parameter and RNN state update."""
            with tf.name_scope("gradients"):
                if len(subset) == sum(num_var):
                    gradients = nest.flatten([tf.gradients(fx[a], x[num_var[b]:num_var[b] + num_var[b+1]])
                            for a, b in zip(range(len(fx)), range(len(num_var)-1))])
                else:
                    bin_num = np.digitize(subset,var_num) - 1
                    if np.std(bin_num) == 0 or len(bin_num) == 1:
                        gradients = nest.flatten([tf.gradients(fx[bin_num[0]], x)])
                    else:
                        gradients = nest.flatten([tf.gradients(fx[a], x[a])
                            for a in bin_num])
                if not second_derivatives:
                    gradients = [tf.stop_gradient(g) for g in gradients]

            with tf.name_scope("deltas"):
                deltas, state_next = zip(*[net(g, s) for g, s in zip(gradients, state)])
                state_next = list(state_next)

            ratio = sum([tf.reduce_mean(tf.div(d,g)) for d, g in zip(deltas, gradients)])/len(gradients)

            return deltas, state_next, ratio

        def time_step(t, fx_array, fx_array_opt, lr_optimizee, x, state):
            """While loop body."""
            x_next = list(x)
            state_next = []
            ratio = []

            with tf.name_scope("fx"):
                fx = [util._make_with_custom_variables(a, x[num_var[b]:num_var[b] + num_var[b+1]])
                       for a, b in zip(make_loss.values(), range(len(num_var)-1))]

            with tf.name_scope("fx_sum"):
                fxsum = sum(fx[a] for a in range(len(fx)))
                fx_array = fx_array.write(t, fxsum)

            with tf.name_scope("fx_opt"):
                fxopt = fx[0]
                fx_array_opt = fx_array_opt.write(t, fxopt)

            with tf.name_scope("dx"):
                for subset, key, s_i in zip(subsets, net_keys, state):
                    x_i = [x[j] for j in subset]
                    deltas, s_i_next, ratio_i = update(nets[key], fx, x_i, s_i, subset)
                    for idx, j in enumerate(subset):
                        x_next[j] += deltas[idx]
                    state_next.append(s_i_next)
                    ratio.append(ratio_i)

            with tf.name_scope("lr_opt"):
                lr_optimizee = lr_optimizee.write(t, sum(ratio)/len(ratio))

            with tf.name_scope("t_next"):
                t_next = t + 1

            return t_next, fx_array, fx_array_opt, lr_optimizee, x_next, state_next

        # Define the while loop.
        fx_array = tf.TensorArray(tf.float32, size=len_unroll,
                                  clear_after_read=False)
        fx_array_opt = tf.TensorArray(tf.float32, size=len_unroll,
                                      clear_after_read=False)
        lr_optimizee = tf.TensorArray(tf.float32, size=len_unroll-1,
                                      clear_after_read=False)
        _, fx_array, fx_array_opt, lr_optimizee, x_final, s_final = tf.while_loop(
            cond=lambda t, *_: t < len_unroll-1,
            body=time_step,
            loop_vars=(0, fx_array, fx_array_opt, lr_optimizee, x, state),
            parallel_iterations=1,
            swap_memory=True,
            name="unroll")

        with tf.name_scope("fx"):
            fx_final = [util._make_with_custom_variables(a, x_final[num_var[b]:num_var[b] + num_var[b+1]])
                       for a, b in zip(make_loss.values(), range(len(num_var)-1))]

        with tf.name_scope("fx_sum"):
            fxsum = sum(fx_final[a] for a in range(len(fx_final)))
            fx_array = fx_array.write(len_unroll-1, fxsum)

        with tf.name_scope("fx_opt"):
            fxopt = fx_final[0]
            fx_array_opt = fx_array_opt.write(len_unroll-1, fxopt)
            farray = fx_array_opt.stack()

        with tf.name_scope("lr_opt"):
            lr_opt = lr_optimizee.stack()

        loss = tf.reduce_sum(fx_array.stack(), name="loss")

        # Reset the state; should be called at the beginning of an epoch.
        with tf.name_scope("reset"):
            variables = (nest.flatten(state) +
                         x + constants)
            # Empty array as part of the reset process.
            reset = [tf.variables_initializer(variables), fx_array.close(), fx_array_opt.close(), lr_optimizee.close()]

        # Operator to update the parameters and the RNN state after our loop, but
        # during an epoch.
        with tf.name_scope("update"):
            update = (nest.flatten(util._nested_assign(x, x_final)) +
                      nest.flatten(util._nested_assign(state, s_final)))

        # Log internal variables.
        for k, net in nets.items():
            print("Optimizer '{}' variables".format(k))
            print([op.name for op in snt.get_variables_in_module(net)])

        return MetaLoss(loss, update, reset, fxopt, farray, lr_opt, x_final)

    def meta_minimize(self, make_loss, len_unroll, learning_rate=0.01, **kwargs):
        """Returns an operator minimizing the meta-loss.

        Args:
          make_loss: Callable which returns the optimizee loss; note that this
              should create its ops in the default graph.
          len_unroll: Number of steps to unroll.
          learning_rate: Learning rate for the Adam optimizer.
          **kwargs: keyword arguments forwarded to meta_loss.

        Returns:
          namedtuple containing (step, update, reset, fx, x)
        """
        info = self.meta_loss(make_loss, len_unroll, **kwargs)
        optimizer = tf.train.AdamOptimizer(learning_rate)
        step = optimizer.minimize(info.loss)
        return MetaStep(step, *info[0:])
