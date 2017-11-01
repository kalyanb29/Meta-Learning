import os
import tensorflow as tf
import numpy as np
from datetime import datetime
from keras import layers, models

def wrap_add_weight_function(obj, replace_vars={}):
    def tensor_add_weight(name,
                   shape,
                   dtype=None,
                   initializer=None,
                   regularizer=None,
                   trainable=True,
                   constraint=None):
        if dtype is None: dtype = tf.float32
        if name in replace_vars:
            weight = replace_vars[name]
        else:
            weight = tf.placeholder(dtype, shape=shape, name=name)
        obj.input_weights[name] = weight
        return weight

    obj.add_weight = tensor_add_weight
    obj.input_weights = {}
    return obj


def f_update(g_in, state_in):
    """
    Update states using gradients and calculate parameter update

    :param state_in:
        states for one parameter, shape: (n_param, n_state)
    :param g_in:
        gradients for one parameter, shape: (n_0, ..., n_m)
        such that \prod_i n_i = n_param

    :return:
    """
    state_shape = state_in.get_shape()
    assert np.prod(g_in.shape) == state_shape[0]

    # Flatten gradients
    g_flat = tf.reshape(g_in, [-1, 1])

    # Create independent parameter inputs:
    # shape: (n_param, n_state + 1)
    state_inputs = tf.concat([state_in, g_flat], axis=1)

    # State update network
    state_out = layers.Dense(int(state_shape[1]), activation='tanh')(state_inputs)

    # Parameter update network
    u_out = layers.Dense(1, activation='tanh')(state_out)
    u_out = tf.reshape(u_out, tf.shape(g_in))
    return u_out, state_out


def g_optimizee(d_in, parameters_in, n_units=[], activations=[]):
    """
    Optimizee function

    :param d_in:
        Data inputs (i.e. a mini-batch), shape (n_batch, n_in)
    :param parameters_in:
        Parameter/weight values to use
    :param n_units:
        Number of units in each layer
    :param activations:
        Activations for each layer
    :return:
        The output of the optimizee network
    """
    n_layers = len(n_units)
    in_ii = d_in
    for ii in range(n_layers):
        n_l = n_units[ii]
        act = activations[ii]
        g_l = layers.Dense(n_l, activation=act)
        replace_vars = {
            'kernel': parameters_in["k%d"%ii],
            'bias': parameters_in["b%d"%ii]
        }

        in_ii = wrap_add_weight_function(g_l, replace_vars)(in_ii)
    return in_ii


if __name__=="__main__":
    output_location = r"."
    start_time = datetime.now()
    log_location = os.path.join(output_location, 'log_{:%Y-%m-%d_%H-%M}'.format(start_time))

    if not tf.gfile.Exists(log_location):
        tf.gfile.MakeDirs(log_location)

    ### CONFIG ###
    n_state = 6
    n_param = 2
    n_unroll = 2

    no_out = 1
    no_hidden_1 = 5
    optimizee_spec = {
        "n_units": [no_hidden_1, no_out],
        "activations": ["relu", "softmax"],
    }
    ### END CONFIG ###

    # Intitial "parameters"
    with tf.name_scope('opt_params'):
        p_0 = {
            "k0": tf.zeros((n_param, no_hidden_1), name="k0"),
            "b0": tf.zeros((no_hidden_1), name="b0"),
            "k1": tf.zeros((no_hidden_1, no_out), name="k1"),
            "b1": tf.zeros((no_out), name="b1"),
        }

    # Initial states
    with tf.name_scope('states'):
        s_0 = {}
        for p_name in p_0:
            np_ii = tf.size(p_0[p_name])
            s_0[p_name] = tf.zeros((np_ii,n_state), name='s_%s_0' % p_name)
        print("Initial States:", s_0)

    # Data for each step
    d_0 = tf.zeros((1,n_param), name='d_0')
    data = [d_0] * n_unroll

    # Unroll graph over timesteps
    s_ii = s_0
    p_ii = p_0
    states = []
    params = []
    for ii in range(n_unroll):
        print("\nLoop %d"%ii)
        with tf.name_scope('iter_%d'%ii):
            print("Parameters:", p_ii)
            # Optimizee function
            with tf.name_scope('optimizee'):
                o_1 = g_optimizee(d_0, p_ii, **optimizee_spec)

            # Optimizee loss function (currently a dummy function)
            with tf.name_scope('loss'):
                l_1 = tf.reduce_sum(o_1, axis=0)

            # Gradients of optimizee loss for each parameter, averaged over mini-batch.
            grads_ii = {}
            for p_name in p_ii:
                # Recurrent function to update state and output parameter updates
                grads_ii[p_name] = tf.gradients(o_1, p_ii[p_name])[0]
            print("Gradients:", grads_ii)

            # Update parameters
            p_next = {}
            s_next = {}
            for p_name in p_ii:
                with tf.name_scope('param_%s'%p_name):
                    # Inputs to update function for parameter:
                    grad_p = grads_ii[p_name]
                    s_p = s_ii[p_name]

                    # Recurrent function to update state and output parameter updates
                    with tf.name_scope('rnn'):
                        u_p, s_p_next = f_update(grad_p, s_p)

                    # Store state
                    s_next[p_name] = s_p_next
                    p_next[p_name] = u_p + p_ii[p_name]

            # Next iteration use these as input state and parameters
            s_ii = s_next
            p_ii = p_next

    input_vals = {}

    outputs = list(p_ii.values())

    # Summary information

    # Setup logging
    g = tf.get_default_graph()

    var_init = tf.variables_initializer(tf.global_variables())
    with tf.Session() as s:
        graph_writer = tf.summary.FileWriter(log_location, s.graph)

        s.run(var_init)
        out = s.run(outputs, input_vals)
        print("output:", out)

    graph_writer.close()