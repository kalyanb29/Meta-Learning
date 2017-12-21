from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import mock
import collections

import sys
import tarfile
import zipfile

import urllib
import numpy as np
from six.moves import xrange
import tensorflow as tf

import problems
import networks

def run_epoch(sess, cost_op, farray, ops, reset, num_unrolls):
    """Runs one optimization epoch."""
    losstot = []
    sess.run(reset)
    for _ in range(num_unrolls):
        farr, cost = [sess.run([farray, cost_op] + ops)[j] for j in range(2)]
        losstot.append(np.log10(farr))
    return np.reshape(losstot, -1), cost


def print_stats(header, total_error, total_time):
    """Prints experiment statistics."""
    print(header)
    print("Log Mean Final Error: {:.2f}".format(np.log10(total_error)))
    print("Mean epoch time: {:.2f} s".format(total_time))


def get_net_path(name, path):
    return None if path is None else os.path.join(path, name + ".l2l")


def get_default_net_config(name, path):
    return {
        "net": "CoordinateWiseDeepLSTM",
        "net_options": {
            "layers": (20, 20),
            "preprocess_name": "LogAndSign",
            "preprocess_options": {"k": 10},
            "scale": 0.01,
        },
        "net_path": get_net_path(name, path)
    }


def get_config(problem_name, path=None):
    """Returns problem configuration."""
    if problem_name == "simple":
        problem = problems.simple()
        net_config = {"cw": {
            "net": "CoordinateWiseDeepLSTM",
            "net_options": {"layers": (), "initializer": "zeros"},
            "net_path": get_net_path("cw", path)
        }}
        net_assignments = None
    elif problem_name == "simple-multi":
        problem = problems.simple_multi_optimizer()
        net_config = {
            "cw": {
                "net": "CoordinateWiseDeepLSTM",
                "net_options": {"layers": (), "initializer": "zeros"},
                "net_path": get_net_path("cw", path)
            },
            "adam": {
                "net": "Adam",
                "net_options": {"learning_rate": 0.1}
            }
        }
        net_assignments = [("cw", ["x_0"]), ("adam", ["x_1"]), ("cw", ["v"])]
    elif problem_name == "quadratic":
        problem = problems.quadratic(batch_size=128, num_dims=10)
        net_config = {"cw": {
            "net": "CoordinateWiseDeepLSTM",
            "net_options": {"layers": (20, 20),
                            "preprocess_name": "LogAndSign",
                            "preprocess_options": {"k": 5},
                            },
            "net_path": get_net_path("cw", path)
        }}
        net_assignments = None
    elif problem_name == "mnist":
        mode = "train" if path is None else "test"
        problem = problems.mnist(layers=(20,), mode=mode)
        net_config = {"cw": get_default_net_config("cw", path)}
        net_assignments = None
    elif problem_name == "cifar":
        mode = "train" if path is None else "test"
        problem = problems.cifar10("cifar10",
                                   conv_channels=(16, 16, 16),
                                   linear_layers=(32,),
                                   mode=mode)
        net_config = {"cw": get_default_net_config("cw", path)}
        net_assignments = None
    elif problem_name == "cifar-multi":
        mode = "train" if path is None else "test"
        problem = problems.cifar10("cifar10",
                                   conv_channels=(16, 16, 16),
                                   linear_layers=(32,),
                                   mode=mode)
        net_config = {
            "conv": get_default_net_config("conv", path),
            "fc": get_default_net_config("fc", path),
            "cw": get_default_net_config("cw", path)
        }
        conv_vars = ["conv_net_2d/conv_2d_{}/w".format(i) for i in xrange(3)]
        fc_vars = ["conv_net_2d/conv_2d_{}/b".format(i) for i in xrange(3)]
        fc_vars += ["conv_net_2d/batch_norm_{}/beta".format(i) for i in xrange(3)]
        fc_vars += ["mlp/linear_{}/w".format(i) for i in xrange(2)]
        fc_vars += ["mlp/linear_{}/b".format(i) for i in xrange(2)]
        fc_vars += ["mlp/batch_norm/beta"]
        net_assignments = [("conv", conv_vars), ("fc", fc_vars), ("cw", ["v"])]
    else:
        raise ValueError("{} is not a valid problem".format(problem_name))

    return problem, net_config, net_assignments

def _nested_assign(ref, value):
    """Returns a nested collection of TensorFlow assign operations.

    Args:
      ref: Nested collection of TensorFlow variables.
      value: Values to be assigned to the variables. Must have the same structure
          as `ref`.

    Returns:
      Nested collection (same structure as `ref`) of TensorFlow assign operations.

    Raises:
      ValueError: If `ref` and `values` have different structures.
    """
    if isinstance(ref, list) or isinstance(ref, tuple):
        if len(ref) != len(value):
            raise ValueError("ref and value have different lengths.")
        result = [_nested_assign(r, v) for r, v in zip(ref, value)]
        if isinstance(ref, tuple):
            return tuple(result)
        return result
    else:
        return tf.assign(ref, value)


def _nested_variable(init, name=None, trainable=False):
    """Returns a nested collection of TensorFlow variables.

    Args:
      init: Nested collection of TensorFlow initializers.
      name: Variable name.
      trainable: Make variables trainable (`False` by default).

    Returns:
      Nested collection (same structure as `init`) of TensorFlow variables.
    """
    if isinstance(init, list) or isinstance(init, tuple):
        result = [_nested_variable(i, name, trainable) for i in init]
        if isinstance(init, tuple):
            return tuple(result)
        return result
    else:
        return tf.Variable(init, name=name, trainable=trainable)


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
    """Calls func, returning any variables created, but ignoring its return value.

    Args:
      func: Function to be called.

    Returns:
      A tuple (variables, constants) where the first element is a list of
      trainable variables and the second is the non-trainable variables.
    """
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
    """Calls func and replaces any trainable variables.

    This returns the output of func, but whenever `get_variable` is called it
    will replace any trainable variables with the tensors in `variables`, in the
    same order. Non-trainable variables will re-use any variables already
    created.

    Args:
      func: Function to be called.
      variables: A list of tensors replacing the trainable variables.

    Returns:
      The return value of func is returned.
    """
    variables = collections.deque(variables)

    def custom_getter(getter, name, **kwargs):
        if kwargs["trainable"]:
            return variables.popleft()
        else:
            kwargs["reuse"] = True
            return getter(name, **kwargs)

    return _wrap_variable_creation(func, custom_getter)

def _make_nets(variables, config, net_assignments):
    """Creates the optimizer networks.

    Args:
      variables: A list of variables to be optimized.
      config: A dictionary of network configurations, each of which will be
          passed to networks.Factory to construct a single optimizer net.
      net_assignments: A list of tuples where each tuple is of the form (netid,
          variable_names) and is used to assign variables to networks. netid must
          be a key in config.

    Returns:
      A tuple (nets, keys, subsets) where nets is a dictionary of created
      optimizer nets such that the net with key keys[i] should be applied to the
      subset of variables listed in subsets[i].

    Raises:
      ValueError: If net_assignments is None and the configuration defines more
          than one network.
    """
    # create a dictionary which maps a variable name to its index within the
    # list of variables.
    name_to_index = dict((v.name.split(":")[0], i)
                         for i, v in enumerate(variables))

    if net_assignments is None:
        if len(config) != 1:
            raise ValueError("Default net_assignments can only be used if there is "
                             "a single net config.")

        with tf.variable_scope("vars_optimizer"):
            key = next(iter(config))
            kwargs = config[key]
            net = networks.factory(**kwargs)

        nets = {key: net}
        keys = [key]
        subsets = [range(len(variables))]
    else:
        nets = {}
        keys = []
        subsets = []
        with tf.variable_scope("vars_optimizer"):
            for key, names in net_assignments:
                if key in nets:
                    raise ValueError("Repeated netid in net_assigments.")
                nets[key] = networks.factory(**config[key])
                subset = [name_to_index[name] for name in names]
                keys.append(key)
                subsets.append(subset)
                print("Net: {}, Subset: {}".format(key, subset))

    # subsets should be a list of disjoint subsets (as lists!) of the variables
    # and nets should be a list of networks to apply to each subset.
    return nets, keys, subsets