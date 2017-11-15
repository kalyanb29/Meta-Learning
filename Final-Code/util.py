import tensorflow as tf
import collections
import mock
import numpy as np

def _wrap_variable_creation(func, custom_getter):
    """Provides a custom getter for all variable creations."""
    original_get_variable = tf.get_variable
    def custom_get_variable(*args, **kwargs):
        if hasattr(kwargs, "custom_getter"):
            raise AttributeError("Custom getters are not supported for optimizee variables.")
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

def run_epoch(sess, num_iter, arraycost, cost_op, ops, reset):
    sess.run(reset)
    costepoch = []
    """Runs one optimization epoch."""
    for _ in range(num_iter):
        cost, loss = [sess.run([arraycost, cost_op] + ops)[j] for j in range(2)]
        costepoch.append(np.log10(cost))
    return np.reshape(costepoch, -1), loss

def print_stats(header, total_error_optimizee, total_time):
    """Prints experiment statistics."""
    print(header)
    print("Mean Final Error Optimizee: {:.2f}".format(total_error_optimizee))
    print("Mean epoch time: {:.2f} s".format(total_time))