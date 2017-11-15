import tensorflow as tf
import numpy as np

def log_encode(x, p=10.0):
    xa = tf.log(tf.maximum(tf.abs(x), np.exp(-p))) / p
    xb = tf.clip_by_value(x * np.exp(p), -1, 1)
    return tf.stack([xa, xb], axis=1)

def update_tanh(x,alpha = 0.1):
    return alpha*tf.tanh(x)