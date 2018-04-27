from fastdtwt import fastdtwt
import tensorflow as tf

a1 = tf.Variable([[1.2,2.3,3.5],[3.0,4.0,5.0],[4.0, 12.0, 14.3]], dtype=tf.float32)
# a1indices = tf.where(tf.not_equal(a1, 0.0))
b1 = tf.Variable([[1.0,2.0,3.0],[10.3, 15.1, 17.5]], dtype=tf.float32)
# b1indices = tf.where(tf.not_equal(b1, 0.0))
# a = tf.SparseTensor(indices=a1indices, values=tf.gather_nd(a1, a1indices), dense_shape=a1.get_shape())
# b = tf.SparseTensor(indices=b1indices, values=tf.gather_nd(b1, b1indices), dense_shape=b1.get_shape())
distancet,patht = fastdtwt(a1, b1)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    dist = sess.run(distancet)
    print(dist)