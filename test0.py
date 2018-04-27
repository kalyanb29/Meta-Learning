import numpy as np
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
a1 = [[1.1,2.3,3.5],[3.0,4.0,5.0],[4.0, 12.0, 14.3]]
# a1indices = tf.where(tf.not_equal(a1, 0.0))
b1 = [[1.0,2.0,3.0],[10.3, 15.1, 17.5]]

distance,path = fastdtw(a1, b1)
print(distance)
