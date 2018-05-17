import numpy as np
import pylab as plt
import networkx as nx
import tensorflow as tf
import tensorflow.contrib.slim as slim


tf.reset_default_graph() # clear the tensorflow graph




indices1 = tf.constant([[0],[3],[4]])
updates1 = tf.constant([[0.00117441],[0.00428762],[0.00125854],[0.00681501],[0.00341065],[0.00134786]])
updates1 = tf.gather_nd(params=updates1, indices=indices1)
updates1 = tf.reshape(updates1, [3])
shape1 = tf.constant([6])

indices = tf.constant([[4], [3], [1], [7]]) # Shape is 4 1
updates = tf.constant([9, 10, 11, 12])
shape = tf.constant([8])

#scatter = updates1
#scatter = tf.scatter_nd(indices, updates, shape)
scatter = tf.scatter_nd(indices1, updates1, shape1)




init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    print(sess.run(scatter))