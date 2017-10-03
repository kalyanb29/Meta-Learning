import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.utils import np_utils
X_train = np.random.uniform(0,1,[224,30])
y_train = np.power(X_train[:,1:11],2)
#X_train = tf.convert_to_tensor(X_train,dtype = tf.float32)
#y_train = tf.convert_to_tensor(y_train,dtype = tf.float32)
def trainOptimizer():
    g = tf.Graph()
    ### BEGIN: GRAPH CONSTRUCTION ###
    with g.as_default():
        #f = np.load('mnist.npz')
        # X_train, y_train = f['x_train'], f['y_train']
        # X_test, y_test = f['x_test'], f['y_test']
        # f.close()
        # num_pixels = X_train.shape[1] * X_train.shape[2]
        # X_train = X_train.reshape(X_train.shape[0], num_pixels).astype('float32')
        # X_test = X_test.reshape(X_test.shape[0], num_pixels).astype('float32')
        #y_test = tf.convert_to_tensor(np_utils.to_categorical(y_test),dtype = tf.float32)
        seed = 7
        np.random.seed(seed)
        Optimizee_steps = 5
        Optimizer_steps = 2
        batch_size = 100
        hidden_layer = 10
        lstm_layer = 2
        n_input = 30
        n_output = 10
        n_hidden1 = 20
        num_var = 4
        unroll_nn = 5
        loss = 0
        X = tf.placeholder(tf.float32, [None, n_input])
        y = tf.placeholder(tf.float32, [None, n_output])
        W_in = tf.truncated_normal([n_input, n_hidden1], stddev=0.1)
        B_in = tf.truncated_normal([n_hidden1], stddev=0.1)
        W_out = tf.truncated_normal([n_hidden1, n_output], stddev=0.1)
        B_out = tf.truncated_normal([n_output], stddev=0.1)
        for t in range(Optimizer_steps):
            x_s = tf.slice(X, [int((220-batch_size)*t/Optimizer_steps), 0], [batch_size, n_input])
            y_s = tf.slice(y, [int((220-batch_size)*t/Optimizer_steps), 0], [batch_size, n_output])
            # random sampling of one instance of the quadratic function
            layer_1 = tf.add(tf.matmul(x_s, W_in), B_in)
            out_layer = tf.add(tf.matmul(layer_1, W_out), B_out)
            f = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = out_layer,labels=y_s))
            sum_f = 0
            g_new_list = [[] for _ in range(num_var)]
            grad_f = [tf.reshape(tf.gradients(f, W_in)[0], [-1, 1]),
                      tf.reshape(tf.gradients(f, B_in)[0], [-1, 1]),
                      tf.reshape(tf.gradients(f, W_out)[0], [-1, 1]),
                      tf.reshape(tf.gradients(f, B_out)[0], [-1, 1])]
            for j in range(len(grad_f)):
                cell = tf.contrib.rnn.MultiRNNCell(
                    [tf.contrib.rnn.LSTMCell(hidden_layer) for _ in range(lstm_layer)])
                state = [cell.zero_state(1, tf.float32) for i in range(grad_f[j].get_shape().as_list()[0])]
                for i in range(grad_f[j].get_shape().as_list()[0]):
                    grad_f_t = tf.slice(grad_f[j], begin=[i, 0], size=[1, 1])
                    for k in range(unroll_nn):
                        if k > 0: tf.get_variable_scope().reuse_variables()
                        cell_out, state[i] = cell(grad_f_t, state[i])
                        softmax_w = tf.get_variable("softmax_w", [hidden_layer, 1])
                        softmax_b = tf.get_variable("softmax_b", [1])
                        g_new_i = tf.add(tf.matmul(cell_out, softmax_w), softmax_b)

                    g_new_list[j].append(g_new_i)

            g_new = g_new_list
            W_in = tf.add(W_in, tf.reshape(g_new[0], [n_input, n_hidden1]))
            B_in = tf.add(B_in, tf.reshape(g_new[1], [n_hidden1]))
            W_out = tf.add(W_out, tf.reshape(g_new[2], [n_hidden1, n_output]))
            B_out = tf.add(B_out, tf.reshape(g_new[3], [n_output]))
            # update parameter
            layer_1_at_t = tf.add(tf.matmul(x_s, W_in), B_in)
            out_layer_at_t = tf.add(tf.matmul(layer_1_at_t, W_out), B_out)
            f_at_theta_t = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = out_layer_at_t,labels=y_s))
            sum_f = sum_f + f_at_theta_t 
    
            loss += sum_f
        
        loss = loss / Optimizer_steps

        tvars = tf.trainable_variables() # should be just the variable inside the RNN
        grads = tf.gradients(loss, tvars)
        grads =  tf.clip_by_global_norm(grads,1.0)
        lr = 0.001 # Technically I need to do random search to decide this
        optimizer = tf.train.GradientDescentOptimizer(lr).minimize(loss)
        train_op = optimizer.apply_gradients(zip(grads, tvars))
        
        ###  END: GRAPH CONSTRUCTION ###
        
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for epoch in range(Optimizee_steps):
                cost, _ = sess.run([loss, train_op],feed_dict={X:X_train,y:y_train})
                print("Epoch %d : loss %f" % (epoch, cost))
            
            print("Saving the trained model...")
            saver = tf.train.Saver()
            saver.save(sess, "model", global_step=0)

            import pickle
            import time
            print("Extracting variables...")
            now = time.time()
            variable_dict = {}
            for var in tf.trainable_variables():
                print(var.name)
                print(var.eval())
                variable_dict[var.name] = var.eval()
            print("elapsed time: {0}".format(time.time()-now))
            with open("variable_dict.pickle", "wb") as f:
                pickle.dump(variable_dict,f)


if __name__ == "__main__":
    trainOptimizer()
        


