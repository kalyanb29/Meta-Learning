import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from keras import backend as K
from keras.optimizers import Optimizer
from keras.callbacks import Callback
from keras import layers, initializers, models
from keras.models import Sequential, Model
from keras.utils import np_utils
# f = np.load('mnist.npz')
# X_train, y_train = f['x_train'], f['y_train']
# X_test, y_test = f['x_test'], f['y_test']
# f.close()
# num_pixels = X_train.shape[1] * X_train.shape[2]
# X_train = X_train.reshape(X_train.shape[0], num_pixels).astype('float32')
# X_test = X_test.reshape(X_test.shape[0], num_pixels).astype('float32')
# X_train = X_train / 255
# X_test = X_test / 255
# y_train = np_utils.to_categorical(y_train)
# y_test = np_utils.to_categorical(y_test)
# Testing
X_train = np.random.uniform(0,1,[224,10])
y_train = np.power(X_train[:,1:6],2)

seed = 7
np.random.seed(seed)
Optimizee_steps = 100
batch_size = 50
n_input = X_train.shape[1]
n_output = y_train.shape[1]
n_hidden1 = 7
batch_num = 3
class MetaOpt(Optimizer):
    def __init__(self,**kwargs):
        super(MetaOpt,self).__init__(**kwargs)
        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, dtype='int64', name='iterations')
            self.Optimizer_steps = 20
            self.lstmunit = 2
            self.hidden_size = 20
            self.unroll_nn = 20
            self.lr = 0.001
    def get_updates(self,loss,params):
        grads = self.get_gradients(loss,params)
        self.updates = [K.update_add(self.iterations,1)]
        self.Optimizer_weight = [[] for _ in range(2*(1+self.lstmunit))]
        g_new = self.update_param(grads)
        for p,g in zip(params,g_new):
            new_p = p + g
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)
            self.updates.append(K.update(p, new_p))

        return self.updates

    def get_config(self):
        base_config = super(MetaOpt, self).get_config()

        return dict(list(base_config.items()))

    def update_param(self, grads):
        with tf.variable_scope('metaoptvar') as scope:
            g_new_list = [[] for _ in range(len(grads))]
            for j in range(len(grads)):
                cell = tf.contrib.rnn.MultiRNNCell(
                    [tf.contrib.rnn.BasicLSTMCell(self.hidden_size) for _ in range(self.lstmunit)])
                gradsj = tf.reshape(grads[j], [-1, 1])
                state = [cell.zero_state(1, tf.float32) for i in range(gradsj.get_shape().as_list()[0])]
                softmax_w = tf.get_variable("softmax_w", [self.hidden_size, 1])
                softmax_b = tf.get_variable("softmax_b", [1])
                scope.reuse_variables()
                for i in range(gradsj.get_shape().as_list()[0]):
                    grad_f_t = tf.slice(gradsj, begin=[i, 0], size=[1, 1])
                    cell_out, state[i] = cell(grad_f_t, state[i])
                    g_new_i = tf.add(tf.matmul(cell_out, softmax_w), softmax_b)
                    g_new_list[j].append(g_new_i)

                g_new_list[j] = tf.reshape(g_new_list[j], grads[j].shape)

            g_new = g_new_list
            self.Optimizer_weight = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='metaoptvar')
            self.Optimizer_grad = tf.gradients(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='metaoptvar')
        return g_new


class UpdateOpt(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        if epoch % (self.model.optimizer.Optimizer_steps+1) == 0 and epoch!=0:
            tvars = [K.get_value(self.model.optimizer.Optimizer_weight[i]) for i in range(len(self.model.optimizer.Optimizer_weight))]
            grads = [tf.gradients(K.mean(self.losses),tvars[i])[0] for i in range(len(tvars))]
            clipped_grads = tf.clip_by_global_norm(grads, 5.0)
            tvars_new = [tvars[i] - self.model.optimizer.lr*clipped_grads[i] for i in range(len(tvars))]
            K.set_value(self.model.optimizer.Optimizer_weight,tvars_new)
            self.losses = []
        else:
            self.losses.append(logs.get('loss'))
        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return

X = np.zeros((batch_num, batch_size, n_input), np.float32)
Y = np.zeros((batch_num, batch_size, n_output), np.float32)
cp = 0
for ii in range(batch_num):
    X[ii] = X_train[cp: cp + batch_size]
    Y[ii] = y_train[cp + 1: cp + batch_size + 1]
    cp += 10
metaopt = MetaOpt()
Metaupdate = UpdateOpt()
model = Sequential()
model.add(layers.Dense(n_hidden1,input_shape = (batch_size,n_input),activation='sigmoid'))
model.add(layers.Dense(n_output,input_shape = (batch_size,n_hidden1),activation='sigmoid'))
model.compile(optimizer = metaopt, loss='mean_squared_error', metrics=['accuracy'])
history = model.fit(X, Y, batch_size=batch_size, epochs=Optimizee_steps,callbacks=[Metaupdate])
plt.plot(history.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
model.save("Meta_Learner.h5")
