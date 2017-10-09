import tensorflow as tf
from keras import backend as K
from keras import optimizers, layers, initializers, models
from keras.models import Sequential, Model
from keras.utils import np_utils, to_categorical
import keras.preprocessing.text as textpp
import os
import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(linewidth=140)
nlp_data_dir = r"C:\Data\nlp"
hidden_size = 512
max_sequence_length = 100
filter_syms = '0123456789"#$%&()*+-/:<=>?@[\\]^_`{|}~\t\n'
sentence_terminators = '.:;'
texts = []
labels_index = {}
labels = []
for name in sorted(os.listdir(nlp_data_dir)):
    filepath = os.path.join(nlp_data_dir, name)
    with open(filepath, 'r', encoding='latin1') as f:
        text = f.read()

    # Remove unwanted symbols
    text = text.replace("\n", " ")
    text = text.translate({ord(c): "" for c in filter_syms})

    # Convert to lowercase
    text = text.lower()

    texts.append(text)
tokenizer = textpp.Tokenizer(char_level=True)
tokenizer.fit_on_texts(text)
text_1hot = tokenizer.texts_to_matrix(text)
n_chars = text_1hot.shape[1]
token_lookup = dict(zip(tokenizer.word_index.values(), tokenizer.word_index.keys()))
num_sentences = text_1hot.shape[0]//max_sequence_length
mb_increment = max_sequence_length

X = np.zeros((num_sentences, max_sequence_length, n_chars), np.float32)
Y = np.zeros((num_sentences, max_sequence_length, n_chars), np.float32)
cp = 0
for ii in range(num_sentences):
    X[ii] = text_1hot[cp: cp + max_sequence_length]
    Y[ii] = text_1hot[cp + 1: cp + max_sequence_length + 1]
    cp += mb_increment

x = tf.placeholder(tf.float32,[None,max_sequence_length,n_chars])
y_ = tf.placeholder(tf.float32,[None,max_sequence_length,n_chars])
y_out = []
cell = tf.contrib.rnn.MultiRNNCell(
                     [tf.contrib.rnn.BasicLSTMCell(hidden_size) for _ in range(2)])
cell_out, state = tf.nn.dynamic_rnn(cell,x, dtype = tf.float32)
loss = 0.0
for i in range(max_sequence_length):
    if i > 0:tf.get_variable_scope().reuse_variables()
    cell_out_slice = cell_out[:,i]
    softmax_w = tf.get_variable("softmax_w", [hidden_size, n_chars])
    softmax_b = tf.get_variable("softmax_b", [n_chars])
    output_slice = tf.add(tf.matmul(cell_out_slice, softmax_w), softmax_b)
    y_slice = y_[:,i]
    y_out.append(output_slice)
    loss += tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = output_slice,labels=y_slice))
loss = loss/max_sequence_length
train_op = tf.train.GradientDescentOptimizer(0.001).minimize(loss)
output_op = tf.stack(y_out,axis = 1)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(100):
        chosen_idx = np.random.choice(X.shape[0], replace=False, size=20)
        x_s = X[chosen_idx,]
        y_s = Y[chosen_idx]
        cost, _ = sess.run([loss, train_op], feed_dict={x: x_s, y_: y_s})
        print("Epoch %d : loss %f" % (epoch, cost))
    saver = tf.train.Saver()
    saver.save(sess, "model", global_step=0)

tf.reset_default_graph()
saver = tf.train.Saver()
with tf.Session() as sess:
  # Restore variables from disk.
    saver.restore(sess, "model")
    next_char = np.random.randint(n_chars)
    for kk in range(10):
        cseed = np.zeros((1, max_sequence_length, n_chars), np.int32)
        for ii in range(0, max_sequence_length):
            cseed[0,ii,next_char] = 1
            out_emb = sess.run(output_op,feed_dict ={x:cseed})
            next_char = np.random.choice(n_chars, p=out_emb[0,ii])
    print("".join([token_lookup.get(x,"") for x in cseed[0].argmax(axis=1)]))

# model = Sequential()
# model.add(layers.LSTM(hidden_size,input_shape=(max_sequence_length, n_chars),return_sequences=True))
# model.add(layers.LSTM(hidden_size,input_shape=(max_sequence_length, n_chars),return_sequences=True))
# model.add(layers.TimeDistributed(layers.Dense(n_chars, activation='softmax')))
# model.compile('adam', 'categorical_crossentropy', metrics=['accuracy'])
# # # tvar = model.trainable_weights
# history = model.fit(X, Y, batch_size=20, epochs=100)
# # model.save("char_rnn_model_l2_h512.h5")
# model = models.load_model("char_rnn_model_l2_h512.h5")
# next_char = np.random.randint(n_chars)
# for kk in range(10):
#    cseed = np.zeros((1, max_sequence_length, n_chars), np.int32)
#    for ii in range(0, sequence_length):
#        cseed[0,ii,next_char] = 1
#        out_emb = model.predict(cseed)
#        next_char = np.random.choice(n_chars, p=out_emb[0,ii])
# print("".join([token_lookup.get(x,"") for x in cseed[0].argmax(axis=1)]))