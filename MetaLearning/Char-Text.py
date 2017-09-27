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

x = tf.placeholder(tf.float32,[num_sentences, max_sequence_length, n_chars])
y_ = tf.placeholder(tf.float32,[num_sentences, max_sequence_length, n_chars])
cell = tf.nn.rnn_cell.LSTMCell(hidden_size,state_is_tuple=True)
val,state = tf.nn.dynamic_rnn(cell,x,dtype = tf.float32)
val = tf.transpose(val,[1,0,2])
last = tf.gather(val,int(val.get.shape()[0]) - 1)
weight = tf.Variable(tf.truncated_normal([hidden_size,int(y_.get_shape()[1])]))
bias = tf.Variable(tf.constant(0.1,shape = [y_.get_shape()[1]]))
prediction = tf.nn.softmax(tf.matmul(last, weight) + bias)
cross_entropy = -tf.reduce_sum(target * tf.log(tf.clip_by_value(prediction,1e-10,1.0)))
optimizer = tf.train.AdamOptimizer()
minimize = optimizer.minimize(cross_entropy)
mistakes = tf.not_equal(tf.argmax(target, 1), tf.argmax(prediction, 1))
error = tf.reduce_mean(tf.cast(mistakes, tf.float32))

init_op = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init_op)
batch_size = 20
no_of_batches = int(len(X)/batch_size)
epoch = 10
for i in range(epoch):
    ptr = 0
    for j in range(no_of_batches):
        inp, out = X[ptr:ptr+batch_size], Y[ptr:ptr+batch_size]
        ptr+=batch_size
        sess.run(minimize,{x: inp, y_: out})
    print("Epoch - ",str(i))
# incorrect = sess.run(error,{x: test_input, y_: test_output})
# print('Epoch {:2d} error {:3.1f}%'.format(i + 1, 100 * incorrect))
# sess.close()

#     y = tf.matmul(output,softmax_w) + softmax_b
# model = Sequential()
# model.add(layers.LSTM(hidden_size,
#                            input_shape=(max_sequence_length, n_chars),
#                            return_sequences=True))
# model.add(layers.LSTM(hidden_size,
#                            input_shape=(max_sequence_length, n_chars),
#                            return_sequences=True))
# model.add(layers.TimeDistributed(layers.Dense(n_chars, activation='softmax')))
# model.compile('adam', 'categorical_crossentropy', metrics=['accuracy'])
# model.summary()
# model.fit(X, Y, batch_size=20, epochs=10)
# model.save("char_rnn_model_l2_h512.h5")
# model = models.load_model("char_rnn_model_l2_h512.h5")
# sequence_length = 60
# next_char = np.random.randint(n_chars)
# for kk in range(10):
#     cseed = np.zeros((1, max_sequence_length, n_chars), np.int32)
#     for ii in range(0, sequence_length):
#         cseed[0,ii,next_char] = 1
#         out_emb = model.predict(cseed)
#         next_char = np.random.choice(n_chars, p=out_emb[0,ii])
#
#     print("".join([token_lookup.get(x,"") for x in cseed[0].argmax(axis=1)]))