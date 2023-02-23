import pandas as pd 
from sklearn.model_selection import train_test_split

from tensorflow import keras
from sklearn.preprocessing import StandardScaler
import numpy as np
import tensorflow as tf
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, RepeatVector, TimeDistributed

data = pd.read_csv('PWR_Dataset.csv') 
print (data.head())

y = data.Pred  
X = data.drop('Pred', axis=1) 
train, test, train_label, test_label = train_test_split(X,y,test_size=0.2)

from keras.models import Sequential 
from keras.layers import Dense 
 
# Call CNN model 
model = Sequential() 
model.add(Dense(12, input_dim=20, activation='relu')) 
model.add(Dense(8, activation='relu')) 

# Compile model 
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) 

# Fit the model 
history=model.fit(train, train_label, epochs=20, batch_size=32) 


rnn_layers = 2
embedding_size = 128
hidden_size = 128
input_dropout = 0.2
learning_rate = 0.01
max_grad_norm = 5
num_epochs = 500001
batch_size = 20
seq_length = 10


def main():
    logging.basicConfig(stream=sys.stdout,
                        format='%(asctime)s %(levelname)s:%(message)s',
                        level=logging.INFO,
                        datefmt='%I:%M:%S')
    with codecs.open(data, 'r') as f:
        text = f.read()
    train_text = text
    vocab_index_dict, index_vocab_dict, vocab_size = create_vocab(text)

    train_batches = BatchGenerator(train_text, batch_size, seq_length, vocab_size, vocab_index_dict)
    graph = tf.Graph()
    with graph.as_default():
        input_data = tf.placeholder(tf.int64, [batch_size, seq_length], name='inputs')
        input_targets = tf.placeholder(tf.int64, [batch_size, seq_length], name='targets')
        tf_learning_rate = tf.constant(learning_rate)

        embedding = tf.get_variable('embedding', [vocab_size, embedding_size])
        inputs = tf.nn.embedding_lookup(embedding, input_data)
        sliced_inputs = [tf.squeeze(input_, [1]) for input_ in
                         tf.split(axis=1, num_or_size_splits=seq_length, value=inputs)]

        weights = tf.Variable(tf.random_normal([2 * hidden_size, vocab_size]))
        biases = tf.Variable(tf.random_normal([vocab_size]))

        lstm_fw_cell = rnn.BasicLSTMCell(hidden_size, forget_bias=1.0)
        lstm_bw_cell = rnn.BasicLSTMCell(hidden_size, forget_bias=1.0)
        outputs, _, _ = rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, sliced_inputs, dtype=tf.float32)

        flat_outputs = tf.reshape(tf.concat(axis=1, values=outputs), [-1, 2 * hidden_size])
        flat_targets = tf.reshape(tf.concat(axis=1, values=input_targets), [-1])
        logits = tf.matmul(flat_outputs, weights) + biases
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=flat_targets)
        mean_loss = tf.reduce_mean(loss)

        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(mean_loss, tvars), max_grad_norm)
        optimizer = tf.train.AdamOptimizer(tf_learning_rate)
        train_op = optimizer.apply_gradients(zip(grads, tvars))

        prediction = tf.nn.softmax(logits)
        correct_pred = tf.equal(tf.argmax(prediction, 1), flat_targets)
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    with tf.Session(graph=graph) as session:
        tf.global_variables_initializer().run()
        for i in range(num_epochs):
            data2 = train_batches.next()
            inputs = np.array(data2[:-1]).transpose()
            targets = np.array(data2[1:]).transpose()
            ops = [mean_loss, train_op, tf_learning_rate, accuracy]
            feed_dict = {input_data: inputs, input_targets: targets}
            average_loss, __, lr, acc = session.run(ops, feed_dict)
            if i % 100 == 0:
                logging.info("average loss: %.5f,accuracy: %.3f", average_loss, acc)


def create_vocab(text):
    unique_chars = list(set(text))
    print(unique_chars)
    vocab_size = len(unique_chars)
    vocab_index_dict = {}
    index_vocab_dict = {}
    for i, char in enumerate(unique_chars):
        vocab_index_dict[char] = i
        index_vocab_dict[i] = char
    return vocab_index_dict, index_vocab_dict, vocab_size


class BatchGenerator(object):
    def __init__(self, text, batch_size, seq_length, vocab_size, vocab_index_dict):
        self._text = text
        self._text_size = len(text)
        self._batch_size = batch_size
        self.vocab_size = vocab_size
        self.seq_length = seq_length
        self.vocab_index_dict = vocab_index_dict

        segment = self._text_size // batch_size

        self._cursor = [offset * segment for offset in range(batch_size)]
        self._last_batch = self._next_batch()

    def _next_batch(self):
        batch = np.zeros(shape=(self._batch_size), dtype=np.float)
        for b in range(self._batch_size):
            batch[b] = self.vocab_index_dict[self._text[self._cursor[b]]]
            self._cursor[b] = (self._cursor[b] + 1) % self._text_size
        return batch

    def next(self):
        batches = [self._last_batch]
        for step in range(self.seq_length):
            batches.append(self._next_batch())
        self._last_batch = batches[-1]
        return batches


if __name__ == '__main__':
    main()
    
# evaluate the model 
scores = model.evaluate(test, test_label) 
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

plt.plot(history.history['loss'], label='Loss')
plt.plot(history.history['accuracy'], label='Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend();

