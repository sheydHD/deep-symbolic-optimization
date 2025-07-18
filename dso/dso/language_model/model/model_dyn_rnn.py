"""Model architecture of default (saved) LanguageModel"""

import tensorflow as tf
import tensorflow_addons as tfa
tf.compat.v1.disable_v2_behavior()

# rnn = tf.nn.rnn_cell
from tensorflow.compat.v1 import nn

class LanguageModel:
    def __init__(self, vocabulary_size, embedding_size, num_layers, num_hidden, mode='train'):
        self.embedding_size = embedding_size
        self.num_layers = num_layers
        self.num_hidden = num_hidden

        self.x = tf.compat.v1.placeholder(tf.int32, [None, None], name="x") # whole seq + seq len
        self.keep_prob = tf.compat.v1.placeholder(tf.float32, [], name="keep_prob")
        self.batch_size = tf.shape(self.x)[0]

        if mode == 'train':
            self.lm_input = self.x[:, :-2]
            self.seq_len = self.x[:, -1]
        elif mode == 'predict':
            self.lm_input = self.x[:,:]
            self.seq_len = tf.reduce_sum(tf.sign(self.lm_input), 1)

        self.logits=tf.Variable(2.0, name="logits")

        # embedding, one-hot encoding
        # if embedding:
        with tf.compat.v1.name_scope("embedding"):
            init_embeddings = tf.random.uniform([vocabulary_size, self.embedding_size])
            embeddings = tf.compat.v1.get_variable("embeddings", initializer=init_embeddings)
            lm_input_emb = tf.nn.embedding_lookup(embeddings, self.lm_input)

        with tf.compat.v1.variable_scope("rnn"):
            def make_cell():
                cell = nn.rnn_cell.BasicRNNCell(self.num_hidden)
                cell = nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=self.keep_prob)
                return cell

            cell = nn.rnn_cell.MultiRNNCell([make_cell() for _ in range(self.num_layers)])

            self.initial_state = cell.zero_state(self.batch_size, dtype=tf.float32)

            # rnn_outputs: [batch_size, max_len, num_hidden(cell output)]
            rnn_outputs, self.last_state = tf.compat.v1.nn.dynamic_rnn(
                cell=cell, 
                initial_state=self.initial_state,
                inputs=lm_input_emb,
                sequence_length=self.seq_len, 
                dtype=tf.float32)

        # with tf.name_scope("output"):
        self.logits = tf.compat.v1.layers.dense(rnn_outputs, vocabulary_size)


        with tf.compat.v1.name_scope("loss"):
            if mode == "train":
                target = self.x[:, 1:-1]
            elif mode == "predict":
                target = self.x[:, :]

            self.loss = tfa.seq2seq.sequence_loss(
                logits=self.logits,
                targets=target,
                weights=tf.sequence_mask(self.seq_len, tf.shape(self.x)[1] - 2, dtype=tf.float32),
                average_across_timesteps=True,
                average_across_batch=True
            )

