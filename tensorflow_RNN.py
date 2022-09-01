import pickle

import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib import layers
import pandas as pd
import numpy as np
import time


def read_data():
    with open('data/yelp/yelp_data', 'rb') as f:
        data_x, data_y = pickle.load(f)
        length = len(data_x)
        print(length)
        train_x, test_x = data_x[5000:-5000], data_x[:5000] + data_x[-5000:]
        train_y, test_y = data_y[5000:-5000], data_y[:5000] + data_y[-5000:]
    return train_x, train_y, test_x, test_y

def length(sequences):
    used = tf.sign(tf.reduce_max(tf.abs(sequences), reduction_indices=2))
    seq_len = tf.reduce_sum(used, reduction_indices=1)
    return tf.cast(seq_len, tf.int32)

def batch(inputs):
  batch_size = len(inputs)
  print("batch size", batch_size)
  #print(inputs)
  document_sizes = np.array([len(doc) for doc in inputs], dtype=np.int32)
  document_size = document_sizes.max()
  sentence_sizes_ = [[len(sent) for sent in doc] for doc in inputs]
  sentence_size = 0
  for i in sentence_sizes_:
    for j in i:
      if j > sentence_size:
        sentence_size = j;	

  #sentence_size = max(map(max, sentence_sizes_))
  b = np.zeros(shape=[batch_size, document_size, sentence_size], dtype=np.int32) # == PAD
  sentence_sizes = np.zeros(shape=[batch_size, document_size], dtype=np.int32)
  for i, document in enumerate(inputs):
    for j, sentence in enumerate(document):
      sentence_sizes[i, j] = sentence_sizes_[i][j]
      for k, word in enumerate(sentence):
        b[i, j, k] = word
  return b, document_size, sentence_size, batch_size    

class model_RNN_Model():
    def __init__(self,
                vocab_size,
                embedding_size,
                classes,
                dropout_keep_proba,
                hidden_size,
                is_training=None):
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.classes = classes
        self.hidden_size = hidden_size
        self.dropout_keep_proba = dropout_keep_proba
        self.is_training = is_training
        with tf.name_scope('placeholder'):
            self.input_x = tf.placeholder(tf.int32, [None, None, None], name='input_x')
            self.input_y = tf.placeholder(tf.float32, [None, self.classes], name='input_y')
            self.max_sentence_num = tf.placeholder(tf.int32, name='max_sentence_num')
            self.max_sentence_length = tf.placeholder(tf.int32, name='max_sentence_length')
            self.batch_size = tf.placeholder(tf.int32, name='batch_size')
        self.build_model()

    def embedding_layer(self):
        with tf.name_scope("embedding"):
            embedding_mat = tf.Variable(tf.truncated_normal((self.vocab_size, self.embedding_size)))
            word_embedded = tf.nn.embedding_lookup(embedding_mat, self.input_x)
        self.embedding_layer = word_embedded
        #return word_embedded

    def biRNN(self, inputs, name):
        with tf.variable_scope(name):
            GRU_cell_fw = rnn.GRUCell(self.hidden_size)
            GRU_cell_bw = rnn.GRUCell(self.hidden_size)
            ((fw_outputs, bw_outputs), (_, _)) = tf.nn.bidirectional_dynamic_rnn(cell_fw=GRU_cell_fw,
                                                                                 cell_bw=GRU_cell_bw,
                                                                                 inputs=inputs,
                                                                                 sequence_length=length(inputs),
                                                                                 dtype=tf.float32)
            outputs = tf.concat((fw_outputs, bw_outputs), 2)
        
        return outputs

    def attention_layer(self, inputs, name):
        """
        Performs task-specific attention reduction, using learned
        attention context vector (constant within task of interest).
        Args:
            inputs: Tensor of shape [batch_size, units, input_size]
                `input_size` must be static (known)
                `units` axis will be attended over (reduced from output)
                `batch_size` will be preserved
            output_size: Size of output's inner (feature) dimension
        Returns:
            outputs: Tensor of shape [batch_size, output_dim].
        """
        assert len(inputs.get_shape()) == 3 and inputs.get_shape()[-1].value is not None

        with tf.variable_scope(name) as scope:
            attention_context_vector = tf.Variable(tf.truncated_normal([self.hidden_size * 2]), name='attention_context_vector')
            input_projection = layers.fully_connected(inputs, self.hidden_size * 2,
                                                      activation_fn=tf.nn.tanh,
                                                      scope=scope)

            vector_attn = tf.reduce_sum(tf.multiply(input_projection, attention_context_vector), axis=2, keep_dims=True)
            attention_weights = tf.nn.softmax(vector_attn, dim=1)
            weighted_projection = tf.multiply(input_projection, attention_weights)
            outputs = tf.reduce_sum(weighted_projection, axis=1)

        return outputs

    def sentence_layer(self):
        with tf.name_scope("sent2vec"):
            word_embedded = tf.reshape(self.embedding_layer, [-1, self.max_sentence_length, self.embedding_size])
            word_encoded = self.biRNN(word_embedded, name='word_encoder')
            sent_vec = self.attention_layer(word_encoded, name='word_attention')
            sent_vec = layers.dropout(sent_vec,
                                      keep_prob=self.dropout_keep_proba,
                                      is_training=self.is_training,
                                      )
        self.sent_vec = sent_vec

    def doc_layer(self):
        with tf.name_scope("doc2vec"):
            sent_vec = tf.reshape(self.sent_vec, [-1, self.max_sentence_num, self.hidden_size*2])
            doc_encoded = self.biRNN(sent_vec, name='sent_encoder')
            doc_vec = self.attention_layer(doc_encoded, name='sent_attention')
        self.doc_vec = doc_vec

    def build_model(self):
        self.embedding_layer()
        self.sentence_layer()
        self.doc_layer()
        with tf.name_scope('doc_classification'):
            out = layers.fully_connected(inputs=self.doc_vec, num_outputs=self.classes, activation_fn=None)
        self.logits  = out
        self.prediction = tf.argmax(self.logits, axis=-1)

    def set_training_state(self, state):
    	self.is_training = state


# Data loading params
tf.flags.DEFINE_integer("vocab_size", 50000, "vocabulary size")
tf.flags.DEFINE_integer("classes", 2, "number of classes")
tf.flags.DEFINE_integer("embedding_size", 200, "Dimensionality of character embedding (default: 200)")
tf.flags.DEFINE_integer("hidden_size", 50, "Dimensionality of GRU hidden layer (default: 50)")
tf.flags.DEFINE_integer("batch_size", 32, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 1, "Number of training epochs (default: 50)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")
tf.flags.DEFINE_integer("evaluate_every", 100, "evaluate every this many batches")
tf.flags.DEFINE_float("learning_rate", 0.01, "learning rate")
tf.flags.DEFINE_float("grad_clip", 5, "grad clip to prevent gradient explode")
tf.flags.DEFINE_float("dropout_keep_proba", 0.5, "drop out probability")
tf.flags.DEFINE_bool("is_training", True, "training or ")

FLAGS = tf.flags.FLAGS
train_x, train_y, dev_x, dev_y = read_data()
print("data load finished")
with tf.Session() as sess:
    model = model_RNN_Model(
                    vocab_size=FLAGS.vocab_size,
                    embedding_size = FLAGS.embedding_size,
                    classes=FLAGS.classes,
                    hidden_size = FLAGS.hidden_size,
                    dropout_keep_proba = FLAGS.dropout_keep_proba,
                    is_training = FLAGS.is_training
                    )

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=model.input_y,
                                                                      logits=model.logits,
                                                                      name='loss'))
    tf.summary.scalar('loss', loss)
    with tf.name_scope('accuracy'):
        label = tf.argmax(model.input_y, axis=1, name='label')
        acc = tf.reduce_mean(tf.cast(tf.equal(model.prediction, label), tf.float32))

    tf.summary.scalar('accuracy', acc)	
    global_step = tf.Variable(0, trainable=False)
    optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
    tvars = tf.trainable_variables()
    grads, global_norm  = tf.clip_by_global_norm(tf.gradients(loss, tvars), FLAGS.grad_clip)
    tf.summary.scalar('global_grad_norm', global_norm)
    grads_and_vars = tuple(zip(grads, tvars))
    train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    train_summary_op =  tf.summary.merge_all()

    def train_step(x_batch, y_batch, document_sizes, sentence_sizes, batch_size, epoch):
        feed_dict = {
            model.input_x: x_batch,
            model.input_y: y_batch,
            model.max_sentence_num: document_sizes,
            model.max_sentence_length: sentence_sizes,
            model.batch_size: batch_size
        }
        _, step, summaries, cost, accuracy = sess.run([train_op, global_step, train_summary_op, loss, acc], feed_dict)
        #sess.run(train_op, feed_dict)
        time_str = str(int(time.time()))
        print("{}: epoch {}, step {}, loss {:g}, acc {:g}".format(time_str, epoch, step, cost, accuracy))
        #train_summary_writer.add_summary(summaries, step)
        return step

    for epoch in range(FLAGS.num_epochs):
        print('current epoch %s' % (epoch + 1))
        #for i in range(0, 190000, FLAGS.batch_size):
        temp = 5120 * FLAGS.batch_size
        for i in range(temp, 190000, FLAGS.batch_size):
            x = train_x[i:i + FLAGS.batch_size]
            y = train_y[i:i + FLAGS.batch_size]
            #x_batch =  tf.train.batch([x[0]], batch_size=1, shapes=[30,30], name="x_batch")
            b, ds, ss, bs = batch(x)
            print(ds,ss)
            step = train_step(b, y, ds, ss, bs, epoch)
            #if step % FLAGS.evaluate_every == 0:
            #    dev_step(dev_x, dev_y, dev_summary_writer)
    saver.save(sess, "data/relp/recomm_model.ckpt")
    #begin to predict
    predictions = []
    labels = []
    model.is_training = False
    for i in range(0, 10000, FLAGS.batch_size):
        x = test_x[i:i + FLAGS.batch_size]
        y = test_y[i:i + FLAGS.batch_size]
        #x_batch =  tf.train.batch([x[0]], batch_size=1, shapes=[30,30], name="x_batch")
        b, ds, ss, bs = batch(x)
        print(ds,ss)
        labels.extend(y);
        feed_dict = {
            model.input_x: b,            
            model.max_sentence_num: ds,
            model.max_sentence_length: ss,
            model.batch_size: bs
        }
        predictions.extend(sess.run(model.prediction, feed_dict))
    df = pd.DataFrame({'predictions': predictions, 'labels': labels})

    print('Accuracy:')
    print((df['predictions'] == df['labels']).mean())





