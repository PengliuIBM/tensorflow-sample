import tensorflow as tf
import pandas as pd
import numpy as np

# Loading the dataset in memory
data = pd.read_csv("data/train.csv")
#data = data.values

test = pd.read_csv("data/test.csv")
#testX = test.values
#for test

# Separating the labels from the training set
#trainX = data[:, 1:]
#labely = data[:, :1]

dfTrainFeatureVectors = data.drop(['label'], axis=1)
trainFeatureVectors = dfTrainFeatureVectors.values.astype(dtype=np.float32)
trainFeatureVectorsConvoFormat = trainFeatureVectors.reshape(42000, 28, 28, 1)
trainLabelsList = data['label'].tolist()
ohTrainLabelsTensor = tf.one_hot(trainLabelsList, depth=10)
ohTrainLabelsNdarray = tf.Session().run(ohTrainLabelsTensor).astype(dtype=np.float64)

testFeatureVectors = test.values.astype(dtype=np.float32)
testFeatureVectorsConvoFormat = testFeatureVectors.reshape(28000, 28, 28, 1)

# Parameters
learning_rate = 0.001
training_iters = 10000
#batch_size = 128
batch_size = 100
display_step = 100
# Network Parameters
n_input = 784 # MNIST data input (img shape: 28*28)
n_classes = 10 # MNIST total classes (0-9 digits)
dropout = 0.75 # Dropout, probability to keep units
# tf Graph input
#x = tf.placeholder(tf.float32, [None, n_input])
x = tf.placeholder(tf.float32, [None, 28, 28, 1])
y = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)

# Create some wrappers for simplicity
def conv2d(x, W, b, strides=1):
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)
def maxpool2d(x, k=2):
# MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],padding='SAME')
# Create model
def conv_net(x, weights, biases, dropout):
    # Reshape input picture
    x = tf.reshape(x, shape=[-1, 28, 28, 1])
    # Convolution Layer
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    # Max Pooling (down-sampling)
    conv1 = maxpool2d(conv1, k=2)
    # Convolution Layer
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    # Max Pooling (down-sampling)
    conv2 = maxpool2d(conv2, k=2)
    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    # Apply Dropout
    fc1 = tf.nn.dropout(fc1, dropout)
    # Output, class prediction
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out
# Store layers weight & bias
weights = {
    # 5x5 conv, 1 input, 32 outputs
    'wc1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
    # 5x5 conv, 32 inputs, 64 outputs
    'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
    # fully connected, 7*7*64 inputs, 1024 outputs
    'wd1': tf.Variable(tf.random_normal([7*7*64, 1024])),
    # 1024 inputs, 10 outputs (class prediction)
    'out': tf.Variable(tf.random_normal([1024, n_classes]))
}
biases = {
    'bc1': tf.Variable(tf.random_normal([32])),
    'bc2': tf.Variable(tf.random_normal([64])),
    'bd1': tf.Variable(tf.random_normal([1024])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}
def next_batch(idx,size,val,label, c_len):
    startIndex = (idx * size) % c_len	# 42000 is the size of the train.csv data set
    endIndex = startIndex + size
    batch_X = val[startIndex : endIndex]
    batch_Y = label[startIndex : endIndex] if label is not None else None
    return batch_X, batch_Y

# Construct model
pred = conv_net(x, weights, biases, keep_prob)
# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = pred, labels = y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
# Evaluate model
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
predictions = tf.argmax(tf.nn.softmax(pred), 1)

saver = tf.train.Saver()
# Initializing the variables
init = tf.global_variables_initializer()
# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    step = 0
    # Keep training until reach max iterations
    while step < training_iters:
        batch_x, batch_y = next_batch(step, batch_size, trainFeatureVectorsConvoFormat, ohTrainLabelsNdarray,42000)
        # Run optimization op (backprop)
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y,keep_prob: dropout})
        #sess.run(optimizer, feed_dict={X: batch_x, Y_: batch_y, lr: learning_rate, pkeep: 0.75})
        if step % display_step == 0:
        # Calculate batch loss and accuracy
            loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x,y: batch_y, keep_prob: 1.})
            print ("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
            "{:.6f}".format(loss) + ", Training Accuracy= " + \
            "{:.5f}".format(acc))
        step += 1
    saver.save(sess, "data/dr_model.ckpt")
    print("Optimization Finished!")
    # Calculate accuracy for 256 mnist test images
    print("Testing Accuracy:")
    # Get predictions on test data
    step = 0 
    pdcol1 = pd.Series([])
    pdcol2 = pd.Series([])
    #saver.restore(sess, "data/dr_model.ckpt") 
    '''
      I have to divide prediction task into small parts in order for my humble 4-GB memroy laptop
      to run the program without crash. 
    '''   
    while step < 280:
        batch_x, batch_y = next_batch(step, batch_size, testFeatureVectorsConvoFormat, None,28000)
        p = sess.run([predictions], {x: batch_x, keep_prob: 1.0})
        idx = step*batch_size
        pdcol1 = pdcol1.append(pd.Series(range(1+idx, len(p[0]) + 1+idx)),ignore_index = True)
        pdcol2 = pdcol2.append(pd.Series(p[0]) ,ignore_index = True)
        print(step, idx)
        step += 1
    print("Predict end")
    '''
    results = pd.DataFrame({'ImageId': pd.Series(range(1, len(p[0]) + 1)), 'Label': pd.Series(p[0])})
    results.to_csv('data/results.csv', index=False)
    '''
    # Write predictions to csv file
    results = pd.DataFrame({'ImageId': pdcol1, 'Label': pdcol2})
    results.to_csv('data/results.csv', index=False)
    print("End")
