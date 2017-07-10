import tensorflow as tf

'introduce a amount of noise to a variables initialization for better performance'
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

'give the variable a slight positive bias to avoid dead neurons'
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

'do convolution, whatever that is (figure this out)'
def conv2d(x,W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

'do pooling (figure out what this is also)'
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

#download and read MNIST test data automatically
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

#None: the number of training samples, unlimited
#784: the pixels of the 28x28 pixel image flattened to a 1D array
x = tf.placeholder(tf.float32, shape=[None, 784])

#define the goal state
y_ = tf.placeholder(tf.float32, [None, 10])

#create first convolutional layer
W_convolution1 = weight_variable([5,5,1,32])
b_convolution1 = bias_variable([32])

#reformat image as a 4d tensor (with width, height, #color channels)
x_image = tf.reshape(x, [-1, 28, 28, 1])

#convolve, bias, apply the ReLU function, and max pool the image (apply 1st layer)
h_convolution1 = tf.nn.relu(conv2d(x_image, W_convolution1) + b_convolution1)
h_pool1 = max_pool_2x2(h_convolution1)

#create and apply a second layer for a deep nn
W_convolution2 = weight_variable([5, 5, 32, 64])
b_convolution2 = bias_variable([64])
h_convolution2 = tf.nn.relu(conv2d(h_pool1, W_convolution2) + b_convolution2)
h_pool2 = max_pool_2x2(h_convolution2)

#add a fully-connected layer of 1024 neurons to process the whole image
#first reshape the tensor from the pooling layer into a batch of vectors
#then multiply by weight matrix, add bias, and apply ReLU as before
W_fullyconnected1 = weight_variable([7*7*64, 1024])
b_fullyconnected1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fullyconnected1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fullyconnected1) + b_fullyconnected1)

#apply dropout to reduce overfitting
keep_probability = tf.placeholder(tf.float32)
h_fullyconnected1_drop = tf.nn.dropout(h_fullyconnected1, keep_probability)

#add the output layer
W_fullyconnected2 = weight_variable([1024, 10])
b_fullyconnected2 = bias_variable([10])

y_convolution = tf.matmul(h_fullyconnected1_drop, W_fullyconnected2)+b_fullyconnected2

#create a checkpoint file saver to save a snapshot of the model for recovery and
#further training later
saver = tf.train.Saver()

#train and evaluate
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    labels=y_, logits=y_convolution))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_convolution, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(20000):
        batch = mnist.train.next_batch(50)
        if i%100 == 0:
            train_accuracy = accuracy.eval(feed_dict={
                x: batch[0], y_: batch[1], keep_probability: 1.0})
            print("step ", i, ", training accuracy ", '{0:.4f}'.format(train_accuracy))
            saver.save(sess, "MNISTdeepnnsave/MNISTdeepnnsave", global_step = i)
            #can restore later with saver.restore(sess, directory)
        train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_probability: 0.5})

    print("test accuracy ", '{0:.4f}'.format(accuracy.eval(feed_dict={x:mnist.test.images,
        y_: mnist.test.labels, keep_probability: 1.0})))
