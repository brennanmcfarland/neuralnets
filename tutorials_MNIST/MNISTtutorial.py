import tensorflow as tf
#download and read MNIST test data automatically
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

#None: the number of training samples, unlimited
#784: the pixels of the 28x28 pixel image flattened to a 1D array
x = tf.placeholder(tf.float32, [None, 784])

#initialize the weights to 0
#10 possible values (0-9)
W = tf.Variable(tf.zeros([784, 10]))
#initialize the biases to 0
b = tf.Variable(tf.zeros([10]))

#define our softmax model
#matmul: matrix multiply input vector with weight matrix
#then add the bias vector to the result
y = tf.nn.softmax(tf.matmul(x, W)+b)

#define the goal state
y_ = tf.placeholder(tf.float32, [None, 10])

#implement cross-entropy function manually
#don't actually use this because it's numerically unstable
cross_entropy = tf.reduce_mean(-tf.reduce_sum(
    y_*tf.log(y), reduction_indices=[1]))

#implemnet the cross-entropy function automatically
cross_entropy = tf.reduce_mean(
tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

#define a training step for the network
train_step = tf.train.GradientDescentOptimizer(0.05).minimize(cross_entropy)

#launch the model in an interactive session
sess = tf.InteractiveSession()

#initialize variables
tf.global_variables_initializer().run()

#train the network
for _ in range(10000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

#determine the number of correct guesses and the aggregate accuracy
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print(sess.run(accuracy, feed_dict={x: mnist.test.images,
    y_: mnist.test.labels}))
