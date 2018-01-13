import math
import tensorflow as tf
import numpy as np


def graph_function(xmin,xmax,xres,function,*args):
    """takes a given mathematical function and returns a set of points to graph"""

    x, y = [], []
    i=0
    while xmin+i*xres <= xmax:
        x.append(xmin+i*xres)
        y.append(function(x[i], *args))
        i += 1
    print('goal function dim: ', i)
    return [x, y]


def quadratic(x, a, b, c):
    return a*x** + b*x + c


def test_quadratic(x):
    return quadratic(x, 3, 2, 1)


def model(features, labels, mode):
    # build linear model
    a = tf.get_variable("a", [1], dtype=tf.float64)
    b = tf.get_variable("b", [1], dtype=tf.float64)
    c = tf.get_variable("c", [1], dtype=tf.float64)
    y = a*features['x']*features['x'] + b*features['x'] + c
    # loss sub-graph
    loss = tf.reduce_sum(tf.square(y-labels))
    # create the sub-graph for training
    global_step = tf.train.get_global_step()
    optimizer = tf.train.GradientDescentOptimizer(.01)
    train = tf.group(optimizer.minimize(loss), tf.assign_add(global_step, 1))
    # connect subgraphs to the appropriate functionality
    return tf.contrib.learn.ModelFnOps(mode=mode, predictions=y, loss=loss, train_op=train)


print('starting tensorflow...')
sess = tf.Session()
estimator = tf.contrib.learn.Estimator(model_fn=model)

# define data sets
train_range = range(100)
eval_range = range(100, 200)
x_train = np.array(train_range)
y_train = np.array([test_quadratic(x) for x in train_range])
x_eval = np.array(eval_range)
y_eval = np.array([test_quadratic(x) for x in eval_range])
input_fn = tf.contrib.learn.io.numpy_input_fn({"x": x_train}, y_train, len(x_train), num_epochs=1000)
eval_input_fn = tf.contrib.learn.io.numpy_input_fn({"x": x_eval}, y_eval, len(x_eval), num_epochs=1000)

#train
estimator.fit(input_fn=input_fn, steps=1000)

#evalutate accuracy
train_loss = estimator.evaluate(input_fn=input_fn)
eval_loss = estimator.evaluate(input_fn=eval_input_fn)
print("training set loss: ", train_loss)
print("test set loss: ", eval_loss)
graph_function(0, 200, 1, quadratic, a, b, c)
graph_function(0, 200, 1, test_quadratic)
