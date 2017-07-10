import tensorflow as tf
import numpy as np

#start a session
sess = tf.Session()

#create constant nodes and print their type
node1 = tf.constant(3.0, dtype=tf.float32)
node2 = tf.constant(4.0)
print(node1, node2)

#evaluate the nodes
print(sess.run([node1, node2]))

#create a third node as the sum of the first 2 and evaluate
node3 = tf.add(node1, node2)
print(sess.run(node3))

#create placeholders (parameters) and evaluate their sum
node1 = tf.placeholder(tf.float32)
node2 = tf.placeholder(tf.float32)
node3 = node1 + node2
print(sess.run(node3, {node1:[2,4], node2:[1,5]}))

#use tf variables
m = tf.Variable([1], dtype=tf.float32)
b = tf.Variable([-1], dtype=tf.float32)
x = tf.placeholder(tf.float32)
linear_model = m*x+b
init = tf.global_variables_initializer() #must have these two
sess.run(init)                           #lines to init the variables

#evaluate the tf varibles above
print(sess.run(linear_model, {x:[0,1,2,3,4,5]}))

#define a loss function
y = tf.placeholder(tf.float32) #the target state
delta_squared = tf.square(linear_model-y) #square of errors
loss = tf.reduce_sum(delta_squared) #sum of squared errors
print(sess.run(loss, {x:[0,1,2,3], y:[1,1,1,1]}))

#adjust the linear model parameters to fit better an evaluate
newm = tf.assign(m, [0])
newb = tf.assign(b, [1])
sess.run([newm, newb])
print(sess.run(loss, {x:[0,1,2,3], y:[1,1,1,1]}))

#set the optimizer (learning algorithm)
#and the goal (to minimize the loss function)
optimizer = tf.train.GradientDescentOptimizer(.01)
train = optimizer.minimize(loss)

#starting with bad inputs, train towards goal
sess.run(init) #reinitialize variables to default values
for i in range(1000):
    sess.run(train, {x:[2,3,4,5], y:[3,4,5,6]})

#and print the results
finm, finb, finloss = sess.run([m,b,loss], {x:[2,3,4,5], y:[3,4,5,6]})
print(finm, finb, finloss)

#do the same thing, only this time with contrib.learn for simplicity
#############################################################
def model(features, labels, mode):
    #build linear model and predict values
    m = tf.get_variable("m", [1], dtype=tf.float64)
    b = tf.get_variable("b", [1], dtype=tf.float64)
    y = m*features['x']+b
    #loss sub-graph
    loss = tf.reduce_sum(tf.square(y-labels))
    #create the sub-graph for training
    global_step = tf.train.get_global_step()
    optimizer = tf.train.GradientDescentOptimizer(.01)
    train = tf.group(optimizer.minimize(loss),
        tf.assign_add(global_step, 1))
    #connect these subgraphs to the appropriate functionality
    return tf.contrib.learn.ModelFnOps(
    mode=mode, predictions=y, loss=loss, train_op=train
    )

estimator = tf.contrib.learn.Estimator(model_fn=model)

#define data sets
x_train = np.array([1.,2.,3.,4.])
y_train = np.array([0.,-1.,-2.,-3.])
x_eval = np.array([2.,5.,8.,1.])
y_eval = np.array([-1.01, -4.2, -20., 0.])
input_fn = tf.contrib.learn.io.numpy_input_fn({"x": x_train},
    y_train, 4, num_epochs=1000)

#train
estimator.fit(input_fn=input_fn, steps=1000)
#evaluate accuracy
train_loss = estimator.evaluate(input_fn=input_fn)
#eval_loss = estimator.evaluate(input_fn=eval_input_fn)

print("train loss: ", train_loss)
#print("eval loss: ", eval_loss)
