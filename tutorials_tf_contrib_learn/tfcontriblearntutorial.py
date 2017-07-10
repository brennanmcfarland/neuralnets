from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import urllib.request

import numpy as np
import tensorflow as tf

IRIS_TRAINING = "iris_training.csv"
IRIS_TRAINING_URL = "http://download.tensorflow.org/data/iris_training.csv"

IRIS_TEST = "iris_test.csv"
IRIS_TEST_URL = "http://download.tensorflow.org/data/iris_test.csv"

#the datasets must be downloaded to the same directory for this to work

#load datasets into Dataset objects
#the Datasets are tuples, with fields data and target (correct output)
training_set = tf.contrib.learn.datasets.base.load_csv_with_header(
    filename = IRIS_TRAINING,
    target_dtype = np.int,
    features_dtype = np.float32)
test_set = tf.contrib.learn.datasets.base.load_csv_with_header(
    filename = IRIS_TEST,
    target_dtype = np.int,
    features_dtype = np.float32)

#specify that all features have real-value data
#feature columns: data type for features in data set
#4 dimensions for 4 features in the data set
feature_columns = [tf.contrib.layers.real_valued_column("", dimension=4)]

#build a 3 layer DNN with 10,20,10 units in a layer, for classification
#feature columns: the set of feature columns defined above
#hidden_units: hidden layers of 10, 20, and 10 neurons respectively
#n_classes: the 3 target classes we're classifying into
#model_dir: checkpoint save data directory
classifier = tf.contrib.learn.DNNClassifier(
    feature_columns = feature_columns,
    hidden_units = [10, 20, 10],
    n_classes = 3,
    model_dir = "/tmp/iris_model")

#define the training inputs, just a set of constants
def get_train_inputs():
    x = tf.constant(training_set.data)
    y = tf.constant(training_set.target)
    return x,y

#train the model to fit
#the state of the model is preserved in classifier, so the two lines below could
#be combined into 1 with twice as many steps, but it's here as an example
# of how it can be split up with the sam results
classifier.fit(x=training_set.data, y=training_set.target, steps=2000)
classifier.fit(x=training_set.data, y=training_set.target, steps=2000)

#define test inputs
def get_test_inputs():
    x = tf.constant(test_set.data)
    y = tf.constant(test_set.target)
    return x,y

#evaluate accuracy
accuracy_score = classifier.evaluate(input_fn=get_test_inputs, steps=1)["accuracy"]
print("\nTest Accuracy: {0:f}\n".format(accuracy_score))

#classify two new samples
def new_samples():
    return np.array([[6.4,3.2,4.5,1.5],[5.8,3.1,5.0,1.7]], dtype=np.float32)

predictions = list(classifier.predict_classes(input_fn=new_samples))

print("New samples, predictions: {}\n".format(predictions))
