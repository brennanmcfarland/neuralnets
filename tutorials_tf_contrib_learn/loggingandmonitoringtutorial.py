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

#set logging verbosity
tf.logging.set_verbosity(tf.logging.INFO)

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

#define validation metrics to monitor
validation_metrics = {
    "accuracy":
        tf.contrib.learn.MetricSpec(
            metric_fn=tf.contrib.metrics.streaming_accuracy,
            prediction_key=tf.contrib.learn.PredictionKey.CLASSES),
    "precision":
        tf.contrib.learn.MetricSpec(
            metric_fn=tf.contrib.metrics.streaming_precision,
            prediction_key=tf.contrib.learn.PredictionKey.CLASSES),
    "recall":
        tf.contrib.learn.MetricSpec(
            metric_fn=tf.contrib.metrics.streaming_recall,
            prediction_key=tf.contrib.learn.PredictionKey.CLASSES)
}

#create a ValidationMonitor to evaluate against the test data every 100 steps
#early stopping: stop if the loss has not improved for 200 steps
validation_monitor = tf.contrib.learn.monitors.ValidationMonitor(
    test_set.data,
    test_set.target,
    every_n_steps=100,
    metrics=validation_metrics,
    early_stopping_metric="loss",
    early_stopping_metric_minimize=True,
    early_stopping_rounds=200)

#build a 3 layer DNN with 10,20,10 units in a layer, for classification
#feature columns: the set of feature columns defined above
#hidden_units: hidden layers of 10, 20, and 10 neurons respectively
#n_classes: the 3 target classes we're classifying into
#model_dir: checkpoint save data directory
#the model will automatically pick up where it left off with whatever is in the
#checkpoint directory; to start over, just empty the directory
#config: setting it to create a checkpoint every second
classifier = tf.contrib.learn.DNNClassifier(
    feature_columns = feature_columns,
    hidden_units = [10, 20, 10],
    n_classes = 3,
    model_dir = "/tmp/iris_model",
    config=tf.contrib.learn.RunConfig(save_checkpoints_secs=1))

#define the training inputs, just a set of constants
def get_train_inputs():
    x = tf.constant(training_set.data)
    y = tf.constant(training_set.target)
    return x,y

#train the model to fit
classifier.fit(x=training_set.data,
                y=training_set.target,
                steps=4000,
                monitors=[validation_monitor])

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
