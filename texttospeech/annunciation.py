from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np
import tensorflow as tf

#TODO: right now it just looks word by word to determine if it should be a long
#or short a, need to actually have it check for each instance of a, including
#multiple in the same word or none in a given word

class PronunciationDetermination:
    """responsible for determining the pronunciation of text"""

    def load_data(self, input_path, training_path):
        """load the text to be evaluated"""
        #TODO: finish implementing this

        #parse the text into words
        input_file = open(input_path, "r")
        wordstrings = input_file.read().split()
        words = map(list, wordstrings)

        #feed the words' unicode values into a tensor
        words_tensor = tf.constant(wordstrings)
        #TODO: may need to convert words to their unicode values first, not sure

        #convert that tensor to a dataset
        training_set = tf.contrib.learn.from_tensor_slices(training_words)

        #parse the training file into a list of ints (1=long, 0=short)
        training_file = open(training_path, "r")
        vowelstrings = training_file.read().split()
        vowels = map(int, vowelstrings) #may need to change to floats to work
        vowels_tensor = tf.constant(vowels)

    def train(self):
        #1 dimension for the 1 feature, the unicode values of the chars
        feature_columns = [tf.contrib.layers.real_valued_column("", dimension=1)]

        #create the model
        classifier = tf.contrib.learn.DNNClassifier(
            feature_columns = feature_columns,
            hidden_units = [10, 20, 10],
            n_classes = 2,
            model_dir = "/tmp/annunciation_model",
            config=tf.contrib.learn.RunConfig(save_checkpoints_secs=10))
        #train the model to fit
        classifier.fit(x=words,
                        y=vowels,
                        steps=4000)
        #evaluate accuracy TODO: do this on test data
        accuracy_score = classifier.evaluate(input_fn=load_data, steps=1)["accuracy"]
        print("\nTest Accuracy: {0:f}\n".format(accuracy_score))

    def __init__(self, input_path, training_path):
        """constructor

        input_path: the path string to the input text file
        """

        #set logging verbosity
        tf.logging.set_verbosity(tf.logging.INFO)
        words, vowels = self.load_data(input_path, training_path)
        #self.train()
    #TODO: refactor the below into this class

def main():
    annunciator = PronunciationDetermination("training_input.txt", "training_output.txt")

if __name__ == "__main__":
    main()

#define validation metrics to monitor
#validation_metrics = {
#    "accuracy":
#        tf.contrib.learn.MetricSpec(
#            metric_fn=tf.contrib.metrics.streaming_accuracy,
#            prediction_key=tf.contrib.learn.PredictionKey.CLASSES),
#    "precision":
#        tf.contrib.learn.MetricSpec(
#            metric_fn=tf.contrib.metrics.streaming_precision,
#            prediction_key=tf.contrib.learn.PredictionKey.CLASSES),
#    "recall":
#        tf.contrib.learn.MetricSpec(
#            metric_fn=tf.contrib.metrics.streaming_recall,
#            prediction_key=tf.contrib.learn.PredictionKey.CLASSES)
#}
#
#create a ValidationMonitor to evaluate against the test data every 100 steps
#early stopping: stop if the loss has not improved for 200 steps
#validation_monitor = tf.contrib.learn.monitors.ValidationMonitor(
#    test_set.data,
#    test_set.target,
#    every_n_steps=100,
#    metrics=validation_metrics,
#    early_stopping_metric="loss",
#    early_stopping_metric_minimize=True,
#    early_stopping_rounds=200)
#
#classify two new samples
#def new_samples():
#    return np.array([[6.4,3.2,4.5,1.5],[5.8,3.1,5.0,1.7]], dtype=np.float32)
#
#predictions = list(classifier.predict_classes(input_fn=new_samples))
#
#print("New samples, predictions: {}\n".format(predictions))
