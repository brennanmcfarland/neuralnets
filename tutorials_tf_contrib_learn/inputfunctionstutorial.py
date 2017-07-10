from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools

import pandas as pd
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)

#define column names
COLUMNS = ["crim", "zn", "indus", "nox", "rm", "age", "dis", "tax", "ptratio",
    "medv"]
FEATURES = ["crim", "zn", "indus", "nox", "rm", "age", "dis", "tax", "ptratio"]
LABEL = "medv"

#read input data from CSVs
training_set = pd.read_csv("boston_train.csv", skipinitialspace=True,
    skiprows=1, names=COLUMNS)
test_set = pd.read_csv("boston_test.csv", skipinitialspace=True,
    skiprows=1, names=COLUMNS)
prediction_set = pd.read_csv("boston_predict.csv", skipinitialspace=True,
    skiprows=1, names=COLUMNS)

#create a list of FeatureColumns for the input data, specifying features to train w/
feature_cols = [tf.contrib.layers.real_valued_column(k) for k in FEATURES]

#create the regression model to be used
#hidden_units: # nodes for each hidden layer
#feature_columns: list of feature columns being used (defined above)
regressor = tf.contrib.learn.DNNRegressor(feature_columns=feature_cols,
                                            hidden_units=[10, 10],
                                            model_dir="/tmp/boston_model")

#define the input function
#converts pandas Dataframes to Tensors of feature columns and labels
def input_fn(data_set):
    feature_cols = {k: tf.constant(data_set[k].values)
                    for k in FEATURES}
    labels = tf.constant(data_set[LABEL].values)
    return feature_cols, labels

#train the network
regressor.fit(input_fn=lambda: input_fn(training_set), steps=5000)

#evaluate
ev = regressor.evaluate(input_fn=lambda: input_fn(test_set), steps=1)
loss_score = ev["loss"]
print("loss: {0:f}".format(loss_score))

#predict from the prediction set
#predict() returns an iterator, so it must be converted to a list for output
y = regressor.predict(input_fn=lambda: input_fn(prediction_set))
predictions = list(itertools.islice(y, 6))
print("predictions: {}".format(str(predictions)))
