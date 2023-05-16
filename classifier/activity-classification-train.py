# -*- coding: utf-8 -*-
"""
This is the script used to train an activity recognition
classifier on audio decibel data.

"""

import os
import sys
import numpy as np
import sklearn
from sklearn.tree import export_graphviz
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
from sklearn.model_selection import KFold

from features import extract_features
from util import slidingWindow, normalize, reset_vars
import pickle

import labels


# %%---------------------------------------------------------------------------
#
#                 Load Data From Disk
#
# -----------------------------------------------------------------------------

print("Loading data...")
sys.stdout.flush()
data_file = 'data/all_labeled_data.csv'
data = np.genfromtxt(data_file, delimiter=',')
print("Loaded {} raw labelled activity data samples.".format(len(data)))
sys.stdout.flush()

# %%---------------------------------------------------------------------------
#
#                    Pre-processing
#
# -----------------------------------------------------------------------------

print("Normalizing decibel data...")
sys.stdout.flush()
reset_vars()
normalized = np.asarray([normalize(data[i,2]) for i in range(len(data))])
normalized_data_with_timestamps = np.append(data[:,0:2],normalized.reshape(-1,1),axis=1)
data = np.append(normalized_data_with_timestamps, data[:,-1:], axis=1)

data = np.nan_to_num(data)


# %%---------------------------------------------------------------------------
#
#                Extract Features & Labels
#
# -----------------------------------------------------------------------------

window_size = 300
step_size = 150

print("Extracting features and labels for window size {} and step size {}...".format(window_size, step_size))
sys.stdout.flush()

X = []
Y = []
feature_names = []
for i,window_with_timestamp_and_label in slidingWindow(data, window_size, step_size):
    window = window_with_timestamp_and_label[:,1:-1]
    feature_names, x = extract_features(window)
    print(x)  # Print the entire feature vector

    X.append(x)
    Y.append(window_with_timestamp_and_label[10, -1])
   
X = np.asarray(X)
Y = np.asarray(Y)
n_features = len(X)
   
print("Finished feature extraction over {} windows".format(len(X)))
print("Unique labels found: {}".format(set(Y)))
print("\n")
sys.stdout.flush()

# %%---------------------------------------------------------------------------
#
#		                Train & Evaluate Classifier
#
# -----------------------------------------------------------------------------


# TODO: split data into train and test datasets using 10-fold cross validation
# Set the number of folds
n_folds = 10

# Split the data into train and test sets using k-fold cross-validation
kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
"""
TODO: iterating over each fold, fit a decision tree classifier on the training set.
Then predict the class labels for the test set and compute the confusion matrix
using predicted labels and ground truth values. Print the accuracy, precision and recall
for each fold.
"""
fold = 1
total_acc = 0
total_prec = 0
total_rec = 0

# train the SAME tree over and over (.fit)
tree = DecisionTreeClassifier(criterion="entropy", max_depth=3)


for train_index, test_index in kf.split(X):
    print("Fold:", fold)

    # Split the data into training and testing sets for this fold
    print(X)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = Y[train_index], Y[test_index]

    # Fit a decision tree classifier to the training data
    tree.fit(X_train, y_train)

    # Use the trained classifier to predict labels for the test data
    y_pred = tree.predict(X_test)

    # Compute the confusion matrix for this fold
    conf_matrix = confusion_matrix(y_test, y_pred)

    # Compute accuracy, precision, and recall for this fold
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted')
    rec = recall_score(y_test, y_pred, average='weighted')

    total_acc += acc
    total_prec += prec
    total_rec += rec

    # Print the results for this fold
    print("Confusion matrix:\n", conf_matrix)
    print("Accuracy:", acc)
    print("Precision:", prec)
    print("Recall:", rec)
    print()
    fold += 1

# TODO: calculate and print the average accuracy, precision and recall values over all 10 folds

print (f"avg acc: {total_acc / n_folds}")
print (f"avg prec: {total_prec / n_folds}")
print (f"avg rec: {total_rec / n_folds}")

# TODO: train the decision tree classifier on entire dataset
# create a decision tree classifier with the desired parameters
#clf = DecisionTreeClassifier(criterion='entropy', max_depth=5)

# fit the classifier to the entire dataset
#clf.fit(X, Y)

# TODO: Save the decision tree visualization to disk - replace 'tree' with your decision tree and run the below line
export_graphviz(tree, out_file='tree.dot', feature_names = feature_names)

# TODO: Save the classifier to disk - replace 'tree' with your decision tree and run the below line
print("saving classifier model...")
with open('classifier.pickle', 'wb') as f:
    pickle.dump(tree, f)