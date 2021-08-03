import numpy as np

from sklearn.metrics import (accuracy_score, 
    confusion_matrix, 
    recall_score, 
    precision_score, 
    f1_score,
    log_loss
)


def print_shapes(X_train, y_train, X_val, y_val, X_test, y_test):
    print("Train shapes - data: {}, target: {}".format(X_train.shape, y_train.shape))
    print("Val shapes - data: {}, target: {}".format(X_val.shape, y_val.shape))
    print("Test shapes - data: {}, target: {}".format(X_test.shape, y_test.shape))

def print_class_distribution(y_train, y_val, y_test):
    print("Classes distribution")
    print("Train set: ", np.bincount(y_train) / len(y_train))
    print("Val set: ", np.bincount(y_val) / len(y_val))
    print("Test set: ", np.bincount(y_test) / len(y_test))


def get_scores(y_test, y_pred):
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    precision = precision_score(y_test, y_pred, average='weighted')
    return acc, cm, f1, recall, precision


def print_scores(acc, cm, f1, recall, precision):
    print("Accuracy: ", acc)
    print("Confusion matrix: ", cm)
    print("F1 score:", f1)
    print("precision score:", precision)
    print("recall score:", recall)

