import numpy as np


def print_shapes(X_train, y_train, X_val, y_val, X_test, y_test):
    print("Train shapes - data: {}, target: {}".format(X_train.shape, y_train.shape))
    print("Val shapes - data: {}, target: {}".format(X_val.shape, y_val.shape))
    print("Test shapes - data: {}, target: {}".format(X_test.shape, y_test.shape))

def print_class_distribution(y_train, y_val, y_test):
    print("Classes distribution")
    print("Train set: ", np.bincount(y_train) / len(y_train))
    print("Val set: ", np.bincount(y_val) / len(y_val))
    print("Test set: ", np.bincount(y_test) / len(y_test))

