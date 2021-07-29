from comet_ml import Experiment

import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.construct import rand
from sklearn.datasets import load_wine, load_breast_cancer, fetch_covtype
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import (accuracy_score, 
    confusion_matrix, 
    recall_score, 
    precision_score, 
    f1_score
)

from config import api_key

# Setting the API key (saved as environment variable)
# experiment = Experiment(
#     api_key=api_key,
#     project_name='covtype')

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


random_state = 31

covtype = fetch_covtype()
data, target = covtype["data"], covtype["target"]

# print("Feature names: ", covtype.feature_names)
# print("Class names: ", covtype.target_names)
# print("Shape of data: ", data.shape)
# print("Shape of target: ", target.shape)
# print("Unique classes: ", np.unique(target))
target = target -1
unique = list(np.unique(target))
# print("Samples per class: {}".format(
#     {str(k):n for k, n in zip(np.array(unique), np.bincount(target))} 
# ))

print("Target distribution: ", np.bincount(target) / len(target))

n = 25000        # number of samples

def shuffle_arrays(arr1, arr2):
    np.random.seed(32)
    p = np.random.permutation(len(arr2))
    # print("Shuffled indices", p[:n])
    X = arr1[p]
    y = arr2[p]
    return X[:n], y[:n]

data, target = shuffle_arrays(data, target)
# print("Shuffeled and filtered Target distribution: ", np.bincount(target) / len(target))

X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=2500, stratify=target, random_state=31)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=2500, stratify=y_train, random_state=31)
# print("y train class distribution: ", np.bincount(y_train) / len(y_train))
# print("y val class distribution: ", np.bincount(y_val) / len(y_val))

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

rf = RandomForestClassifier(n_estimators=10, random_state=random_state)
sgd = SGDClassifier(random_state=random_state)

lr = LogisticRegression(multi_class="multinomial")

# n_epochs = 200
# acc_train, acc_val = [], []

# for epoch in range(n_epochs):
#     lr.fit(X_train_scaled, y_train)
#     y_train_pred = lr.predict(X_train_scaled)
#     y_val_pred = lr.predict(X_val_scaled)
#     acc_train.append(accuracy_score(y_train, y_train_pred))
#     acc_val.append(accuracy_score(y_val, y_val_pred))

# plt.plot(acc_train)
# plt.plot(acc_val)
# plt.show()

# print("Training random forest")
# rf.fit(X_train_scaled, y_train)
# y_pred_rf = rf.predict(X_test_scaled)
# y_probas_rf = rf.predict(X_test_scaled)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))

def my_decision_func(samples, coefs, intercept):
    X = np.dot(samples, np.transpose(coefs)) + intercept
    return X

samples = 20
X_samples = X_val_scaled[:samples]
y_samples = y_val[:samples]

print("Training Logistic Regression")
lr.fit(X_train_scaled, y_train)

testres = my_decision_func(X_samples, lr.coef_, lr.intercept_)
# print(testres)


y_pred_lr = lr.predict(X_val_scaled)
dec_func_lr = lr.decision_function(X_samples)
print("My values")
# print("Decision Function: ", dec_func_lr)
y_probas = lr.predict_proba(X_samples)
y_probas_log = lr.predict_log_proba(X_samples)

my_probas = softmax(testres)
# print("Decision to softmax: ", my_probas)
my_pred_classes = np.argmax(my_probas, axis=1)
print("Classes from argmax(probas): ", my_pred_classes)
print("Pred classes lr.predic(val): ", y_pred_lr[:samples])
print("True classes: ", y_samples)

print(" -- "*20)
pred_classes = np.argmax(y_probas, axis=1)
print("Classes from argmax(probas): ", pred_classes)
print("Pred classes lr.predic(val): ", y_pred_lr[:samples])
print("True classes: ", y_samples)
# print()
assert False

good_pred = np.array(pred_classes == y_samples).astype(int)
print(good_pred)
acc = np.sum(good_pred) / samples
print("Acc: ", acc)

# print("Coef_ and intercept", lr.coef_, lr.intercept_)
print("Coef_ and intercept", lr.coef_.shape, lr.intercept_.shape)
print("X val samples shape", X_samples.shape)
print(" -- "*20)
print("Dec function shape", dec_func_lr.shape)


# print("Predict Proba: ", y_probas)
# print("Sum Predict Proba: ", np.sum(y_probas))
# print("Predict log Proba: ", y_probas_log)
# print("Predict proba np.log: ", np.log(lr.predict_proba(X_samples)))

# print("Training SGD Classifier")
# sgd.fit(X_train_scaled, y_train)
# y_pred_sgd = sgd.predict(X_test_scaled)

# acc_rf, cm_rf, f1_rf, recall_rf, precision_rf = get_scores(y_test, y_pred_rf)
# acc_lr, cm_lr, f1_lr, recall_lr, precision_lr = get_scores(y_test, y_pred_lr)
# acc_sgd, cm_sgd, f1_sgd, recall_sgd, precision_sgd = get_scores(y_test, y_pred_sgd)

# params = {"model_type1": "rf",
#           "scaler": "standard scaler",
#           "stratify": True
#           }

# metrics = {"acc_rf": acc_rf,
#            "f1_rf": f1_rf,
#            "recall_rf": recall_rf,
#            "precision_rf": precision_rf
#            }

# experiment.log_metrics(metrics)
# experiment.log_confusion_matrix(matrix=cm_rf, file_name="cm_rf.json")
# experiment.log_parameters(params)


# params = {"model_type2": "lg",
#           "scaler": "standard scaler",
#           "stratify": True
#           }

# metrics = {"acc_lr": acc_lr,
#            "f1_lr": f1_lr,
#            "recall_lr": recall_lr,
#            "precision_lr": precision_lr
#            }

# experiment.log_metrics(metrics)
# experiment.log_confusion_matrix(matrix=cm_lr, file_name="cm_lr.json")
# experiment.log_parameters(params)

# params = {"model_type3": "sgd",
#           "scaler": "standard scaler",
#           "stratify": True
#           }

# metrics = {"acc_sgd": acc_sgd,
#            "f1_sgd": f1_sgd,
#            "recall_sgd": recall_sgd,
#            "precision_sgd": precision_sgd
#            }

# experiment.log_metrics(metrics)
# experiment.log_confusion_matrix(matrix=cm_sgd, file_name="cm_sgd.json")
# experiment.log_parameters(params)
# # test where I can find the curve
# experiment.log_curve("curves/my_curve", x=[1, 2, 3, 4, 5],
#   y=[10, 20, 30, 40, 50])
# experiment.end()
