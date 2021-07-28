from comet_ml import Experiment

import numpy as np
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
experiment = Experiment(
    api_key=api_key,
    project_name='covtype')

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

n = 20000        # number of samples

def shuffle_arrays(arr1, arr2):
    np.random.seed(32)
    p = np.random.permutation(len(arr2))
    # print("Shuffled indices", p[:n])
    X = arr1[p]
    y = arr2[p]
    return X[:n], y[:n]

data, target = shuffle_arrays(data, target)
# print("Shuffeled and filtered Target distribution: ", np.bincount(target) / len(target))

X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, stratify=target, random_state=31)
# print("y train class distribution: ", np.bincount(y_train) / len(y_train))

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

rf = RandomForestClassifier(n_estimators=10, random_state=random_state)
lr = LogisticRegression(C=1, max_iter=100)
sgd = SGDClassifier(random_state=31)

print("Training random forest")
rf.fit(X_train_scaled, y_train)
y_pred_rf = rf.predict(X_test_scaled)
print("Training Logistic Regression")
lr.fit(X_train_scaled, y_train)
y_pred_lr = lr.predict(X_test_scaled)
print("Training SGD Classifier")
sgd.fit(X_train_scaled, y_train)
y_pred_sgd = sgd.predict(X_test_scaled)

acc_rf, cm_rf, f1_rf, recall_rf, precision_rf = get_scores(y_test, y_pred_rf)
acc_lr, cm_lr, f1_lr, recall_lr, precision_lr = get_scores(y_test, y_pred_lr)
acc_sgd, cm_sgd, f1_sgd, recall_sgd, precision_sgd = get_scores(y_test, y_pred_sgd)

params = {"model_type": "rf",
          "scaler": "standard scaler",
          "stratify": True
          }

metrics = {"acc_rf": acc_rf,
           "f1_rf": f1_rf,
           "recall_rf": recall_rf,
           "precision_rf": precision_rf
           }

experiment.log_metrics(metrics)
experiment.log_confusion_matrix(matrix=cm_rf, file_name="cm_rf.json")
experiment.log_parameters(params)


params = {"model_type": "lg",
          "scaler": "standard scaler",
          "stratify": True
          }

metrics = {"acc_lr": acc_lr,
           "f1_lr": f1_lr,
           "recall_lr": recall_lr,
           "precision_lr": precision_lr
           }

experiment.log_metrics(metrics)
experiment.log_confusion_matrix(matrix=cm_lr, file_name="cm_lr.json")
experiment.log_parameters(params)

params = {"model_type": "sgd",
          "scaler": "standard scaler",
          "stratify": True
          }

metrics = {"acc_sgd": acc_sgd,
           "f1_sgd": f1_sgd,
           "recall_sgd": recall_sgd,
           "precision_sgd": precision_sgd
           }

experiment.log_metrics(metrics)
experiment.log_confusion_matrix(matrix=cm_sgd, file_name="cm_sgd.json")
experiment.log_parameters(params)
