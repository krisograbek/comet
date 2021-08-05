from comet_ml import Experiment

import pprint
import pickle
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
from numpy.lib.function_base import average
from sklearn.datasets import load_wine, load_breast_cancer, fetch_covtype
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, 
    confusion_matrix, 
    recall_score, 
    precision_score, 
    f1_score,
    log_loss,
    roc_auc_score,
    make_scorer
)

from helpers import (
    print_shapes,
    print_class_distribution,
    get_scores,
    print_scores,
    log_learning_curve
)
from utils_numpy import compute_scores
from config import api_key

# Setting the API key (saved as environment variable)
experiment = Experiment(
    api_key=api_key,
    project_name='covtype')

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

sample_size = 100000
test_size, val_size = 10000, 10000
sample_size = 20000
test_size, val_size = 2000, 2000

_, X_train, _, y_train = train_test_split(data, target, stratify=target, test_size=sample_size, random_state=random_state)
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, stratify=y_train, test_size=test_size, random_state=random_state)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, stratify=y_train, test_size=test_size, random_state=random_state)

print_shapes(X_train, y_train, X_val, y_val, X_test, y_test)
print_class_distribution(y_train, y_val, y_test)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train[:, :10])
X_val_scaled = scaler.transform(X_val[:, :10])
X_test_scaled = scaler.transform(X_test[:, :10])
X_train_scaled = np.concatenate([X_train_scaled, X_train[:, 10:]], axis=1)
X_val_scaled = np.concatenate([X_val_scaled, X_val[:, 10:]], axis=1)
X_test_scaled = np.concatenate([X_test_scaled, X_test[:, 10:]], axis=1)


#####  METRICS  #####

# lr = LogisticRegression(C=100)

# lr.fit(X_train_scaled, y_train)
# y_pred = lr.predict(X_val_scaled)

# print("Actual classes: ", np.bincount(y_val))
# print("Predicted classes: ", np.bincount(y_pred))

# # average parameter for multiclass
# average = "weighted"
# acc, prec, recall, f1 = get_scores(y_val, y_pred, average=average)
# print(average, " scores: ")
# print_scores(acc, prec, recall, f1)

# weights_val = np.bincount(y_val) / len(y_val)
# cm = confusion_matrix(y_val, y_pred)
# compute_scores(cm, average=average, weights=weights_val)


#####  METRICS  #####


#####  HYPERPARAMETER TUNING  ###########


average = "macro"
# sgd = SGDClassifier()
rf = RandomForestClassifier()
lr = LogisticRegression()
knn = KNeighborsClassifier()

f1 = make_scorer(f1_score, average=average)
prec = make_scorer(precision_score, average=average)
recall = make_scorer(recall_score, average=average)

scoring = {
    "f1": f1,
    "precision": prec,
    "recall": recall
}

fit_time_rf = 0
fit_time_lr = 0
fit_time_knn = 0
n_iter = 25

train_mode = False

if train_mode is True:
    print( "Training Random Forest")
    param_random = {
        'n_estimators': [5, 10, 20, 50, 100],
        'max_depth': [3, 7, 15, 25, 50, None],
        'max_features': ['auto', 'sqrt'],
        'min_samples_leaf': [1, 2, 4],
        'min_samples_split': [2, 5, 10],
        'bootstrap': [True, False]
    }

    rnd = RandomizedSearchCV(rf,
                            param_distributions=param_random,
                            cv=5,
                            n_iter=n_iter,
                            random_state=random_state,
                            scoring=scoring,
                            refit="f1",
                            n_jobs=-1
    )

    rf_fit_begin = dt.datetime.now()
    rnd.fit(X_train_scaled, y_train)
    fit_time_rf = dt.datetime.now() - rf_fit_begin
    params_rf = rnd.best_params_
    best_rf = rnd.best_estimator_

    print("Best params Random Forest:", params_rf)

    with open("models/best_rf.pkl", "wb") as f:
        pickle.dump(best_rf, f)

    # mean_f1 = rnd.cv_results_["mean_test_f1"]
    # mean_precision = rnd.cv_results_["mean_test_precision"]
    # mean_recall = rnd.cv_results_["mean_test_recall"]

    # print("Mean F1: ", mean_f1)
    # print("Mean precision: ", mean_precision)
    # print("Mean recall: ", mean_recall)

else:
    print("Loading Pretrained Random Forest")
    with open("models/best_rf.pkl", "rb") as f:
        best_rf = pickle.load(f)

    params_rf = best_rf.get_params()

y_pred_rf = best_rf.predict(X_val_scaled)
acc_rf, precision_rf, recall_rf, f1_rf, cm_rf = get_scores(y_val, y_pred_rf, average=average)
print("Scores Random Forest")
print_scores(acc_rf, precision_rf, recall_rf, f1_rf)

print(best_rf.get_params())
print(type(best_rf))

params = {"model_type1": "rf",
          "scaler": "standard scaler",
          "stratify": True,
          "best_params_rf": params_rf,
          "rndz_params": param_random if train_mode else [],
          "training_time_rf": fit_time_rf,
          "n_iter_rf": 20
          }

metrics = {"acc_rf": acc_rf,
           "f1_rf": f1_rf,
           "recall_rf": recall_rf,
           "precision_rf": precision_rf
           }

experiment.log_metrics(metrics)
experiment.log_confusion_matrix(matrix=cm_rf, file_name="cm_rf.json")
experiment.log_parameters(params)


####  LOGISTIC REGRESSION  ####

print(" -- "*25)
print("####  LOGISTIC REGRESSION  ####")

if train_mode is True:

    grid_params_lr = {
        'C': [0.1, 1, 10, 100],
        'max_iter': np.linspace(100, 500, 3)
    }
    # init grid search for Logistic Regression
    grid_lr = GridSearchCV(lr, grid_params_lr, scoring=scoring, refit="f1", n_jobs=-1)
    before_lr = dt.datetime.now()
    grid_lr.fit(X_train_scaled, y_train)
    fit_time_lr = dt.datetime.now() - before_lr
    params_lr =  grid_lr.best_params_
    best_lr =  grid_lr.best_estimator_

    with open("models/best_lr.pkl", "wb") as f:
        pickle.dump(best_lr, f)

else:
    with open("models/best_lr.pkl", "rb") as f:
        best_lr = pickle.load(f)

    params_lr = best_lr.get_params()

y_pred_lr = best_lr.predict(X_val_scaled)
acc_lr, precision_lr, recall_lr, f1_lr, cm_lr = get_scores(y_val, y_pred_lr, average=average)
print("Scores Logistic Regression")
print_scores(acc_lr, precision_lr, recall_lr, f1_lr)


params = {"model_type2": "lr",
          "scaler": "standard scaler",
          "stratify": True,
          "best_params_lr": params_lr,
          "grid_params_lr": grid_params_lr if train_mode else [],
          "training_time_lr": fit_time_lr
          }

metrics = {"acc_lr": acc_lr,
           "f1_lr": f1_lr,
           "recall_lr": recall_lr,
           "precision_lr": precision_lr
           }

experiment.log_metrics(metrics)
experiment.log_confusion_matrix(matrix=cm_lr, file_name="cm_lr.json")
experiment.log_parameters(params)


####  K-Nearest Neighbors  ####

print(" -- "*25)
print("####  K-Nearest Neighbors  ####")

if train_mode is True:

    grid_params_knn = { 'n_neighbors':[1,3,5,7,9],
                        # 'leaf_size':[1,3,5],
                        # 'algorithm':['auto', 'kd_tree'],
                        'metric': ['euclidean', 'manhattan', 'minkowski']
    }
    # init grid search for Logistic Regression
    grid_knn = GridSearchCV(knn, grid_params_knn, scoring=scoring, refit="f1", n_jobs=-1)
    before_knn = dt.datetime.now()
    grid_knn.fit(X_train_scaled, y_train)
    fit_time_knn = dt.datetime.now() - before_knn
    params_knn =  grid_knn.best_params_
    best_knn =  grid_knn.best_estimator_

    with open("models/best_knn.pkl", "wb") as f:
        pickle.dump(best_knn, f)

else:
    with open("models/best_knn.pkl", "rb") as f:
        best_knn = pickle.load(f)

    params_knn = best_knn.get_params()

y_pred_knn = best_knn.predict(X_val_scaled)
acc_knn, precision_knn, recall_knn, f1_knn, cm_knn = get_scores(y_val, y_pred_knn, average=average)
print("Scores K-Nearest Neighbors")
print_scores(acc_knn, precision_knn, recall_knn, f1_knn)

params = {"model_type2": "knn",
          "scaler": "standard scaler",
          "stratify": True,
          "best_params_knn": params_knn,
          "grid_params_knn": grid_params_knn if train_mode else [],
          "training_time_knn": fit_time_knn
          }

metrics = {"acc_knn": acc_knn,
           "f1_knn": f1_knn,
           "recall_knn": recall_knn,
           "precision_knn": precision_knn
           }

experiment.log_metrics(metrics)
experiment.log_confusion_matrix(matrix=cm_knn, file_name="cm_knn.json")
experiment.log_parameters(params)


# pp = pprint.PrettyPrinter(indent=4)
# pp.pprint(grid_lr.cv_results_)

# scores = grid_lr.cv_results_['mean_test_score']
# params = grid_lr.cv_results_['params']

# for score, param in zip(scores, params):
#     print("param: {} - score: {} ".format(param, score))
    
# best_lr = grid_lr.best_estimator_
# # print("Feature importances:", np.sort(best_lr.feature_importances_))

# y_pred_lr = best_lr.predict(X_train_scaled)
# acc_lr, cm_lr, f1_lr, recall_lr, precision_lr = get_scores(y_train, y_pred_lr)
# print("Training scores")
# # print_scores(acc_lr, cm_lr, f1_lr, recall_lr, precision_lr)

# print(" -- "*25)

# y_pred_lr = best_lr.predict(X_val_scaled)
# acc_lr, cm_lr, f1_lr, recall_lr, precision_lr = get_scores(y_val, y_pred_lr)
# print("Validation scores")
# print_scores(acc_lr, cm_lr, f1_lr, recall_lr, precision_lr)


# log_learning_curve(experiment, "loss", "loss_train", n_epochs, loss_train)

# params = {"model_type2": "lg",
#           "scaler": "standard scaler",
#           "stratify": True
#           }


# print("Training SGD Classifier")
# sgd.fit(X_train_scaled, y_train)
# y_pred_sgd = sgd.predict(X_test_scaled)




#####  HYPERPARAMETER TUNING  ###########



# acc_rf, cm_rf, f1_rf, recall_rf, precision_rf = get_scores(y_test, y_pred_rf)
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
experiment.end()
