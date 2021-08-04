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
# experiment = Experiment(
#     api_key=api_key,
#     project_name='covtype')

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


rf = RandomForestClassifier()
sgd = SGDClassifier()
lr = LogisticRegression()

average = "macro"

if False:

    param_random = {
        'n_estimators': [5, 10, 20, 50, 100, 200],
        'max_depth': [3, 7, 15, 25, 50, None],
        'max_features': ['auto', 'sqrt'],
        'min_samples_leaf': [1, 2, 4],
        'min_samples_split': [2, 5, 10],
        'bootstrap': [True, False]
    }


    f1 = make_scorer(f1_score, average=average)
    prec = make_scorer(precision_score, average=average)
    recall = make_scorer(recall_score, average=average)

    scoring = {
        "f1": f1,
        "precision": prec,
        "recall": recall
    }

    rnd = RandomizedSearchCV(rf,
                        param_distributions=param_random,
                        cv=5,
                        n_iter=20,
                        random_state=random_state,
                        scoring=scoring,
                        refit="f1",
                        n_jobs=-1)


    rnd.fit(X_train_scaled, y_train)

    best_params_rf = rnd.best_params_
    best_rf = rnd.best_estimator_

    print("Best params Random Forest:", best_params_rf)



    with open("models/best_rf.pkl", "wb") as f:
        pickle.dump(best_rf, f)

    pp = pprint.PrettyPrinter(indent=4)
    mean_f1 = rnd.cv_results_["mean_test_f1"]
    mean_precision = rnd.cv_results_["mean_test_precision"]
    mean_recall = rnd.cv_results_["mean_test_recall"]

    print("Mean F1: ", mean_f1)
    print("Mean precision: ", mean_precision)
    print("Mean recall: ", mean_recall)

else:
    with open("models/best_rf.pkl", "rb") as f:
        best_rf = pickle.load(f)

y_pred_rf = best_rf.predict(X_val_scaled)

acc_rf, prec_rf, recall_rf, f1_rf = get_scores(y_val, y_pred_rf, average=average)
print_scores(acc_rf, prec_rf, recall_rf, f1_rf)

print(best_rf.get_params())



# grid_params_lr = {
#     # 'C': [0.001,0.01,0.1,1,10,100,1000]
#     'C': [0.01, 0.1, 1, 10, 100]
#     # 'max_iter': np.linspace(100, 500, 3)
# }
# # init grid search for Logistic Regression
# grid_lr = GridSearchCV(lr, grid_params_lr, scoring=scoring, refit="f1", n_jobs=-1)
# before = dt.datetime.now()
# grid_lr.fit(X_train_scaled, y_train)
# print("Training time: ", dt.datetime.now() - before)
# print("Best LogReg params:", grid_lr.best_params_)

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
# # test where I can find the curve
# experiment.log_curve("curves/my_curve", x=[1, 2, 3, 4, 5],
#   y=[10, 20, 30, 40, 50])
# experiment.end()
