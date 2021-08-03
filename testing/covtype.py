from comet_ml import Experiment

import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
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
    roc_auc_score
)

from helpers import (
    print_shapes,
    print_class_distribution,
    get_scores
)
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

# print(X_train[:3])

X_train_scaled = scaler.fit_transform(X_train[:, :10])
X_val_scaled = scaler.transform(X_val[:, :10])
X_test_scaled = scaler.transform(X_test[:, :10])
X_train_scaled = np.concatenate([X_train_scaled, X_train[:, 10:]], axis=1)
X_val_scaled = np.concatenate([X_val_scaled, X_val[:, 10:]], axis=1)
X_test_scaled = np.concatenate([X_test_scaled, X_test[:, 10:]], axis=1)

lr = LogisticRegression(warm_start=True, max_iter=1)


# grid_params_lr = {
#     # 'penalty': ['l1', 'l2'], 
#     # 'C': [0.001,0.01,0.1,1,10,100,1000]
#     'C': [1, 5, 10, 20, 50]
# }

# # init grid search for Logistic Regression
# grid_lr = GridSearchCV(lr, grid_params_lr)
# grid_lr.fit(X_train_scaled, y_train)

# print("Best LogReg params:", grid_lr.best_params_)
# best_lr = grid_lr.best_estimator_
# # print("Feature importances:", np.sort(best_lr.feature_importances_))

# y_pred_lr = best_lr.predict(X_train_scaled)
# acc_lr, cm_lr, f1_lr, recall_lr, precision_lr = get_scores(y_train, y_pred_lr)
# print("Training scores")
# print_scores(acc_lr, cm_lr, f1_lr, recall_lr, precision_lr)

# print(" -- "*25)

# y_pred_lr = best_lr.predict(X_val_scaled)
# acc_lr, cm_lr, f1_lr, recall_lr, precision_lr = get_scores(y_val, y_pred_lr)
# print("Validation scores")
# print_scores(acc_lr, cm_lr, f1_lr, recall_lr, precision_lr)

# rf = RandomForestClassifier(n_estimators=10, random_state=random_state)
# sgd = SGDClassifier(random_state=random_state)

n_epochs = 100
f1_train, f1_val = [], []
acc_train, acc_val = [], []
roc_train, roc_val = [], []
loss_train, loss_val = [], []

for epoch in range(n_epochs):
    # before = dt.datetime.now()
    lr.fit(X_train_scaled, y_train)
    y_train_pred = lr.predict(X_train_scaled)
    y_val_pred = lr.predict(X_val_scaled)
    
    acc_tr, _, f1_tr, recall_tr, precision_tr =  get_scores(y_train, y_train_pred)
    acc_v, _, f1_v, recall_v, precision_v =  get_scores(y_val, y_val_pred)


    y_probas_train = lr.predict_proba(X_train_scaled)
    y_probas_val = lr.predict_proba(X_val_scaled)
    loss_train.append(log_loss(y_train, y_probas_train))
    loss_val.append(log_loss(y_val, y_probas_val))
    # print(dt.datetime.now() - before)
    
    f1_train.append(f1_tr)
    f1_val.append(f1_v)
    acc_train.append(acc_tr)
    acc_val.append(acc_v)
    roc_train.append(roc_auc_score(y_train, y_probas_train, multi_class="ovo"))
    roc_val.append(roc_auc_score(y_val, y_probas_val, multi_class="ovo"))
    # f1_train.append(f1_tr)
    # f1_val.append(f1_val)

# experiment.log_asset_data(loss_train, name="train loss", epoch=range(n_epochs))
experiment.log_curve("curves/train_loss", x=range(n_epochs), y=loss_train)
experiment.log_curve("curves/val_loss", x=range(n_epochs), y=loss_val)
experiment.log_curve("curves/train_f1", x=range(n_epochs), y=f1_train)
experiment.log_curve("curves/val_f1", x=range(n_epochs), y=f1_val)
experiment.log_curve("curves/train_acc", x=range(n_epochs), y=acc_train)
experiment.log_curve("curves/val_acc", x=range(n_epochs), y=acc_val)
experiment.log_curve("curves/train_roc", x=range(n_epochs), y=roc_train)
experiment.log_curve("curves/val_roc", x=range(n_epochs), y=roc_val)
experiment.end()

# params = {"model_type2": "lg",
#           "scaler": "standard scaler",
#           "stratify": True
#           }

# plt.plot(loss_train, label = "train loss")
# plt.plot(loss_val, label = "val loss")
# plt.legend(loc="upper right")
# plt.show()

# plt.plot(f1_train, label = "train f1")
# plt.plot(f1_val, label = "val f1")
# plt.legend(loc="lower right")
# plt.show()

# print("Training random forest")
# rf.fit(X_train_scaled, y_train)
# y_pred_rf = rf.predict(X_test_scaled)
# y_probas_rf = rf.predict(X_test_scaled)



# print("Predict Proba: ", y_probas)
# print("Sum Predict Proba: ", np.sum(y_probas))
# print("Predict log Proba: ", y_probas_log)
# print("Predict proba np.log: ", np.log(lr.predict_proba(X_samples)))

# print("Training SGD Classifier")
# sgd.fit(X_train_scaled, y_train)
# y_pred_sgd = sgd.predict(X_test_scaled)

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
