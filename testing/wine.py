from comet_ml import Experiment

import numpy as np
from sklearn.datasets import load_wine, load_breast_cancer, fetch_covtype
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
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
    # or
    # api_key=os.environ.get("COMET_API_KEY"),
    project_name='comet-examples')

random_state = 31

wine = load_breast_cancer()
# wine = load_wine()
# wine = fetch_covtype()
data, target = wine["data"], wine["target"]

print("Feature names: ", wine.feature_names)
print("Class names: ", wine.target_names)
print("Shape of data: ", data.shape)
print("Shape of target: ", target.shape)
print("Samples per class: {}".format(
    {k:n for k, n in zip(wine.target_names, np.bincount(target))} 
))

X_train, X_test, y_train, y_test = train_test_split(
    data, target, 
    stratify=target,
    random_state=random_state,
    test_size=0.2
)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

rf = RandomForestClassifier(n_estimators=10, random_state=random_state)

print(" -- "*25)
print(rf.get_params())
rf.fit(X_train_scaled, y_train)
param_grid = {
    'n_estimators': [5, 10, 20, 50, 100],
    'max_depth': [3, 5, 7, 9, None]
}

grid = GridSearchCV(rf,
                    param_grid=param_grid,
                    cv=10,
                    # random_state=random_state,
                    n_jobs=-1)

grid.fit(X_train_scaled, y_train)

print(grid.best_params_)

y_pred = grid.predict(X_test_scaled)

acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
precision = precision_score(y_test, y_pred, average='weighted')


print("Grid search")
print("Accuracy", acc)
print("Confusion Matrix", cm)

param_random = {
    'n_estimators': [5, 10, 20, 50, 100, 200],
    'max_depth': [3, 7, 15, 25, 50, None],
    'max_features': ['auto', 'sqrt'],
    'min_samples_leaf': [1, 2, 4],
    'min_samples_split': [2, 5, 10],
    'bootstrap': [True, False]
}

rnd = RandomizedSearchCV(rf,
                    param_distributions=param_random,
                    cv=10,
                    n_iter=25,
                    random_state=random_state,
                    n_jobs=-1)

rnd.fit(X_train_scaled, y_train)

print("Random best params: ", rnd.best_params_)

y_pred2 = rnd.predict(X_test_scaled)
acc2 = accuracy_score(y_test, y_pred2)
cm2 = confusion_matrix(y_test, y_pred2)
f1_2 = f1_score(y_test, y_pred2, average='weighted')
recall2 = recall_score(y_test, y_pred2, average='weighted')
precision2 = precision_score(y_test, y_pred2, average='weighted')


print(" -- " *25)
print("Randomized search")

print("Accuracy", acc2)
print("Confusion Matrix", cm2)

# best_rf = rnd.best_estimator_
# if acc > acc2:
#     print("Grid was better")
#     best_rf = grid.best_estimator_
# else:
#     print("Random was better")

# rf2 = RandomForestClassifier(**rnd.best_params_)
# # best_rf = rnd.best_estimator_ if (acc2 > acc) else grid.best_estimator_

# # best_rf.fit(X_train_scaled, y_train)

# y_pred3 = best_rf.predict(X_test_scaled)
# acc3 = accuracy_score(y_test, y_pred3)
# cm3 = confusion_matrix(y_test, y_pred3)
# f1 = f1_score(y_test, y_pred3)
# recall = recall_score(y_test, y_pred3)
# precision = precision_score(y_test, y_pred3)

params = {"random_state": random_state,
          "model_type": "grid",
          "scaler": "standard scaler",
          "param_grid": str(param_grid),
          "best_params_grid": grid.best_params_,
          "stratify": True
          }

metrics = {"f1": f1,
           "recall": recall,
           "precision": precision
           }

# experiment.log_metric(name="acc", value = acc2)
experiment.log_metrics(metrics)
experiment.log_confusion_matrix(matrix=cm)
experiment.log_parameters(params)

params = {"random_state": random_state,
          "model_type": "rnd",
          "scaler": "standard scaler",
          "param_random": str(param_random),
          "best_params_random": rnd.best_params_,
          "stratify": True
          }

metrics = {"f12": f1_2,
           "recall2": recall2,
           "precision2": precision2
           }

# experiment.log_metric(name="acc", value = acc2)
experiment.log_metrics(metrics)
experiment.log_confusion_matrix(matrix=cm2, file_name="conf_matrix_rnd.json")
experiment.log_parameters(params)

experiment.end()

# print(" -- " *35)
# print("RandomForest with best params from randomized")

# print("Accuracy", acc3)
# print("Confusion Matrix", cm3)

for name, importance in zip(wine.feature_names, rnd.best_estimator_.feature_importances_):
    print(name, "=", importance)
