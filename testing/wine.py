import numpy as np
from sklearn.datasets import load_wine, load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix


random_state = 31

# wine = load_breast_cancer()
wine = load_wine()
data, target = wine["data"], wine["target"]

print("Feature names: ", wine.feature_names)
print("Shape of data: ", data.shape)
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
# rf.fit(X_train_scaled, y_train)
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

print(" -- " *25)
print("Randomized search")

print("Accuracy", acc2)
print("Confusion Matrix", cm2)

best_rf = rnd.best_estimator_
if acc > acc2:
    print("Grid was better")
    best_rf = grid.best_estimator_
else:
    print("Random was better")

rf2 = RandomForestClassifier(**rnd.best_params_)
# best_rf = rnd.best_estimator_ if (acc2 > acc) else grid.best_estimator_

# best_rf.fit(X_train_scaled, y_train)

y_pred3 = best_rf.predict(X_test_scaled)
acc3 = accuracy_score(y_test, y_pred3)
cm3 = confusion_matrix(y_test, y_pred3)

print(" -- " *35)
print("RandomForest with best params from randomized")

print("Accuracy", acc3)
print("Confusion Matrix", cm3)

for name, importance in zip(wine.feature_names, best_rf.feature_importances_):
    print(name, "=", importance)
