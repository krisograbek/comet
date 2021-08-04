import numpy as np
import matplotlib.pyplot as plt


from sklearn.metrics import (accuracy_score, 
    confusion_matrix, 
    recall_score, 
    precision_score, 
    f1_score,
    log_loss,
    roc_auc_score
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


def get_scores(y_test, y_pred, average="weighted"):
    print("Returning Validation scores with average - ", average)
    acc = accuracy_score(y_test, y_pred)
    # cm = confusion_matrix(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average=average)
    recall = recall_score(y_test, y_pred, average=average)
    f1 = f1_score(y_test, y_pred, average=average)
    return acc, precision, recall, f1


def print_scores(acc, precision, recall, f1):
    print("Accuracy: ", acc)
    print("precision score:", precision)
    print("recall score:", recall)
    print("F1 score:", f1)


def get_learning_curves(estimator, X_train, X_val, y_train, y_val, n_epochs = 10):
    f1_train, f1_val = [], []
    acc_train, acc_val = [], []
    roc_train, roc_val = [], []
    loss_train, loss_val = [], []

    for epoch in range(n_epochs):
        # before = dt.datetime.now()
        estimator.fit(X_train, y_train)
        y_train_pred = estimator.predict(X_train)
        y_val_pred = estimator.predict(X_val)
        
        acc_tr, _, f1_tr, recall_tr, precision_tr =  get_scores(y_train, y_train_pred)
        acc_v, _, f1_v, recall_v, precision_v =  get_scores(y_val, y_val_pred)


        y_probas_train = estimator.predict_proba(X_train)
        y_probas_val = estimator.predict_proba(X_val)
        loss_train.append(log_loss(y_train, y_probas_train))
        loss_val.append(log_loss(y_val, y_probas_val))
        # print(dt.datetime.now() - before)
        
        f1_train.append(f1_tr)
        f1_val.append(f1_v)
        acc_train.append(acc_tr)
        acc_val.append(acc_v)
        roc_train.append(roc_auc_score(y_train, y_probas_train, multi_class="ovo"))
        roc_val.append(roc_auc_score(y_val, y_probas_val, multi_class="ovo"))


    plt.plot(loss_train, label = "train loss")
    plt.plot(loss_val, label = "val loss")
    plt.legend(loc="upper right")
    plt.show()

    plt.plot(f1_train, label = "train f1")
    plt.plot(f1_val, label = "val f1")
    plt.legend(loc="lower right")
    plt.show()

def log_learning_curve(experiment, dirname, curve_type, n_epochs, y_type):
    data_path = dirname + "/" + curve_type
    print(data_path)
    # experiment.log_curve(data_path, x=range(n_epochs), y=y_type)
    # experiment.log_curve("curves/train_loss", x=range(n_epochs), y=loss_train)
    # experiment.log_curve("curves/val_loss", x=range(n_epochs), y=loss_val)
    # experiment.log_curve("curves/train_f1", x=range(n_epochs), y=f1_train)
    # experiment.log_curve("curves/val_f1", x=range(n_epochs), y=f1_val)
    # experiment.log_curve("curves/train_acc", x=range(n_epochs), y=acc_train)
    # experiment.log_curve("curves/val_acc", x=range(n_epochs), y=acc_val)
    # experiment.log_curve("curves/train_roc", x=range(n_epochs), y=roc_train)
    # experiment.log_curve("curves/val_roc", x=range(n_epochs), y=roc_val)
    # experiment.end()
