import numpy as np


def compute_tp_fp_fn(cm):
    
    results = np.empty([cm.shape[0], 3])
    for i in range(cm.shape[0]):
        tp = cm[i, i]
        fn = np.sum(cm[i, :]) - tp
        fp = np.sum(cm[:, i]) - tp
        results[i, :] = np.array([tp, fn, fp]).astype(np.int16)
    return results


def compute_precision(tp, fp, **kwargs):
    prec = tp / (tp + fp)
    weights = kwargs["weights"]
    if kwargs["average"] == "macro":
        prec = np.mean(prec)
    elif kwargs["average"] == "micro":
        prec = np.sum(tp) / (np.sum(tp) + np.sum(fp))
    elif kwargs["average"] == "weighted":
        prec = np.dot(weights, prec)
    else:
        prec = np.mean(prec)
    # print("Precision computed: ", prec)
    return prec

def compute_recall(tp, fn, **kwargs):
    recall = tp / (tp + fn)
    weights = kwargs["weights"]
    if kwargs["average"] == "macro":
        recall = np.mean(recall)
    elif kwargs["average"] == "micro":
        recall = np.sum(tp) / (np.sum(tp) + np.sum(fn))
    elif kwargs["average"] == "weighted":
        recall = np.dot(weights, recall)
    else:
        recall = np.mean(recall)
    # print("Recall computed: ", recall)
    return recall

def compute_f1(tp, fp, fn, **kwargs):
    f1 = tp / (tp + ((fn + fp)/2))
    weights = kwargs["weights"]
    if kwargs["average"] == "macro":
        f1 = np.mean(f1)
    elif kwargs["average"] == "micro":
        f1 = np.sum(tp) / (np.sum(tp) + ((np.sum(fp)+np.sum(fn))/2))
    elif kwargs["average"] == "weighted":
        f1 = np.dot(weights, f1)
    else:
        f1 = np.mean(f1)
    # print("f1 computed: ", f1)
    return f1

def compute_scores(cm, **kwargs):
    print(" -- " * 25)
    print("Computing scores for type:", kwargs["average"])
    results = compute_tp_fp_fn(cm)
    tp = results[:, 0]
    fn = results[:, 1]
    fp = results[:, 2]
    precision = compute_precision(tp, fp, **kwargs)
    recall = compute_recall(tp, fn, **kwargs)
    f1 = compute_f1(tp, fp, fn, **kwargs)
    print("Precision computed: ", precision)
    print("recall computed: ", recall)
    print("f1 computed: ", f1)

