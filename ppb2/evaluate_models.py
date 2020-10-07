import os 
import argparse 

import numpy as np
import pandas as pd 

from sklearn.metrics import pairwise_distances

from skmultilearn.model_selection import IterativeStratification

from sklearn.neighbors import NearestNeighbors
from sklearn.naive_bayes import BernoulliNB
from sklearn.multioutput import MultiOutputClassifier

from sklearn.metrics import (roc_auc_score, 
    f1_score, 
    accuracy_score,
    matthews_corrcoef, 
    precision_score, 
    recall_score)

from scipy import sparse as sp


# def fit_nb(nb, queries, X, y):
#     assert len(y.shape) == 2
#     n_queries = queries.shape[0]
#     n_targets = y.shape[1]
#     zero_idx = (1-y).all(axis=0)
#     one_idx = y.all(axis=0)
#     pred = np.empty((n_queries, n_targets), )
#     pred[:, zero_idx] = 0
#     pred[:, one_idx] = 1

#     idx = ~np.logical_or(zero_idx, one_idx)
#     if idx.any():
#         nb.fit(X, y[:, idx])
#         assert all([len(nb_.classes_) == 2
#             for nb_ in nb.estimators_])
#         assert all([nb_.classes_[1] 
#             for nb_ in nb.estimators_])
#         probs = nb.predict_proba(queries)
#         if probs.shape[1] != idx.sum():
#             assert idx.sum() == 1
#             probs = probs[:,1]
#         # pred[:, idx] = np.column_stack([probs[:,1] 
#         #     for probs in nb.predict_proba(queries)])
#         pred[:, idx] = probs
#     return pred

# def nb_prediction(nb, queries, X, Y):

#     n_queries = queries.shape[0]
#     n_targets = Y.shape[-1]
#     predictions = np.zeros((n_queries, n_targets))

#     if len(X.shape) == 3:
#         for query_idx in range(n_queries):

#             predictions[query_idx] = fit_nb(nb, 
#                 queries[query_idx:query_idx+1],
#                 X[query_idx],
#                 Y[query_idx])

#             print ("completed fitting NB for query num",
#                 query_idx+1)
           
#     else:
#         predictions = fit_nb(nb, 
#             queries, X, Y)
#     return predictions


def compute_measures(labels, 
    probs, 
    threshold=.5):  

    assert len(labels.shape) == len(probs.shape) == 1

    roc = roc_auc_score(labels, probs,)

    thresholds = np.linspace(probs.min(), 
        probs.max(), num=50, endpoint=False)
    mccs = np.array(
        [matthews_corrcoef(labels, probs > threshold) 
        for threshold in thresholds])
    idx = mccs.argmax()
    threshold = thresholds[idx]

    pred = probs > threshold
    assert not pred.all()
    assert not (1-pred).all(), (probs.min(), probs.max(),
        threshold)
    f1 = f1_score(labels, pred, )
    accuracy = accuracy_score(labels, pred, )
    # mcc = matthews_corrcoef(labels, pred)
    mcc = mccs[idx]
    precision = precision_score(labels, pred, )
    recall = recall_score(labels, pred, )

    return roc, threshold, f1, accuracy, mcc, precision, recall

def cross_validation(X, Y, model,
    n_splits=30,
    test_size=100,
    ks=(5, 50, 500, ),
    batch_size=1000, 
    ):

    n_compounds = X.shape[0]
    n_targets = Y.shape[1]
    n_ks = len(ks)
    max_k = max(ks)

    metric_names = [
        "ROC", "threshold", "f1", "accuracy", 
        "MCC", "precision", "recall"
    ]

    if "nn" in model:
        models = [model + "_k={}".format(k) for k in ks]
    else:
        models = ["NB"]

    n_models, n_metrics = len(models), len(metric_names) 

    results = np.zeros((n_splits, n_models, n_metrics))

    if "nb" in model:
        alpha = 1.
        print ("using naive bayes with laplace smoothing")
        print ("alpha =", alpha)
        nb_model = BernoulliNB(alpha=alpha) 
        nb_model = MultiOutputClassifier(nb_model, 
            n_jobs=-1)

    if "nn" in model:
        print ("building KNN model")
        nbrs = NearestNeighbors(n_neighbors=max_k,
            metric="jaccard", 
            algorithm="brute",                
            n_jobs=-1)

    for split in range(n_splits):
        print ("processing split", split+1)
        
        np.random.seed(split)
        idx = np.random.permutation(n_compounds)
        test_idx = idx[:test_size]
        train_idx = idx[test_size:]
    # print ("building multi label stratifier")
    # stratifier = IterativeStratification(
        # n_splits=n_splits,
        # order=1, # ensure balanced selection across all targets
        # random_state=0
        # )

    # for split, (train_idx, test_idx) in enumerate(stratifier.split(X, y)):

    # train_idx, test_idx = next(stratifier.split(X, Y))

        X_train = X[train_idx]
        X_test = X[test_idx]

        y_train = Y[train_idx]
        y_test = Y[test_idx]

        assert not y_train.all(axis=-1).any()
        assert not (1-y_train).all(axis=-1).any()

        assert not y_test.all(axis=-1).any()
        assert not (1-y_test).all(axis=-1).any()
        
        n_queries = len(test_idx)

        predictions = np.empty((n_models, n_queries, n_targets))

        if model == "nb":
            print ("Fitting NB on complete data")
            predictions[0] =  nb_prediction(
                nb_model,
                X_test, 
                X_train, 
                y_train)
            print ("done")
        else:

            assert "nn" in model
            nbrs.fit(X_train)

            print ("using KNN")
            print ("chunking queries for KNN")

            n_queries = len(test_idx)
            n_batches = int (np.ceil(n_queries / batch_size))

            print ("num queries", n_queries)
            print ("batch size", batch_size)
            print ("num batches", n_batches)

            for batch_num in range(n_batches):

                batch_idx = range(batch_num*batch_size,
                    min((batch_num+1)*batch_size, n_queries))
                queries = X_test[batch_idx]

                print ("computing", max_k,
                    "closest neighbours for batch")
                # dists_sorted = sort_by_distance(
                #     queries, 
                #     X_train,
                #     max_k=max(ks))
                dists_sorted = nbrs.kneighbors(queries, 
                    return_distance=False)

                for k_idx in range(n_ks):
                    k = ks[k_idx]
                    print ("k =", k)
                    idx = dists_sorted[:,:k]
                    k_nearest_labels = y_train[idx]

                    assert k_nearest_labels.shape[1] == k

                    if "nb" in model:
                        print ("knn+nb")
                        k_nearest_samples = X_train[idx]
                        pred = nb_prediction(
                            nb_model,
                            queries, 
                            k_nearest_samples, 
                            k_nearest_labels)

                    else:
                        print ("knn")
                        pred = k_nearest_labels.mean(1) # vote across neighbours

                    predictions[k_idx, batch_idx] = pred

                print ("completed batch", batch_num+1, 
                    "/", n_batches)

        print ("computing results for split", split+1)
        results[split] = np.array(
            [[compute_measures(y_test[query_idx, :], 
                predictions[model_idx, query_idx, :])
                for query_idx in range(n_queries)]
            for model_idx in range(n_models)]).mean(1) # mean over queries
        print ("completed split", split+1)
        print ()

    results = results.mean(0) # mean over splits
    
    results = pd.DataFrame(results, 
        index=pd.Series(models, name="model"),
        columns=pd.Series(metric_names, name="metric"))

    return results


def load_fingerprints(fp, 
    fingerprint_dir=os.path.join("data", )):
    if fp == "mqn":
        filename = os.path.join(fingerprint_dir,
            "MQN_fingerprint.npy")
    elif fp == "xfp":
        filename = os.path.join(fingerprint_dir,
            "Xfp_fingerprint.npy")
    elif fp == "ecfp4":
        filename = os.path.join(fingerprint_dir,
            "Extended_connectivity_fingerprint_(ECfp4).npy")
    elif fp == "morg2":
        filename = os.path.join(fingerprint_dir,
            "MORG2_fingerprint.npy")    
    elif fp == "morg3":
        filename = os.path.join(fingerprint_dir,
            "MORG3_fingerprint.npy")
    elif fp == "rdk":
        filename = os.path.join(fingerprint_dir,
            "RDK_fingerprint.npy")
    elif fp == "circular":
        filename = os.path.join(fingerprint_dir,
            "circular_fingerprint.npy")
    elif fp == "MACCS":
        filename = os.path.join(fingerprint_dir,
            "MACCS_fingerprint.npy")
    else:
        raise Exception

    print ("loading fingerprints from", filename)
    return np.load(filename)

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--fp", default="ecfp4",
        choices=["mqn", "xfp", "ecfp4", 
            "morg2", "morg3", "rdk", "circular", "MACCS"])

    parser.add_argument("--model", default="nn+nb",
        choices=["nn", "nb", "nn+nb"])

    return parser.parse_args()

def main():

    args = parse_args()

    data_dir = os.path.join("data", )
    
    X = load_fingerprints(args.fp,
        fingerprint_dir=data_dir)
    print ("training fingerprints shape is", 
        X.shape)
    assert X.dtype == bool

    labels_filename = os.path.join(data_dir,
        "targets.npz")
    print ("loading labels from", labels_filename)
    y = sp.load_npz(labels_filename).A

    assert X.shape[0] == y.shape[0]
    
    # select targets with most actives?
    # counts = y.sum(0).flatten()
    # idx = counts.argsort()[::-1][:100]
    # y = y[:,idx]
    # idx = y.any(axis=-1)
    # X = X[idx]
    # y = y[idx]
    print ("labels shape is", y.shape)

    raise SystemExit

    print ("model is", args.model)

    cross_validation_results = cross_validation(X, y, args.model,
        n_splits=10,
        test_size=100,
        ks=(3, 5, 25, 50, 2000))

    results_filename = os.path.join(".", 
        "{}-{}-results.csv".format(args.fp, 
        args.model,))
    print ("saving results to", results_filename)
    cross_validation_results.to_csv(results_filename)

   
if __name__ == "__main__":
    main()