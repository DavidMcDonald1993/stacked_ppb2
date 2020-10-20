import os 
import argparse 

import numpy as np
import pandas as pd 

from sklearn.metrics import (roc_auc_score, 
    average_precision_score,
    f1_score,
    precision_score, 
    recall_score)

# from sklearn.model_selection import KFold
from skmultilearn.model_selection import IterativeStratification

from scipy import sparse as sp

from models import build_model, get_model_filename, load_model
from split_data import read_smiles, load_labels

import pickle as pkl

def compute_p_at_k(labels,
    pred, 
    probs, 
    k=5):

    # mask out any samples with less than k positives
    valid_compounds = labels.sum(axis=1) >= k
    assert valid_compounds.sum() > 0
    labels = labels[valid_compounds]
    pred = pred[valid_compounds]
    probs = probs[valid_compounds]

    # precision at k
    idx = probs.argsort(axis=1)[:, -k:]
    assert idx.shape[0] == n_queries
    assert idx.shape[1] == k
    p_at_k = np.array([
        precision_score(labels_[idx_], pred_[idx_])
        for labels_, pred_, idx_ in 
            zip(labels, pred, idx)
    ])

    assert len(p_at_k) == valid_compounds.sum()

    return p_at_k

def compute_measures(
    labels,
    pred, 
    probs, 
    k=5):  

    print ("computing classification evaluaiton metrics")

    assert len(labels.shape) == len(pred.shape) == len(probs.shape) == 2
    # assert (labels.sum(axis=1) >= k).all(), "must have at least k positive examples"
    n_queries = labels.shape[0]

    p_at_k = compute_p_at_k(labels, pred, probs, k=k)

    labels = labels.T
    pred = pred.T
    probs = probs.T

    roc = roc_auc_score(labels, probs, average=None)
    ap = average_precision_score(labels, probs, average=None)

    f1 = f1_score(labels, pred, average=None )
    precision = precision_score(labels, pred, average=None )
    recall = recall_score(labels, pred, average=None )

    assert len(roc) == len(ap) == len(f1) == len(precision) == len(recall) == n_queries

    return (p_at_k.mean(), roc.mean(), ap.mean(), f1.mean(), 
        precision.mean(), recall.mean())

def cross_validation(
    args,
    n_splits=5,
    k=5):

    n_compounds = X.shape[0]
    n_targets = Y.shape[1]

    metric_names = [
        "p@{}".format(k),
        "ROC", "AP", 
        "f1", 
        "precision", "recall"
    ]

    n_metrics = len(metric_names) 

    results = np.zeros((n_splits, n_metrics))

    # split = IterativeStratification(n_splits=n_splits, order=1, random_state=0)

    # for split, (train_idx, test_idx) in enumerate(split.split(X, Y)):
    #     print ("processing fold", split+1, "/", n_splits)
        
    #     X_train = X[train_idx]
    #     X_test = X[test_idx]

    #     y_train = Y[train_idx]
    #     y_test = Y[test_idx]

    #     assert not y_train.all(axis=-1).any()
    #     assert not (1-y_train).all(axis=-1).any()

    #     assert not y_test.all(axis=-1).any()    
    #     assert not (1-y_test).all(axis=-1).any()
        
    #     n_queries = len(test_idx)

    #     model.fit(X_train, y_train)

    for split in range(n_splits):
        print ("processing fold", split+1, "/", n_splits)

        model_filename = os.path.join("models", "split_{}".format(split),
            get_model_filename(args))
        assert os.path.exists(model_filename)
        model = load_model(model_filename)

        X_test_filename = os.path.join("splits", "split_{}".format(split),
            "test.smi")
        assert os.path.exists(X_test_filename)
        X = read_smiles(X_test_filename)

        Y_test_filename = os.path.join("splits", "split_{}".format(split),
            "test.npz")
        assert os.path.exists(Y_test_filename)
        Y_test = load_labels(Y_test_filename).A

        predictions = model.predict(X_test)
        probs = model.predict_proba(X_test)
        assert isinstance(probs, np.ndarray)
        assert Y_test.shape[0] == predictions.shape[0] == probs.shape[0]
        assert Y_test.shape[1] == predictions.shape[1] == probs.shape[1]

        print ("computing results for split", split+1)
        results[split] = compute_measures(Y_test, predictions, probs)
        print ("completed fold", split+1)
        print ("#########################")
        print ()

    results = pd.DataFrame(results,
        columns=pd.Series(metric_names, name="metric"))

    print ("computing mean over all folds")
  
    # mean over all folds
    results = results.mean(0)

    print ("beginning evaluation on test data")

    # compute test metrics
    model_filename = os.path.join("models", "complete",
        get_model_filename(args))
    assert os.path.exists(model_filename)
    model = load_model(model_filename)

    X_test_filename = os.path.join("splits", "complete",
        "test.smi")
    assert os.path.exists(X_test_filename)
    X = read_smiles(X_test_filename)

    Y_test_filename = os.path.join("splits", "complete",
        "test.npz")
    assert os.path.exists(Y_test_filename)
    Y_test = load_labels(Y_test_filename).A

    predictions = model.predict(X_test)
    probs = model.predict_proba(X_test)
    assert isinstance(probs, np.ndarray)
    assert y_test.shape[0] == predictions.shape[0] == probs.shape[0]
    assert y_test.shape[1] == predictions.shape[1] == probs.shape[1]

    test_results = compute_measures(Y_test, 
        predictions,
        probs)
    test_results = pd.Series(test_results, 
        index=pd.Series(["test-{}".format(metric) for metric in metric_names],
            name="metric"))

    return results.append(test_results)


def parse_args():
    parser = argparse.ArgumentParser()

    # parser.add_argument("--fp", default="morg2",
    #     choices=["mqn", "xfp", "ecfp4", 
    #         "morg2", "morg3", "rdk", "rdk_maccs",
    #         "circular", "maccs"])

    parser.add_argument("--model", default=["morg2-nn+nb"], nargs="+",
        # choices=["nn", "nb", "nn+nb", 
            # "svc", "lr", "bag", "stack"]
            )

    return parser.parse_args()

def main():

    args = parse_args()

    cross_validation_results = cross_validation(
        args.model,
        n_splits=5,
        k=5)

    if args.model == "stack":
        name = "{}-({})".format(
            args.model, "&".join((name 
                    for name, _ in model.classifiers)))
    else:
        name = "{}-{}".format(
            args.fp, args.model)
    cross_validation_results.name = name
    print (cross_validation_results)

    results_dir = os.path.join("results", )
    os.makedirs(results_dir, exist_ok=True)
   
    results_filename = os.path.join(results_dir, 
        "{}-results.pkl".format(name))
    print ("picking results to", results_filename)
    with open(results_filename, "wb") as f:
        pkl.dump(cross_validation_results, f, pkl.HIGHEST_PROTOCOL)

   
if __name__ == "__main__":
    main()