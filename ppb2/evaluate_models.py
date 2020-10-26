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

import multiprocessing as mp

def _compute_p_at_k(labels, pred, idx):
    return precision_score(labels[idx], pred[idx])

def compute_p_at_k(
    labels,
    pred, 
    probs,
    k=5,
    n_proc=8):

    print ("computing precision at", k, "with", n_proc, "processes")

    # mask out any samples with less than k positives
    valid_compounds = labels.sum(axis=1) >= k
    n_valid = valid_compounds.sum()
    assert n_valid > 0
    print ("number of compounds with >=", k, "targets:", n_valid)
    labels = labels[valid_compounds]
    pred = pred[valid_compounds]
    probs = probs[valid_compounds]

    # precision at k
    idx = probs.argsort(axis=1)[:, -k:]
    assert idx.shape[1] == k
    assert idx.shape[0] == n_valid
    if n_proc > 1:
        with mp.Pool(processes=n_proc) as p:
            p_at_k = np.array(p.starmap(_compute_p_at_k,
                iterable=zip(labels, pred, idx)))
    else:
        p_at_k = np.array([
            precision_score(labels_[idx_], pred_[idx_])
            for labels_, pred_, idx_ in 
                zip(labels, pred, idx)
        ])

    assert len(p_at_k) == n_valid

    print ("computed precision at", k)

    return p_at_k

def compute_measures(
    labels,
    pred, 
    probs, 
    k=5,
    n_proc=8):  

    print ("computing classification evaluation metrics")

    assert len(labels.shape) == len(pred.shape) == len(probs.shape) == 2
    n_queries = labels.shape[0]
    assert pred.shape[0] == probs.shape[0] == n_queries

    p_at_k = compute_p_at_k(labels, pred, probs, k=k, n_proc=n_proc)

    # labels = labels.T
    # pred = pred.T
    # probs = probs.T
    # # shape is now n_targets x n_compounds

    # roc = roc_auc_score(labels, probs, average=None)
    with mp.Pool(processes=n_proc) as p:
        roc = np.array(p.starmap(roc_auc_score, 
            iterable=zip(labels, probs)))
    print ("computed ROC")
    # ap = average_precision_score(labels, probs, average=None)
    with mp.Pool(processes=n_proc) as p:
        ap = np.array(p.starmap(average_precision_score, 
            iterable=zip(labels, probs)))
    print ("computed AP")

    # f1 = f1_score(labels, pred, average=None )
    with mp.Pool(processes=n_proc) as p:
        f1 = np.array(p.starmap(f1_score, 
            iterable=zip(labels, pred)))
    print ("computed F1")
    # precision = precision_score(labels, pred, average=None )
    with mp.Pool(processes=n_proc) as p:
        precision = np.array(p.starmap(precision_score, 
            iterable=zip(labels, pred)))
    print ("computed precision")
    # recall = recall_score(labels, pred, average=None )
    with mp.Pool(processes=n_proc) as p:
        recall = np.array(p.starmap(recall_score, 
            iterable=zip(labels, pred)))
    print ("computed recall")

    assert len(roc) == len(ap) == len(f1) == len(precision) == len(recall) == n_queries

    return (p_at_k.mean(), roc.mean(), ap.mean(), f1.mean(), 
        precision.mean(), recall.mean())

def validate(args, split_name):

    model_filename = os.path.join("models", 
        split_name,
        get_model_filename(args))
    assert os.path.exists(model_filename)
    model = load_model(model_filename)
    model.set_n_proc(args.n_proc)

    X_test_filename = os.path.join("splits", 
        split_name,
        "test.smi")
    assert os.path.exists(X_test_filename)
    X_test = read_smiles(X_test_filename)

    Y_test_filename = os.path.join("splits", 
        split_name,
        "test.npz")
    assert os.path.exists(Y_test_filename)
    Y_test = load_labels(Y_test_filename).A

    predictions = model.predict(X_test)
    probs = model.predict_proba(X_test)
    assert isinstance(probs, np.ndarray)
    assert Y_test.shape[0] == predictions.shape[0] == probs.shape[0]

    # ensure targets have at least one hit / miss
    idx = np.logical_and(Y_test.any(axis=0,), (1-Y_test).any(axis=0))
    Y_test = Y_test[:,idx]
    predictions = predictions[:,idx]
    probs = probs[:,idx]

    # ensure compounds hit at least one target
    idx = np.logical_and(Y_test.any(axis=1,), (1-Y_test).any(axis=1))
    Y_test = Y_test[idx]
    predictions = predictions[idx]
    probs = probs[idx]

    assert Y_test.any(axis=0).all()
    assert (1-Y_test).any(axis=0).all()
    assert Y_test.any(axis=1).all()
    assert (1-Y_test).any(axis=1).all()

    return compute_measures(Y_test, 
        predictions,
        probs)

def cross_validation(
    args,
    n_splits=5,
    k=5):

    n_proc = args.n_proc

    metric_names = [
        "p@{}".format(k),
        "ROC", "AP", 
        "f1", 
        "precision", "recall"
    ]

    n_metrics = len(metric_names) 

    results = np.zeros((n_splits, n_metrics))

    for split in range(n_splits):
        print ("processing fold", split+1, "/", n_splits)

        # model_filename = os.path.join("models", "split_{}".format(split),
        #     get_model_filename(args))
        # assert os.path.exists(model_filename)
        # model = load_model(model_filename)
        # model.set_n_proc(n_proc)

        # X_test_filename = os.path.join("splits", "split_{}".format(split),
        #     "test.smi")
        # assert os.path.exists(X_test_filename)
        # X_test = read_smiles(X_test_filename)

        # Y_test_filename = os.path.join("splits", "split_{}".format(split),
        #     "test.npz")
        # assert os.path.exists(Y_test_filename)
        # Y_test = load_labels(Y_test_filename).A

        # predictions = model.predict(X_test)
        # probs = model.predict_proba(X_test)
        # assert isinstance(probs, np.ndarray)
        # assert Y_test.shape[0] == predictions.shape[0] == probs.shape[0]
        # assert Y_test.shape[1] == predictions.shape[1] == probs.shape[1]

        print ("computing results for split", split+1)
        # results[split] = compute_measures(Y_test, predictions, probs, n_proc=n_proc)
        results[split] = validate(args, "split_{}".format(split))
        print ("completed fold", split+1)
        print ("#########################")
        print ()

    results = pd.DataFrame(results,
        columns=pd.Series(metric_names, name="metric"))

    print ("computing mean over all folds")
    print ()
  
    # mean over all folds
    results = results.mean(0)

    print ("evaluating on test data")

    # # compute test metrics
    # model_filename = os.path.join("models", "complete",
    #     get_model_filename(args))
    # assert os.path.exists(model_filename)
    # model = load_model(model_filename)

    # X_test_filename = os.path.join("splits", "complete",
    #     "test.smi")
    # assert os.path.exists(X_test_filename)
    # X_test = read_smiles(X_test_filename)

    # Y_test_filename = os.path.join("splits", "complete",
    #     "test.npz")
    # assert os.path.exists(Y_test_filename)
    # Y_test = load_labels(Y_test_filename).A

    # predictions = model.predict(X_test)
    # probs = model.predict_proba(X_test)
    # assert isinstance(probs, np.ndarray)
    # assert Y_test.shape[0] == predictions.shape[0] == probs.shape[0]
    # # assert Y_test.shape[1] == predictions.shape[1] == probs.shape[1]

    # idx = np.logical_and(Y_test.any(axis=0,), (1-Y_test).any(axis=0))
    # Y_test = Y_test[:,idx]
    # predictions = predictions[:,idx]
    # probs = probs[:,idx]

    # idx = np.logical_and(Y_test.any(axis=1,), (1-Y_test).any(axis=1))
    # Y_test = Y_test[idx]
    # predictions = predictions[idx]
    # probs = probs[idx]

    # assert Y_test.any(axis=0).all()
    # assert (1-Y_test).any(axis=0).all()
    # assert Y_test.any(axis=1).all()
    # assert (1-Y_test).any(axis=1).all()

    # test_results = compute_measures(Y_test, 
    #     predictions,
    #     probs)
    test_results = validate(args, "complete")
    test_results = pd.Series(test_results, 
        index=pd.Series(["test-{}".format(metric) for metric in metric_names],
            name="metric"))

    return results.append(test_results)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", 
        default=["morg2-nn+nb"], nargs="+",)

    parser.add_argument("--n_proc", default=8, type=int)

    return parser.parse_args()

def main():

    args = parse_args()

    cross_validation_results = cross_validation(
        args,
        n_splits=5,
        k=5)

    if args.model == "stack":
        name = "stack-({})".format(
            "&".join(args.model[1:]))
    else:
        name = args.model[0]
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