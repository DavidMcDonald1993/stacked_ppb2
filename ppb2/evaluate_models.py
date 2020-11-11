import os 
import argparse 

import numpy as np
import pandas as pd 

from sklearn.metrics import (roc_auc_score, 
    average_precision_score,
    f1_score,
    precision_score, 
    recall_score)

from scipy import sparse as sp

from models import get_model_name, load_model
from data_utils import read_smiles, load_labels, filter_data

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

    p_at_k = compute_p_at_k(labels, pred, 
        probs, k=k, n_proc=n_proc)

    if n_proc > 1:

        with mp.Pool(processes=n_proc) as p:
            roc = np.array(p.starmap(roc_auc_score, 
                iterable=zip(labels, probs)))
        print ("computed ROC")

        with mp.Pool(processes=n_proc) as p:
        ap = np.array(p.starmap(average_precision_score, 
            iterable=zip(labels, probs)))
        print ("computed AP")

        with mp.Pool(processes=n_proc) as p:
            f1 = np.array(p.starmap(f1_score, 
                iterable=zip(labels, pred)))
        print ("computed F1")

        with mp.Pool(processes=n_proc) as p:
            precision = np.array(p.starmap(precision_score, 
                iterable=zip(labels, pred)))
        print ("computed precision")

        with mp.Pool(processes=n_proc) as p:
            recall = np.array(p.starmap(recall_score, 
                iterable=zip(labels, pred)))
        print ("computed recall")

    else:

        labels = labels.T
        pred = pred.T
        probs = probs.T
        # # shape is now n_targets x n_compounds

        roc = roc_auc_score(labels, probs, average=None)
        print ("computed ROC")
    
        ap = average_precision_score(labels, probs, average=None)
        print ("computed AP")

        f1 = f1_score(labels, pred, average=None )
        print ("computed F1")

        precision = precision_score(labels, pred, average=None )
        print ("computed precision")

        recall = recall_score(labels, pred, average=None )
        print ("computed recall")
        
    assert len(roc) == len(ap) == len(f1) == len(precision) == len(recall) == n_queries

    return (p_at_k.mean(), roc.mean(), ap.mean(), f1.mean(), 
        precision.mean(), recall.mean())

def validate(args, split_name):

    # model_filename = os.path.join("models", 
    #     split_name,
    #     get_model_filename(args))
    # assert os.path.exists(model_filename)
    # model = load_model(model_filename)
    # model.set_n_proc(args.n_proc)

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

    # X_test, Y_test = filter_data(
    #     X_test, Y_test, 
    #     min_actives=0, min_hits=1)

    # assert Y_test.any(axis=0).all()
    # assert (1-Y_test).any(axis=0).all()
    assert Y_test.any(axis=1).all()
    assert (1-Y_test).any(axis=1).all()

    # predictions = model.predict(X_test)
    # probs = model.predict_proba(X_test)
    prediction_dir = os.path.join("predictions", 
        split_name, "{}-test".format(get_model_name(args)))
    
    prediction_filename = os.path.join(prediction_dir, 
        "predictions.csv.gz")
    assert os.path.exists(prediction_filename)
    predictions = pd.read_csv(prediction_filename, index_col=0)
    predictions = predictions.values

    probs_filename = os.path.join(prediction_dir, 
        "probs.csv.gz")
    assert os.path.exists(probs_filename)
    probs = pd.read_csv(probs_filename, index_col=0)
    probs = probs.values

    assert isinstance(probs, np.ndarray)
    assert X_test.shape[0] == Y_test.shape[0] == predictions.shape[0] == probs.shape[0]

    return compute_measures(
        Y_test, 
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

        print ("computing results for split", split+1)
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

    test_results = validate(args, "complete")
    test_results = pd.Series(test_results, 
        index=pd.Series(["test-{}".format(metric) for metric in metric_names],
            name="metric"))

    return results.append(test_results)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", 
        default=["morg2-nn+nb"], nargs="+",)

    parser.add_argument("--n_proc", default=1, type=int)

    return parser.parse_args()

def main():

    args = parse_args()

    cross_validation_results = cross_validation(
        args,
        n_splits=5,
        k=5)

    name = get_model_name(args)
    cross_validation_results.name = name

    print ("results:")
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