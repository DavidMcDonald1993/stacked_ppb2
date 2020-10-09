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

from get_fingerprints import load_labels, read_smiles
from models import StackedPPB2, PPB2

import pickle as pkl



def compute_measures(
    labels,
    pred, 
    probs, 
    ):  

    assert len(labels.shape) == len(pred.shape) == len(probs.shape) == 2
    n_queries = labels.shape[0]

    labels = labels.T
    pred = pred.T
    probs = probs.T

    roc = roc_auc_score(labels, probs, average=None)
    ap = average_precision_score(labels, probs, average=None)

    f1 = f1_score(labels, pred, average=None )
    precision = precision_score(labels, pred, average=None )
    recall = recall_score(labels, pred, average=None )

    assert len(roc) == len(ap) == len(f1) == len(precision) == len(recall) == n_queries

    return (roc.mean(), ap.mean(), f1.mean(), 
        precision.mean(), recall.mean())

def cross_validation(
    X, Y, 
    model,
    n_splits=5,
    ):

    n_compounds = X.shape[0]
    n_targets = Y.shape[1]

    metric_names = [
        "ROC", "AP", 
        "f1", 
        "precision", "recall"
    ]

    n_metrics = len(metric_names) 

    results = np.zeros((n_splits, n_metrics))

    # split = KFold(n_splits=n_splits, random_state=0)
    split = IterativeStratification(n_splits=n_splits, order=1, random_state=0)

    for split, (train_idx, test_idx) in enumerate(split.split(X, Y)):
        print ("processing split", split+1, "/", n_splits)
        
        X_train = X[train_idx]
        X_test = X[test_idx]

        y_train = Y[train_idx]
        y_test = Y[test_idx]

        assert not y_train.all(axis=-1).any()
        assert not (1-y_train).all(axis=-1).any()

        assert not y_test.all(axis=-1).any()    
        assert not (1-y_test).all(axis=-1).any()
        
        n_queries = len(test_idx)

        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        probs = model.predict_proba(X_test)
        assert isinstance(probs, np.ndarray)
        assert y_test.shape[0] == predictions.shape[0] == probs.shape[0]
        assert y_test.shape[1] == predictions.shape[1] == probs.shape[1]


        print ("computing results for split", split+1)
        results[split] = compute_measures(y_test, predictions, probs)
        print ("completed split", split+1)
        print ()

    results = pd.DataFrame(results,
        columns=pd.Series(metric_names, name="metric"))

    return results.mean(0)

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--fp", default="ecfp4",
        choices=["mqn", "xfp", "ecfp4", 
            "morg2", "morg3", "rdk", "circular", "maccs"])

    parser.add_argument("--model", default="nn+nb",
        choices=["nn", "nb", "nn+nb", 
            "svc", "lr", "bag", "stack"])

    return parser.parse_args()

def main():

    args = parse_args()

    data_dir = os.path.join("data", )
    
    smiles_filename = os.path.join(data_dir, 
        "compounds.smi")
    X = read_smiles(smiles_filename)
    y = load_labels().A

    # idx = range(1000)

    # X = X[idx]
    # y = y[idx, ]

    # idx = np.logical_and(y.sum(0) >= 10, (1-y).sum(0) >=10)
    # y = y[:,idx]

    assert X.shape[0] == y.shape[0]
    
    print ("model is", args.model)
    if args.model == "stack":
        model = StackedPPB2()
    else:
        model = PPB2(fp=args.fp, model_name=args.model)

    cross_validation_results = cross_validation(
        X, y, model,
        n_splits=5)
    cross_validation_results.name = "{}-{}".format(
        args.fp, args.model)

    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)

    results_filename = os.path.join(results_dir, 
        "{}-{}-results.pkl".format(args.fp, 
        args.model,))
    print ("picking results to", results_filename)
    with open(results_filename, "wb") as f:
        pkl.dump(cross_validation_results, f, pkl.HIGHEST_PROTOCOL)

   
if __name__ == "__main__":
    main()