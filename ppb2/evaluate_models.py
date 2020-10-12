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
    k=5):  

    assert len(labels.shape) == len(pred.shape) == len(probs.shape) == 2
    assert (labels.sum(axis=1) >= k).all(), "must have at least k positive examples"
    n_queries = labels.shape[0]

    # precision at k
    idx = probs.argsort(axis=1)[:, -k:]
    assert idx.shape[0] == n_queries
    assert idx.shape[1] == k
    p_at_k = np.array([
        precision_score(labels_[idx_], pred_[idx_])
        for labels_, pred_, idx_ in 
            zip(labels, pred, idx)
    ])

    labels = labels.T
    pred = pred.T
    probs = probs.T

    roc = roc_auc_score(labels, probs, average=None)
    ap = average_precision_score(labels, probs, average=None)

    f1 = f1_score(labels, pred, average=None )
    precision = precision_score(labels, pred, average=None )
    recall = recall_score(labels, pred, average=None )

    assert len(p_at_k) == len(roc) == len(ap) == len(f1) == len(precision) == len(recall) == n_queries

    return (p_at_k.mean(), roc.mean(), ap.mean(), f1.mean(), 
        precision.mean(), recall.mean())

def cross_validation(
    X, Y, 
    model,
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

def filter_training_data(X, y, 
    min_compounds=10, 
    min_target_hits=10):
    print ("filtering training data to ensure",
        "at least", min_compounds, "compounds hit each target and",
        "each compound hits", min_target_hits, "targets")

    while (y.sum(axis=0) < min_compounds).any() or\
        (y.sum(axis=1) < min_target_hits).any():

        # filter targets
        idx = np.logical_and(
            y.sum(axis=0) >= min_compounds, 
            (1-y).sum(axis=0) >= min_compounds)
        y = y[:,idx]

        # filter compounds
        idx = y.sum(axis=1) >= min_target_hits
        X = X[idx]
        y = y[idx, ]


    assert (y.sum(axis=0) >= min_compounds).all()
    assert ((1-y).sum(axis=0) >= min_compounds).all()
    assert (y.sum(axis=1) >= min_target_hits).all()
    assert ((1-y).sum(axis=1) >= min_target_hits).all()

    print ("num compounds after filtering:", X.shape[0])
    print ("num targets after filtering:", y.shape[1])

    return X, y

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--fp", default="morg2",
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

    # filter out for compounds that hit at least min_hits targets
    min_hits = 10

    # remove any targets that are hit/not hit by less than 10 compounds
    min_compounds = 10

    X, y = filter_training_data(X, y, 
        min_compounds=min_compounds, 
        min_target_hits=min_hits)

    assert X.shape[0] == y.shape[0]
    assert y.any(axis=1).all()
    assert (1-y).any(axis=1).all()
    assert not y.all(axis=0).any()
    assert not (1-y).all(axis=0).any()

    print ("model is", args.model)
    if args.model == "stack":
        model = StackedPPB2(
            fps=["maccs", "rdk", "morg2"],
            models=["nn+nb"]
        )
    else:
        model = PPB2(fp=args.fp, model_name=args.model)

    cross_validation_results = cross_validation(
        X, y, model,
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