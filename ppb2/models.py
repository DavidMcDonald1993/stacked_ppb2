import numpy as np
import pandas as pd

import itertools
import functools

from sklearn.base import BaseEstimator, ClassifierMixin

from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.multiclass import OneVsRestClassifier

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegressionCV

from sklearn.ensemble import BaggingClassifier#, StackingClassifier

from sklearn.base import clone

from sklearn.utils.validation import check_is_fitted

from sklearn.metrics import pairwise_distances

from sklearn.model_selection import KFold, StratifiedKFold
from skmultilearn.model_selection import IterativeStratification

from sklearn.exceptions import NotFittedError

from get_fingerprints import compute_fp, load_training_fingerprints

# from multi_label_stack import StackingClassifier
import multiprocessing as mp

import pickle as pkl

def build_model(args):
    model = args.model
    assert isinstance(model, list)
    n_proc = args.n_proc

    print ("model is", model[0])
    if model[0] == "stack":
        return StackedPPB2(
            models=model[1:],
            n_proc=n_proc
        )
    else:
        return PPB2(model=model[0],
            n_proc=n_proc)

def get_model_filename(args):
    model = args.model
    assert isinstance(model, list)
    if model[0] == "stack":
        return "stack-({}).pkl".format("&".join(model[1:]))
    else:
        return "{}.pkl".format(model[0])

def save_model(model, model_filename):
    print ("pickling model to", model_filename)
    with open(model_filename, "wb") as f:
        pkl.dump(model, f, pkl.HIGHEST_PROTOCOL)

def load_model(model_filename):
    print ("reading model from", model_filename)
    with open(model_filename, "rb") as f:
        return pkl.load(f)

class StackedPPB2(BaseEstimator, ClassifierMixin):  
    """Stacked PPB2 model"""

    def __init__(self, 
        # fps=["morg2", "rdk", "maccs"], 
        models=["morg2-nn+nb", "morg3-nn+nb"],
        n_splits=5,
        stack_method="predict_proba",
        final_estimator=LogisticRegressionCV(),
        n_proc=8,
        passthrough=False):

        # assert len(fps) == len(models)

        self.classifiers = [
            # ("{}-{}".format(fp, model), 
            #     PPB2(fp=fp, model_name=model))
            # for fp, model in zip(fps, models)
            (model, PPB2(model=model, n_proc=n_proc))
                for model in models
        ]
        assert len(self.classifiers)  == len(models)

        print ("building stacked PPB2 classifier",
            "using the following models:", self.classifiers)

        self.n_splits = n_splits
        assert stack_method in {"predict_proba", "predict"}
        self.stack_method = stack_method
        self.final_estimator = final_estimator
        self.n_proc = n_proc
        self.passthrough = passthrough
        if passthrough:
            raise NotImplementedError

        # self.model = StackingClassifier(classifiers,
        #     n_jobs=1,
        #     passthrough=True,
        #     verbose=True,
        # )

    def fit(self, X, y):
        """
       
        """
        assert isinstance(X, pd.Series)
        assert (X.dtype==pd.StringDtype()), "X should be a vector of smiles"

        assert y.any(axis=0).all(), "At least one positive example is needed"
        assert (1-y).any(axis=0).all(), "At least one negative example is needed"

        if  len(y.shape) == 1:
            print ("fitting in the single-target setting")
            self.multi_label = False
            self.split = StratifiedKFold(n_splits=self.n_splits)

            meta_preds = np.empty((X.shape[0], len(self.classifiers), ))

        else:
            print ("fitting in the multi-target setting")
            print ("number of targets:", y.shape[1])
            self.multi_label = True
            self.split = IterativeStratification(n_splits=self.n_splits,
                order=1)
            self.n_targets = y.shape[1]

            meta_preds = np.empty((X.shape[0], len(self.classifiers), self.n_targets, ))
            
        for i, (name, classifier) in enumerate(self.classifiers):
            print ("fitting classifier:", name)
            for split_no, (train, test) in enumerate(self.split.split(X, y)):
                print ("processing split", split_no+1, "/", self.n_splits)
                classifier.fit(X[train], y[train])
                if self.stack_method == "predict_proba":
                    meta_preds[test, i] = classifier.predict_proba(X[test]) # multi target predict probs (for positive class)
                else:
                    meta_preds[test, i] = classifier.predict(X[test]) # multi target predict
                print ("completed split", split_no+1, "/", self.n_splits)
                print ()
            print ("completed classifier", name, )
            print ()

        if not isinstance(y, np.ndarray):
            y = y.A

        if self.multi_label:
            print ("fitting final estimators")
            if not isinstance(self.final_estimator, list):
                self.final_estimator = [
                    clone(self.final_estimator) 
                        for _ in range(self.n_targets)]

            for target_id in range(self.n_targets):
                self.final_estimator[target_id].fit(meta_preds[...,target_id], y[:,target_id])
        else:

            print ("fitting final estimator")
            self.final_estimator.fit(meta_preds, y)

        print ("completed fitting of final estimators")
        print ()

        return self

    def predict(self, X):
        assert isinstance(X, pd.Series)
        assert (X.dtype==pd.StringDtype()), "X should be a vector of smiles"

        if self.multi_label:
            meta_preds = np.empty((X.shape[0], len(self.classifiers), self.n_targets, ))
        else: 
            meta_preds = np.empty((X.shape[0], len(self.classifiers), ))

        for i, (name, classifier) in enumerate(self.classifiers):
            print ("performing prediction with classifier:", name)
            assert classifier.check_is_fitted()
            if self.stack_method == "predict_proba":
                meta_preds[:,i] = classifier.predict_proba(X)
            else:
                meta_preds[:,i] = classifier.predict(X)
        
        # final estimator
        if self.multi_label:
            final_pred = np.empty((X.shape[0], self.n_targets))
            for target_id in range(self.n_targets):
                check_is_fitted(self.final_estimator[target_id])
                final_pred[:,target_id] = self.final_estimator[target_id].predict(meta_preds[..., target_id])
            return final_pred
        else:
            check_is_fitted(self.final_estimator)
            return self.final_estimator.predict(meta_preds)

        # return self.model.predict(X)

    def predict_proba(self, X):
        assert isinstance(X, pd.Series)
        assert (X.dtype==pd.StringDtype()), "X should be a vector of smiles"

        if self.multi_label:
            meta_preds = np.empty((X.shape[0], len(self.classifiers), self.n_targets, ))

        else: 
            meta_preds = np.empty((X.shape[0], len(self.classifiers), ))

        for i, (name, classifier) in enumerate(self.classifiers):
            assert classifier.check_is_fitted()
            if self.stack_method == "predict_proba":
                meta_preds[:,i] = classifier.predict_proba(X)
            else:
                meta_preds[:,i] = classifier.predict(X)
        
        # final estimator
        if self.multi_label:
            final_pred = np.empty((X.shape[0], self.n_targets))
            for target_id in range(self.n_targets):
                check_is_fitted(self.final_estimator[target_id])
                assert self.final_estimator[target_id].classes_[1]
                final_pred[:,target_id] = self.final_estimator[target_id].predict_proba(meta_preds[..., target_id])[:,1]
            return final_pred
        else:
            check_is_fitted(self.final_estimator)
            assert self.final_estimator.classes_[1]
            return self.final_estimator.predict_proba(meta_preds)[:, 1]

        # return self.model.predict_proba(X)

    def set_n_proc(self, n_proc):
        self.n_proc = n_proc
        for _, classifier in self.classifiers:
            classifier.set_n_proc(n_proc)

class PPB2(BaseEstimator, ClassifierMixin):  
    """PPB2 model"""
    
    def __init__(self, 
        model="morg2&nn+nb",
        n_proc=8,
        k=200):
        model = model.split("-")
        assert len(model) == 2
        self.fp = model[0]
        assert self.fp in {"rdk", "morg2", "morg3", "rdk_maccs",
            "circular", "maccs"}
        self.model_name = model[1]
        assert self.model_name in {"nn", "nb", "nn+nb",
            "bag", "lr", "svc"}
        self.n_proc = n_proc
        self.k = k
        
    def fit(self, X, y):
        """
        """
        assert isinstance(X, pd.Series)
        assert (X.dtype==pd.StringDtype()), "X should be a vector of smiles"

        assert X.shape[0] == y.shape[0]

        print ("fitting PPB2 model", "({})".format(self.model_name),
            "to", X.shape[0], "SMILES")

        if  len(y.shape) == 1:
            print ("fitting in the single-target setting")
            self.multi_label = False
        else:
            print ("fitting in the multi-target setting")
            print ("number of targets:", y.shape[1])
            self.multi_label = True

        model_name = self.model_name   
        if model_name == "nn":
            self.model = KNeighborsClassifier(
                n_neighbors=self.k,
                metric="jaccard", 
                algorithm="brute", 
                n_jobs=self.n_proc)
        elif model_name == "nb":
            self.model = BernoulliNB(alpha=1.)
        elif model_name == "svc":
            self.model = SVC(probability=True)
        elif model_name == "bag":
            self.model = BaggingClassifier(n_jobs=1)
        elif model_name == "lr":
            self.model = LogisticRegressionCV(
                max_iter=1000,
                n_jobs=1)
        if self.multi_label and model_name in {"nb", "svc", "bag", "lr"}:
            self.model = OneVsRestClassifier(
                self.model,
                n_jobs=self.n_proc)

        if model_name == "nn+nb": # keep references for local NB fitting
            self.X = X 
            self.y = y
            self.model = None

        # covert X to fingerprint
        X = load_training_fingerprints(X, self.fp,)

        assert X.shape[0] == y.shape[0]

        if self.model is not None:
            print ("fitting model to", 
                X.shape[0], "fingerprints", 
                "for", y.shape[1], "targets",
                "using", self.model.n_jobs, "cores")
            self.model.fit(X, y)

        return self

    def set_k(self, k):
        self.k = k 
        if isinstance(self.model, KNeighborsClassifier):
            self.model.n_neighbors = k

    def _determine_k_closest_samples(self, X):
        # assert isinstance(self.model, KNeighborsClassifier)
        # assert self.k == self.model.n_neighbors
        # print ("determining", self.k, 
        #     "nearest compounds to each query")
        # idx = self.model.kneighbors(X,
        #     return_distance=False)
        # print ("neighbours determined")

        training_samples = load_training_fingerprints(self.X, self.fp)
        training_labels = self.y
        if not isinstance(training_labels, np.ndarray):
            training_labels = training_labels.A

        print ("determining", self.k, 
            "nearest compounds to each query")
        dists = pairwise_distances(X, training_samples, 
            metric="jaccard", n_jobs=-1, )
        idx = dists.argsort(axis=-1)[:,:self.k]
        print ("neighbours determined")

        assert idx.shape[0] == X.shape[0]
        assert idx.shape[1] == self.k

        k_nearest_samples = training_samples[idx]
        k_nearest_labels = training_labels[idx]

        return k_nearest_samples, k_nearest_labels

    def _fit_nb(self,
        query, X, y, 
        mode="predict"):

        if len(query.shape) == 1:
            query = query[None, :]
        
        assert query.shape[0] == 1

        if self.multi_label:
            assert len(y.shape) == 2
            nb = OneVsRestClassifier(
                BernoulliNB(alpha=1.),
                n_jobs=1)

            pred = np.zeros(y.shape[1])
            # probs = np.zeros(y.shape[1])
            
            ones_idx = y.all(axis=0)
            zeros_idx = (1-y).all(axis=0)

            pred[ones_idx] = 1
            # probs[ones_idx] = 1

            # only fit on targets with pos and neg examples
            idx = ~np.logical_or(ones_idx, zeros_idx)
            if idx.any():
                nb.fit(X, y[:,idx])
                pred[idx] = (nb.predict(query)[0] if mode=="predict"
                    else nb.predict_proba(query)[0])
                # pred[idx] = nb.predict(query)[0]
                # probs[idx] = nb.predict_proba(query)[0]
            # return pred, probs
            return pred

        else:
            assert len(y.shape) == 1
            nb = BernoulliNB(alpha=1.)

            if y.all():
                return 1, 1
            if (1-y).all():
                return 0, 0

            nb.fit(X, y)

            return (nb.predict(query)[0] if mode=="predict"
                else nb.predict_proba(query)[0])
            # return (nb.predict(query)[0], nb.predict_proba(query)[0])

    def _local_nb_prediction(self, 
        queries, X, y,
        mode="predict"):
        print ("fitting unique NB models for each query",
            "in mode", mode)

        n_queries = queries.shape[0]
        if self.multi_label:
            assert len(y.shape) == 3
            n_targets = y.shape[-1]
            # predictions = np.zeros((n_queries, n_targets))

        # else:
            # predictions = np.zeros((n_queries, ))
        # probs = np.zeros_like(predictions)

        # for query_idx in range(n_queries):

        #     # predictions[query_idx], probs[query_idx] = self._fit_nb(
        #     predictions[query_idx] = self._fit_nb(
        #         queries[query_idx:query_idx+1],
        #         X[query_idx],
        #         y[query_idx],
        #         mode=mode)

        #     if query_idx > 0 and query_idx % 50 == 0:
        #         print ("completed fitting NB for query",
        #             query_idx, "/", n_queries,
        #             "in mode", mode)
        
        with mp.Pool(processes=self.n_proc) as p:
            predictions = p.starmap(
                functools.partial(self._fit_nb, mode=mode),
                zip(queries, X, y))

        predictions = np.array(predictions)
        assert predictions.shape[0] == n_queries

        if self.multi_label:
            assert predictions.shape[1] == n_targets

        return predictions#, probs

 
    def predict(self, X):
        print ("predicting for", X.shape[0], 
            "query molecules")
        X = compute_fp(X, self.fp, n_proc=self.n_proc)
        print ("performing prediction")
        if self.model_name == "nn+nb":

            k_nearest_samples, k_nearest_labels = \
                self._determine_k_closest_samples(X)
           
            return self._local_nb_prediction(
                X,
                k_nearest_samples,
                k_nearest_labels,
                mode="predict")#[0] # return predictions only
        else:
            return self.model.predict(X)

    def predict_proba(self, X):
        print ("predicting probabilities for", X.shape[0], 
            "query molecules")
        X = compute_fp(X, self.fp, n_proc=self.n_proc)
        if self.model_name == "nn+nb":

            k_nearest_samples, k_nearest_labels = \
                self._determine_k_closest_samples(X)
           
            return self._local_nb_prediction(
                X,
                k_nearest_samples,
                k_nearest_labels,
                mode="predict_proba")#[1] # return probs only

        if self.model_name == "nn":
            probs = self.model.predict_proba(X)
            classes = self.model.classes_
            return np.hstack([probs[:,idx] if idx.any() else 1-probs
                for probs, idx in zip(probs, classes)])

        else:
            return self.model.predict_proba(X)

    def check_is_fitted(self):
        if self.model is None:
            return True
        try:
             check_is_fitted(self.model)
             return True
        except NotFittedError:
            return False

    def __str__(self):
        return "PPB2({}-{})".format(self.fp, self.model_name)

    def set_n_proc(self, n_proc):
        self.n_proc = n_proc
        self.model.n_jobs = n_proc