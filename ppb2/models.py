import os

import numpy as np
import pandas as pd
from scipy import sparse as sp

import functools

from sklearn.base import BaseEstimator, ClassifierMixin

from sklearn.dummy import DummyClassifier

from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.multiclass import OneVsRestClassifier

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, RidgeClassifierCV

from sklearn.ensemble import (BaggingClassifier, ExtraTreesClassifier, 
    AdaBoostClassifier, GradientBoostingClassifier)

from xgboost import XGBClassifier

from sklearn.base import clone

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.utils.validation import check_is_fitted

from sklearn.metrics import pairwise_distances

from sklearn.model_selection import StratifiedKFold
from skmultilearn.model_selection import IterativeStratification

from sklearn.exceptions import NotFittedError

from get_fingerprints import compute_fp, load_training_fingerprints

import multiprocessing as mp

import pickle as pkl

import gzip

from joblib import parallel_backend

dense_input = {"nn", "lda"}
support_multi_label = {"nn", "etc", }

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

def get_model_name(args):
    model = args.model
    assert isinstance(model, list)
    if model[0] == "stack":
        return "stack-({})".format("&".join(model[1:]))
    else:
        return model[0]

def get_model_filename(args):
    return get_model_name(args) + ".pkl.gz"

def save_model(model, model_filename):
    assert model_filename.endswith(".pkl.gz")
    print ("pickling model to", model_filename)
    with gzip.open(model_filename, "wb") as f:
        pkl.dump(model, f, pkl.HIGHEST_PROTOCOL)

def load_model(model_filename):
    assert model_filename.endswith(".pkl.gz")
    assert os.path.exists(model_filename), model_filename
    print ("reading model from", model_filename)
    with gzip.open(model_filename, "rb") as f:
        return pkl.load(f)

class StackedPPB2(BaseEstimator, ClassifierMixin):  
    """Stacked PPB2 model"""

    def __init__(self, 
        models=["morg2-nn+nb", "morg3-nn+nb"],
        n_splits=5,
        stack_method="predict_proba",
        final_estimator=LogisticRegression(max_iter=1000),
        n_proc=8,
        passthrough=False):

        self.classifiers = [
            (model, PPB2(model=model, n_proc=n_proc))
                for model in models
        ]
        assert len(self.classifiers)  == len(models)

        print ("building stacked PPB2 classifier",
            "using the following models:", )
        for model_name, classifier in self.classifiers:
            print (model_name, classifier)
        print ()

        self.n_splits = n_splits
        assert stack_method in {"predict_proba", "predict"}
        self.stack_method = stack_method
        self.final_estimator = final_estimator
        self.n_proc = n_proc
        self.passthrough = passthrough
        if passthrough:
            raise NotImplementedError

    def fit(self, X, y):
        """
       
        """
        assert isinstance(X, pd.Series)

        assert y.any(axis=0).all(), "At least one positive example is needed"
        assert (1-y).any(axis=0).all(), "At least one negative example is needed"

        print ("Fitting meta-estimators using cross-validation")

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
            print ("fitting meta estimators")
            if not isinstance(self.final_estimator, list):
                self.final_estimator = [
                    clone(self.final_estimator) 
                        for _ in range(self.n_targets)]

            for target_id in range(self.n_targets):
                with parallel_backend('threading', n_jobs=self.n_proc):
                    self.final_estimator[target_id].fit(
                        meta_preds[...,target_id], y[:,target_id])

                print ("completed fitting meta estimator for target", 
                    target_id+1, "/", self.n_targets, "targets")
        else:

            print ("fitting meta estimator")
            self.final_estimator.fit(meta_preds, y)

        print ("completed fitting of meta estimator(s)")
        print ()
        
        print ("fitting base estimator(s) using full training set")
        for i, (name, classifier) in enumerate(self.classifiers):
            print ("fitting classifier", name)
            classifier.fit(X, y)
            print ("completed classifier", name, )
            print ()

        print ()

        return self

    def predict(self, X):
        assert isinstance(X, pd.Series)
        # assert (X.dtype==pd.StringDtype()), "X should be a vector of smiles"

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
                with parallel_backend('threading', n_jobs=self.n_proc):
                    final_pred[:,target_id] = self.final_estimator[target_id].predict(meta_preds[..., target_id])
            return final_pred
        else:
            check_is_fitted(self.final_estimator)
            return self.final_estimator.predict(meta_preds)

    def predict_proba(self, X):
        assert isinstance(X, pd.Series)
        # assert (X.dtype==pd.StringDtype()), "X should be a vector of smiles"

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
                with parallel_backend('threading', n_jobs=self.n_proc):
                    final_pred[:,target_id] = self.final_estimator[target_id].predict_proba(meta_preds[..., target_id])[:,1]
            return final_pred
        else:
            check_is_fitted(self.final_estimator)
            assert self.final_estimator.classes_[1]
            with parallel_backend('threading', n_jobs=self.n_proc):
                return self.final_estimator.predict_proba(meta_preds)[:, 1]

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
        assert self.model_name in {"dum", "nn", "nb", "nn+nb",
            "bag", "lr", "svc", "etc", "ridge", "ada", "gb", "lda",
            "xgc"}
        self.n_proc = n_proc
        self.k = k

        model_name = self.model_name  
        if model_name == "dum":
            self.model = DummyClassifier(
                strategy="stratified")
        elif model_name == "nn":
            self.model = KNeighborsClassifier(
                n_neighbors=self.k,
                metric="jaccard", 
                algorithm="brute", 
                n_jobs=self.n_proc)
        elif model_name == "nb":
            self.model = BernoulliNB(alpha=1.)
        elif model_name == "nn+nb":
            self.model = None
        elif model_name == "svc":
            self.model = SVC(probability=True)
        elif model_name == "bag":
            self.model = BaggingClassifier(
                n_jobs=self.n_proc)
        elif model_name == "lr":
            self.model = LogisticRegressionCV(
                max_iter=1000,
                n_jobs=self.n_proc)
        elif model_name == "ada":
            self.model = AdaBoostClassifier()
        elif model_name == "gb":
            self.model = GradientBoostingClassifier()
        elif model_name == "lda":
            self.model = LinearDiscriminantAnalysis()
        elif model_name == "etc":
            self.model = ExtraTreesClassifier(
                n_estimators=500,
                bootstrap=True, 
                max_features="log2",
                min_samples_split=10,
                max_depth=5,
                min_samples_leaf=3,
                verbose=True,
                n_jobs=n_proc) # capable of multilabel classification out of the box
        elif model_name == "ridge":
            self.model = RidgeClassifierCV()
        elif model_name == "xgc":
            self.model = XGBClassifier()
        else:
            raise Exception
        
    def fit(self, X, y):
        """
        """
        assert isinstance(X, pd.Series)
        # assert (X.dtype==pd.StringDtype()), "X should be a vector of smiles"

        assert X.shape[0] == y.shape[0]

        print ("fitting PPB2 model", 
            "({}-{})".format(self.fp, self.model_name),
            "to", X.shape[0], "SMILES")

        if  len(y.shape) == 1:
            print ("fitting in the single-target setting")
            self.multi_label = False
        else:
            print ("fitting in the multi-target setting")
            print ("number of targets:", y.shape[1])
            self.multi_label = True

        if self.multi_label and self.model_name not in support_multi_label.union({"nn+nb"}):
            self.model = OneVsRestClassifier( # wrap classifier in OneVsRestClassifier for multi-label case
                self.model,
                n_jobs=self.n_proc)

        # covert X to fingerprint
        X = load_training_fingerprints(X, self.fp,)

        if self.model_name in dense_input: # cannot handle sparse input
            X = X.A

        if self.model_name == "nn+nb": # keep training data references for local NB fitting
            self.X = X
            self.y = y

        assert X.shape[0] == y.shape[0]

        if self.model is not None:
            print ("fitting", self.model_name, "model to", 
                X.shape[0], self.fp, "fingerprints", 
                "for", y.shape[1], "targets",
                "using", self.n_proc, "core(s)")

            with parallel_backend('threading', n_jobs=self.n_proc):
                self.model.fit(X, y)

        return self

    # def _determine_k_closest_samples(self, X, chunksize=1000):
    #     if not isinstance(X, np.ndarray): # dense needed for jaccard distance
    #         X = X.A

    #     # training_samples = load_training_fingerprints(self.X, self.fp)
    #     training_samples = self.X
    #     if not isinstance(training_samples, np.ndarray):
    #         training_samples = training_samples.A
    #     training_labels = self.y
    #     if not isinstance(training_labels, np.ndarray):
    #         training_labels = training_labels.A

    #     print ("determining", self.k, 
    #         "nearest compounds to each query")
    #     n_queries = X.shape[0]
    #     n_chunks = n_queries // chunksize + 1
    #     print ("chunking queries with chunksize", chunksize,)
    #     print ("number of chunks:", n_chunks)
    #     # idx = np.empty((n_queries, self.k))

    #     for chunk in range(n_chunks):

    #         chunk_queries = X[chunk*chunksize:(chunk+1)*chunksize]
        
    #         dists = pairwise_distances(
    #                 chunk_queries,
    #                 training_samples, 
    #             metric="jaccard", n_jobs=self.n_proc, )
    #         # idx[chunk*chunksize:(chunk+1)*chunksize] = \
    #         idx =  dists.argsort(axis=-1)[:,:self.k] # smallest k distances
        
    #         k_nearest_samples = training_samples[idx] # return dense
    #         k_nearest_labels = training_labels[idx]

    #         yield (chunk_queries, 
    #             k_nearest_samples, k_nearest_labels)

    #         print ("completed chunk", chunk+1)

        # print ("closest", self.k, "neighbours determined")

        # assert idx.shape[0] == X.shape[0]
        # assert idx.shape[1] == self.k

        # k_nearest_samples = training_samples[idx] # return dense
        # k_nearest_labels = training_labels[idx]

        # return k_nearest_samples, k_nearest_labels

    def _fit_local_nb(self,
        query,
        mode="predict",
        alpha=1.):

        if len(query.shape) == 1:
            query = query[None, :]

        X = self.X 
        y = self.y 

        assert isinstance (query, sp.csr_matrix)
        assert query.dtype == bool
        assert isinstance (X, sp.csr_matrix)
        assert X.dtype == bool

        # sparse jaccard distance
        assert query.shape[1] == X.shape[1]
        dists = pairwise_distances(query.A, X.A, 
            metric="jaccard", n_jobs=1)
        idx = dists.argsort()[0, :self.k]

        assert query.shape[0] == 1

        X = X[idx]
        y = y[idx]

        n_targets = y.shape[-1]

        pred = np.zeros(n_targets)
        ones_idx = y.all(axis=0)
        zeros_idx = (1-y).all(axis=0)

        # set prediction for classes where only positive class
        # is seen
        pred[ones_idx] = 1

        # only fit on targets with pos and neg examples
        idx = ~np.logical_or(ones_idx, zeros_idx)
        if idx.any():
            nb = BernoulliNB(alpha=alpha)
            if idx.sum() > 1:
                nb = OneVsRestClassifier(nb, n_jobs=1)
            y_ = y[:,idx]
            if idx.sum() == 1:
                y_ = y_.flatten()
            nb.fit(X, y_)
            pred_ = (nb.predict(query)[0] 
                if mode=="predict"
                    else nb.predict_proba(query)[0])
            if idx.sum() == 1 and mode != "predict":
                assert pred_.shape[0] == 2
                assert nb.classes_.any()
                pred_ = pred_[nb.classes_==1]
            pred[idx] = pred_
        return pred

    def _local_nb_prediction(self, 
        queries, 
        # X, y,
        mode="predict"):
        print ("fitting unique NB models for each query",
            "in mode", mode)

        n_queries = queries.shape[0]

        with mp.Pool(processes=self.n_proc) as p:
            predictions = p.map(
                functools.partial(self._fit_local_nb, mode=mode),
                (query for query in queries))

        predictions = np.array(predictions)
        assert predictions.shape[0] == n_queries

        if self.multi_label:
            assert predictions.shape[1] == self.y.shape[-1]

        return predictions

    def predict(self, X):
        print ("predicting for", X.shape[0], 
            "query molecules")
        X = compute_fp(X, self.fp, n_proc=self.n_proc)
        print ("performing prediction",
            "using", self.n_proc, "processes")

        if self.model_name == "nn+nb":

            # X = X.astype(int)
            
            # with mp.Pool(processes=).

            # self._fit_local_nb(X[0])

            # return np.vstack([
            #     self._local_nb_prediction(
            #         X_,
            #         k_nearest_samples,
            #         k_nearest_labels,
            #         mode="predict")
            #         for X_, k_nearest_samples, k_nearest_labels
            #             in self._determine_k_closest_samples(X)
            # ])

            # k_nearest_samples, k_nearest_labels = \
            #     self._determine_k_closest_samples(X)

            return self._local_nb_prediction(
                X,
            #     k_nearest_samples,
            #     k_nearest_labels,
                mode="predict")
        else:
            if self.model_name in dense_input \
                and not isinstance(X, np.ndarray):
                X = X.A
            assert hasattr(self.model, "predict")

            with parallel_backend('threading', n_jobs=self.n_proc):
                return self.model.predict(X)

    def predict_proba(self, X):
        print ("predicting probabilities for", X.shape[0], 
            "query molecules")
        X = compute_fp(X, self.fp, n_proc=self.n_proc)
        print ("performing probability prediction",
            "using", self.n_proc, "processes")
        if self.model_name == "nn+nb":
            
            # X = X.astype(int)
            # k_nearest_samples, k_nearest_labels = \
                # self._determine_k_closest_samples(X)
           
            return self._local_nb_prediction(
                X,
                # k_nearest_samples,
                # k_nearest_labels,
                mode="predict_proba")

        if self.model_name in dense_input \
            and not isinstance(X, np.ndarray):
            X = X.A

        if self.model_name in support_multi_label:
            with parallel_backend('threading', n_jobs=self.n_proc):
                probs = self.model.predict_proba(X) # handle missing classes correctly
            classes = self.model.classes_
            return np.hstack([probs[:,idx] if idx.any() else 1-probs
                for probs, idx in zip(probs, classes)]) # check for existence of positive class

        else:
            assert isinstance(self.model, OneVsRestClassifier)
            if hasattr(self.model, "predict_proba"):
                with parallel_backend('threading', n_jobs=self.n_proc):
                    return self.model.predict_proba(X)
            elif hasattr(self.model, "decision_function"):
                print ("predicting with decision function")
                with parallel_backend('threading', n_jobs=self.n_proc):
                    return self.model.decision_function(X)
            else:
                raise Exception

    def decision_function(self, X):
        print ("predicting probabilities for", X.shape[0], 
            "query molecules")
        X = compute_fp(X, self.fp, n_proc=self.n_proc)
        print ("determining decision function",
            "using", self.n_proc, "processes")
        if self.model_name == "nn+nb":
           
            return self._local_nb_prediction(
                X,
                mode="predict_proba") # NB does not have a decision function

        if self.model_name in dense_input \
            and not isinstance(X, np.ndarray):
            X = X.A

        if self.model_name in support_multi_label: # k neigbours has no decision function
            with parallel_backend('threading', n_jobs=self.n_proc):
                probs = self.model.predict_proba(X) # handle missing classes correctly
            classes = self.model.classes_
            return np.hstack([probs[:,idx] if idx.any() else 1-probs
                for probs, idx in zip(probs, classes)]) # check for existence of positive class

        else:
            assert isinstance(self.model, OneVsRestClassifier)

            if hasattr(self.model, "decision_function"):
                with parallel_backend('threading', n_jobs=self.n_proc):
                    return self.model.decision_function(X)
            elif hasattr(self.model, "predict_proba"):
                print ("predicting using probability")
                with parallel_backend('threading', n_jobs=self.n_proc):
                    return self.model.predict_proba(X)
            else:
                raise Exception

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
        if self.model is not None:
            self.model.n_jobs = n_proc

    def set_k(self, k):
        self.k = k 
        if isinstance(self.model, KNeighborsClassifier):
            self.model.n_neighbors = k