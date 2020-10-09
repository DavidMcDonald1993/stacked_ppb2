import numpy as np
import pandas as pd

import itertools

from sklearn.base import BaseEstimator, ClassifierMixin

from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.multiclass import OneVsRestClassifier

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegressionCV

from sklearn.ensemble import BaggingClassifier#, StackingClassifier

from sklearn.base import clone

from sklearn.utils.validation import check_is_fitted

from sklearn.model_selection import KFold, StratifiedKFold
from skmultilearn.model_selection import IterativeStratification

from sklearn.exceptions import NotFittedError

from get_fingerprints import compute_fp, load_training_fingerprints

# from multi_label_stack import StackingClassifier

class StackedPPB2(BaseEstimator, ClassifierMixin):  
    """Stacked PPB2 model"""

    def __init__(self, 
        fps=["morg2", "rdk", "maccs"], 
        models=["nn+nb", ],
        n_splits=5,
        final_estimator=LogisticRegressionCV()):

        self.classifiers = [
            ("{}-{}".format(fp, model), 
                PPB2(fp=fp, model_name=model))
            for fp, model in itertools.product(fps, models)
        ]
        assert len(self.classifiers) == len(fps) * len(models)

        print ("building stacked PPB2 classifier",
            "using the following models:", self.classifiers)

        self.n_splits = n_splits
        
        self.final_estimator = final_estimator

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
            # self.split = KFold(n_splits=self.n_splits)
            self.split = IterativeStratification(n_splits=self.n_splits,
                order=1)
            self.n_targets = y.shape[1]

            meta_preds = np.empty((X.shape[0], len(self.classifiers), self.n_targets, ))
            
        for i, (name, classifier) in enumerate(self.classifiers):
            print ("fitting classifier:", name)
            for split_no, (train, test) in enumerate(self.split.split(X, y)):
                print ("processing split", split_no+1, "/", self.n_splits)
                classifier.fit(X[train], y[train])
                meta_preds[test, i] = classifier.predict(X[test]) # multi target precit

        if not isinstance(y, np.ndarray):
            y = y.A

        if self.multi_label:
            print ("fitting final estimators")
            self.final_estimator = [
                clone(self.final_estimator) 
                    for _ in range(self.n_targets)]

            for target_id in range(self.n_targets):
                self.final_estimator[target_id].fit(meta_preds[...,target_id], y[:,target_id])
        else:

            print ("fitting final estimator")
            self.final_estimator.fit(meta_preds, y)

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

class PPB2(BaseEstimator, ClassifierMixin):  
    """PPB2 model"""
    
    def __init__(self, 
        fp="morg2", 
        model_name="nn+nb", 
        k=200):
        self.fp = fp
        self.model_name = model_name
        self.k = k
        
    def fit(self, X, y):
        """
        """
        assert isinstance(X, pd.Series)
        assert (X.dtype==pd.StringDtype()), "X should be a vector of smiles"

        assert X.shape[0] == y.shape[0]

        print ("fitting PPB2 model to", X.shape[0], "SMILES")

        if  len(y.shape) == 1:
            print ("fitting in the single-target setting")
            self.multi_label = False
        else:
            print ("fitting in the multi-target setting")
            print ("number of targets:", y.shape[1])
            self.multi_label = True

        model_name = self.model_name   
        if "nn" in model_name:
            self.model = KNeighborsClassifier(
                n_neighbors=self.k,
                metric="jaccard", 
                algorithm="brute", 
                n_jobs=-1)
        elif model_name == "nb":
            self.model = BernoulliNB(alpha=1.)
        elif model_name == "svc":
            self.model = SVC(probability=True)
        elif model_name == "bag":
            self.model = BaggingClassifier(n_jobs=-1)
        elif model_name == "lr":
            self.model = LogisticRegressionCV(
                max_iter=1000,
                # multi_class="ovr",
                n_jobs=1)
        if self.multi_label and model_name in {"nb", "svc", "bag", "lr"}:
            self.model = OneVsRestClassifier(
                self.model,
                n_jobs=-1)

        if model_name == "nn+nb": # keep references for local NB fitting
            self.X = X 
            self.y = y

        # covert X to fingerprint
        X = load_training_fingerprints(X, self.fp)

        assert X.shape[0] == y.shape[0]

        print ("fitting model to fingerprints")
        self.model.fit(X, y)

        return self

    def set_k(self, k):
        self.k = k 
        if isinstance(self.model, KNeighborsClassifier):
            self.model.n_neighbors = k

    def _determine_k_closest_samples(self, X):
        assert isinstance(self.model, KNeighborsClassifier)
        assert self.k == self.model.n_neighbors
        print ("determining", self.k, 
            "nearest compounds to each query")
        idx = self.model.kneighbors(X,
            return_distance=False)

        training_samples = load_training_fingerprints(self.X, self.fp)
        training_labels = self.y
        if not isinstance(training_labels, np.ndarray):
            training_labels = training_labels.A

        k_nearest_samples = training_samples[idx]
        k_nearest_labels = training_labels[idx]

        return k_nearest_samples, k_nearest_labels

    def _fit_nb(self,
        query, X, y, 
        mode="predict"):
        
        assert query.shape[0] == 1

        if self.multi_label:
            assert len(y.shape) == 2
            nb = OneVsRestClassifier(
                BernoulliNB(alpha=1.),
                n_jobs=-1)

            pred = np.zeros(y.shape[1])
            ones_idx = y.all(axis=0)
            zeros_idx = (1-y).all(axis=0)

            pred[ones_idx] = 1

            # only fit on targets with pos and neg examples
            idx = ~np.logical_or(ones_idx, zeros_idx)
            if idx.any():
                nb.fit(X, y[:,idx])
                pred[idx] = (nb.predict(query)[0] if mode=="predict"
                    else nb.predict_proba(query)[0])
            return pred

        else:
            assert len(y.shape) == 1
            nb = BernoulliNB(alpha=1.)

            if y.all():
                return 1
            if (1-y).all():
                return 0

            nb.fit(X, y)

            return (nb.predict(query)[0] if mode=="predict"
                else nb.predict_proba(query)[0])

    def _local_nb_prediction(self, 
        queries, X, y,
        mode="predict"):

        n_queries = queries.shape[0]
        if self.multi_label:
            assert len(y.shape) == 3
            n_targets = y.shape[-1]
            predictions = np.zeros((n_queries, n_targets))

        else:
            predictions = np.zeros((n_queries, ))

        for query_idx in range(n_queries):

            predictions[query_idx] = self._fit_nb(
                queries[query_idx:query_idx+1],
                X[query_idx],
                y[query_idx],
                mode=mode)

            # print ("completed fitting NB for query",
                # query_idx+1)
            
        return predictions

 
    def predict(self, X):
        print ("predicting for", X.shape[0], 
            "query molecules")
        X = compute_fp(X, self.fp)
        print ("performing prediction")
        if self.model_name == "nn+nb":

            k_nearest_samples, k_nearest_labels = \
                self._determine_k_closest_samples(X)
           
            return self._local_nb_prediction(
                X,
                k_nearest_samples,
                k_nearest_labels,
                mode="predict")
        else:
            return self.model.predict(X)

    def predict_proba(self, X):
        print ("predicting probabilities for", X.shape[0], 
            "query molecules")
        X = compute_fp(X, self.fp)
        if self.model_name == "nn+nb":

            k_nearest_samples, k_nearest_labels = \
                self._determine_k_closest_samples(X)
           
            return self._local_nb_prediction(
                X,
                k_nearest_samples,
                k_nearest_labels,
                mode="prob")

        if self.model_name == "nn":
            probs = self.model.predict_proba(X)
            classes = self.model.classes_
            return np.hstack([probs[:,idx] if idx.any() else 1-probs
                for probs, idx in zip(probs, classes)])

        else:
            return self.model.predict_proba(X)

    def check_is_fitted(self):
        try:
             check_is_fitted(self.model)
             return True
        except NotFittedError:
            return False

    def __str__(self):
        return "Model name: " + self.model_name + ", fp: " + self.fp +\
            ", k = {}".format(self.k)