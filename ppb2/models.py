import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, ClassifierMixin

from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.multiclass import OneVsRestClassifier

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegressionCV

from sklearn.ensemble import StackingClassifier, BaggingClassifier

import itertools

from get_fingerprints import compute_fp, load_training_fingerprints

###################

# def fit_nb(nb, queries, X, y):
#     assert len(y.shape) == 1
#     print (y.shape)
#     raise SystemExit
#     n_queries = queries.shape[0]
#     # n_targets = y.shape[1]
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
#         pred_query = nb.predict(queries)
#         pred[:, idx] = pred_query
#     return pred

def fit_nb(nb, query, X, y,
    mode="predict"):
    assert len(y.shape) == 1
    assert query.shape[0] == 1

    if y.all():
        return (1 if mode in ("predict", "prob")
            else 0)
    if (1-y).all():
        return (0 if mode in ("predict", "prob")
            else -np.inf)
    nb.fit(X, y)
    return (nb.predict(query)[0] if mode=="predict"
        else np.predict_proba(query)[0,1] if mode=="prob"
        else np.predict_log_proba(query)[0,1])

def local_nb_prediction(nb, queries, X, y,
    mode="predict"):

    n_queries = queries.shape[0]
    # n_targets = Y.shape[-1]
    assert len(y.shape) == 2
    predictions = np.zeros((n_queries, ))

    # if len(X.shape) == 3:
    for query_idx in range(n_queries):

        predictions[query_idx] = fit_nb(nb, 
            queries[query_idx:query_idx+1],
            X[query_idx],
            y[query_idx],
            mode=mode)

        # print ("completed fitting NB for query",
            # query_idx+1)
           
    # else:
    #     predictions = fit_nb(nb, 
    #         queries, X, y)
    return predictions

class StackedPPB2(BaseEstimator, ClassifierMixin):  
    """Stacked PPB2 model"""

    def __init__(self, 
        fps=["morg2", "rdk", "maccs"], 
        models=["nn+nb", ]):

        classifiers = [
            ("{}-{}".format(fp, model), 
                PPB2(fp=fp, model_name=model))
            for fp, model in itertools.product(fps, models)
        ]
        assert len(classifiers) == len(fps) * len(models)

        print ("building stacked PPB2 classifier",
            "using the following models:", classifiers)

        self.model = StackingClassifier(classifiers,
            # final_estimator=LogisticRegression()
        )

    def fit(self, X, y):
        """
       
        """
        assert isinstance(X, pd.Series)
        assert (X.dtype==pd.StringDtype()), "X should be a vector of smiles"

        self.model.fit(X, y)

        return self

    def predict(self, X):
        assert isinstance(X, pd.Series)
        assert (X.dtype==pd.StringDtype()), "X should be a vector of smiles"

        return self.model.predict(X)

    def predict_proba(self, X):
        assert isinstance(X, pd.Series)
        assert (X.dtype==pd.StringDtype()), "X should be a vector of smiles"

        return self.model.predict_proba(X)

    def predict_log_proba(self, X):
        assert isinstance(X, pd.Series)
        assert (X.dtype==pd.StringDtype()), "X should be a vector of smiles"

        return self.model.predict_log_proba(X)

class PPB2(BaseEstimator, ClassifierMixin):  
    """PPB2 model"""
    
    def __init__(self, 
        fp="morg2", 
        model_name="nn+nb", 
        k=200):
        self.fp = fp
        self.model_name = model_name
        self.k = k
        
        if "nn" in model_name:
            self.model = KNeighborsClassifier(
                n_neighbors=k,
                metric="jaccard", 
                # algorithm="brute", 
                n_jobs=-1)
        elif model_name == "nb":
            self.model = BernoulliNB(alpha=1.)
        elif model_name == "svc":
            self.model = SVC(probability=True)
        elif _name == "bag":
            self.model = BaggingClassifier(n_jobs=-1)
        elif mode_name == "lr":
            self.model = LogisticRegressionCV(n_jobs=-1)
        
    def fit(self, X, y):
        """
        """
        assert isinstance(X, pd.Series)
        assert (X.dtype==pd.StringDtype()), "X should be a vector of smiles"

        assert len(y.shape) == 1, "single target only"

        assert X.shape[0] == y.shape[0]

        if  self.model_name == "nn+nb":
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

        k_nearest_samples = training_samples[idx]
        k_nearest_labels = training_labels[idx]

        return k_nearest_samples, k_nearest_labels

 
    def predict(self, X):
        print ("predicting for", X.shape[0], 
            "query molecules")
        X = compute_fp(X, self.fp)
        if self.model_name == "nn+nb":

            k_nearest_samples, k_nearest_labels = \
                self._determine_k_closest_samples(X)
           
            return local_nb_prediction(BernoulliNB(alpha=1.),
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
           
            return local_nb_prediction(BernoulliNB(alpha=1.),
                X,
                k_nearest_samples,
                k_nearest_labels,
                mode="prob")
        else:
            return self.model.predict_proba(X)

    def predict_log_proba(self, X):
        print ("predicting log probabilities for", X.shape[0], 
            "query molecules")
        X = compute_fp(X, self.fp)
        if self.model_name == "nn+nb":

            k_nearest_samples, k_nearest_labels = \
                self._determine_k_closest_samples(X)
           
            return local_nb_prediction(BernoulliNB(alpha=1.),
                X,
                k_nearest_samples,
                k_nearest_labels,
                mode="log")
        else:
            return self.model.predict_log_proba(X)


    def __str__(self):
        return "Model name: " + self.model_name + ", fp: " + self.fp +\
            ", k = {}".format(self.k)