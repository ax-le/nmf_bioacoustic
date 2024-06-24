import sklearn.feature_selection
import numpy as np
from sklearn.cluster import DBSCAN # HDBSCAN or OPTICS also possible, to test
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score


class SourceCountEstimator(BaseEstimator):

    def __init__(self, *, var_divide=10, eps=0.8, metric='correlation'):

        self.var_divide = var_divide
        self.eps = eps
        self.metric = metric

    def fit(self, all_H, annotations):
        all_estimations = []
        for H in all_H:
            estimation = estimate_number_sources(H, var_divide=self.var_divide, eps=self.eps, metric=self.metric)
            all_estimations.append(estimation)

        return all_estimations
    
    def predict(self, all_H):
        all_estimations = []
        for H in all_H:
            estimation = estimate_number_sources(H, var_divide=self.var_divide, eps=self.eps, metric=self.metric)
            all_estimations.append(estimation)

        return all_estimations

    def score(self, estimations, annotations):

        return accuracy_score(annotations, estimations)


def estimate_number_sources(H, var_divide = 10, eps = 0.7, metric = 'correlation'):
    # This function is used to return the number of species in the current audio file

    # First, remove all features with a variance lower than the mean variance divided by 10
    H_cropped = threshold_H(H, var_divide=var_divide)

    # Then, we count the number of species
    number_species = DBSCAN_count(H_cropped, eps=eps, metric=metric)

    return number_species

def threshold_H(H, var_divide=10):
    var = np.var(H, axis=1)
    threshold = np.mean(var)/var_divide

    # Maybe tune the threshold not to be too low, it can lead to massive data loss
    thresh = sklearn.feature_selection.VarianceThreshold(threshold=threshold)
    H_cropped = thresh.fit_transform(H.T).T

    return H_cropped

def DBSCAN_count(H, eps = 0.7, metric = 'correlation'): # jensenshannon, mahalanobis, cityblock
    db = DBSCAN(eps=eps, min_samples=1, metric=metric)
    db.fit(H)
    labels = db.labels_
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    return n_clusters_

def compute_difference(estimation, annotation):
    return np.abs(estimation - annotation)
