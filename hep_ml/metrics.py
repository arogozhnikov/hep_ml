"""
About

this module contains different metrics of uniformity
and the metrics of quality as well (which support weights, actually)

"""

from __future__ import division, print_function

import numpy
from sklearn.base import BaseEstimator
from sklearn.neighbors.unsupervised import NearestNeighbors

from .commonutils import take_features, check_xyw, weighted_quantile
from . import metrics_utils as ut


__author__ = 'Alex Rogozhnikov'

"""
README on flatness

this metrics are unfortunately more complicated than usual ones
and require more information: not only predictions and classes,
but also mass (or other variables along which we want to hav uniformity)

Here we compute the different metrics of uniformity of predictions:

SDE  - the standard deviation of efficiency
Theil- Theil index of Efficiency (Theil index is used in economics)
KS   - based on Kolmogorov-Smirnov distance between distributions
CVM  - based on Cramer-von Mises similarity between distributions

uniform_label:
  1, if you want to measure non-uniformity in signal predictions
  0, if background.

Metrics are following `REP` conventions (first fit, then compute metrics on same dataset).
For these metrics 'fit' stage is crucial, since it precomputes information from dataset X,
which is quite long and better to do this once.
"""


class AbstractMetric(BaseEstimator):
    def fit(self, X, y, sample_weight=None):
        """
        If metrics needs some initial heavy computations,
        this can be done here.
        interface is the same as for all REP estimators
        """
        return self

    def __call__(self, y, proba, sample_weight):
        """
        Compute value of metrics
        :param proba: numpy.array of shape [n_samples, n_classes]
            with predicted probabilities (typically returned by predict_proba)
        Events should be passed in the same order, as to method fit
        """
        raise NotImplementedError('To be derived by descendant')


class AbstractBinMetric(AbstractMetric):
    def __init__(self, n_bins, uniform_features, uniform_label=0):
        """
        Abstract class for bin-based metrics of uniformity.

        :param n_bins: int, number of bins along each axis
        :param uniform_features: list of strings, features along which uniformity is desired ()
        :param uniform_label: int, label of class in which uniformity is desired
            (typically, 0 is bck, 1 is signal)
        """
        self.uniform_label = uniform_label
        self.uniform_features = uniform_features
        self.n_bins = n_bins

    def fit(self, X, y, sample_weight=None):
        """ Prepare different things for fast computation of metrics """
        X, y, sample_weight = check_xyw(X, y, sample_weight=sample_weight)
        self._mask = numpy.array(y == self.uniform_label)
        assert sum(self._mask) > 0, 'No event of class, along which uniformity is desired'
        self._masked_weight = sample_weight[self._mask]

        X_part = numpy.array(take_features(X, self.uniform_features))[self._mask, :]
        self._bin_indices = ut.compute_bin_indices(X_part=X_part, n_bins=self.n_bins)
        self._bin_weights = ut.compute_bin_weights(bin_indices=self._bin_indices,
                                                   sample_weight=self._masked_weight)
        return self


class BinBasedSDE(AbstractBinMetric):
    def __init__(self, uniform_features, n_bins=10, uniform_label=0, target_rcp=None, power=2.):
        AbstractBinMetric.__init__(self, n_bins=n_bins,
                                   uniform_features=uniform_features,
                                   uniform_label=uniform_label)
        self.power = power
        self.target_rcp = target_rcp

    def __call__(self, y, proba, sample_weight):
        y_pred = proba[self._mask, self.uniform_label]
        if self.target_rcp is None:
            self.target_rcp = [0.5, 0.6, 0.7, 0.8, 0.9]

        result = 0.
        cuts = weighted_quantile(y_pred, self.target_rcp, sample_weight=self._masked_weight)
        for cut in cuts:
            bin_efficiencies = ut.compute_bin_efficiencies(y_pred, bin_indices=self._bin_indices,
                                                           cut=cut, sample_weight=self._masked_weight)
            result += ut.weighted_deviation(bin_efficiencies, weights=self._bin_weights, power=self.power)

        return (result / len(cuts)) ** (1. / self.power)


class BinBasedTheil(AbstractBinMetric):
    def __init__(self, uniform_features, n_bins=10, uniform_label=0, target_rcp=None, power=2.):
        AbstractBinMetric.__init__(self, n_bins=n_bins,
                                   uniform_features=uniform_features,
                                   uniform_label=uniform_label)
        self.power = power
        self.target_rcp = target_rcp

    def __call__(self, y, proba, sample_weight):
        y_pred = proba[self._mask, self.uniform_label]
        if self.target_rcp is None:
            self.target_rcp = [0.5, 0.6, 0.7, 0.8, 0.9]

        result = 0.
        cuts = weighted_quantile(y_pred, self.target_rcp, sample_weight=self._masked_weight)
        for cut in cuts:
            bin_efficiencies = ut.compute_bin_efficiencies(y_pred, bin_indices=self._bin_indices,
                                                           cut=cut, sample_weight=self._masked_weight)
            result += ut.theil(bin_efficiencies, weights=self._bin_weights)
        return result / len(cuts)


class BinBasedCvM(AbstractBinMetric):
    def __init__(self, uniform_features, n_bins=10, uniform_label=0, power=2.):
        AbstractBinMetric.__init__(self, n_bins=n_bins,
                                   uniform_features=uniform_features,
                                   uniform_label=uniform_label)
        self.power = power

    def __call__(self, y, proba, sample_weight):
        y_pred = proba[self._mask, self.uniform_label]
        global_data, global_weight, global_cdf = ut.prepare_distribution(y_pred, weights=self._masked_weight)

        result = 0.
        for bin, bin_weight in enumerate(self._bin_weights):
            if bin_weight <= 0:
                continue
            bin_mask = self._bin_indices == bin
            local_distribution = y_pred[bin_mask]
            local_weights = self._masked_weight[bin_mask]
            result += bin_weight * ut._cvm_2samp_fast(global_data, local_distribution,
                                                      global_weight, local_weights, global_cdf)
        return result


class AbstractKnnMetric(AbstractMetric):
    def __init__(self, uniform_features, n_neighbours=50, uniform_label=0):
        """
        Abstract class for knn-based metrics of uniformity.

        :param n_neighbours: int, number of neighbours
        :param uniform_features: list of strings, features along which uniformity is desired ()
        :param uniform_label: int, label of class in which uniformity is desired
            (typically, 0 is bck, 1 is signal)
        """
        self.uniform_label = uniform_label
        self.uniform_features = uniform_features
        self.n_neighbours = n_neighbours

    # noinspection PyAttributeOutsideInit
    def fit(self, X, y, sample_weight=None):
        """ Prepare different things for fast computation of metrics """
        X, y, sample_weight = check_xyw(X, y, sample_weight=sample_weight)
        self._label_mask = numpy.array(y == self.uniform_label)
        assert sum(self._label_mask) > 0, 'No events of uniform class!'
        # weights of events
        self._masked_weight = sample_weight[self._label_mask]

        X_part = numpy.array(take_features(X, self.uniform_features))[self._label_mask, :]
        # computing knn indices
        neighbours = NearestNeighbors(n_neighbors=self.n_neighbours, algorithm='kd_tree').fit(X_part)
        _, self._groups_indices = neighbours.kneighbors(X_part)
        self._group_matrix = ut.group_indices_to_groups_matrix(self._groups_indices, n_events=len(X_part))
        # self._group_weights = ut.compute_group_weights_by_indices(self._groups_indices,
        #                                                           sample_weight=self._masked_weight)
        self._group_weights = ut.compute_group_weights(self._group_matrix, sample_weight=self._masked_weight)
        # self._divided_weights = ut.compute_divided_weight_by_indices(self._groups_indices,
        #                                                              sample_weight=self._masked_weight)
        self._divided_weights = ut.compute_divided_weight(self._group_matrix, sample_weight=self._masked_weight)
        return self


class KnnBasedSDE(AbstractKnnMetric):
    def __init__(self, uniform_features, n_neighbours=50, uniform_label=0, target_rcp=None, power=2.):
        AbstractKnnMetric.__init__(self, n_neighbours=n_neighbours,
                                   uniform_features=uniform_features,
                                   uniform_label=uniform_label)
        self.power = power
        self.target_rcp = target_rcp

    def __call__(self, y, proba, sample_weight):
        y_pred = proba[self._label_mask, self.uniform_label]
        if self.target_rcp is None:
            self.target_rcp = [0.5, 0.6, 0.7, 0.8, 0.9]
        self.target_rcp = numpy.array(self.target_rcp)

        result = 0.
        cuts = weighted_quantile(y_pred, quantiles=1 - self.target_rcp, sample_weight=self._masked_weight)
        for cut in cuts:
            groups_efficiencies = ut.compute_group_efficiencies(y_pred, groups_matrix=self._group_matrix, cut=cut,
                                                                divided_weight=self._divided_weights)
            result += ut.weighted_deviation(groups_efficiencies, weights=self._group_weights, power=self.power)
        return (result / len(cuts)) ** (1. / self.power)


class KnnBasedTheil(AbstractKnnMetric):
    def __init__(self, uniform_features, n_neighbours=50, uniform_label=0, target_rcp=None, power=2.):
        AbstractKnnMetric.__init__(self, n_neighbours=n_neighbours,
                                   uniform_features=uniform_features,
                                   uniform_label=uniform_label)
        self.target_rcp = target_rcp

    def __call__(self, y, proba, sample_weight):
        y_pred = proba[self._label_mask, self.uniform_label]
        if self.target_rcp is None:
            self.target_rcp = [0.5, 0.6, 0.7, 0.8, 0.9]
        self.target_rcp = numpy.array(self.target_rcp)

        result = 0.
        cuts = weighted_quantile(y_pred, quantiles=1 - self.target_rcp, sample_weight=self._masked_weight)
        for cut in cuts:
            groups_efficiencies = ut.compute_group_efficiencies(y_pred, groups_matrix=self._group_matrix, cut=cut,
                                                                divided_weight=self._divided_weights)
            result += ut.theil(groups_efficiencies, weights=self._group_weights)
        return result / len(cuts)


class KnnBasedCvM(AbstractKnnMetric):
    def __init__(self, uniform_features, n_neighbours=50, uniform_label=0, power=2.):
        AbstractKnnMetric.__init__(self, n_neighbours=n_neighbours,
                                   uniform_features=uniform_features,
                                   uniform_label=uniform_label)
        self.power = power

    def __call__(self, y, proba, sample_weight):
        y_pred = proba[self._label_mask, self.uniform_label]

        result = 0.
        global_data, global_sample_weight, global_cdf = ut.prepare_distribution(y_pred, weights=self._masked_weight)
        for group, group_weight in zip(self._groups_indices, self._group_weights):
            local_distribution = y_pred[group]
            local_sample_weights = self._masked_weight[group]
            result += group_weight * ut._cvm_2samp_fast(global_data, local_distribution,
                                                        global_sample_weight, local_sample_weights, global_cdf)
        return result



