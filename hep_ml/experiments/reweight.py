"""
This module contains reweighting algorithms.
Reweighting is procedure of finding such weights for original distribution,
that make distribution of one or several variables identical in old distribution and target distribution.

Fitted algorithm can be used to predict weights later for any event.

"""
from __future__ import division, print_function, absolute_import

from sklearn.base import BaseEstimator
from scipy.ndimage import gaussian_filter
from ..commonutils import check_sample_weight, weighted_quantile
from .. import gradientboosting as gb
import numpy

__author__ = 'Alex Rogozhnikov'


def bincount_nd(x, weights, shape):
    """
    Does the same thing as numpy.bincount, but allows binning in several integer variables.
    :param x: numpy.array of shape [n_samples, n_features] with non-negative integers
    :param weights: weights of samples, array of shape [n_samples]
    :param shape: shape of result, should be greater, then maximal value
    :return: weighted number of event in each bin, of shape=shape
    """
    assert len(weights) == len(x), 'length of weight is different: {} {}'.format(len(x), len(weights))
    assert x.shape[1] == len(shape), 'wrong length of shape: {} {}'.format(x.shape[1], len(shape))
    maximals = numpy.max(x, axis=0)
    assert numpy.all(maximals < shape), 'smaller shape: {} {}'.format(maximals, shape)
    bin_indices = numpy.zeros(len(x), dtype=int)
    for i, axis_length in enumerate(shape):
        bin_indices *= axis_length
        bin_indices += x[:, i]
    result = numpy.bincount(bin_indices, weights=weights, minlength=numpy.prod(shape))
    return result.reshape(shape)


class ReweighterMixin(object):
    n_features_ = None

    def normalize_input(self, data, weights):
        weights = check_sample_weight(data, sample_weight=weights, normalize=True)
        data = numpy.array(data)
        if len(data.shape) == 1:
            data = data[:, numpy.newaxis]
        if self.n_features_ is None:
            self.n_features_ = data.shape[1]
        assert self.n_features_ == data.shape[1], \
            'number of features is wrong: {} {}'.format(self.n_features_, data.shape[1])
        return data, weights

    def fit(self, original, target, original_weight, target_weight):
        raise NotImplementedError('To be overriden in descendants')

    def predict_weights(self, original, original_weight=None):
        raise NotImplementedError('To be overriden in descendants')


class BinsReweighter(BaseEstimator, ReweighterMixin):
    def __init__(self, n_bins=200, n_neighs=3.):
        """
        Use bins for reweighting.
        :param int n_bins: how many bins to use for each input variable.
        :param int n_neighs: size of smearing

        This method works fine 1d/2d histograms, while usually being quite unstable for higher dimensions.
        """
        self.n_percentiles = n_bins
        self.n_neighs = n_neighs

    def compute_bin_indices(self, data):
        bin_indices = []
        for axis, axis_edges in enumerate(self.edges):
            bin_indices.append(numpy.searchsorted(axis_edges, data[:, axis]))
        return numpy.array(bin_indices).T

    def fit(self, original, target, original_weight=None, target_weight=None):
        """
        Prepare reweighting formula by finding coefficients.

        :param original: values from original distribution, array-like of shape [n_samples, n_features]
        :param target: values from target distribution, array-like of shape [n_samples, n_features]
        :param original_weight: weights for samples of original distributions
        :param target_weight: weights for samples of original distributions
        :return: self
        """
        self.n_features_ = None
        original, original_weight = self.normalize_input(original, original_weight)
        target, target_weight = self.normalize_input(target, target_weight)
        target_perc = numpy.linspace(0, 1, self.n_percentiles + 1)[1:-1]
        self.edges = []
        for axis in range(self.n_features_):
            self.edges.append(weighted_quantile(target[:, axis], quantiles=target_perc, sample_weight=target_weight))

        bins_weights = []
        for data, weights in [(original, original_weight), (target, target_weight)]:
            bin_indices = self.compute_bin_indices(data)
            bin_w = bincount_nd(bin_indices, weights=weights, shape=[self.n_percentiles] * self.n_features_)
            bins_weights.append(gaussian_filter(bin_w, sigma=self.n_neighs, truncate=2.5))
        bin_orig_weights, bin_targ_weights = bins_weights
        self.transition = bin_targ_weights / (bin_orig_weights + 1.)
        return self

    def predict_weights(self, original, original_weight=None):
        """
        Returns corrected weights

        :param original: values from original distribution of shape [n_samples, n_features]
        :param original_weight: weights of samples before reweighting.
        :return: numpy.array of shape [n_samples] with new weights.
        """
        original, original_weight = self.normalize_input(original, original_weight)
        bin_indices = self.compute_bin_indices(original)
        results = self.transition[tuple(bin_indices.T)] * original_weight
        return results


class GBReweighter(BaseEstimator, ReweighterMixin):
    def __init__(self,
                 n_estimators=40,
                 learning_rate=0.2,
                 max_depth=4,
                 min_samples_leaf=1000,
                 other_args=None):
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.other_args = other_args

    def fit(self, original, target, original_weight=None, target_weight=None):
        """
        Prepare reweighting formula by finding coefficients.

        :param original: values from original distribution, array-like of shape [n_samples, n_features]
        :param target: values from target distribution, array-like of shape [n_samples, n_features]
        :param original_weight: weights for samples of original distributions
        :param target_weight: weights for samples of original distributions
        :return: self
        """
        self.n_features_ = None
        if self.other_args is None:
            self.other_args = {}
        original, original_weight = self.normalize_input(original, original_weight)
        target, target_weight = self.normalize_input(target, target_weight)
        original_weight /= numpy.sum(original_weight)
        target_weight /= numpy.sum(target_weight)
        self.gb = gb.GradientBoostingClassifier(loss=gb.AdaLossFunction(),
                                                n_estimators=self.n_estimators,
                                                max_depth=self.max_depth,
                                                min_samples_leaf=self.min_samples_leaf,
                                                learning_rate=self.learning_rate,
                                                ** self.other_args)
        data = numpy.vstack([original, target])
        target = numpy.array([0] * len(original) + [1] * len(target))
        weights = numpy.hstack([original_weight, target_weight])
        self.gb.fit(data, target, sample_weight=weights)
        return self

    def predict_weights(self, original, original_weight=None):
        """
        Returns corrected weights

        :param original: values from original distribution of shape [n_samples, n_features]
        :param original_weight: weights of samples before reweighting.
        :return: numpy.array of shape [n_samples] with new weights.
        """
        original, original_weight = self.normalize_input(original, original_weight)
        multipliers = numpy.exp(2. * self.gb.decision_function(original))
        return multipliers * original_weight


class GBReweighterNew(BaseEstimator, ReweighterMixin):
    _reg = 10.

    def __init__(self,
                 n_estimators=100,
                 learning_rate=0.2):
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators

    def fit(self, original, target, original_weight=None, target_weight=None):
        """
        Prepare reweighting formula by finding coefficients.

        :param original: values from original distribution, array-like of shape [n_samples, n_features]
        :param target: values from target distribution, array-like of shape [n_samples, n_features]
        :param original_weight: weights for samples of original distributions
        :param target_weight: weights for samples of original distributions
        :return: self
        """
        self.n_features_ = None
        original, original_weight = self.normalize_input(original, original_weight)
        target, target_weight = self.normalize_input(target, target_weight)
        original_weight /= numpy.sum(original_weight)
        target_weight /= numpy.sum(target_weight)

        data = numpy.vstack([original, target])
        target = numpy.array([0] * len(original) + [1] * len(target))
        weights = numpy.hstack([original_weight, target_weight])
        weights /= numpy.mean(weights)

        _reg = self._reg
        # list: feature, cut, [value0, value1]
        self.estimators = []
        pred = numpy.zeros(len(data))
        for _ in range(self.n_estimators):

            feature_id = numpy.random.choice(range(self.n_features_))
            _p = numpy.argsort(data[:, feature_id])
            _y = target[_p]
            _w = weights[_p] * numpy.exp((1 - _y) * pred[_p])
            left_w_plus = numpy.cumsum(_w * _y)
            left_w_minus = numpy.cumsum(_w * (1 - _y))
            right_w_plus = left_w_plus[-1] - left_w_plus
            right_w_minus = left_w_minus[-1] - left_w_minus
            binned_chi2 = (left_w_minus - left_w_plus) ** 2 / (left_w_minus + left_w_plus + _reg)
            binned_chi2 += (right_w_minus - right_w_plus) ** 2 / (right_w_minus + right_w_plus + _reg)
            _index = numpy.argmax(binned_chi2)
            index = _p[_index]
            optimal_binned_chi2 = numpy.max(binned_chi2)
            thresh = data[index, feature_id]

            left_value = - numpy.log((left_w_minus[_index] + _reg) / (left_w_plus[_index] + _reg))
            right_value = - numpy.log((right_w_minus[_index] + _reg) / (right_w_plus[_index] + _reg))

            estimator = [feature_id, thresh, [left_value, right_value]]
            self.estimators.append(estimator)
            pred += self.learning_rate * self.predict_estimator(data, estimator)

        return self

    def predict_estimator(self, data, estimator):
        feature_id, threshold, values = estimator
        return numpy.take(values, data[:, feature_id] > threshold)

    def predict_weights(self, original, original_weight=None):
        """
        Returns corrected weights

        :param original: values from original distribution of shape [n_samples, n_features]
        :param original_weight: weights of samples before reweighting.
        :return: numpy.array of shape [n_samples] with new weights.
        """
        original, original_weight = self.normalize_input(original, original_weight)
        pred = numpy.zeros(len(original))
        for estimator in self.estimators:
            pred += self.learning_rate * self.predict_estimator(original, estimator)

        multipliers = numpy.exp(pred)
        return multipliers * original_weight