"""
**hep_ml.reweight** contains reweighting algorithms.

Reweighting is procedure of finding such weights for original distribution,
that make distribution of one or several variables identical in original distribution and target distribution.

Typical application of this technique in HEP is reweighting of Monte-Carlo simulation results to minimize
disagreement between simulated data and real data.
Frequently the reweighting rule is trained on one part of data (normalization channel)
and applied to different (signal channel).

Remark: if each variable has identical distribution in two samples,
this doesn't imply that multidimensional distributions are equal (almost surely they aren't).
Aim of reweighters is to get identical multidimensional distributions.

Algorithms are implemented as estimators, fitting and reweighting stages are split.
Fitted reweighter can be applied many times to different data, pickled and so on.


Examples
________

The most common use case is reweighting of Monte-Carlo simulations results to sPlotted real data.
(original weights are all equal to 1 and could be skipped, but left here for example)

>>> from hep_ml.reweight import BinsReweighter, GBReweighter
>>> original_weights = numpy.ones(len(MC_data))
>>> reweighter = BinsReweighter(n_bins=100, n_neighs=3)
>>> reweighter.fit(original=MC_data, target=RealData,
>>>                original_weight=original_weights, target_weight=sWeights)
>>> MC_weights = reweighter.predict_weights(MC_data, original_weight=original_weights)

The same example for `GBReweighter`:

>>> reweighter = GBReweighter(max_depth=2, gb_args={'subsample': 0.5})
>>> reweighter.fit(original=MC_data, target=RealData, target_weight=sWeights)
>>> MC_weights = reweighter.predict_weights(MC_data)

"""
from __future__ import division, print_function, absolute_import

from sklearn.base import BaseEstimator
from scipy.ndimage import gaussian_filter
import numpy

from .commonutils import check_sample_weight, weighted_quantile
from . import gradientboosting as gb
from . import losses

__author__ = 'Alex Rogozhnikov'
__all__ = ['BinsReweighter', 'GBReweighter']


def _bincount_nd(x, weights, shape):
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

    result = numpy.zeros(shape, dtype=float)
    numpy.add.at(result, tuple(x.T), weights)
    return result


class ReweighterMixin(object):
    """Supplementary class which shows the interface of reweighter.
     Reweighters should be derived from this class."""
    n_features_ = None

    def _normalize_input(self, data, weights):
        """ Normalize input of reweighter
        :param data: array like of shape [n_samples] or [n_samples, n_features]
        :param weights: array-like of shape [n_samples] or None
        :return: tuple with
            data - numpy.array of shape [n_samples, n_features]
            weights - numpy.array of shape [n_samples] with mean = 1.
        """
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
        Use bins for reweighting. Bins' edges are computed using quantiles along each axis
        (which is better than bins of even size).

        This method works fine for 1d/2d histograms,
        while being unstable or inaccurate for higher dimensions.

        To make computed rule more smooth and stable, after computing weights in bins,
        gaussian filter is applied (so reweighting coefficient also includes information from neighbouring bins).

        :param int n_bins: how many bins to use for each input variable.
        :param float n_neighs: size of gaussian filter (in bins).
            This parameter is responsible for tradeoff between stability of rule and accuracy of predictions.
            With increase of n_neighs the reweighting rule becomes more stable.

        """
        self.n_bins = n_bins
        self.n_neighs = n_neighs
        # if number of events in bins is less than this value, number of events is clipped.
        self.min_in_the_bin = 1.

    def compute_bin_indices(self, data):
        """
        Compute id of bin along each axis.

        :param data: data, array-like of shape [n_samples, n_features]
            with the same order of features as in training
        :return: numpy.array of shape [n_samples, n_features] with integers, each from [0, n_bins - 1]
        """
        bin_indices = []
        for axis, axis_edges in enumerate(self.edges):
            bin_indices.append(numpy.searchsorted(axis_edges, data[:, axis]))
        return numpy.array(bin_indices).T

    def fit(self, original, target, original_weight=None, target_weight=None):
        """
        Prepare reweighting formula by computing histograms.

        :param original: values from original distribution, array-like of shape [n_samples, n_features]
        :param target: values from target distribution, array-like of shape [n_samples, n_features]
        :param original_weight: weights for samples of original distributions
        :param target_weight: weights for samples of original distributions
        :return: self
        """
        self.n_features_ = None
        original, original_weight = self._normalize_input(original, original_weight)
        target, target_weight = self._normalize_input(target, target_weight)
        target_perc = numpy.linspace(0, 1, self.n_bins + 1)[1:-1]
        self.edges = []
        for axis in range(self.n_features_):
            self.edges.append(weighted_quantile(target[:, axis], quantiles=target_perc, sample_weight=target_weight))

        bins_weights = []
        for data, weights in [(original, original_weight), (target, target_weight)]:
            bin_indices = self.compute_bin_indices(data)
            bin_w = _bincount_nd(bin_indices, weights=weights, shape=[self.n_bins] * self.n_features_)
            smeared_weights = gaussian_filter(bin_w, sigma=self.n_neighs, truncate=2.5)
            bins_weights.append(smeared_weights.clip(self.min_in_the_bin))
        bin_orig_weights, bin_targ_weights = bins_weights
        self.transition = bin_targ_weights / bin_orig_weights
        return self

    def predict_weights(self, original, original_weight=None):
        """
        Returns corrected weights. Result is computed as original_weight * reweighter_multipliers.

        :param original: values from original distribution of shape [n_samples, n_features]
        :param original_weight: weights of samples before reweighting.
        :return: numpy.array of shape [n_samples] with new weights.
        """
        original, original_weight = self._normalize_input(original, original_weight)
        bin_indices = self.compute_bin_indices(original)
        results = self.transition[tuple(bin_indices.T)] * original_weight
        return results


class GBReweighter(BaseEstimator, ReweighterMixin):
    def __init__(self,
                 n_estimators=40,
                 learning_rate=0.2,
                 max_depth=3,
                 min_samples_leaf=200,
                 gb_args=None):
        """
        Gradient Boosted Reweighter - a reweighter algorithm based on ensemble of regression trees.
        Parameters have the same role, as in gradient boosting.
        Special loss function is used, trees are trained to maximize symmetrized binned chi-squared statistics.

        Training takes much more time than for bin-based versions, but `GBReweighter` is capable
        to work in high dimensions while keeping reweighting rule reliable and precise
        (and even smooth if many trees are used).

        :param n_estimators: number of trees
        :param learning_rate: float from [0, 1]. Lesser learning rate requires more trees,
            but makes reweighting rule more stable.
        :param max_depth: maximal depth of trees
        :param min_samples_leaf: minimal number of events in the leaf. If many
        :param gb_args: other parameters passed to gradient boosting.
            See :class:`hep_ml.gradientboosting.UGradientBoostingClassifier`
        """
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.gb_args = gb_args

    def fit(self, original, target, original_weight=None, target_weight=None):
        """
        Prepare reweighting formula by training sequence of trees.

        :param original: values from original distribution, array-like of shape [n_samples, n_features]
        :param target: values from target distribution, array-like of shape [n_samples, n_features]
        :param original_weight: weights for samples of original distributions
        :param target_weight: weights for samples of original distributions
        :return: self
        """
        self.n_features_ = None
        if self.gb_args is None:
            self.gb_args = {}
        original, original_weight = self._normalize_input(original, original_weight)
        target, target_weight = self._normalize_input(target, target_weight)

        self.gb = gb.UGradientBoostingClassifier(loss=losses.ReweightLossFunction(),
                                                 n_estimators=self.n_estimators,
                                                 max_depth=self.max_depth,
                                                 min_samples_leaf=self.min_samples_leaf,
                                                 learning_rate=self.learning_rate,
                                                 **self.gb_args)
        data = numpy.vstack([original, target])
        target = numpy.array([1] * len(original) + [0] * len(target))
        weights = numpy.hstack([original_weight, target_weight])
        self.gb.fit(data, target, sample_weight=weights)
        return self

    def predict_weights(self, original, original_weight=None):
        """
        Returns corrected weights. Result is computed as original_weight * reweighter_multipliers.

        :param original: values from original distribution of shape [n_samples, n_features]
        :param original_weight: weights of samples before reweighting.
        :return: numpy.array of shape [n_samples] with new weights.
        """
        original, original_weight = self._normalize_input(original, original_weight)
        multipliers = numpy.exp(self.gb.decision_function(original))
        return multipliers * original_weight
