from __future__ import division, print_function, absolute_import

import numpy

from hep_ml.reweight import BinsReweighter, GBReweighter, FoldingReweighter
from hep_ml.metrics_utils import ks_2samp_weighted

__author__ = 'Alex Rogozhnikov'


def weighted_covariance(data, weights):
    if len(data.shape) == 1:
        data = data[:, numpy.newaxis]
    data = data - numpy.mean(data, axis=0, keepdims=True)
    weights = weights * 1. / weights.sum()
    return numpy.einsum('ij, ik, i -> jk', data, data, weights)


def check_reweighter(n_dimensions, n_samples, reweighter, folding=False):
    mean_original = numpy.random.normal(size=n_dimensions)
    cov_original = numpy.diag([1.] * n_dimensions)

    mean_target = numpy.random.mtrand.multivariate_normal(mean=mean_original, cov=cov_original)
    cov_target = cov_original * 0.4 + numpy.ones([n_dimensions, n_dimensions]) * 0.2

    original = numpy.random.mtrand.multivariate_normal(mean=mean_original, cov=cov_original, size=n_samples + 1)
    original_weight = numpy.ones(n_samples + 1)

    target = numpy.random.mtrand.multivariate_normal(mean=mean_target, cov=cov_target, size=n_samples)
    target_weight = numpy.ones(n_samples)

    reweighter.fit(original, target, original_weight=original_weight, target_weight=target_weight)
    new_weights_array = []
    new_weights_array.append(reweighter.predict_weights(original, original_weight=original_weight))
    if folding:
        def mean_vote(x):
            return numpy.mean(x, axis=0)

        new_weights_array.append(reweighter.predict_weights(original, original_weight=original_weight,
                                                            vote_function=mean_vote))

    for new_weights in new_weights_array:
        av_orig = numpy.average(original, weights=original_weight, axis=0)
        print('WAS', av_orig)
        av_now = numpy.average(original, weights=new_weights, axis=0)
        print('NOW:', av_now)
        av_ideal = numpy.average(target, weights=target_weight, axis=0)
        print('IDEAL:', av_ideal)

        print('COVARIANCE')
        print('WAS', weighted_covariance(original, original_weight))
        print('NOW', weighted_covariance(original, new_weights))
        print('IDEAL', weighted_covariance(target, target_weight))

        assert numpy.all(abs(av_now - av_ideal) < abs(av_orig - av_ideal)), 'averages are wrong'
        for dim in range(n_dimensions):
            diff1 = ks_2samp_weighted(original[:, dim], target[:, dim], original_weight, target_weight)
            diff2 = ks_2samp_weighted(original[:, dim], target[:, dim], new_weights, target_weight)
            print('KS', diff1, diff2)
            assert diff2 < diff1, 'Differences {} {}'.format(diff1, diff2)


def test_reweighter_1d():
    reweighter = BinsReweighter(n_bins=200, n_neighs=2)
    check_reweighter(n_dimensions=1, n_samples=100000, reweighter=reweighter)


def test_gb_reweighter_1d():
    reweighter = GBReweighter(n_estimators=100, max_depth=2)
    check_reweighter(n_dimensions=1, n_samples=100000, reweighter=reweighter)


def test_reweighter_2d():
    reweighter = BinsReweighter(n_bins=20, n_neighs=2)
    check_reweighter(n_dimensions=2, n_samples=1000000, reweighter=reweighter)


def test_gb_reweighter_2d():
    reweighter = GBReweighter(max_depth=3, n_estimators=30, learning_rate=0.3, gb_args=dict(subsample=0.3))
    check_reweighter(n_dimensions=2, n_samples=200000, reweighter=reweighter)


def test_folding_gb_reweighter():
    reweighter = FoldingReweighter(GBReweighter(n_estimators=20, max_depth=2, learning_rate=0.1), n_folds=3)
    check_reweighter(n_dimensions=2, n_samples=200000, reweighter=reweighter, folding=True)


def test_folding_bins_reweighter():
    reweighter = FoldingReweighter(BinsReweighter(n_bins=20, n_neighs=2), n_folds=3)
    check_reweighter(n_dimensions=2, n_samples=1000000, reweighter=reweighter, folding=True)
