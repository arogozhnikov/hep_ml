from __future__ import division, print_function, absolute_import

import numpy

from hep_ml.experiments.reweight import BinsReweighter, GBReweighter
from hep_ml.metrics_utils import ks_2samp_weighted

__author__ = 'Alex Rogozhnikov'


def weighted_covar(data, weights):
    if len(data.shape) == 1:
        data = data[:, numpy.newaxis]
    data = data - numpy.mean(data, axis=0, keepdims=True)
    weights = weights * 1. / weights.sum()
    return numpy.einsum('ij, ik, i -> jk', data, data, weights)


def check_reweighter(dimension, n_samples, reweighter):
    mean_original = numpy.random.normal(size=dimension)
    cov_original = numpy.diag([1.] * dimension)

    mean_target = numpy.random.mtrand.multivariate_normal(mean=mean_original, cov=cov_original)
    cov_target = cov_original * 0.4 + numpy.ones([dimension, dimension]) * 0.2

    original = numpy.random.mtrand.multivariate_normal(mean=mean_original, cov=cov_original, size=n_samples + 1)
    original_weight = numpy.ones(n_samples + 1)

    target = numpy.random.mtrand.multivariate_normal(mean=mean_target, cov=cov_target, size=n_samples)
    target_weight = numpy.ones(n_samples)

    reweighter.fit(original, target, original_weight=original_weight, target_weight=target_weight)
    new_weights = reweighter.predict_weights(original, original_weight=original_weight)

    av_orig = numpy.average(original, weights=original_weight, axis=0)
    print('WAS', av_orig)
    av_now = numpy.average(original, weights=new_weights, axis=0)
    print('NOW:', av_now)
    av_ideal = numpy.average(target, weights=target_weight, axis=0)
    print('IDEAL:', av_ideal)

    print('COVARIATION')
    print('WAS', weighted_covar(original, original_weight))
    print('NOW', weighted_covar(original, new_weights))
    print('IDEAL', weighted_covar(target, target_weight))

    assert numpy.all(abs(av_now - av_ideal) < abs(av_orig - av_ideal)), 'deviation is wrong'
    if dimension == 1:
        diff1 = ks_2samp_weighted(original.flatten(), target.flatten(), original_weight, target_weight)
        diff2 = ks_2samp_weighted(original.flatten(), target.flatten(), new_weights, target_weight)
        assert diff2 < diff1, 'Differences {} {}'.format(diff1, diff2)


def test_reweighter_1d():
    reweighter = BinsReweighter(n_bins=200, n_neighs=2)
    check_reweighter(dimension=1, n_samples=100000, reweighter=reweighter)


def test_gb_reweighter_1d():
    reweighter = GBReweighter(n_estimators=100, max_depth=2)
    check_reweighter(dimension=1, n_samples=100000, reweighter=reweighter)


def test_reweighter_2d():
    reweighter = BinsReweighter(n_bins=20, n_neighs=2)
    check_reweighter(dimension=2, n_samples=1000000, reweighter=reweighter)


def test_gb_reweighter_2d():
    reweighter = GBReweighter(max_depth=3, n_estimators=30, learning_rate=0.3, other_args=dict(subsample=0.3))
    check_reweighter(dimension=2, n_samples=1000000, reweighter=reweighter)
