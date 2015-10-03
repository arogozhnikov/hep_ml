from __future__ import division, print_function, absolute_import
import numpy
from hep_ml import splot

__author__ = 'Alex Rogozhnikov'


def test_splot(n_samples=100):
    """
    Checking basic properties of sWeights based on small samples
    """
    for weights in [None, numpy.ones(n_samples), numpy.random.exponential(size=n_samples)]:
        # None is the same as sample_weight = 1
        real_weights = weights
        if real_weights is None:
            real_weights = numpy.ones(n_samples)

        for n_classes in [2, 3, 4, 5]:
            p = numpy.random.random([n_samples, n_classes])
            p /= numpy.sum(p, axis=1, keepdims=True)

            sWeights = splot.compute_sweights(probabilities=p, sample_weight=weights)
            initial_stats = numpy.sum(p * real_weights[:, numpy.newaxis], axis=0)

            # How much inside each bin
            prob_collection = numpy.dot(sWeights.T, p)
            assert numpy.allclose(prob_collection, numpy.diag(initial_stats)), \
                'wrong reconstruction after reweighting (biased)'
            assert numpy.allclose(sWeights.sum(axis=0), initial_stats), \
                'sum of sWeights should be equal to weighted amount of expected samples'
            assert numpy.allclose(sWeights.sum(axis=1), real_weights), \
                'sum of sWeights should be equal to initial weight of sample'
