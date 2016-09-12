"""
tests for hep_ml.speedup module
"""
from __future__ import division, print_function, absolute_import

import numpy
import pandas
import time
from hep_ml.speedup import LookupClassifier
from hep_ml.commonutils import generate_sample
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from collections import OrderedDict
from nose.tools import raises

__author__ = 'Alex Rogozhnikov'


def test_lookup(n_samples=10000, n_features=7, n_bins=8):
    X, y = generate_sample(n_samples=n_samples, n_features=n_features, distance=0.6)

    base_estimator = GradientBoostingClassifier()
    clf = LookupClassifier(base_estimator=base_estimator, n_bins=n_bins, keep_trained_estimator=True).fit(X, y)
    p = clf.predict_proba(X)
    assert roc_auc_score(y, p[:, 1]) > 0.8, 'quality of classification is too low'
    assert p.shape == (n_samples, 2)
    assert numpy.allclose(p.sum(axis=1), 1), 'probabilities are not summed up to 1'

    # checking conversions
    lookup_size = n_bins ** n_features
    lookup_indices = numpy.arange(lookup_size, dtype=int)
    bins_indices = clf.convert_lookup_index_to_bins(lookup_indices=lookup_indices)
    lookup_indices2 = clf.convert_bins_to_lookup_index(bins_indices=bins_indices)
    assert numpy.allclose(lookup_indices, lookup_indices2), 'something wrong with conversions'
    assert len(clf._lookup_table) == n_bins ** n_features, 'wrong size of lookup table'

    # checking speed
    X = pandas.concat([X] * 10)
    start = time.time()
    p1 = clf.trained_estimator.predict_proba(clf.transform(X))
    time_old = time.time() - start
    start = time.time()
    p2 = clf.predict_proba(X)
    time_new = time.time() - start
    print(time_old, ' now takes ', time_new)
    assert numpy.allclose(p1, p2), "pipeline doesn't work as expected"


def test_sizes(n_samples=10000, n_features=4, n_bins=8):
    X, y = generate_sample(n_samples=n_samples, n_features=n_features, distance=0.6)

    base_estimator = GradientBoostingClassifier(n_estimators=1)
    clf = LookupClassifier(base_estimator=base_estimator, n_bins=n_bins).fit(X, y)

    bin_indices = clf.transform(X)
    assert numpy.allclose(numpy.max(bin_indices, axis=0) + 1, n_bins)

    maximals = OrderedDict()
    for column in X.columns:
        maximals[column] = numpy.random.randint(low=n_bins // 2, high=n_bins)

    clf = LookupClassifier(base_estimator=base_estimator, n_bins=maximals).fit(X, y)
    bin_indices = clf.transform(X)
    assert numpy.allclose(numpy.max(bin_indices, axis=0) + 1, list(maximals.values()))

    assert numpy.allclose(numpy.min(bin_indices, axis=0), 0)


@raises(ValueError)
def test_raising_exception():
    X, y = generate_sample(n_samples=100, n_features=10)
    LookupClassifier(GradientBoostingClassifier(), n_bins=16).fit(X, y)


def test_classifier_with_dataframe():
    try:
        from rep.estimators import SklearnClassifier
        clf = SklearnClassifier(GradientBoostingClassifier(n_estimators=1))
        X, y = generate_sample(n_samples=100, n_features=4)
        for X_ in [X, pandas.DataFrame(X)]:
            lookup = LookupClassifier(clf, n_bins=16).fit(X_, y)
            lookup.predict_proba(X)
    except ImportError:
        print('expected fail: yandex/rep not installed')
