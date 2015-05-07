from __future__ import division, print_function, absolute_import

import numpy
import pandas
from numpy.random.mtrand import RandomState
from scipy.stats import ks_2samp
from hep_ml.commonutils import generate_sample

from hep_ml.metrics_utils import compute_sde_on_bins, \
    compute_sde_on_groups, compute_theil_on_bins, compute_theil_on_groups, \
    prepare_distibution, _ks_2samp_fast, ks_2samp_weighted, bin_based_ks, \
    groups_based_ks, cvm_2samp, _cvm_2samp_fast, bin_based_cvm, group_based_cvm

from hep_ml.metrics import KnnBasedSDE, KnnBasedTheil, KnnBasedCvM, \
    BinBasedSDE, BinBasedTheil, BinBasedCvM

from hep_ml.metrics_utils import bin_to_group_indices, compute_bin_indices
from hep_ml.commonutils import compute_knn_indices_of_signal
import hep_ml.metrics_utils as ut


__author__ = 'Alex Rogozhnikov'

random = RandomState()


def generate_binned_dataset(n_samples, n_bins):
    """useful function, generates dataset with bins, groups, random weights.
    This is used to test correlation functions. """
    random = RandomState()
    y = random.uniform(size=n_samples) > 0.5
    pred = random.uniform(size=(n_samples, 2))
    weights = random.exponential(size=(n_samples,))
    bins = random.randint(0, n_bins, n_samples)
    groups = bin_to_group_indices(bin_indices=bins, mask=(y == 1))
    return y, pred, weights, bins, groups


def test_bin_to_group_indices(size=100, bins=10):
    bin_indices = RandomState().randint(0, bins, size=size)
    mask = RandomState().randint(0, 2, size=size) > 0.5
    group_indices = bin_to_group_indices(bin_indices, mask=mask)
    assert numpy.sum([len(group) for group in group_indices]) == numpy.sum(mask)
    a = numpy.sort(numpy.concatenate(group_indices))
    b = numpy.where(mask > 0.5)[0]
    assert numpy.all(a == b), 'group indices are computed wrongly'


def test_bins(size=500, n_bins=10):
    columns = ['var1', 'var2']
    df = pandas.DataFrame(random.uniform(size=(size, 2)), columns=columns)
    x_limits = numpy.linspace(0, 1, n_bins + 1)[1:-1]
    bins = compute_bin_indices(df[columns].values, bin_limits=[x_limits, x_limits])
    assert numpy.all(0 <= bins) and numpy.all(bins < n_bins * n_bins), "the bins with wrong indices appeared"


def test_compare_sde_computations(n_samples=1000, n_bins=10):
    y, pred, weights, bins, groups = generate_binned_dataset(n_samples=n_samples, n_bins=n_bins)
    target_efficiencies = RandomState().uniform(size=3)
    a = compute_sde_on_bins(pred[:, 1], mask=(y == 1), bin_indices=bins,
                            target_efficiencies=target_efficiencies, sample_weight=weights)
    b = compute_sde_on_groups(pred[:, 1], mask=(y == 1), groups_indices=groups,
                              target_efficiencies=target_efficiencies, sample_weight=weights)
    assert numpy.allclose(a, b)


def test_theil(n_samples=1000, n_bins=10):
    y, pred, weights, bins, groups = generate_binned_dataset(n_samples=n_samples, n_bins=n_bins)
    a = compute_theil_on_bins(pred[:, 1], y == 1, bins, [0.5, 0.78], sample_weight=weights)
    b = compute_theil_on_groups(pred[:, 1], y == 1, groups, [0.5, 0.78], sample_weight=weights)
    assert numpy.allclose(a, b)


def test_ks2samp_fast(size=1000):
    y1 = RandomState().uniform(size=size)
    y2 = y1[RandomState().uniform(size=size) > 0.5]
    a = ks_2samp(y1, y2)[0]
    prep_data, prep_weights, prep_F = prepare_distibution(y1, numpy.ones(len(y1)))
    b = _ks_2samp_fast(prep_data, y2, prep_weights, numpy.ones(len(y2)), F1=prep_F)
    c = _ks_2samp_fast(prep_data, y2, prep_weights, numpy.ones(len(y2)), F1=prep_F)
    d = ks_2samp_weighted(y1, y2, numpy.ones(len(y1)) / 3, numpy.ones(len(y2)) / 4)
    assert numpy.allclose(a, b, rtol=1e-2, atol=1e-3)
    assert numpy.allclose(b, c)
    assert numpy.allclose(b, d)
    print('ks2samp is ok')


def test_ks(n_samples=1000, n_bins=10):
    y, pred, weights, bins, groups = generate_binned_dataset(n_samples=n_samples, n_bins=n_bins)
    mask = y == 1
    a = bin_based_ks(pred[:, 1], mask=mask, sample_weight=weights, bin_indices=bins)
    b = groups_based_ks(pred[:, 1], mask=mask, sample_weight=weights, groups_indices=groups)
    assert numpy.allclose(a, b)


def test_fast_cvm(n_samples=1000):
    random = RandomState()
    data1 = random.uniform(size=n_samples)
    weights1 = random.uniform(size=n_samples)
    mask = random.uniform(size=n_samples) > 0.5
    data2 = data1[mask]
    weights2 = weights1[mask]
    a = cvm_2samp(data1, data2, weights1, weights2)
    prepared_data1, prepared_weights1, F1 = prepare_distibution(data1, weights1)
    b = _cvm_2samp_fast(prepared_data1, data2, prepared_weights1, weights2, F1=F1)
    assert numpy.allclose(a, b)


def test_cvm(size=1000):
    y_pred = random.uniform(size=size)
    y = random.uniform(size=size) > 0.5
    sample_weight = random.exponential(size=size)
    bin_indices = random.randint(0, 10, size=size)
    mask = y == 1
    groups_indices = bin_to_group_indices(bin_indices=bin_indices, mask=mask)
    cvm1 = bin_based_cvm(y_pred[mask], sample_weight=sample_weight[mask], bin_indices=bin_indices[mask])
    cvm2 = group_based_cvm(y_pred, mask=mask, sample_weight=sample_weight, groups_indices=groups_indices)
    assert numpy.allclose(cvm1, cvm2)


def test_cvm_sde_limit(size=2000):
    """ Checks that in the limit CvM coincides with MSE """
    effs = numpy.linspace(0, 1, 2000)
    y_pred = random.uniform(size=size)
    y = random.uniform(size=size) > 0.5
    sample_weight = random.exponential(size=size)
    bin_indices = random.randint(0, 10, size=size)
    y_pred += bin_indices * random.uniform()
    mask = y == 1

    val1 = bin_based_cvm(y_pred[mask], sample_weight=sample_weight[mask], bin_indices=bin_indices[mask])
    val2 = compute_sde_on_bins(y_pred, mask=mask, bin_indices=bin_indices, target_efficiencies=effs,
                               sample_weight=sample_weight)

    assert numpy.allclose(val1, val2 ** 2, atol=1e-3, rtol=1e-2)


def test_new_metrics(n_samples=2000, knn=50):
    X, y = generate_sample(n_samples=n_samples, n_features=10)
    sample_weight = numpy.random.exponential(size=n_samples) ** 0.
    predictions = numpy.random.random(size=[n_samples, 2])
    predictions /= predictions.sum(axis=1, keepdims=True)
    predictions *= 1000.

    # Checking SDE
    features = X.columns[:1]
    sde_val1 = sde(y, predictions, X, uniform_variables=features, sample_weight=sample_weight, label=0, knn=knn)
    sde2 = KnnBasedSDE(n_neighbours=knn, uniform_features=features, uniform_label=0, )
    sde2.fit(X, y, sample_weight=sample_weight)
    sde_val2 = sde2(y, predictions, sample_weight=sample_weight)

    assert sde_val1 == sde_val2, 'SDE values are different'

    # Checking CVM
    features = X.columns[:1]
    cvm_val1 = cvm_flatness(y, predictions, X, uniform_variables=features, sample_weight=sample_weight, label=0,
                            knn=knn)
    cvm2 = KnnBasedCvM(n_neighbours=knn, uniform_features=features, uniform_label=0, )
    cvm2.fit(X, y, sample_weight=sample_weight)
    cvm_val2 = cvm2(y, predictions, sample_weight=sample_weight)

    assert cvm_val1 == cvm_val2, 'CvM values are different'


def test_metrics_clear(n_samples=2000, knn=50, uniform_class=0):
    """
    Testing that after deleting all inappropriate events (events of other class),
    metrics stays the same
    """
    X, y = generate_sample(n_samples=n_samples, n_features=10)
    sample_weight = numpy.random.exponential(size=n_samples)
    predictions = numpy.random.random(size=[n_samples, 2])
    predictions /= predictions.sum(axis=1, keepdims=True)
    features = X.columns[:1]

    mask = (y == uniform_class)
    X_clear = X.ix[mask, :]
    y_clear = y[mask]
    sample_weight_clear = sample_weight[mask]
    predictions_clear = predictions[mask]

    for function in [sde, theil_flatness, cvm_flatness]:
        flatness_val = function(y, predictions, X, uniform_variables=features, sample_weight=sample_weight, label=0,
                                knn=knn)
        flatness_val_clear = function(y_clear, predictions_clear, X_clear, uniform_variables=features,
                                      sample_weight=sample_weight_clear, label=0, knn=knn)
        assert flatness_val == flatness_val_clear, 'after deleting other class, the metrics changed'

    for class_ in [KnnBasedSDE, KnnBasedTheil, KnnBasedCvM]:
        metric1 = class_(n_neighbours=knn, uniform_features=features, uniform_label=0, )
        metric1.fit(X, y, sample_weight=sample_weight)
        flatness_val1 = metric1(y, predictions, sample_weight)

        metric2 = class_(n_neighbours=knn, uniform_features=features, uniform_label=0, )
        metric2.fit(X_clear, y_clear, sample_weight=sample_weight_clear)
        flatness_val2 = metric2(y_clear, predictions_clear, sample_weight_clear)
        assert flatness_val1 == flatness_val2, 'after deleting other class, the metrics changed'


def test_workability(n_samples=2000, knn=50, uniform_label=0, n_bins=10):
    """Simply checks that metrics are working """
    X, y = generate_sample(n_samples=n_samples, n_features=10)
    sample_weight = numpy.random.exponential(size=n_samples)
    predictions = numpy.random.random(size=[n_samples, 2])
    predictions /= predictions.sum(axis=1, keepdims=True)
    features = X.columns[:1]

    for class_ in [KnnBasedSDE, KnnBasedTheil, KnnBasedCvM]:
        metric = class_(n_neighbours=knn, uniform_features=features, uniform_label=uniform_label, )
        metric.fit(X, y, sample_weight=sample_weight)
        flatness_val_ = metric(y, predictions, sample_weight)

    for class_ in [BinBasedSDE, BinBasedTheil, BinBasedCvM]:
        metric = class_(n_bins=n_bins, uniform_features=features, uniform_label=uniform_label, )
        metric.fit(X, y, sample_weight=sample_weight)
        flatness_val_ = metric(y, predictions, sample_weight)


# region Uniformity metrics (old version, reference code for comparison)

"""
Comments on the old interface:

Mask is needed to show the events of needed class,
for instance, if we want to compute the uniformity on signal predictions,
mask should be True on signal events and False on the others.

y_score in usually predicted probabilities of event being a needed class.

So, if I want to compute efficiency on signal, I put:
  mask = y == 1
  y_pred = clf.predict_proba[:, 1]

If want to do it for bck:
  mask = y == 0
  y_pred = clf.predict_proba[:, 0]

"""


def sde(y, proba, X, uniform_variables, sample_weight=None, label=1, knn=30):
    """ The most simple way to compute SDE, this is however very slow
    if you need to recompute SDE many times
    :param y: real classes of events, shape = [n_samples]
    :param proba: predicted probabilities, shape = [n_samples, n_classes]
    :param X: pandas.DataFrame with uniform features
    :param uniform_variables: features, along which uniformity is desired, list of strings
    :param sample_weight: weights of events, shape = [n_samples]
    :param label: class, for which uniformity is measured (usually, 0 is bck, 1 is signal)
    :param knn: number of nearest neighbours used in knn

    Example of usage:
    proba = classifier.predict_proba(testX)
    sde(testY, proba=proba, X=testX, uniform_variables=['mass'])
    """
    assert len(y) == len(proba) == len(X), 'Different lengths'
    X = pandas.DataFrame(X)
    mask = y == label
    groups = compute_knn_indices_of_signal(X[uniform_variables], is_signal=mask, n_neighbours=knn)
    groups = groups[mask, :]

    return ut.compute_sde_on_groups(proba[:, label], mask=mask, groups_indices=groups,
                                    target_efficiencies=[0.5, 0.6, 0.7, 0.8, 0.9], sample_weight=sample_weight)


def theil_flatness(y, proba, X, uniform_variables, sample_weight=None, label=1, knn=30):
    """This is ready-to-use function, and it is quite slow to use many times"""

    mask = y == label
    groups_indices = compute_knn_indices_of_signal(X[uniform_variables], is_signal=mask, n_neighbours=knn)[mask, :]
    return ut.compute_theil_on_groups(proba[:, label], mask=mask, groups_indices=groups_indices,
                                      target_efficiencies=[0.5, 0.6, 0.7, 0.8, 0.9], sample_weight=sample_weight)


def cvm_flatness(y, proba, X, uniform_variables, sample_weight=None, label=1, knn=30):
    """ The most simple way to compute Cramer-von Mises flatness, this is however very slow
    if you need to compute it many times
    :param y: real classes of events, shape = [n_samples]
    :param proba: predicted probabilities, shape = [n_samples, n_classes]
    :param X: pandas.DataFrame with uniform features (i.e. test dataset)
    :param uniform_variables: features, along which uniformity is desired, list of strings
    :param sample_weight: weights of events, shape = [n_samples]
    :param label: class, for which uniformity is measured (usually, 0 is bck, 1 is signal)
    :param knn: number of nearest neighbours used in knn

    Example of usage:
    proba = classifier.predict_proba(testX)
    cvm_flatness(testY, proba=proba, X=testX, uniform_variables=['mass'])
    """
    assert len(y) == len(proba) == len(X), 'Different lengths'
    X = pandas.DataFrame(X)

    signal_mask = y == label
    groups_indices = compute_knn_indices_of_signal(X[uniform_variables], is_signal=signal_mask, n_neighbours=knn)
    groups_indices = groups_indices[signal_mask, :]

    return ut.group_based_cvm(proba[:, label], mask=signal_mask, groups_indices=groups_indices,
                              sample_weight=sample_weight)


# endregion
