from __future__ import division, print_function, absolute_import

import numpy
import pandas
from sklearn.utils import column_or_1d

from hep_ml.commonutils import check_sample_weight, compute_cut_for_efficiency, compute_knn_indices_of_signal
from hep_ml.metrics_utils import compute_bin_weights, compute_bin_efficiencies, weighted_deviation, \
    compute_group_efficiencies_by_indices, theil, prepare_distribution, _ks_2samp_fast, compute_cdf, \
    _cvm_2samp_fast


__author__ = 'Alex Rogozhnikov'


def compute_sde_on_bins(y_pred, mask, bin_indices, target_efficiencies, power=2., sample_weight=None):
    # ignoring events from other classes
    sample_weight = check_sample_weight(y_pred, sample_weight=sample_weight)
    y_pred = y_pred[mask]
    bin_indices = bin_indices[mask]
    sample_weight = sample_weight[mask]

    bin_weights = compute_bin_weights(bin_indices=bin_indices, sample_weight=sample_weight)
    cuts = compute_cut_for_efficiency(target_efficiencies, mask=numpy.ones(len(y_pred), dtype=bool),
                                      y_pred=y_pred, sample_weight=sample_weight)

    result = 0.
    for cut in cuts:
        bin_efficiencies = compute_bin_efficiencies(y_pred, bin_indices=bin_indices,
                                                    cut=cut, sample_weight=sample_weight)
        result += weighted_deviation(bin_efficiencies, weights=bin_weights, power=power)

    return (result / len(cuts)) ** (1. / power)


def compute_theil_on_bins(y_pred, mask, bin_indices, target_efficiencies, sample_weight):
    y_pred = column_or_1d(y_pred)
    sample_weight = check_sample_weight(y_pred, sample_weight=sample_weight)

    # ignoring events from other classes
    y_pred = y_pred[mask]
    bin_indices = bin_indices[mask]
    sample_weight = sample_weight[mask]

    bin_weights = compute_bin_weights(bin_indices=bin_indices, sample_weight=sample_weight)
    cuts = compute_cut_for_efficiency(target_efficiencies, mask=numpy.ones(len(y_pred), dtype=bool),
                                      y_pred=y_pred, sample_weight=sample_weight)
    result = 0.
    for cut in cuts:
        bin_efficiencies = compute_bin_efficiencies(y_pred, bin_indices=bin_indices,
                                                    cut=cut, sample_weight=sample_weight)
        result += theil(bin_efficiencies, weights=bin_weights)
    return result / len(cuts)


def compute_sde_on_groups(y_pred, mask, groups_indices, target_efficiencies, sample_weight=None, power=2.):
    y_pred = column_or_1d(y_pred)
    sample_weight = check_sample_weight(y_pred, sample_weight=sample_weight)
    group_weights = compute_group_weights_by_indices(groups_indices, sample_weight=sample_weight)
    divided_weight = compute_divided_weight_by_indices(groups_indices, sample_weight=sample_weight * mask)

    cuts = compute_cut_for_efficiency(target_efficiencies, mask=mask, y_pred=y_pred, sample_weight=sample_weight)

    sde = 0.
    for cut in cuts:
        group_efficiencies = compute_group_efficiencies_by_indices(y_pred, groups_indices=groups_indices,
                                                        cut=cut, divided_weight=divided_weight)
        # print('FROM SDE function', cut, group_efficiencies)
        sde += weighted_deviation(group_efficiencies, weights=group_weights, power=power)
    return (sde / len(cuts)) ** (1. / power)


def compute_theil_on_groups(y_pred, mask, groups_indices, target_efficiencies, sample_weight):
    y_pred = column_or_1d(y_pred)
    sample_weight = check_sample_weight(y_pred, sample_weight=sample_weight)
    groups_weights = compute_group_weights_by_indices(groups_indices, sample_weight=sample_weight)
    divided_weight = compute_divided_weight_by_indices(groups_indices, sample_weight=sample_weight * mask)
    cuts = compute_cut_for_efficiency(target_efficiencies, mask=mask,
                                      y_pred=y_pred, sample_weight=sample_weight)

    result = 0.
    for cut in cuts:
        groups_efficiencies = compute_group_efficiencies_by_indices(y_pred, groups_indices=groups_indices,
                                                         cut=cut, divided_weight=divided_weight)
        result += theil(groups_efficiencies, groups_weights)
    return result / len(cuts)


def bin_based_ks(y_pred, mask, sample_weight, bin_indices):
    """Kolmogorov-Smirnov flatness on bins"""
    assert len(y_pred) == len(sample_weight) == len(bin_indices) == len(mask)
    y_pred = y_pred[mask]
    sample_weight = sample_weight[mask]
    bin_indices = bin_indices[mask]

    bin_weights = compute_bin_weights(bin_indices=bin_indices, sample_weight=sample_weight)
    prepared_data, prepared_weight, prep_F = prepare_distribution(y_pred, weights=sample_weight)

    result = 0.
    for bin, bin_weight in enumerate(bin_weights):
        if bin_weight <= 0:
            continue
        local_distribution = y_pred[bin_indices == bin]
        local_weights = sample_weight[bin_indices == bin]
        result += bin_weight * \
                  _ks_2samp_fast(prepared_data, local_distribution, prepared_weight, local_weights, prep_F)
    return result


def groups_based_ks(y_pred, mask, sample_weight, groups_indices):
    """Kolmogorov-Smirnov flatness on groups """
    assert len(y_pred) == len(sample_weight) == len(mask)
    group_weights = compute_group_weights_by_indices(groups_indices, sample_weight=sample_weight)
    prepared_data, prepared_weight, prep_F = prepare_distribution(y_pred[mask], weights=sample_weight[mask])

    result = 0.
    for group_weight, group_indices in zip(group_weights, groups_indices):
        local_distribution = y_pred[group_indices]
        local_weights = sample_weight[group_indices]
        result += group_weight * \
                  _ks_2samp_fast(prepared_data, local_distribution, prepared_weight, local_weights, prep_F)
    return result


def cvm_2samp(data1, data2, weights1=None, weights2=None, power=2.):
    """Computes Cramer-von Mises similarity on 2 samples,
    CvM = \int |F_2 - F_1|^p dF_1
    This implementation sorts the arrays each time,
    so inside loops it will be slow"""
    weights1 = check_sample_weight(data1, sample_weight=weights1)
    weights2 = check_sample_weight(data2, sample_weight=weights2)
    weights1 /= numpy.sum(weights1)
    weights2 /= numpy.sum(weights2)
    data = numpy.unique(numpy.concatenate([data1, data2]))
    bins = numpy.append(data, data[-1] + 1)
    weights1_new = numpy.histogram(data1, bins=bins, weights=weights1)[0]
    weights2_new = numpy.histogram(data2, bins=bins, weights=weights2)[0]
    F1 = compute_cdf(weights1_new)
    F2 = compute_cdf(weights2_new)
    return numpy.average(numpy.abs(F1 - F2) ** power, weights=weights1_new)


def bin_based_cvm(y_pred, sample_weight, bin_indices):
    """Cramer-von Mises similarity, quite slow meanwhile"""
    assert len(y_pred) == len(sample_weight) == len(bin_indices)
    bin_weights = compute_bin_weights(bin_indices=bin_indices, sample_weight=sample_weight)

    result = 0.
    global_data, global_weight, global_F = prepare_distribution(y_pred, weights=sample_weight)

    for bin, bin_weight in enumerate(bin_weights):
        if bin_weight <= 0:
            continue
        bin_mask = bin_indices == bin
        local_distribution = y_pred[bin_mask]
        local_weights = sample_weight[bin_mask]
        result += bin_weight * _cvm_2samp_fast(global_data, local_distribution,
                                               global_weight, local_weights, global_F)

    return result


def group_based_cvm(y_pred, mask, sample_weight, groups_indices):
    y_pred = column_or_1d(y_pred)
    sample_weight = check_sample_weight(y_pred, sample_weight=sample_weight)
    group_weights = compute_group_weights_by_indices(groups_indices, sample_weight=sample_weight)

    result = 0.
    global_data, global_weight, global_F = prepare_distribution(y_pred[mask], weights=sample_weight[mask])
    for group, group_weight in zip(groups_indices, group_weights):
        local_distribution = y_pred[group]
        local_weights = sample_weight[group]
        result += group_weight * _cvm_2samp_fast(global_data, local_distribution,
                                                 global_weight, local_weights, global_F)
    return result

# endregion

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


def sde(y, proba, X, uniform_features, sample_weight=None, label=1, knn=30):
    """ The most simple way to compute SDE, this is however very slow
    if you need to recompute SDE many times
    :param y: real classes of events, shape = [n_samples]
    :param proba: predicted probabilities, shape = [n_samples, n_classes]
    :param X: pandas.DataFrame with uniform features
    :param uniform_features: features, along which uniformity is desired, list of strings
    :param sample_weight: weights of events, shape = [n_samples]
    :param label: class, for which uniformity is measured (usually, 0 is bck, 1 is signal)
    :param knn: number of nearest neighbours used in knn

    Example of usage:
    proba = classifier.predict_proba(testX)
    sde(testY, proba=proba, X=testX, uniform_features=['mass'])
    """
    assert len(y) == len(proba) == len(X), 'Different lengths'
    X = pandas.DataFrame(X)
    mask = y == label
    groups = compute_knn_indices_of_signal(X[uniform_features], is_signal=mask, n_neighbours=knn)
    groups = groups[mask, :]

    return compute_sde_on_groups(proba[:, label], mask=mask, groups_indices=groups,
                                 target_efficiencies=[0.5, 0.6, 0.7, 0.8, 0.9], sample_weight=sample_weight)


def theil_flatness(y, proba, X, uniform_features, sample_weight=None, label=1, knn=30):
    """This is ready-to-use function, and it is quite slow to use many times"""

    mask = y == label
    groups_indices = compute_knn_indices_of_signal(X[uniform_features], is_signal=mask, n_neighbours=knn)[mask, :]
    return compute_theil_on_groups(proba[:, label], mask=mask, groups_indices=groups_indices,
                                   target_efficiencies=[0.5, 0.6, 0.7, 0.8, 0.9], sample_weight=sample_weight)


def cvm_flatness(y, proba, X, uniform_features, sample_weight=None, label=1, knn=30):
    """ The most simple way to compute Cramer-von Mises flatness, this is however very slow
    if you need to compute it many times
    :param y: real classes of events, shape = [n_samples]
    :param proba: predicted probabilities, shape = [n_samples, n_classes]
    :param X: pandas.DataFrame with uniform features (i.e. test dataset)
    :param uniform_features: features, along which uniformity is desired, list of strings
    :param sample_weight: weights of events, shape = [n_samples]
    :param label: class, for which uniformity is measured (usually, 0 is bck, 1 is signal)
    :param knn: number of nearest neighbours used in knn

    Example of usage:
    proba = classifier.predict_proba(testX)
    cvm_flatness(testY, proba=proba, X=testX, uniform_features=['mass'])
    """
    assert len(y) == len(proba) == len(X), 'Different lengths'
    X = pandas.DataFrame(X)

    signal_mask = y == label
    groups_indices = compute_knn_indices_of_signal(X[uniform_features], is_signal=signal_mask, n_neighbours=knn)
    groups_indices = groups_indices[signal_mask, :]

    return group_based_cvm(proba[:, label], mask=signal_mask, groups_indices=groups_indices,
                           sample_weight=sample_weight)


# endregion

def compute_group_weights_by_indices(group_indices, sample_weight):
    """
    Group weight = sum of divided weights of indices inside that group.
    """
    divided_weight = compute_divided_weight_by_indices(group_indices, sample_weight=sample_weight)
    result = numpy.zeros(len(group_indices))
    for i, group in enumerate(group_indices):
        result[i] = numpy.sum(divided_weight[group])
    return result / numpy.sum(result)


def compute_divided_weight_by_indices(group_indices, sample_weight):
    """Divided weight takes into account that different events
    are met different number of times """
    indices = numpy.concatenate(group_indices)
    occurences = numpy.bincount(indices, minlength=len(sample_weight))
    return sample_weight / numpy.maximum(occurences, 1)