from __future__ import division, print_function, absolute_import

import numpy
import pandas
from scipy.stats import ks_2samp

from numpy.random.mtrand import RandomState
from hep_ml.commonutils import generate_sample
from hep_ml.metrics_utils import prepare_distribution, _ks_2samp_fast, ks_2samp_weighted, _cvm_2samp_fast, \
    group_indices_to_groups_matrix
from hep_ml.metrics import KnnBasedSDE, KnnBasedTheil, KnnBasedCvM, \
    BinBasedSDE, BinBasedTheil, BinBasedCvM
from hep_ml.metrics_utils import bin_to_group_indices, compute_bin_indices
from tests._metrics_oldimplementation import compute_sde_on_bins, compute_sde_on_groups, compute_theil_on_bins, \
    compute_theil_on_groups, bin_based_ks, groups_based_ks, cvm_2samp, bin_based_cvm, group_based_cvm, sde, \
    cvm_flatness, theil_flatness

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


def test_groups_matrix(size=1000, bins=4):
    bin_indices = RandomState().randint(0, bins, size=size)
    mask = RandomState().randint(0, 2, size=size) > 0.5
    n_signal_events = numpy.sum(mask)
    group_indices = bin_to_group_indices(bin_indices, mask=mask)
    assert numpy.sum([len(group) for group in group_indices]) == n_signal_events
    group_matrix = group_indices_to_groups_matrix(group_indices, n_events=size)
    assert group_matrix.sum() == n_signal_events
    for event_id, (is_signal, bin) in enumerate(zip(mask, bin_indices)):
        assert group_matrix[bin, event_id] == is_signal


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
    prep_data, prep_weights, prep_F = prepare_distribution(y1, numpy.ones(len(y1)))
    b = _ks_2samp_fast(prep_data, y2, prep_weights, numpy.ones(len(y2)), cdf1=prep_F)
    c = _ks_2samp_fast(prep_data, y2, prep_weights, numpy.ones(len(y2)), cdf1=prep_F)
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


def test_ks2samp(n_samples1=100, n_samples2=100):
    """
    checking that KS can be computed with ROC curve
    """
    data1 = numpy.random.normal(size=n_samples1)
    weights1 = numpy.random.random(size=n_samples1)
    data2 = numpy.random.normal(size=n_samples2)
    weights2 = numpy.random.random(size=n_samples2)

    print(weights1.sum(), 'SUM')

    KS = ks_2samp_weighted(data1, data2, weights1=weights1, weights2=weights2)

    # alternative way to check
    labels = [0] * len(data1) + [1] * len(data2)
    data = numpy.concatenate([data1, data2])
    weights = numpy.concatenate([weights1, weights2])
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(labels, data, sample_weight=weights)
    KS2 = numpy.max(numpy.abs(symmetrize(fpr) - symmetrize(tpr)))
    print(KS, KS2)
    print(weights1.sum(), 'SUM')
    assert numpy.allclose(KS, KS2), 'different values of KS'


def symmetrize(F):
    return 0.5 * (F + numpy.insert(F[:-1], 0, [0]))


def test_cvm2samp(n_samples1=100, n_samples2=100):
    data1 = numpy.random.normal(size=n_samples1)
    weights1 = numpy.random.random(size=n_samples1)
    data2 = numpy.random.normal(size=n_samples2)
    weights2 = numpy.random.random(size=n_samples2)

    CVM = cvm_2samp(data1, data2, weights1=weights1, weights2=weights2)

    # alternative way to check
    labels = [0] * len(data1) + [1] * len(data2)
    data = numpy.concatenate([data1, data2])
    weights = numpy.concatenate([weights1, weights2])
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(labels, data, sample_weight=weights)
    # data1 corresponds to
    weights1 = numpy.diff(numpy.insert(fpr, 0, [0]))
    CVM2 = numpy.sum(weights1 * (symmetrize(fpr) - symmetrize(tpr)) ** 2)
    print(CVM, CVM2)
    assert numpy.allclose(CVM, CVM2), 'different values of CVM'


def test_fast_cvm(n_samples=1000):
    random = RandomState()
    data1 = random.uniform(size=n_samples)
    weights1 = random.uniform(size=n_samples)
    mask = random.uniform(size=n_samples) > 0.5
    data2 = data1[mask]
    weights2 = weights1[mask]
    a = cvm_2samp(data1, data2, weights1, weights2)
    prepared_data1, prepared_weights1, F1 = prepare_distribution(data1, weights1)
    b = _cvm_2samp_fast(prepared_data1, data2, prepared_weights1, weights2, cdf1=F1)
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
    predictions_orig = numpy.random.random(size=[n_samples, 2])

    for shift in [0.1, 0.2]:
        predictions = predictions_orig.copy()
        predictions[:, 1] += shift * y
        predictions /= predictions.sum(axis=1, keepdims=True)

        # Checking SDE
        features = X.columns[:1]
        sde_val1 = sde(y, predictions, X, uniform_features=features, sample_weight=sample_weight, label=0, knn=knn)
        sde_metric = KnnBasedSDE(n_neighbours=knn, uniform_features=features, uniform_label=0, )
        sde_metric.fit(X, y, sample_weight=sample_weight)
        sde_val2 = sde_metric(y, predictions, sample_weight=sample_weight)

        assert numpy.allclose(sde_val1, sde_val2), 'SDE values are different'

        # Checking theil
        theil_val1 = theil_flatness(y, predictions, X, uniform_features=features, sample_weight=sample_weight,
                                    label=0, knn=knn)
        theil_metric = KnnBasedTheil(n_neighbours=knn, uniform_features=features, uniform_label=0, )
        theil_metric.fit(X, y, sample_weight=sample_weight)
        theil_val2 = theil_metric(y, predictions, sample_weight=sample_weight)
        print(theil_val1, theil_val2)
        assert numpy.allclose(theil_val1, theil_val2), 'Theil values are different'

        # Checking CVM
        features = X.columns[:1]
        cvm_val1 = cvm_flatness(y, predictions, X, uniform_features=features, sample_weight=sample_weight, label=0,
                                knn=knn)
        cvm_metric = KnnBasedCvM(n_neighbours=knn, uniform_features=features, uniform_label=0, )
        cvm_metric.fit(X, y, sample_weight=sample_weight)
        cvm_val2 = cvm_metric(y, predictions, sample_weight=sample_weight)

        assert numpy.allclose(cvm_val1, cvm_val2), 'CvM values are different'


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
    X_clear = X[mask]
    y_clear = y[mask]
    sample_weight_clear = sample_weight[mask]
    predictions_clear = predictions[mask]

    for function in [sde, theil_flatness, cvm_flatness]:
        flatness_val = function(y, predictions, X, uniform_features=features, sample_weight=sample_weight, label=0,
                                knn=knn)
        flatness_val_clear = function(y_clear, predictions_clear, X_clear, uniform_features=features,
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
