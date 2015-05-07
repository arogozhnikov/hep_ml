from __future__ import division, print_function, absolute_import

import numpy
import pandas
from numpy.random.mtrand import RandomState
from sklearn.metrics.pairwise import pairwise_distances
from hep_ml import commonutils
from hep_ml.commonutils import weighted_quantile, build_normalizer, \
    compute_cut_for_efficiency, generate_sample, compute_knn_indices_of_signal, \
    compute_knn_indices_of_same_class


def test_splitting(n_rows=10, n_columns=8):
    column_names = ['col' + str(i) for i in range(n_columns)]
    signal_df = pandas.DataFrame(numpy.ones([n_rows, n_columns]), columns=column_names)
    bg_df = pandas.DataFrame(numpy.zeros([n_rows, n_columns]), columns=column_names)

    common_X = pandas.concat([signal_df, bg_df], ignore_index=True)
    common_y = numpy.concatenate([numpy.ones(len(signal_df)), numpy.zeros(len(bg_df))])

    trainX, testX, trainY, testY = commonutils.train_test_split(common_X, common_y)

    for (index, row), label in zip(trainX.iterrows(), trainY):
        assert numpy.all(row == label), 'wrong data partition'
    for (index, row), label in zip(testX.iterrows(), testY):
        assert numpy.all(row == label), 'wrong data partition'

    assert (trainX.columns == column_names).all(), 'new column names!'
    assert (testX.columns == column_names).all(), 'new column names!'
    assert len(trainX) + len(testX) == len(common_X), 'new size is strange'


def check_weighted_percentile(size=100, q_size=20):
    random = RandomState()
    array = random.permutation(size)
    quantiles = random.uniform(size=q_size)
    q_permutation = random.permutation(q_size)
    result1 = weighted_quantile(array, quantiles)[q_permutation]
    result2 = weighted_quantile(array, quantiles[q_permutation])
    result3 = weighted_quantile(array[random.permutation(size)], quantiles[q_permutation])
    assert numpy.all(result1 == result2) and numpy.all(result1 == result3), 'breaks on permutations'

    # checks that order is kept
    quantiles = numpy.linspace(0, 1, size * 3)
    x = weighted_quantile(array, quantiles, sample_weight=random.exponential(size=size))
    assert numpy.all(x == numpy.sort(x)), "doesn't preserve order"

    array = numpy.array([0, 1, 2, 5])
    # comparing with simple percentiles
    for x in random.uniform(size=10):
        assert numpy.abs(numpy.percentile(array, x * 100) - weighted_quantile(array, x, old_style=True)) < 1e-7, \
            "doesn't coincide with numpy.percentile"


def test_weighted_percentile():
    check_weighted_percentile(RandomState().randint(4, 40), RandomState().randint(4, 40))
    check_weighted_percentile(100, 20)
    check_weighted_percentile(20, 100)


def test_build_normalizer(checks=10):
    predictions = numpy.array(RandomState().normal(size=2000))
    result = build_normalizer(predictions)(predictions)
    assert numpy.all(result[numpy.argsort(predictions)] == sorted(result))
    assert numpy.all(result >= 0) and numpy.all(result <= 1)
    percentiles = [100 * (i + 1.) / (checks + 1.) for i in range(checks)]
    assert numpy.all(abs(numpy.percentile(result, percentiles) - numpy.array(percentiles) / 100.) < 0.01)

    # testing with weights
    predictions = numpy.exp(predictions / 2)
    weighted_normalizer = build_normalizer(predictions, sample_weight=predictions)
    result = weighted_normalizer(predictions)
    assert numpy.all(result[numpy.argsort(predictions)] == sorted(result))
    assert numpy.all(result >= 0) and numpy.all(result <= 1 + 1e-7)
    predictions = numpy.sort(predictions)
    result = weighted_normalizer(predictions)
    result2 = numpy.cumsum(predictions) / numpy.sum(predictions)
    assert numpy.all(numpy.abs(result - result2) < 0.005)


def test_compute_cut():
    random = RandomState()
    predictions = random.permutation(100)
    labels = numpy.ones(100)
    for eff in [0.1, 0.5, 0.75, 0.99]:
        cut = compute_cut_for_efficiency(eff, labels, predictions)
        assert numpy.sum(predictions > cut) / len(predictions) == eff, 'the cut was set wrongly'

    weights = numpy.array(random.exponential(size=100))
    for eff in random.uniform(size=100):
        cut = compute_cut_for_efficiency(eff, labels, predictions, sample_weight=weights)
        lower = numpy.sum(weights[predictions > cut + 1]) / numpy.sum(weights)
        upper = numpy.sum(weights[predictions > cut - 1]) / numpy.sum(weights)
        assert lower < eff < upper, 'the cut was set wrongly'


def test_compute_knn_indices(n_events=100):
    X, y = generate_sample(n_events, 10, distance=.5)
    is_signal = y > 0.5
    signal_indices = numpy.where(is_signal)[0]
    uniform_columns = X.columns[:1]
    knn_indices = compute_knn_indices_of_signal(X[uniform_columns], is_signal, 10)
    distances = pairwise_distances(X[uniform_columns])
    for i, neighbours in enumerate(knn_indices):
        assert numpy.all(is_signal[neighbours]), "returned indices are not signal"
        not_neighbours = [x for x in signal_indices if not x in neighbours]
        min_dist = numpy.min(distances[i, not_neighbours])
        max_dist = numpy.max(distances[i, neighbours])
        assert min_dist >= max_dist, "distances are set wrongly!"

    knn_all_indices = compute_knn_indices_of_same_class(X[uniform_columns], is_signal, 10)
    for i, neighbours in enumerate(knn_all_indices):
        assert numpy.all(is_signal[neighbours] == is_signal[i]), "returned indices are not signal/bg"

