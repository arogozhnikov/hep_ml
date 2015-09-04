from __future__ import division, print_function, absolute_import
import numpy
from hep_ml.preprocessing import IronTransformer, BinTransformer

__author__ = 'Alex Rogozhnikov'


def assure_monotonic(original, transformed):
    """
    Checks that transformation of data was monotonic
    """
    original = numpy.array(original)
    transformed = numpy.array(transformed)

    assert numpy.shape(original) == numpy.shape(transformed)
    for column in range(original.shape[1]):
        sorter = numpy.argsort(original[:, column])
        sorted_trans = transformed[sorter, column]
        assert numpy.all(numpy.diff(sorted_trans) >= 0)


def test_bin_transform(n_features=10, n_samples=10000):
    """
    Testing BinTransformer
    """
    data = numpy.random.random([n_samples, n_features])

    n_bins = 41

    transformer = BinTransformer(max_bins=n_bins).fit(data)
    result = transformer.transform(data)

    assert numpy.all(result < n_bins)
    assure_monotonic(data, result)

    # check reproducibility
    assert numpy.all(transformer.transform(data) == transformer.transform(data))

    # checking dtype is integer
    numpy_result = numpy.array(result)
    print(numpy_result.dtype)
    assert numpy_result.dtype == 'uint8'


def test_iron_transformer(n_features=10, n_samples=10000):
    """
    Testing FlatTransformer
    """
    data = numpy.random.random([n_samples, n_features])

    transformer = IronTransformer().fit(data)
    result = transformer.transform(data)

    assert numpy.all(result <= 1.)
    assert numpy.all(result >= 0.)
    assure_monotonic(data, result)

    # check reproducibility
    assert numpy.all(transformer.transform(data) == transformer.transform(data))

    # checking dtype is integer
    numpy_result = numpy.array(result)
    assert numpy_result.dtype == float

