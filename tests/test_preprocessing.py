from __future__ import division, print_function, absolute_import
import numpy
from hep_ml.preprocessing import IronTransformer, BinTransformer
from hep_ml.commonutils import generate_sample

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
    data = numpy.random.normal(size=[n_samples, n_features])

    n_bins = 41

    transformer = BinTransformer(max_bins=n_bins).fit(data)
    result = transformer.transform(data)

    assert numpy.all(result < n_bins)
    assert numpy.all(result >= 0)
    assert numpy.allclose(numpy.max(result, axis=0), n_bins - 1)
    assert numpy.allclose(numpy.min(result, axis=0), 0)
    assure_monotonic(data, result)

    # check reproducibility
    assert numpy.all(transformer.transform(data) == transformer.transform(data))

    # checking dtype is integer
    numpy_result = numpy.array(result)
    print(numpy_result.dtype)
    assert numpy_result.dtype == 'uint8'


def test_iron_transformer(n_features=10, n_samples=5000):
    """
    Testing IronTransformer
    """
    import pandas
    data1 = numpy.random.normal(size=[n_samples, n_features])
    data2 = pandas.DataFrame(data=numpy.random.normal(size=[n_samples, n_features]),
                             index=numpy.random.choice(n_samples * 2, size=n_samples, replace=False))
    data3 = numpy.clip(data1, -0.2, 0.2)

    for symmetrize in [True, False]:
        for max_points in [n_samples // 2, n_samples * 2]:
            for data in [data1, data2, data3]:
                data_copy = data.copy()

                transformer = IronTransformer(max_points=max_points, symmetrize=symmetrize).fit(data)
                result = transformer.transform(data)

                assert numpy.all(data == data_copy), 'data was augmented!'

                assert numpy.all(numpy.isfinite(result))  # , numpy.isfinite(result).all()
                assert numpy.all(result <= 1.)
                if symmetrize:
                    assert numpy.all(result >= -1)
                else:
                    assert numpy.all(result >= 0.)
                assure_monotonic(data, result)

                # check reproducibility
                assert numpy.all(transformer.transform(data) == transformer.transform(data))

                # checking dtype is integer
                numpy_result = numpy.array(result)
                assert numpy_result.dtype == float
                for name, (feature_values, feature_percentiles) in transformer.feature_maps.items():
                    assert len(feature_values) <= max_points


def test_bin_transformer_limits(n_features=10, n_bins=123):
    X, y = generate_sample(n_samples=1999, n_features=n_features)
    X = BinTransformer(max_bins=n_bins).fit_transform(X)
    assert numpy.allclose(X.max(axis=0), n_bins - 1)

    X_orig, y = generate_sample(n_samples=20, n_features=n_features)
    X = BinTransformer(max_bins=n_bins).fit_transform(X_orig)
    assert numpy.allclose(X.min(axis=0), 0)


def test_bin_transformer_extend_to(n_features=10, n_bins=123):
    extended_length = 19
    X, y = generate_sample(n_samples=20, n_features=n_features)
    X1 = BinTransformer(max_bins=n_bins).fit(X).transform(X)
    X2 = BinTransformer(max_bins=n_bins).fit(X).transform(X, extend_to=extended_length)
    assert len(X2) % extended_length == 0, 'wrong shape!'
    assert numpy.allclose(X2[:len(X1)], X1), 'extending does not work as expected!'
