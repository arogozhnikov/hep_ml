"""
**hep_ml.preprocessing** contains useful operations with data.
Algorithms implemented here follow sklearn conventions for transformers and inherited from BaseEstimator and TransformerMixin.

Minor difference compared to sklearn is that transformations preserve names of features in DataFrames
(if it is possible).

See also: sklearn.preprocessing for other useful data transformations.

Examples
--------

Transformers may be used as any other transformer, manually training and applying:

>>> from hep_ml.preprocessing import IronTransformer
>>> transformer = IronTransformer().fit(trainX)
>>> new_trainX = transformer.transform(trainX)
>>> new_testX = transformer.transform(testX)

Apart from this, transformers may be plugged as part of sklearn.Pipeline:

>>> from sklearn.pipeline import Pipeline
>>> from hep_ml.nnet import SimpleNeuralNetwork
>>> clf = Pipeline(['pre', IronTransformer(),
>>>                 'nnet', SimpleNeuralNetwork()])

Also, neural networks support special argument 'scaler'. You can pass any transformer there:

>>> clf = SimpleNeuralNetwork(layers=[10, 8], scaler=IronTransformer())

"""

from __future__ import division, print_function, absolute_import
from collections import OrderedDict

import numpy
import pandas
from sklearn.base import BaseEstimator, TransformerMixin
from .commonutils import check_sample_weight, to_pandas_dataframe, weighted_quantile

__author__ = 'Alex Rogozhnikov'
__all__ = ['BinTransformer', 'IronTransformer']


class BinTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, max_bins=128):
        """
        Bin transformer transforms all features (which are expected to be numerical)
        to small integers.

        This simple transformation, while loosing part of information, can increase speed of some algorithms.

        :param int max_bins: maximal number of bins along each axis.
        """
        self.max_bins = max_bins

    def fit(self, X, y=None, sample_weight=None):
        """Prepare transformation rule, compute bin edges.

        :param X: pandas.DataFrame or numpy.array with data
        :param y: labels, ignored
        :param sample_weight: weights, ignored
        :return: self
        """
        assert self.max_bins < 255, 'Too high number of bins!'
        X = to_pandas_dataframe(X)
        self.percentiles = OrderedDict()
        for column in X.columns:
            values = numpy.array(X[column])
            if len(numpy.unique(values)) < self.max_bins:
                self.percentiles[column] = numpy.unique(values)[:-1]
            else:
                targets = numpy.linspace(0, 100, self.max_bins + 1)[1:-1]
                self.percentiles[column] = numpy.percentile(values, targets)
        return self

    def transform(self, X, extend_to=1):
        """
        :param X: pandas.DataFrame or numpy.array with data
        :param int extend_to: extends number of samples to be divisible by extend_to
        :return: numpy.array with transformed features (names of columns are not preserved),
            dtype is 'int8' for space efficiency.
        """
        X = to_pandas_dataframe(X)
        assert list(X.columns) == list(self.percentiles.keys()), 'Wrong names of columns'
        n_samples = len(X)
        extended_length = ((n_samples + extend_to - 1) // extend_to) * extend_to
        bin_indices = numpy.zeros([extended_length, X.shape[1]], dtype='uint8', order='F')
        for i, column in enumerate(X.columns):
            bin_indices[:n_samples, i] = numpy.searchsorted(self.percentiles[column], numpy.array(X[column]))
        return bin_indices


class IronTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, max_points=10000, symmetrize=False):
        """
        IronTransformer fits one-dimensional transformation for each feature.

        After applying this transformations distribution of each feature turns into uniform.
        This is very handy to work with features with different scale and complex distributions.

        The name of transformer comes from https://en.wikipedia.org/wiki/Clothes_iron,
        which makes anything flat, being applied with enough pressure :)

        Recommended to apply with neural networks and other algorithms sensitive to scale of features.

        :param symmetrize: if True, resulting distribution is uniform in [-1, 1], otherwise in [0, 1]
        :param int max_points: leave so many points in monotonic transformation.
        """
        self.symmetrize = symmetrize
        self.max_points = max_points

    def fit(self, X, y=None, sample_weight=None):
        """Fit formula. Compute set of 1-dimensional transformations.

        :param X: pandas.DataFrame with data
        :param y: ignored
        :param sample_weight: ignored
        :return: self
        """
        X = to_pandas_dataframe(X)
        sample_weight = check_sample_weight(X, sample_weight=sample_weight)
        sample_weight = sample_weight / sample_weight.sum()

        self.feature_maps = OrderedDict()
        for column in X.columns:
            # TODO add support for NaNs
            data = numpy.array(X[column], dtype=float)
            data_unique, indices = numpy.unique(data, return_inverse=True)
            weights_unique = numpy.bincount(indices, weights=sample_weight)

            assert len(weights_unique) == len(data_unique)
            if len(data_unique) < self.max_points:
                feature_quantiles = numpy.cumsum(weights_unique) - weights_unique * 0.5
                self.feature_maps[column] = (data_unique, feature_quantiles)
            else:
                feature_quantiles = numpy.linspace(0, 1, self.max_points)
                feature_values = weighted_quantile(data, quantiles=feature_quantiles, sample_weight=sample_weight)
                feature_values, indices = numpy.unique(feature_values, return_index=True)
                feature_quantiles = feature_quantiles[indices]
                self.feature_maps[column] = (feature_values, feature_quantiles)

        return self

    def transform(self, X):
        """Transform data.

        :param X: pandas.DataFrame with data
        :return: pandas.DataFrame with transformed features
        """
        X = to_pandas_dataframe(X)
        assert list(X.columns) == list(self.feature_maps.keys()), \
            'Columns passed {} are different from expected {}'.format(X.columns, list(self.feature_maps.keys()))

        result = pandas.DataFrame(numpy.zeros(X.shape, dtype=float), columns=X.columns)
        for column, (feature_values, feature_percentiles) in self.feature_maps.items():
            data = numpy.array(X[column], dtype=float)
            result[column] = numpy.interp(data, feature_values, feature_percentiles)

        if self.symmetrize:
            result = 2 * result - 1

        return result
