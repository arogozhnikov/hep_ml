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
from .commonutils import check_sample_weight, to_pandas_dataframe

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
            values = X[column]
            if len(numpy.unique(values)) < self.max_bins:
                self.percentiles[column] = numpy.unique(values)[:-1]
            else:
                targets = numpy.linspace(0, 100, self.max_bins + 1)[1:-1]
                self.percentiles[column] = numpy.percentile(values, targets)
        return self

    def transform(self, X):
        """
        :param X: pandas.DataFrame or numpy.array with data
        :returns: pandas.DataFrame with transformed features (names of columns are preserved),
            dtype is 'int8' for space economy
        """
        X = to_pandas_dataframe(X)
        assert list(X.columns) == list(self.percentiles.keys()), 'Wrong names of columns'
        bin_indices = numpy.zeros(X.shape, dtype='uint8')
        for i, column in enumerate(X.columns):
            bin_indices[:, i] = numpy.searchsorted(self.percentiles[column], X[column])
        return pandas.DataFrame(bin_indices, columns=X.columns)


class IronTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, max_points=10000):
        """
        IronTransformer fits one-dimensional transformation for each feature.

        After applying this transformations distribution of each feature turns into uniform in [0, 1].
        This is very handy to work with features with different scale and complex distributions.

        The name of transformer comes from https://en.wikipedia.org/wiki/Clothes_iron,
        which makes anything flat, being applied with enough pressure :)

        Recommended to apply with neural networks and other algorithms sensitive to scale of features.

        :param int max_points: leave so many points in monotonic transformation.
        """
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

        self.feature_values = numpy.zeros(X.shape, dtype=float)
        self.feature_percentiles = numpy.zeros(X.shape, dtype=float)

        self.feature_maps = OrderedDict()
        random_state = numpy.random.RandomState(seed=42)
        for column in X.columns:
            # adding little noise for symmetrical gaps around concentrations.
            data = X[column] + random_state.normal(0, scale=1e-10, size=len(X))
            order = numpy.argsort(data)
            feature_percentiles = numpy.cumsum(sample_weight[order])
            feature_values = data[order]
            self.feature_maps[column] = (feature_values, feature_percentiles)

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
            result[column] = numpy.interp(X[column], feature_values, feature_percentiles)

        return result
