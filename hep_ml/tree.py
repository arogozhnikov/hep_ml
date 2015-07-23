from __future__ import division, print_function, absolute_import
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree.tree import DTYPE
import numpy
"""
A wrapper over regression trees is presented here.
This one isn't actually needed by itself, but is an important part of gradient boosting.

GBDT uses (attention!) **transform** method, which returns not
predictions, but indices of leaves for all samples.
"""

__author__ = 'Alex Rogozhnikov'


class ClusteringTree(TransformerMixin):
    """
    Trivial wrapper over different decision trees
    """
    def transform(self, X):
        """
        Return indices of leaves, to which each event belongs.
        :param X: numpy.array of shape [n_samples, n_features]
        :return: [n_samples] with indices
        """
        raise NotImplementedError('should be overriden in descendant')

    def predict(self, X):
        """
        Predict values, separately for each leaf.
        """
        raise NotImplementedError('should be overriden in descendant')

    def get_leaf_values(self):
        """
        Return values tree predicts for each of leaves.
        :return: numpy.array of shape [n_samples]
        """
        raise NotImplementedError('should be overriden in descendant')

    @staticmethod
    def prepare_data(X):
        """Convert dataset to the way when no additional work is needed inside fitting or predicting.
        This method is called once to transform dataset.
        """
        raise NotImplementedError('should be overriden in descendant')


class SklearnClusteringTree(DecisionTreeRegressor, ClusteringTree):
    """
    RegressionTree from scikit-learn, which provides transforming interface.
    """

    def transform(self, X):
        return self.tree_.apply(X)

    def get_leaf_values(self):
        return self.tree_.value.flatten()

    @staticmethod
    def prepare_data(X):
        """Converting to the type needed during fitting sklearn trees."""
        return numpy.array(X, dtype=DTYPE)


