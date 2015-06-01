from __future__ import division, print_function, absolute_import
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin
from sklearn.tree import DecisionTreeRegressor

"""
A wrapper over regression trees is presented here.
This one isn't actually needed by itself, but is an important part of gradient boosting.

GBDT uses only (attention!) **transform** method, which returns not
predictions, but indices of values.
"""

__author__ = 'Alex Rogozhnikov'


class ClusteringTree(TransformerMixin):
    """
    Wrapper over different decision trees, which is quite simple

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


class SklearnClusteringTree(DecisionTreeRegressor, ClusteringTree):
    """
    RegressionTree from scikit-learn, which provides transforming interface.
    """

    def transform(self, X):
        return self.tree_.apply(X)

    def get_leaf_values(self):
        return self.tree_.value.flatten()


