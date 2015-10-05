"""
**hep_ml.losses** contains different loss functions to use in gradient boosting.

Apart from standard classification losses, **hep_ml** contains losses for uniform classification
(see :class:`BinFlatnessLossFunction`, :class:`KnnFlatnessLossFunction`, :class:`KnnAdaLossFunction`)
and for ranking (see :class:`RankBoostLossFunction`)

**Interface**

Loss functions inside **hep_ml** are stateful estimators and require initial fitting,
which is done automatically inside gradient boosting.

All loss function should be derived from AbstractLossFunction and implement this interface.


Examples
________

Training gradient boosting, optimizing LogLoss and using all features

>>> from hep_ml.gradientboosting import UGradientBoostingClassifier, LogLossFunction
>>> classifier = UGradientBoostingClassifier(loss=LogLossFunction(), n_estimators=100)
>>> classifier.fit(X, y, sample_weight=sample_weight)

Using composite loss function and subsampling:

>>> loss = CompositeLossFunction()
>>> classifier = UGradientBoostingClassifier(loss=loss, subsample=0.5)

To get uniform predictions in mass in background (note that mass should not present in features):

>>> loss = BinFlatnessLossFunction(uniform_features=['mass'], uniform_label=0, train_features=['pt', 'flight_time'])
>>> classifier = UGradientBoostingClassifier(loss=loss)

To get uniform predictions in both signal and background:

>>> loss = BinFlatnessLossFunction(uniform_features=['mass'], uniform_label=[0, 1], train_features=['pt', 'flight_time'])
>>> classifier = UGradientBoostingClassifier(loss=loss)


"""
from __future__ import division, print_function, absolute_import
import numbers
import warnings

import numpy
import pandas
from scipy import sparse
from scipy.special import expit
from sklearn.utils.validation import check_random_state
from sklearn.base import BaseEstimator

from .commonutils import compute_knn_indices_of_signal, check_sample_weight, check_uniform_label, weighted_quantile
from .metrics_utils import bin_to_group_indices, compute_bin_indices, compute_group_weights, \
    group_indices_to_groups_matrix

__author__ = 'Alex Rogozhnikov'

__all__ = [
    'AbstractLossFunction',
    'MSELossFunction',
    'MAELossFunction',
    'LogLossFunction',
    'AdaLossFunction',
    'CompositeLossFunction',
    'BinFlatnessLossFunction',
    'KnnFlatnessLossFunction',
    'KnnAdaLossFunction',
    'RankBoostLossFunction',
    'ReweightLossFunction'
]


def _compute_positions(y_pred, sample_weight):
    """
    For each event computes it position among other events by prediction.
    position = (weighted) part of elements with lower predictions => position belongs to [0, 1]

    This function is very close to `scipy.stats.rankdata`, but supports weights.
    """
    order = numpy.argsort(y_pred)
    ordered_weights = sample_weight[order]
    ordered_weights /= float(numpy.sum(ordered_weights))
    efficiencies = (numpy.cumsum(ordered_weights) - 0.5 * ordered_weights)
    return efficiencies[numpy.argsort(order)]


class AbstractLossFunction(BaseEstimator):
    """
    This is base class for loss functions used in `hep_ml`.
    Main differences compared to `scikit-learn` loss functions:

    1. losses are stateful, and may require fitting of training data before usage.
    2. thus, when computing gradient, hessian, one shall provide predictions of all events.
    3. losses are object that shall be passed as estimators to gradient boosting (see examples).
    4. only two-class case is supported, and different classes may have different role and meaning.
    """

    def fit(self, X, y, sample_weight):
        """ This method is optional, it is called before all the others."""
        return self

    def negative_gradient(self, y_pred):
        """The y_pred should contain all the events passed to `fit` method,
        moreover, the order should be the same"""
        raise NotImplementedError()

    def __call__(self, y_pred):
        """The y_pred should contain all the events passed to `fit` method,
        moreover, the order should be the same"""
        raise NotImplementedError()

    def prepare_tree_params(self, y_pred):
        """Prepares parameters for regression tree that minimizes MSE

        :param y_pred: contains predictions for all the events passed to `fit` method,
         moreover, the order should be the same
        :return: tuple (tree_target, tree_weight) with target and weight to be used in decision tree
        """
        return self.negative_gradient(y_pred), numpy.ones(len(y_pred))

    def prepare_new_leaves_values(self, terminal_regions, leaf_values, y_pred):
        """
        Method for pruning. Loss function can prepare better values for leaves

        :param terminal_regions: indices of terminal regions of each event.
        :param leaf_values: numpy.array, current mapping of leaf indices to prediction values.
        :param y_pred: predictions before adding new tree.
        :return: numpy.array with new prediction values for all leaves.
        """
        return leaf_values

    def compute_optimal_step(self, y_pred):
        """
        Compute optimal global step. This method is typically used to make optimal step
        before fitting trees to reduce variance.
        :param y_pred: initial predictions, numpy.array of shape [n_samples]
        :return: float
        """
        return 0.


class HessianLossFunction(AbstractLossFunction):
    """Loss function with diagonal hessian, provides uses Newton-Raphson step to update trees. """

    def __init__(self, regularization=5.):
        """
        :param regularization: float, penalty for leaves with few events,
         corresponds roughly to the number of added events of both classes to each leaf.
        """
        self.regularization = regularization

    def fit(self, X, y, sample_weight):
        self.regularization_ = self.regularization * numpy.mean(sample_weight)
        return self

    def hessian(self, y_pred):
        """ Returns diagonal of hessian matrix.
        :param y_pred: numpy.array of shape [n_samples] with events passed in the same order as in `fit`.
        :return: numpy.array of shape [n_sampels] with second derivatives with respect to each prediction.
        """
        raise NotImplementedError('Override this method in loss function.')

    def prepare_tree_params(self, y_pred):
        grad = self.negative_gradient(y_pred)
        hess = self.hessian(y_pred) + 0.01
        return grad / hess, hess

    def prepare_new_leaves_values(self, terminal_regions, leaf_values, y_pred):
        """ This expression comes from optimization of second-order approximation of loss function."""
        min_length = len(leaf_values)
        gradients = self.negative_gradient(y_pred)
        hessians = self.hessian(y_pred)
        nominators = numpy.bincount(terminal_regions, weights=gradients, minlength=min_length)
        denominators = numpy.bincount(terminal_regions, weights=hessians, minlength=min_length)
        return nominators / (denominators + self.regularization_)

    def compute_optimal_step(self, y_pred):
        """
        Optimal step is computed using Newton-Raphson algorithm (10 iterations).
        :param y_pred: predictions (usually, zeros)
        :return: float
        """
        terminal_regions = numpy.zeros(len(y_pred), dtype='int')
        leaf_values = numpy.zeros(shape=1)
        step = 0.
        for _ in range(10):
            step_ = self.prepare_new_leaves_values(terminal_regions, leaf_values=leaf_values, y_pred=y_pred + step)[0]
            step += 0.5 * step_
        return step


# region Classification losses

class AdaLossFunction(HessianLossFunction):
    """ AdaLossFunction is the same as Exponential Loss Function (aka exploss) """

    def fit(self, X, y, sample_weight):
        self.sample_weight = check_sample_weight(y, sample_weight=sample_weight,
                                                 normalize=True, normalize_by_class=True)
        self.y_signed = 2 * y - 1
        HessianLossFunction.fit(self, X, y, sample_weight=self.sample_weight)
        return self

    def __call__(self, y_pred):
        return numpy.sum(self.sample_weight * numpy.exp(- self.y_signed * y_pred))

    def negative_gradient(self, y_pred):
        return self.y_signed * self.sample_weight * numpy.exp(- self.y_signed * y_pred)

    def hessian(self, y_pred):
        return self.sample_weight * numpy.exp(- self.y_signed * y_pred)

    def prepare_tree_params(self, y_pred):
        return self.y_signed, self.hessian(y_pred)


class LogLossFunction(HessianLossFunction):
    """Logistic loss function (logloss), aka binomial deviance, aka cross-entropy,
    aka log-likelihood loss.
    """

    def fit(self, X, y, sample_weight):
        self.sample_weight = check_sample_weight(y, sample_weight=sample_weight,
                                                 normalize=True, normalize_by_class=True)
        self.y_signed = 2 * y - 1
        HessianLossFunction.fit(self, X, y, sample_weight=self.sample_weight)
        return self

    def __call__(self, y_pred):
        return numpy.sum(self.sample_weight * numpy.logaddexp(0, - self.y_signed * y_pred))

    def negative_gradient(self, y_pred):
        return self.y_signed * self.sample_weight * expit(- self.y_signed * y_pred)

    def hessian(self, y_pred):
        expits = expit(self.y_signed * y_pred)
        return self.sample_weight * expits * (1 - expits)

    def prepare_tree_params(self, y_pred):
        return self.y_signed * expit(- self.y_signed * y_pred), self.sample_weight


class CompositeLossFunction(HessianLossFunction):
    """
    Composite loss function is defined as exploss for backgorund events and logloss for signal with proper constants.

    Such kind of loss functions is very useful to optimize AMS or in situations where very clean signal is expected.
    """

    def fit(self, X, y, sample_weight):
        self.y = y
        self.sample_weight = check_sample_weight(y, sample_weight=sample_weight,
                                                 normalize=True, normalize_by_class=True)
        self.y_signed = 2 * y - 1
        self.sig_w = (y == 1) * self.sample_weight
        self.bck_w = (y == 0) * self.sample_weight
        HessianLossFunction.fit(self, X, y, sample_weight=self.sample_weight)
        return self

    def __call__(self, y_pred):
        result = numpy.sum(self.sig_w * numpy.logaddexp(0, -y_pred))
        result += numpy.sum(self.bck_w * numpy.exp(0.5 * y_pred))
        return result

    def negative_gradient(self, y_pred):
        result = self.sig_w * expit(- y_pred)
        result -= 0.5 * self.bck_w * numpy.exp(0.5 * y_pred)
        return result

    def hessian(self, y_pred):
        expits = expit(- y_pred)
        return self.sig_w * expits * (1 - expits) + self.bck_w * 0.25 * numpy.exp(0.5 * y_pred)


# endregion

# region Regression Losses

class MSELossFunction(HessianLossFunction):
    r""" Mean squared error loss function, used for regression.
    :math:`\text{loss} = \sum_i (y_i - \hat{y}_i)^2`
    """

    def fit(self, X, y, sample_weight):
        self.y = y
        self.sample_weight = check_sample_weight(y, sample_weight=sample_weight, normalize=True)
        HessianLossFunction.fit(self, X, y, sample_weight=sample_weight)
        return self

    def __call__(self, y_pred):
        return 0.5 * numpy.sum(self.sample_weight * (self.y - y_pred) ** 2)

    def negative_gradient(self, y_pred):
        return self.sample_weight * (self.y - y_pred)

    def hessian(self, y_pred):
        return self.sample_weight

    def prepare_tree_params(self, y_pred):
        return self.y - y_pred, self.sample_weight

    def compute_optimal_step(self, y_pred):
        return numpy.average(self.y - y_pred, weights=self.sample_weight)


class MAELossFunction(AbstractLossFunction):
    r""" Mean absolute error loss function, used for regression.
    :math:`\text{loss} = \sum_i |y_i - \hat{y}_i|`
    """

    def fit(self, X, y, sample_weight):
        self.y = y
        self.sample_weight = check_sample_weight(y, sample_weight=sample_weight, normalize=True)
        return self

    def __call__(self, y_pred):
        return numpy.sum(self.sample_weight * numpy.abs(self.y - y_pred))

    def negative_gradient(self, y_pred):
        return self.sample_weight * numpy.sign(self.y - y_pred)

    def prepare_tree_params(self, y_pred):
        return numpy.sign(self.y - y_pred), self.sample_weight

    def prepare_new_leaves_values(self, terminal_regions, leaf_values, y_pred):
        # TODO use weighted median
        new_leaf_values = numpy.zeros(len(leaf_values), dtype='float')
        target = (self.y - y_pred)
        for terminal_region in range(len(leaf_values)):
            values = target[terminal_regions == terminal_region]
            values = numpy.insert(values, [0], [0])
            new_leaf_values[terminal_region] = numpy.median(values)
        return new_leaf_values

    def compute_optimal_step(self, y_pred):
        return weighted_quantile(self.y - y_pred, quantiles=[0.5], sample_weight=self.sample_weight)[0]


# endregion RegressionLosses


class RankBoostLossFunction(HessianLossFunction):
    def __init__(self, request_column, penalty_power=1., update_iterations=1):
        r"""RankBoostLossFunction is target of optimization in RankBoost [RB]_ algorithm,
        which was developed for ranking and introduces penalties for wrong order of predictions.

        However, this implementation goes further and there is selection of optimal leaf values based
        on iterative procedure. This implementation also uses matrix decomposition of loss function,
        which is very effective, when labels are from some very limited set (usually it is 0, 1, 2, 3, 4)

        :math:`\text{loss} = \sum_{ij} w_{ij} exp(pred_i - pred_j)`,

        :math:`w_{ij} = ( \alpha + \beta * [query_i = query_j]) R_{label_i, label_j}`, where
        :math:`R_{ij} = 0` if :math:`i \leq j`, else :math:`R_{ij} = (i - j)^{p}`

        :param str request_column: name of column with search query ids. The higher attention is payed
          to samples with same query.
        :param float penalty_power: describes dependence of penalty on the difference between target labels.
        :param int update_iterations: number of minimization steps to provide optimal values in leaves.

        .. [RB] Y. Freund et al. An Efficient Boosting Algorithm for Combining Preferences
        """
        self.update_terations = update_iterations
        self.penalty_power = penalty_power
        self.request_column = request_column
        HessianLossFunction.__init__(self, regularization=0.1)

    def fit(self, X, y, sample_weight):
        self.queries = X[self.request_column]
        self.y = y
        self.possible_queries, normed_queries = numpy.unique(self.queries, return_inverse=True)
        self.possible_ranks, normed_ranks = numpy.unique(self.y, return_inverse=True)

        self.lookups = [normed_ranks, normed_queries * len(self.possible_ranks) + normed_ranks]
        self.minlengths = [len(self.possible_ranks), len(self.possible_ranks) * len(self.possible_queries)]
        self.rank_penalties = numpy.zeros([len(self.possible_ranks), len(self.possible_ranks)], dtype=float)
        for r1 in self.possible_ranks:
            for r2 in self.possible_ranks:
                if r1 < r2:
                    self.rank_penalties[r1, r2] = (r2 - r1) ** self.penalty_power

        self.penalty_matrices = []
        self.penalty_matrices.append(self.rank_penalties / numpy.sqrt(1 + len(y)))
        n_queries = numpy.bincount(normed_queries)
        assert len(n_queries) == len(self.possible_queries)
        self.penalty_matrices.append(
            sparse.block_diag([self.rank_penalties * 1. / numpy.sqrt(1 + nq) for nq in n_queries]))
        HessianLossFunction.fit(self, X, y, sample_weight=sample_weight)

    def __call__(self, y_pred):
        y_pred -= y_pred.mean()
        pos_exponent = numpy.exp(y_pred)
        neg_exponent = numpy.exp(-y_pred)
        result = 0.
        for lookup, length, penalty_matrix in zip(self.lookups, self.minlengths, self.penalty_matrices):
            pos_stats = numpy.bincount(lookup, weights=pos_exponent, minlength=length)
            neg_stats = numpy.bincount(lookup, weights=neg_exponent, minlength=length)
            result += pos_stats.T.dot(penalty_matrix.dot(neg_stats))
        return result

    def negative_gradient(self, y_pred):
        y_pred -= y_pred.mean()
        pos_exponent = numpy.exp(y_pred)
        neg_exponent = numpy.exp(-y_pred)
        gradient = numpy.zeros(len(y_pred), dtype=float)
        for lookup, length, penalty_matrix in zip(self.lookups, self.minlengths, self.penalty_matrices):
            pos_stats = numpy.bincount(lookup, weights=pos_exponent, minlength=length)
            neg_stats = numpy.bincount(lookup, weights=neg_exponent, minlength=length)
            gradient += pos_exponent * penalty_matrix.dot(neg_stats)[lookup]
            gradient -= neg_exponent * penalty_matrix.T.dot(pos_stats)[lookup]
        return - gradient

    def hessian(self, y_pred):
        y_pred -= y_pred.mean()
        pos_exponent = numpy.exp(y_pred)
        neg_exponent = numpy.exp(-y_pred)
        result = numpy.zeros(len(y_pred), dtype=float)
        for lookup, length, penalty_matrix in zip(self.lookups, self.minlengths, self.penalty_matrices):
            pos_stats = numpy.bincount(lookup, weights=pos_exponent, minlength=length)
            neg_stats = numpy.bincount(lookup, weights=neg_exponent, minlength=length)
            result += pos_exponent * penalty_matrix.dot(neg_stats)[lookup]
            result += neg_exponent * penalty_matrix.T.dot(pos_stats)[lookup]
        return result

    def prepare_new_leaves_values(self, terminal_regions, leaf_values, y_pred):
        leaves_values = numpy.zeros(len(leaf_values))
        for _ in range(self.update_terations):
            y_test = y_pred + leaves_values[terminal_regions]
            new_leaves_values = self._prepare_new_leaves_values(terminal_regions, leaves_values, y_test)
            leaves_values = 0.5 * new_leaves_values + leaves_values
        return leaves_values

    def _prepare_new_leaves_values(self, terminal_regions, leaf_values, y_pred):
        """
        For each event we shall represent loss as w_plus * e^{pred} + w_minus * e^{-pred},
        then we are able to construct optimal step.
        Pay attention: this is not an optimal, since we are ignoring,
        that some events belong to the same leaf
        """
        pos_exponent = numpy.exp(y_pred)
        neg_exponent = numpy.exp(-y_pred)
        w_plus = numpy.zeros(len(y_pred), dtype=float)
        w_minus = numpy.zeros(len(y_pred), dtype=float)

        for lookup, length, penalty_matrix in zip(self.lookups, self.minlengths, self.penalty_matrices):
            pos_stats = numpy.bincount(lookup, weights=pos_exponent, minlength=length)
            neg_stats = numpy.bincount(lookup, weights=neg_exponent, minlength=length)
            w_plus += penalty_matrix.dot(neg_stats)[lookup]
            w_minus += penalty_matrix.T.dot(pos_stats)[lookup]

        w_plus_leaf = numpy.bincount(terminal_regions, weights=w_plus * pos_exponent) + self.regularization
        w_minus_leaf = numpy.bincount(terminal_regions, weights=w_minus * neg_exponent) + self.regularization
        return 0.5 * numpy.log(w_minus_leaf / w_plus_leaf)


# region MatrixLossFunction


class AbstractMatrixLossFunction(HessianLossFunction):
    def __init__(self, uniform_features, regularization=5.):
        r"""AbstractMatrixLossFunction is a base class to be inherited by other loss functions,
        which choose the particular A matrix and w vector. The formula of loss is:
        \text{loss} = \sum_i w_i * exp(- \sum_j a_ij y_j score_j)
        """
        self.uniform_features = uniform_features
        # real matrix and vector will be computed during fitting
        self.A = None
        self.A_t = None
        self.w = None
        HessianLossFunction.__init__(self, regularization=regularization)

    def fit(self, X, y, sample_weight):
        """This method is used to compute A matrix and w based on train dataset"""
        assert len(X) == len(y), "different size of arrays"
        A, w = self.compute_parameters(X, y, sample_weight)
        self.A = sparse.csr_matrix(A)
        self.A_t = sparse.csr_matrix(self.A.transpose())
        self.A_t_sq = self.A_t.multiply(self.A_t)
        self.w = numpy.array(w)
        assert A.shape[0] == len(w), "inconsistent sizes"
        assert A.shape[1] == len(X), "wrong size of matrix"
        self.y_signed = numpy.array(2 * y - 1)
        HessianLossFunction.fit(self, X, y, sample_weight=sample_weight)
        return self

    def __call__(self, y_pred):
        """Computing the loss itself"""
        assert len(y_pred) == self.A.shape[1], "something is wrong with sizes"
        exponents = numpy.exp(- self.A.dot(self.y_signed * y_pred))
        return numpy.sum(self.w * exponents)

    def negative_gradient(self, y_pred):
        """Computing negative gradient"""
        assert len(y_pred) == self.A.shape[1], "something is wrong with sizes"
        exponents = numpy.exp(- self.A.dot(self.y_signed * y_pred))
        result = self.A_t.dot(self.w * exponents) * self.y_signed
        return result

    def hessian(self, y_pred):
        assert len(y_pred) == self.A.shape[1], 'something wrong with sizes'
        exponents = numpy.exp(- self.A.dot(self.y_signed * y_pred))
        result = self.A_t_sq.dot(self.w * exponents)
        return result

    def compute_parameters(self, trainX, trainY, trainW):
        """This method should be overloaded in descendant, and should return A, w (matrix and vector)"""
        raise NotImplementedError()

    def prepare_new_leaves_values(self, terminal_regions, leaf_values, y_pred):
        exponents = numpy.exp(- self.A.dot(self.y_signed * y_pred))
        # current approach uses Newton-Raphson step
        # TODO compare with iterative suboptimal choice of value, based on exp(a x) ~ a exp(x)
        regions_matrix = sparse.csc_matrix((self.y_signed, [numpy.arange(len(self.y_signed)), terminal_regions]))
        # Z is matrix of shape [n_exponents, n_terminal_regions]
        # with contributions of each terminal region to each exponent
        Z = self.A.dot(regions_matrix)
        Z = Z.T
        nominator = Z.dot(self.w * exponents)
        denominator = Z.multiply(Z).dot(self.w * exponents)
        return nominator / (denominator + 1e-5)


class KnnAdaLossFunction(AbstractMatrixLossFunction):
    def __init__(self, uniform_features, uniform_label, knn=10, row_norm=1.):
        r"""Modification of AdaLoss to achieve uniformity of predictions

        :math:`\text{loss} = \sum_i w_i * exp(- \sum_j a_{ij} y_j score_j)`

        `A` matrix is square, each row corresponds to a single event in train dataset, in each row we put ones
        to the closest neighbours if this event from uniform class.
        See [BU]_ for details.

        :param list[str] uniform_features: the features, along which uniformity is desired
        :param int|list[int] uniform_label: the label (labels) of 'uniform classes'
        :param int knn: the number of nonzero elements in the row, corresponding to event in 'uniform class'

        .. [BU] A. Rogozhnikov et al, New approaches for boosting to uniformity
            http://arxiv.org/abs/1410.4140

        """
        self.knn = knn
        self.row_norm = row_norm
        self.uniform_label = check_uniform_label(uniform_label)
        AbstractMatrixLossFunction.__init__(self, uniform_features)

    def compute_parameters(self, trainX, trainY, trainW):
        A_parts = []
        w_parts = []
        for label in self.uniform_label:
            label_mask = numpy.array(trainY == label)
            n_label = numpy.sum(label_mask)
            knn_indices = compute_knn_indices_of_signal(trainX[self.uniform_features], label_mask, self.knn)
            knn_indices = knn_indices[label_mask, :]
            ind_ptr = numpy.arange(0, n_label * self.knn + 1, self.knn)
            column_indices = knn_indices.flatten()
            data = numpy.ones(n_label * self.knn, dtype=float) * self.row_norm / self.knn
            A_part = sparse.csr_matrix((data, column_indices, ind_ptr), shape=[n_label, len(trainX)])
            w_part = numpy.mean(numpy.take(trainW, knn_indices), axis=1)
            assert A_part.shape[0] == len(w_part)
            A_parts.append(A_part)
            w_parts.append(w_part)

        for label in set(trainY) - set(self.uniform_label):
            label_mask = trainY == label
            n_label = numpy.sum(label_mask)
            ind_ptr = numpy.arange(0, n_label + 1)
            column_indices = numpy.where(label_mask)[0].flatten()
            data = numpy.ones(n_label, dtype=float) * self.row_norm
            A_part = sparse.csr_matrix((data, column_indices, ind_ptr), shape=[n_label, len(trainX)])
            w_part = trainW[label_mask]
            A_parts.append(A_part)
            w_parts.append(w_part)

        A = sparse.vstack(A_parts, format='csr', dtype=float)
        w = numpy.concatenate(w_parts)
        assert A.shape == (len(trainX), len(trainX))
        return A, w


# endregion


# region ReweightLossFunction


# Mathematically at each stage we
# 0. recompute weights
# 1. normalize ratio between distributions (negatives are in opposite distribution)
# 2. chi2 - changing only sign, weights are the same
# 3. optimal value: simply log (negatives are in the same distribution with sign -)

class ReweightLossFunction(AbstractLossFunction):
    def __init__(self, regularization=5.):
        """
        Loss function used to reweight destributions. Works inside :class:`hep_ml.reweight.GBReweighter`

        Conventions: :math:`y=0` - target distribution, :math:`y=1` - original distribution.

        Weights after look like:

        * :math:`w = w_0` for target distribution
        * :math:`w = w_0 * exp(pred)` for events from original distribution
          (so predictions for target distribution is ignored)

        :param float regularization: roughly, it's number of events added in each leaf to prevent overfitting.
        """
        self.regularization = regularization

    def fit(self, X, y, sample_weight):
        assert numpy.all(numpy.in1d(y, [0, 1]))
        if sample_weight is None:
            self.sample_weight = numpy.ones(len(X), dtype=float)
        else:
            self.sample_weight = numpy.array(sample_weight, dtype=float)
        self.y = y
        # signs encounter transfer to opposite distribution
        self.signs = (2 * y - 1) * numpy.sign(sample_weight)

        self.mask_original = numpy.array(self.y)
        self.mask_target = numpy.array(1 - self.y)
        return self

    def _compute_weights(self, y_pred):
        """We need renormalization at eac step"""
        weights = self.sample_weight * numpy.exp(self.y * y_pred)
        return check_sample_weight(self.y, weights, normalize=True, normalize_by_class=True)

    def __call__(self, *args, **kwargs):
        """ Loss function doesn't have precise expression """
        return 0

    def negative_gradient(self, y_pred):
        return 0.

    def prepare_tree_params(self, y_pred):
        return self.signs, numpy.abs(self._compute_weights(y_pred))

    def prepare_new_leaves_values(self, terminal_regions, leaf_values, y_pred):
        weights = self._compute_weights(y_pred)
        w_target = numpy.bincount(terminal_regions, weights=self.mask_target * weights)
        w_original = numpy.bincount(terminal_regions, weights=self.mask_original * weights)

        # suppressing possibly negative samples
        w_target = w_target.clip(0)
        w_original = w_original.clip(0)

        return numpy.log(w_target + self.regularization) - numpy.log(w_original + self.regularization)


# endregion


# region FlatnessLossFunction


def _exp_margin(margin):
    """ margin = - y_signed * y_pred """
    return numpy.exp(numpy.clip(margin, -1e5, 2))


class AbstractFlatnessLossFunction(AbstractLossFunction):
    """Base class for FlatnessLosses"""

    def __init__(self, uniform_features, uniform_label, power=2., fl_coefficient=3.,
                 allow_wrong_signs=True):

        self.uniform_features = uniform_features
        if isinstance(uniform_label, numbers.Number):
            self.uniform_label = numpy.array([uniform_label])
        else:
            self.uniform_label = numpy.array(uniform_label)
        self.power = power
        self.fl_coefficient = fl_coefficient
        self.allow_wrong_signs = allow_wrong_signs

    def fit(self, X, y, sample_weight=None):
        sample_weight = check_sample_weight(y, sample_weight=sample_weight,
                                            normalize=True, normalize_by_class=True)
        assert len(X) == len(y), 'lengths are different'
        X = pandas.DataFrame(X)

        self.group_indices = dict()
        self.group_matrices = dict()
        self.group_weights = dict()

        occurences = numpy.zeros(len(X))
        for label in self.uniform_label:
            self.group_indices[label] = self._compute_groups_indices(X, y, label=label)
            self.group_matrices[label] = group_indices_to_groups_matrix(self.group_indices[label], len(X))
            self.group_weights[label] = compute_group_weights(self.group_matrices[label], sample_weight=sample_weight)
            for group in self.group_indices[label]:
                occurences[group] += 1

        out_of_bins = (occurences == 0) & numpy.in1d(y, self.uniform_label)
        if numpy.mean(out_of_bins) > 0.01:
            warnings.warn("%i events out of all bins " % numpy.sum(out_of_bins), UserWarning)

        self.y = y
        self.y_signed = 2 * y - 1
        self.sample_weight = numpy.copy(sample_weight)
        self.divided_weight = sample_weight / numpy.maximum(occurences, 1)

        return self

    def _compute_groups_indices(self, X, y, label):
        raise NotImplementedError('To be overriden in descendants.')

    def __call__(self, pred):
        # the actual value does not play any role in boosting
        # optimizing here
        return 0

    def _compute_fl_derivatives(self, y_pred):
        y_pred = numpy.ravel(y_pred)
        neg_gradient = numpy.zeros(len(self.y), dtype=numpy.float)

        for label in self.uniform_label:
            label_mask = self.y == label
            global_positions = numpy.zeros(len(y_pred), dtype=float)
            global_positions[label_mask] = \
                _compute_positions(y_pred[label_mask], sample_weight=self.sample_weight[label_mask])

            for indices_in_bin in self.group_indices[label]:
                local_pos = _compute_positions(y_pred[indices_in_bin],
                                               sample_weight=self.sample_weight[indices_in_bin])
                global_pos = global_positions[indices_in_bin]
                bin_gradient = self.power * numpy.sign(local_pos - global_pos) * \
                               numpy.abs(local_pos - global_pos) ** (self.power - 1)
                neg_gradient[indices_in_bin] += bin_gradient

        neg_gradient *= self.divided_weight
        # check that events outside uniform uniform classes are not touched
        assert numpy.all(neg_gradient[~numpy.in1d(self.y, self.uniform_label)] == 0)
        return neg_gradient

    def negative_gradient(self, y_pred):
        y_signed = self.y_signed
        neg_gradient = self._compute_fl_derivatives(y_pred) * self.fl_coefficient
        # adding ExpLoss
        neg_gradient += y_signed * self.sample_weight * _exp_margin(-y_signed * y_pred)

        if not self.allow_wrong_signs:
            neg_gradient = y_signed * numpy.clip(y_signed * neg_gradient, 0, 1e5)

        return neg_gradient


class BinFlatnessLossFunction(AbstractFlatnessLossFunction):
    def __init__(self, uniform_features, uniform_label, n_bins=10, power=2., fl_coefficient=3.,
                 allow_wrong_signs=True):
        r"""
        This loss function contains separately penalty for non-flatness and for bad prediction quality.
        See [FL]_ for details.

        :math:`\text{loss} =\text{ExpLoss} + c \times \text{FlatnessLoss}`

        FlatnessLoss computed using binning of uniform variables

        :param list[str] uniform_features: names of features, along which we want to obtain uniformity of predictions
        :param int|list[int] uniform_label: the label(s) of classes for which uniformity is desired
        :param int n_bins: number of bins along each variable
        :param float power: the loss contains the difference :math:`| F - F_bin |^p`, where p is power
        :param float fl_coefficient: multiplier for flatness_loss. Controls the tradeoff of quality vs uniformity.
        :param bool allow_wrong_signs: defines whether gradient may different sign from the "sign of class"
            (i.e. may have negative gradient on signal). If False, values will be clipped to zero.

        .. [FL] A. Rogozhnikov et al, New approaches for boosting to uniformity
            http://arxiv.org/abs/1410.4140
        """
        self.n_bins = n_bins
        AbstractFlatnessLossFunction.__init__(self, uniform_features,
                                              uniform_label=uniform_label, power=power,
                                              fl_coefficient=fl_coefficient,
                                              allow_wrong_signs=allow_wrong_signs)

    def _compute_groups_indices(self, X, y, label):
        """Returns a list, each element is events' indices in some group."""
        label_mask = y == label
        extended_bin_limits = []
        for var in self.uniform_features:
            f_min, f_max = numpy.min(X[var][label_mask]), numpy.max(X[var][label_mask])
            extended_bin_limits.append(numpy.linspace(f_min, f_max, 2 * self.n_bins + 1))
        groups_indices = list()
        for shift in [0, 1]:
            bin_limits = []
            for axis_limits in extended_bin_limits:
                bin_limits.append(axis_limits[1 + shift:-1:2])
            bin_indices = compute_bin_indices(X.ix[:, self.uniform_features].values, bin_limits=bin_limits)
            groups_indices += list(bin_to_group_indices(bin_indices, mask=label_mask))
        return groups_indices


class KnnFlatnessLossFunction(AbstractFlatnessLossFunction):
    def __init__(self, uniform_features, uniform_label, n_neighbours=100, power=2., fl_coefficient=3.,
                 max_groups=5000, allow_wrong_signs=True, random_state=42):
        r"""
        This loss function contains separately penalty for non-flatness and for bad prediction quality.
        See [FL]_ for details.

        :math:`\text{loss} = \text{ExpLoss} + c \times \text{FlatnessLoss}`

        FlatnessLoss computed using nearest neighbors in space of uniform features

        :param list[str] uniform_features: names of features, along which we want to obtain uniformity of predictions
        :param int|list[int] uniform_label: the label(s) of classes for which uniformity is desired
        :param int n_neighbours: number of neighbors used in flatness loss
        :param float power: the loss contains the difference :math:`| F - F_bin |^p`, where p is power
        :param float fl_coefficient: multiplier for flatness_loss. Controls the tradeoff of quality vs uniformity.
        :param bool allow_wrong_signs: defines whether gradient may different sign from the "sign of class"
            (i.e. may have negative gradient on signal). If False, values will be clipped to zero.
        :param int max_groups: to limit memory consumption when training sample is large,
            we randomly pick this number of points with their members.

        .. [FL] A. Rogozhnikov et al, New approaches for boosting to uniformity
            http://arxiv.org/abs/1410.4140
        """

        self.n_neighbours = n_neighbours
        self.max_groups = max_groups
        self.random_state = random_state
        AbstractFlatnessLossFunction.__init__(self, uniform_features,
                                              uniform_label=uniform_label, power=power,
                                              fl_coefficient=fl_coefficient,
                                              allow_wrong_signs=allow_wrong_signs)

    def _compute_groups_indices(self, X, y, label):
        mask = y == label
        self.random_state = check_random_state(self.random_state)
        knn_indices = compute_knn_indices_of_signal(X[self.uniform_features], mask,
                                                    n_neighbours=self.n_neighbours)[mask, :]
        if len(knn_indices) > self.max_groups:
            selected_group = self.random_state.choice(len(knn_indices), size=self.max_groups, replace=False)
            return knn_indices[selected_group, :]
        else:
            return knn_indices

# endregion
