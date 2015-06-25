"""
Different loss functions for gradient boosting are defined here.
"""
from __future__ import division, print_function, absolute_import
import numbers
import warnings
from collections import defaultdict

import numpy
import pandas
from scipy import sparse
from scipy.special import expit
from sklearn.utils.validation import check_random_state
from sklearn.base import BaseEstimator

from .commonutils import compute_knn_indices_of_signal, check_sample_weight, check_uniform_label
from .metrics_utils import bin_to_group_indices, compute_bin_indices, compute_group_weights, \
    group_indices_to_groups_matrix


__author__ = 'Alex Rogozhnikov'


def compute_positions(y_pred, sample_weight):
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
    2. thus, when computing gradients, hessians, one shall provide predictions of all events.
    3. losses are object that shall be passed as estimators to gradient boosting (see examples).
    4. only two-class case is supported, and different classes may have different role and meaning.
       (so no class-switching like possible as this was done in sklearn).
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
        """Prepares patameters for regression tree that minimizes MSE
        :param y_pred: contains predictions for all the events passed to `fit` method,
        moreover, the order should be the same
        :return: tuple (residual, sample_weight) with target and weight to be used in decision tree
        """
        return self.negative_gradient(y_pred), None

    def prepare_new_leaves_values(self, terminal_regions, leaf_values,
                                  X, y, y_pred, sample_weight, update_mask, residual):
        """
        Method for pruning. Loss function can prepare better values for leaves

        :param terminal_regions: indices of terminal regions of each event.
        :param leaf_values: numpy.array, current mapping of leaf indices to prediction values.
        :param X: data (same as passed in fit, ignored usually)
        :param y: labels (same as passed in fit)
        :param y_pred: predictions before adding new tree.
        :param sample_weight: weights od samples (same as passed in fit)
        :param update_mask: which events to use during update?
        :param residual: computed value of negative gradient (before adding tree)
        :return: numpy.array with new prediction values for all leaves.
        """
        return leaf_values


class HessianLossFunction(AbstractLossFunction):
    def __init__(self, regularization=5.):
        """Loss function with diagonal hessian, provides uses Newton-Raphson step to update trees.
        :param regularization: float, penalty for leaves with few events,
            corresponds roughly to the number of added events of both classes to each leaf.
        """
        self.regularization = regularization

    def fit(self, X, y, sample_weight):
        self.regularization_ = self.regularization * numpy.mean(sample_weight)
        return self

    def hessian(self, y_pred):
        raise NotImplementedError('Override this method in loss function.')

    def prepare_tree_params(self, y_pred):
        grad = self.negative_gradient(y_pred)
        hess = self.hessian(y_pred) + 0.01
        return grad / hess, hess

    def prepare_new_leaves_values(self, terminal_regions, leaf_values,
                                  X, y, y_pred, sample_weight, update_mask, residual):
        """ This expression comes from optimization of second-order approximation of loss function."""
        minlength = len(leaf_values)
        nominators = numpy.bincount(terminal_regions, weights=residual, minlength=minlength)
        denominators = numpy.bincount(terminal_regions, weights=self.hessian(y_pred), minlength=minlength)
        return nominators / (denominators + self.regularization_)


class AdaLossFunction(HessianLossFunction):
    """ AdaLossFunction is the same as Exponential Loss Function (exploss)"""
    # TODO write better update
    def fit(self, X, y, sample_weight):
        self.y = y
        self.sample_weight = sample_weight
        self.y_signed = 2 * y - 1
        HessianLossFunction.fit(self, X, y, sample_weight=sample_weight)
        return self

    def __call__(self, y_pred):
        return numpy.sum(self.sample_weight * numpy.exp(- self.y_signed * y_pred))

    def negative_gradient(self, y_pred):
        return self.y_signed * self.sample_weight * numpy.exp(- self.y_signed * y_pred)

    def hessian(self, y_pred):
        return self.sample_weight * numpy.exp(- self.y_signed * y_pred)


class BinomialDevianceLossFunction(HessianLossFunction):
    """
    Binomial deviance, aka Logistic loss function (logloss).
    """

    def fit(self, X, y, sample_weight):
        self.y = y
        self.sample_weight = sample_weight
        self.y_signed = 2 * y - 1
        self.adjusted_regularization = numpy.mean(sample_weight) * self.regularization
        HessianLossFunction.fit(self, X, y, sample_weight=sample_weight)
        return self

    def __call__(self, y_pred):
        return numpy.sum(self.sample_weight * numpy.logaddexp(0, - self.y_signed * y_pred))

    def negative_gradient(self, y_pred):
        return self.y_signed * self.sample_weight * expit(- self.y_signed * y_pred)

    def hessian(self, y_pred):
        expits = expit(self.y_signed * y_pred)
        return self.sample_weight * expits * (1 - expits)


class CompositeLossFunction(HessianLossFunction):
    """
    This is exploss for bck and logloss for signal with proper constants.
    Such kind of loss functions is very useful to optimize AMS.
    """

    def fit(self, X, y, sample_weight):
        self.y = y
        self.sample_weight = sample_weight
        self.y_signed = 2 * y - 1
        self.sig_w = (y == 1) * self.sample_weight
        self.bck_w = (y == 0) * self.sample_weight
        HessianLossFunction.fit(self, X, y, sample_weight=sample_weight)
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


class RankBoostLossFunction(HessianLossFunction):
    def __init__(self, request_column, messup_penalty='square', regularization=0.1):
        self.messup_penalty = messup_penalty
        self.request_column = request_column
        HessianLossFunction.__init__(self, regularization=regularization)

    def fit(self, X, y, sample_weight):
        self.queries = X[self.request_column]
        self.y = y
        self.possible_queries, normed_queries = numpy.unique(self.queries, return_inverse=True)
        self.possible_ranks, normed_ranks = numpy.unique(self.y, return_inverse=True)
        self.query_matrices = []

        self.lookups = [normed_ranks, normed_queries * len(self.possible_ranks) + normed_ranks]
        self.minlengths = [len(self.possible_ranks), len(self.possible_ranks) * len(self.possible_queries)]
        self.rank_penalties = numpy.zeros([len(self.possible_ranks), len(self.possible_ranks)], dtype=float)
        for r1 in self.possible_ranks:
            for r2 in self.possible_ranks:
                if r1 < r2:
                    if self.messup_penalty == 'square':
                        self.rank_penalties[r1, r2] = (r2 - r1) ** 2
                    elif self.messup_penalty == 'linear':
                        self.rank_penalties[r1, r2] = r2 - r1
                    else:
                        raise NotImplementedError()

        self.penalty_matrices = []
        self.penalty_matrices.append(self.rank_penalties)
        self.penalty_matrices.append(sparse.block_diag([self.rank_penalties] * len(self.possible_queries)))

    def __call__(self, y_pred):
        """
        loss is defined as  w_ij exp(pred_i - pred_j),
        w_ij is zero if label_i <= label_j
        All the other labels are:

        w_ij = (alpha + beta * [query_i = query_j]) rank_penalty_{type_i, type_j}
        rank_penalty_{ij} is zero if i <= j

        :param y_pred: predictions of shape [n_samples]
        :return: value of loss, float
        """
        y_pred -= y_pred.mean()
        pos_exponent = numpy.exp(y_pred)
        neg_exponent = numpy.exp(-y_pred)
        result = 0.
        for lookup, length, penalty_matrix in zip(self.lookups, self.minlengths, self.penalty_matrices):
            pos_stats = numpy.bincount(lookup, weights=pos_exponent)
            neg_stats = numpy.bincount(lookup, weights=neg_exponent)
            result += pos_stats.T.dot(penalty_matrix.dot(neg_stats))
        # assert numpy.shape(result) == tuple()
        return result

    def negative_gradient(self, y_pred):
        y_pred -= y_pred.mean()
        pos_exponent = numpy.exp(y_pred)
        neg_exponent = numpy.exp(-y_pred)
        gradient = numpy.zeros(len(y_pred), dtype=float)
        for lookup, length, penalty_matrix in zip(self.lookups, self.minlengths, self.penalty_matrices):
            pos_stats = numpy.bincount(lookup, weights=pos_exponent)
            neg_stats = numpy.bincount(lookup, weights=neg_exponent)
            gradient += pos_exponent * penalty_matrix.dot(neg_stats)[lookup]
            gradient -= neg_exponent * penalty_matrix.T.dot(pos_stats)[lookup]
        return - gradient

    def hessian(self, y_pred):
        y_pred -= y_pred.mean()
        pos_exponent = numpy.exp(y_pred)
        neg_exponent = numpy.exp(-y_pred)
        result = numpy.zeros(len(y_pred), dtype=float)
        for lookup, length, penalty_matrix in zip(self.lookups, self.minlengths, self.penalty_matrices):
            pos_stats = numpy.bincount(lookup, weights=pos_exponent)
            neg_stats = numpy.bincount(lookup, weights=neg_exponent)
            result += pos_exponent * penalty_matrix.dot(neg_stats)[lookup]
            result += neg_exponent * penalty_matrix.T.dot(pos_stats)[lookup]
        return result

    def prepare_new_leaves_values(self, terminal_regions, leaf_values,
                                  X, y, y_pred, sample_weight, update_mask, residual):
        """
        For each event we shall represent loss as
        w_plus * e^{pred} + w_minus * e^{-pred},
        then we are able to construct optimal step.
        Pay attention: this is not an optimal, since we are ignoring,
        that some events belong to the same leaf
        """
        pos_exponent = numpy.exp(y_pred)
        neg_exponent = numpy.exp(-y_pred)
        w_plus = numpy.zeros(len(y_pred), dtype=float)
        w_minus = numpy.zeros(len(y_pred), dtype=float)
        for matrix in self.query_matrices:
            pos_stats = matrix.dot(pos_exponent)
            neg_stats = matrix.dot(neg_exponent)
            w_plus += (matrix.T.dot(self.rank_penalties.dot(neg_stats)))
            w_minus += (matrix.T.dot(self.rank_penalties.T.dot(pos_stats)))

        w_plus_leaf = numpy.bincount(terminal_regions, weights=w_plus) + self.regularization
        w_minus_leaf = numpy.bincount(terminal_regions, weights=w_minus) + self.regularization
        return 0.5 * numpy.log(w_minus_leaf / w_plus_leaf)



# region MatrixLossFunction


class AbstractMatrixLossFunction(HessianLossFunction):
    # TODO write better update
    def __init__(self, uniform_variables, regularization=5.):
        """KnnLossFunction is a base class to be inherited by other loss functions,
        which choose the particular A matrix and w vector. The formula of loss is:
        loss = \sum_i w_i * exp(- \sum_j a_ij y_j score_j)
        """
        self.uniform_variables = uniform_variables
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
        self.y_signed = 2 * y - 1
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

    def prepare_new_leaves_values(self, terminal_regions, leaf_values,
                                  X, y, y_pred, sample_weight, update_mask, residual):
        exponents = numpy.exp(- self.A.dot(self.y_signed * y_pred))
        # current approach uses Newton-Raphson step
        # TODO compare with suboptimal choice of value, based on exp(a x) ~ a exp(x)
        regions_matrix = sparse.csc_matrix((self.y_signed, [numpy.arange(len(y)), terminal_regions]))

        Z = self.A.dot(regions_matrix)
        Z = Z.T
        nominator = Z.dot(self.w * exponents)
        denominator = Z.multiply(Z).dot(self.w * exponents)
        return nominator / (denominator + 1e-5)

    # def update_tree_leaf(self, leaf, indices_in_leaf, X, y, y_pred, sample_weight, update_mask, residual):
    #     terminal_region = numpy.zeros(len(X), dtype=float)
    #     terminal_region[indices_in_leaf] += 1
    #     z = self.A.dot(terminal_region * self.y_signed)
    #     # optimal value here by several steps?
    #     alpha = numpy.sum(self.update_exponents * z) / (numpy.sum(self.update_exponents * z * z) + 1e-10)
    #     return alpha


class SimpleKnnLossFunction(AbstractMatrixLossFunction):
    def __init__(self, uniform_variables, knn=10, uniform_label=1, distinguish_classes=True, row_norm=1.):
        """A matrix is square, each row corresponds to a single event in train dataset, in each row we put ones
        to the closest neighbours of that event if this event from class along which we want to have uniform prediction.
        :param list[str] uniform_variables: the features, along which uniformity is desired
        :param int knn: the number of nonzero elements in the row, corresponding to event in 'uniform class'
        :param int|list[int] uniform_label: the label (labels) of 'uniform classes'
        :param bool distinguish_classes: if True, 1's will be placed only for events of same class.
        """
        self.knn = knn
        self.distinguish_classes = distinguish_classes
        self.row_norm = row_norm
        self.uniform_label = check_uniform_label(uniform_label)
        AbstractMatrixLossFunction.__init__(self, uniform_variables)

    def compute_parameters(self, trainX, trainY, trainW):
        A_parts = []
        w_parts = []
        for label in self.uniform_label:
            label_mask = trainY == label
            n_label = numpy.sum(label_mask)
            if self.distinguish_classes:
                mask = label_mask
            else:
                mask = numpy.ones(len(trainY), dtype=numpy.bool)
            knn_indices = compute_knn_indices_of_signal(trainX[self.uniform_variables], mask, self.knn)
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


# region FlatnessLossFunction

def exp_margin(margin):
    """ margin = - y_signed * y_pred """
    return numpy.exp(numpy.clip(margin, -1e5, 2))


class AbstractFlatnessLossFunction(AbstractLossFunction):
    def __init__(self, uniform_variables, uniform_label=1, power=2., ada_coefficient=1.,
                 allow_wrong_signs=True, use_median=False,
                 keep_debug_info=False):
        """
        This loss function contains separately penalty for non-flatness and ada_coefficient.
        The penalty for non-flatness is using bins.

        :type uniform_variables: the vars, along which we want to obtain uniformity
        :type uniform_label: int | list(int), the labels for which we want to obtain uniformity
        :type power: the loss contains the difference | F - F_bin |^p, where p is power
        :type ada_coefficient: coefficient of ada_loss added to this one. The greater the coefficient,
            the less we tend to uniformity.
        :type allow_wrong_signs: defines whether gradient may different sign from the "sign of class"
            (i.e. may have negative gradient on signal). If False, values will be clipped to zero.
        """
        self.uniform_variables = uniform_variables
        if isinstance(uniform_label, numbers.Number):
            self.uniform_label = numpy.array([uniform_label])
        else:
            self.uniform_label = numpy.array(uniform_label)
        self.power = power
        self.ada_coefficient = ada_coefficient
        self.allow_wrong_signs = allow_wrong_signs
        self.keep_debug_info = keep_debug_info
        self.use_median = use_median

    def fit(self, X, y, sample_weight=None):
        sample_weight = check_sample_weight(y, sample_weight=sample_weight)
        assert len(X) == len(y), 'lengths are different'
        X = pandas.DataFrame(X)

        self.group_indices = dict()
        self.group_matrices = dict()
        self.group_weights = dict()

        occurences = numpy.zeros(len(X))
        for label in self.uniform_label:
            self.group_indices[label] = self.compute_groups_indices(X, y, label=label)
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

        if self.keep_debug_info:
            self.debug_dict = defaultdict(list)
        return self

    def compute_groups_indices(self, X, y, label):
        raise NotImplementedError()

    def __call__(self, pred):
        # TODO implement,
        # the actual value does not play any role in boosting, but is interesting
        return 0

    def negative_gradient(self, y_pred):
        y_pred = numpy.ravel(y_pred)
        neg_gradient = numpy.zeros(len(self.y), dtype=numpy.float)

        for label in self.uniform_label:
            label_mask = self.y == label
            global_positions = numpy.zeros(len(y_pred), dtype=float)
            global_positions[label_mask] = \
                compute_positions(y_pred[label_mask], sample_weight=self.sample_weight[label_mask])

            for indices_in_bin in self.group_indices[label]:
                local_pos = compute_positions(y_pred[indices_in_bin],
                                              sample_weight=self.sample_weight[indices_in_bin])
                global_pos = global_positions[indices_in_bin]
                bin_gradient = self.power * numpy.sign(local_pos - global_pos) * \
                               numpy.abs(local_pos - global_pos) ** (self.power - 1)

                neg_gradient[indices_in_bin] += bin_gradient

        neg_gradient *= self.divided_weight

        assert numpy.all(neg_gradient[~numpy.in1d(self.y, self.uniform_label)] == 0)

        y_signed = self.y_signed
        if self.keep_debug_info:
            self.debug_dict['pred'].append(numpy.copy(y_pred))
            self.debug_dict['fl_grad'].append(numpy.copy(neg_gradient))
            self.debug_dict['ada_grad'].append(y_signed * self.sample_weight * exp_margin(-y_signed * y_pred))

        # adding ada
        neg_gradient += self.ada_coefficient * y_signed * self.sample_weight * exp_margin(-y_signed * y_pred)

        if not self.allow_wrong_signs:
            neg_gradient = y_signed * numpy.clip(y_signed * neg_gradient, 0, 1e5)

        return neg_gradient


class BinFlatnessLossFunction(AbstractFlatnessLossFunction):
    def __init__(self, uniform_variables, n_bins=10, uniform_label=1, power=2., ada_coefficient=1.,
                 allow_wrong_signs=True, use_median=False, keep_debug_info=False):
        self.n_bins = n_bins
        AbstractFlatnessLossFunction.__init__(self, uniform_variables,
                                              uniform_label=uniform_label, power=power, ada_coefficient=ada_coefficient,
                                              allow_wrong_signs=allow_wrong_signs, use_median=use_median,
                                              keep_debug_info=keep_debug_info)

    def compute_groups_indices(self, X, y, label):
        """Returns a list, each element is events' indices in some group."""
        label_mask = y == label
        extended_bin_limits = []
        for var in self.uniform_variables:
            f_min, f_max = numpy.min(X[var][label_mask]), numpy.max(X[var][label_mask])
            extended_bin_limits.append(numpy.linspace(f_min, f_max, 2 * self.n_bins + 1))
        groups_indices = list()
        for shift in [0, 1]:
            bin_limits = []
            for axis_limits in extended_bin_limits:
                bin_limits.append(axis_limits[1 + shift:-1:2])
            bin_indices = compute_bin_indices(X.ix[:, self.uniform_variables].values, bin_limits=bin_limits)
            groups_indices += list(bin_to_group_indices(bin_indices, mask=label_mask))
        return groups_indices


class KnnFlatnessLossFunction(AbstractFlatnessLossFunction):
    def __init__(self, uniform_variables, n_neighbours=100, uniform_label=1, power=2., ada_coefficient=1.,
                 max_groups_on_iteration=3000, allow_wrong_signs=True, use_median=False, keep_debug_info=False,
                 random_state=None):
        self.n_neighbours = n_neighbours
        self.max_group_on_iteration = max_groups_on_iteration
        self.random_state = random_state
        AbstractFlatnessLossFunction.__init__(self, uniform_variables,
                                              uniform_label=uniform_label, power=power, ada_coefficient=ada_coefficient,
                                              allow_wrong_signs=allow_wrong_signs, use_median=use_median,
                                              keep_debug_info=keep_debug_info)

    def compute_groups_indices(self, X, y, label):
        mask = y == label
        self.random_state = check_random_state(self.random_state)
        knn_indices = compute_knn_indices_of_signal(X[self.uniform_variables], mask,
                                                    n_neighbours=self.n_neighbours)[mask, :]
        if len(knn_indices) > self.max_group_on_iteration:
            selected_group = self.random_state.choice(len(knn_indices), size=self.max_group_on_iteration)
            return knn_indices[selected_group, :]
        else:
            return knn_indices

# endregion


# region ReweightLossFunction

class ReweightLossFunction(AbstractLossFunction):
    def __init__(self, regularization=5.):
        """
        Loss function used to reweight events. Conventions:
         y=0 - target distribution, y=1 - original distribution.
        Weights after look like:
         w = w_0 for target distribution
         w = w_0 * exp(pred) for events from original distribution
         (so pred for target distribution is ignored)

        :param regularization: roughly, it's number of events added in each leaf to prevent overfitting.
        """
        self.regularization = regularization

    def fit(self, X, y, sample_weight):
        w = check_sample_weight(y, sample_weight=sample_weight)
        self.y = y
        self.y_signed = 2 * y - 1
        self.w = w
        return self

    def _compute_weights(self, y_pred):
        weights = self.w * numpy.exp(self.y * y_pred)
        return check_sample_weight(self.y, weights, normalize=True, normalize_by_class=True)

    def __call__(self, *args, **kwargs):
        """ Loss function doesn't have precise expression """
        return 0

    def negative_gradient(self, y_pred):
        return self.y_signed * self._compute_weights(y_pred)

    def prepare_tree_params(self, y_pred):
        return self.y_signed, self._compute_weights(y_pred=y_pred)

    def prepare_new_leaves_values(self, terminal_regions, leaf_values,
                                  X, y, y_pred, sample_weight, update_mask, residual):
        weights = self._compute_weights(y_pred)
        w_target = numpy.bincount(terminal_regions, weights=weights * (1 - self.y))
        w_original = numpy.bincount(terminal_regions, weights=weights * self.y)
        return numpy.log(w_target + self.regularization) - numpy.log(w_original + self.regularization)


# endregion