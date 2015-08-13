"""
The module contains an implementation of uBoost algorithm.
The main goal of **uBoost** is to fight correlation between predictions and some variables (i.e. mass of particle).

* `uBoostBDT` is a modified version of AdaBoost, that targets to obtain efficiency uniformity at the specified level (global efficiency)
* `uBoostClassifier` is a combination of uBoostBDTs for different efficiencies

This implementation is more advanced than one described in the original paper,
contains smoothing and trains classifiers in threads, has `learning_rate` and `uniforming_rate` parameters,
does automatic weights renormalization and supports SAMME.R modification to use predicted probabilities.

Only binary classification is implemented.

See also: :class:`hep_ml.losses.BinFlatnessLossFunction`, :class:`hep_ml.losses.KnnFlatnessLossFunction`,
:class:`hep_ml.losses.KnnAdaLossFunction`
to fight correlation.

Examples
________

To get uniform prediction in mass for background:

>>> base_tree = DecisionTreeClassifier(max_depth=3)
>>> clf = uBoostClassifier(uniform_features=['mass'], uniform_label=0, base_estimator=base_tree,
>>>                        train_features=['pt', 'flight_time'])
>>> clf.fit(train_data, train_labels, sample_weight=train_weights)
>>> proba = clf.predict_proba(test_data)

To get uniform prediction in Dalitz variables for signal

>>> clf = uBoostClassifier(uniform_features=['mass_12', 'mass_23'], uniform_label=1, base_estimator=base_tree,
>>>                        train_features=['pt', 'flight_time'])
>>> clf.fit(train_data, train_labels, sample_weight=train_weights)
>>> proba = clf.predict_proba(test_data)


"""

# Authors:
# Alex Rogozhnikov <axelr@yandex-team.ru>
# Nikita Kazeev <kazeevn@yandex-team.ru>

from six.moves import zip

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.random import check_random_state

from .commonutils import sigmoid_function, map_on_cluster, \
    compute_knn_indices_of_same_class, compute_cut_for_efficiency, check_xyw
from . import commonutils
from .metrics_utils import compute_group_efficiencies_by_indices


__author__ = "Alex Rogozhnikov, Nikita Kazeev"

__all__ = ["uBoostBDT", "uBoostClassifier"]


class uBoostBDT(BaseEstimator, ClassifierMixin):
    def __init__(self,
                 uniform_features,
                 uniform_label,
                 target_efficiency=0.5,
                 n_neighbors=50,
                 subsample=1.0,
                 base_estimator=None,
                 n_estimators=50,
                 learning_rate=1.,
                 uniforming_rate=1.,
                 train_features=None,
                 smoothing=0.0,
                 random_state=None,
                 algorithm="SAMME"):
        """
        uBoostBDT is AdaBoostClassifier, which is modified to have flat
        efficiency of signal (class=1) along some variables.
        Efficiency is only guaranteed at the cut,
        corresponding to global efficiency == target_efficiency.

        Can be used alone, without uBoostClassifier.

        :param uniform_features: list of strings, names of variables, along which
         flatness is desired

        :param uniform_label: int, label of class on which uniformity is desired
            (typically 0 for background, 1 for signal).

        :param target_efficiency: float, the flatness is obtained at global BDT cut,
            corresponding to global efficiency

        :param n_neighbors: int, (default=50) the number of neighbours,
            which are used to compute local efficiency

        :param subsample: float (default=1.0), part of training dataset used
            to build each base estimator.

        :param base_estimator: classifier, optional (default=DecisionTreeClassifier(max_depth=2))
            The base estimator from which the boosted ensemble is built.
            Support for sample weighting is required, as well as proper
            `classes_` and `n_classes_` attributes.

        :param n_estimators: integer, optional (default=50)
            number of estimators used.

        :param learning_rate: float, optional (default=1.)
            Learning rate shrinks the contribution of each classifier by
            ``learning_rate``. There is a trade-off between ``learning_rate``
            and ``n_estimators``.

        :param uniforming_rate: float, optional (default=1.)
            how much do we take into account the uniformity of signal,
            there is a trade-off between uniforming_rate and the speed of
            uniforming, zero value corresponds to plain AdaBoost

        :param train_features: list of strings, names of variables used in
           fit/predict. If None, all the variables are used
           (including uniform_variables)

        :param smoothing: float, (default=0.), used to smooth computing of local
           efficiencies, 0.0 corresponds to usual uBoost

        :param random_state: int, RandomState instance or None (default None)

        Reference
        ----------
        .. [1] J. Stevens, M. Williams 'uBoost: A boosting method for
            producing uniform selection efficiencies from multivariate classifiers'
        """

        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.uniforming_rate = uniforming_rate
        self.uniform_features = uniform_features
        self.target_efficiency = target_efficiency
        self.n_neighbors = n_neighbors
        self.subsample = subsample
        self.train_features = train_features
        self.smoothing = smoothing
        self.uniform_label = uniform_label
        self.random_state = random_state
        self.algorithm = algorithm

    def fit(self, X, y, sample_weight=None, neighbours_matrix=None):
        """Build a boosted classifier from the training set (X, y).

        :param X: array-like of shape [n_samples, n_features]
        :param y: labels, array of shape [n_samples] with 0 and 1.
        :param sample_weight: array-like of shape [n_samples] or None

        :param neighbours_matrix: array-like of shape [n_samples, n_neighbours],
            each row contains indices of signal neighbours
            (neighbours should be computed for background too),
            if None, this matrix is computed.

        :return: self
        """
        if self.smoothing < 0:
            raise ValueError("Smoothing must be non-negative")
        if not isinstance(self.base_estimator, BaseEstimator):
            raise TypeError("estimator must be a subclass of BaseEstimator")
        if self.n_estimators <= 0:
            raise ValueError("n_estimators must be greater than zero.")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be greater than zero")
        if self.base_estimator is None:
            self.base_estimator = DecisionTreeClassifier(max_depth=2)
        # Check that algorithm is supported
        if self.algorithm not in ('SAMME', 'SAMME.R'):
            raise ValueError("algorithm %s is not supported" % self.algorithm)
        if self.algorithm == 'SAMME.R':
            if not hasattr(self.base_estimator, 'predict_proba'):
                raise TypeError(
                    "uBoostBDT with algorithm='SAMME.R' requires "
                    "that the weak learner have a predict_proba method.\n"
                    "Please change the base estimator or set algorithm='SAMME' instead.")

        assert np.in1d(y, [0, 1]).all(), \
            "only two-class classification is implemented, with labels 0 and 1"
        self.signed_uniform_label = 2 * self.uniform_label - 1

        if neighbours_matrix is not None:
            assert np.shape(neighbours_matrix) == (len(X), self.n_neighbors), \
                "Wrong shape of neighbours_matrix"
            self.knn_indices = neighbours_matrix
        else:
            assert self.uniform_features is not None, \
                "uniform_variables should be set"
            self.knn_indices = compute_knn_indices_of_same_class(
                X.ix[:, self.uniform_features], y, self.n_neighbors)

        sample_weight = commonutils.check_sample_weight(y, sample_weight=sample_weight, normalize=True)
        assert np.all(sample_weight >= 0.), 'the weights should be non-negative'

        # Clear any previous fit results
        self.estimators_ = []
        self.estimator_weights_ = []
        # score cuts correspond to
        # global efficiency == target_efficiency on each iteration.
        self.score_cuts_ = []

        x_train_features = self._get_train_features(X)
        x_train_features, y, sample_weight = check_xyw(x_train_features, y, sample_weight)

        self.random_state_ = check_random_state(self.random_state)

        self._boost(x_train_features, y, sample_weight)

        self.score_cut = self.signed_uniform_label * compute_cut_for_efficiency(
            self.target_efficiency, y == self.uniform_label, self.decision_function(X) * self.signed_uniform_label)
        assert np.allclose(self.score_cut, self.score_cuts_[-1], rtol=1e-10, atol=1e-10), \
            "score cut doesn't appear to coincide with the staged one"
        assert len(self.estimators_) == len(self.estimator_weights_) == len(self.score_cuts_)
        return self

    def _make_estimator(self):
        estimator = clone(self.base_estimator)
        # self.estimators_.append(estimator)
        try:
            estimator.set_params(random_state=self.random_state)
        except ValueError:
            pass
        return estimator

    def _estimator_score(self, estimator, X):
        if self.algorithm == "SAMME":
            return 2 * estimator.predict(X) - 1.
        else:
            p = estimator.predict_proba(X)
            p[p <= 1e-5] = 1e-5
            return np.log(p[:, 1] / p[:, 0])

    @staticmethod
    def _normalize_weight(y, weight):
        # frequently algorithm assigns very big weight to signal events
        # compared to background ones (or visa versa, if want to be uniform in bck)
        return commonutils.check_sample_weight(y, sample_weight=weight, normalize=True, normalize_by_class=True)

    def _compute_uboost_multipliers(self, sample_weight, score, y):
        """Returns uBoost multipliers to sample_weight and computed global cut"""
        signed_score = score * self.signed_uniform_label
        signed_score_cut = compute_cut_for_efficiency(self.target_efficiency, y == self.uniform_label, signed_score)
        global_score_cut = signed_score_cut * self.signed_uniform_label

        local_efficiencies = compute_group_efficiencies_by_indices(signed_score, self.knn_indices, cut=signed_score_cut,
                                                                   smoothing=self.smoothing)

        # pay attention - sample_weight should be used only here
        e_prime = np.average(np.abs(local_efficiencies - self.target_efficiency),
                             weights=sample_weight)

        is_uniform_class = (y == self.uniform_label)

        # beta = np.log((1.0 - e_prime) / e_prime)
        # changed to log(1. / e_prime), otherwise this can lead to the situation
        # where beta is negative (which is a disaster).
        # Mike (uboost author) said he didn't take that into account.
        beta = np.log(1. / e_prime)
        boost_weights = np.exp((self.target_efficiency - local_efficiencies) * is_uniform_class *
                               (beta * self.uniforming_rate))

        return boost_weights, global_score_cut

    def _boost(self, X, y, sample_weight):
        """Implement a single boost using the SAMME or SAMME.R algorithm,
        which is modified in uBoost way"""
        cumulative_score = np.zeros(len(X))
        y_signed = 2 * y - 1
        for iteration in range(self.n_estimators):
            estimator = self._make_estimator()
            mask = _generate_subsample_mask(len(X), self.subsample, self.random_state_)
            estimator.fit(X[mask], y[mask], sample_weight=sample_weight[mask])

            # computing estimator weight
            if self.algorithm == 'SAMME':
                y_pred = estimator.predict(X)

                # Error fraction
                estimator_error = np.average(y_pred != y, weights=sample_weight)
                estimator_error = np.clip(estimator_error, 1e-6, 1. - 1e-6)

                estimator_weight = self.learning_rate * 0.5 * (
                    np.log((1. - estimator_error) / estimator_error))

                score = estimator_weight * (2 * y_pred - 1)
            else:
                estimator_weight = self.learning_rate * 0.5
                score = estimator_weight * self._estimator_score(estimator, X)

            # correcting the weights and score according to predictions
            sample_weight *= np.exp(- y_signed * score)
            sample_weight = self._normalize_weight(y, sample_weight)
            cumulative_score += score

            uboost_multipliers, global_score_cut = \
                self._compute_uboost_multipliers(sample_weight, cumulative_score, y)
            sample_weight *= uboost_multipliers
            sample_weight = self._normalize_weight(y, sample_weight)

            self.score_cuts_.append(global_score_cut)
            self.estimators_.append(estimator)
            self.estimator_weights_.append(estimator_weight)

        # erasing from memory
        self.knn_indices = None

    def _get_train_features(self, X):
        """Gets the DataFrame and returns only columns
           that should be used in fitting / predictions"""
        if self.train_features is None:
            return X
        else:
            return X[self.train_features]

    def staged_decision_function(self, X):
        """Decision function after each stage of boosting.
        Float for each sample, the greater --- the more signal like event is.

        :param X: data, pandas.DataFrame of shape [n_samples, n_features]
        :return: array of shape [n_samples] with floats.
        """
        X = self._get_train_features(X)
        score = np.zeros(len(X))
        for classifier, weight in zip(self.estimators_, self.estimator_weights_):
            score += self._estimator_score(classifier, X) * weight
            yield score

    def decision_function(self, X):
        """Decision function. Float for each sample, the greater --- the more signal like event is.

        :param X: data, pandas.DataFrame of shape [n_samples, n_features]
        :return: array of shape [n_samples] with floats
        """
        return commonutils.take_last(self.staged_decision_function(X))

    def predict(self, X):
        """Predict classes for each sample

        :param X: data, pandas.DataFrame of shape [n_samples, n_features]
        :return: array of shape [n_samples] with predicted classes.
        """
        return np.array(self.decision_function(X) > self.score_cut, dtype=int)

    def predict_proba(self, X):
        """Predict probabilities

        :param X: data, pandas.DataFrame of shape [n_samples, n_features]
        :return: array of shape [n_samples, n_classes] with probabilities.
        """
        return commonutils.score_to_proba(self.decision_function(X))

    def staged_predict_proba(self, X):
        """Predicted probabilities for each sample after each stage of boosting.

        :param X: data, pandas.DataFrame of shape [n_samples, n_features]
        :return: sequence of numpy.arrays of shape [n_samples, n_classes]
        """
        for score in self.staged_decision_function(X):
            yield commonutils.score_to_proba(score)

    def _uboost_predict_score(self, X):
        """Method added specially for uBoostClassifier"""
        return sigmoid_function(self.decision_function(X) - self.score_cut,
                                self.smoothing)

    def _uboost_staged_predict_score(self, X):
        """Method added specially for uBoostClassifier"""
        for cut, score in zip(self.score_cuts_, self.staged_decision_function(X)):
            yield sigmoid_function(score - cut, self.smoothing)

    @property
    def feature_importances_(self):
        """Return the feature importances for `train_features`.

        :return: array of shape [n_features], the order is the same as in `train_features`
        """
        if self.estimators_ is None or len(self.estimators_) == 0:
            raise ValueError("Estimator not fitted,"
                             " call `fit` before `feature_importances_`.")

        return sum(tree.feature_importances_ * weight for tree, weight
                   in zip(self.estimators_, self.estimator_weights_))


def _train_classifier(classifier, X_train_vars, y, sample_weight, neighbours_matrix):
    # supplementary function to train separate parts of uBoost on cluster
    return classifier.fit(X_train_vars, y,
                          sample_weight=sample_weight,
                          neighbours_matrix=neighbours_matrix)


class uBoostClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, uniform_features,
                 uniform_label,
                 train_features=None,
                 n_neighbors=50,
                 efficiency_steps=20,
                 n_estimators=40,
                 base_estimator=None,
                 subsample=1.0,
                 algorithm="SAMME",
                 smoothing=None,
                 n_threads=1,
                 random_state=None):
        """uBoost classifier, an algorithm of boosting targeted to obtain
        flat efficiency in signal along some variables (e.g. mass).

        In principle, uBoost is ensemble of uBoostBDTs. See [1] for details.

        Parameters
        ----------
        :param uniform_features: list of strings, names of variables,
            along which flatness is desired

        :param uniform_label: int,
            tha label of class for which uniformity is desired

        :param train_features: list of strings,
            names of variables used in fit/predict.
            if None, all the variables are used (including uniform_variables)

        :param n_neighbors: int, (default=50) the number of neighbours,
            which are used to compute local efficiency

        :param n_estimators: integer, optional (default=50)
            The maximum number of estimators at which boosting is terminated.
            In case of perfect fit, the learning procedure is stopped early.

        :param efficiency_steps: integer, optional (default=20),
            How many uBoostBDTs should be trained
            (each with its own target_efficiency)

        :param base_estimator: object, optional (default=DecisionTreeClassifier(max_depth=2))
            The base estimator from which the boosted ensemble is built.
            Support for sample weighting is required,
            as well as proper `classes_` and `n_classes_` attributes.

        :param subsample: float (default =1.) part of training dataset used
            to train each base classifier.

        :param smoothing: float, default=None, used to smooth computing of
            local efficiencies, 0.0 corresponds to usual uBoost,

        :param random_state: int, RandomState instance or None, (default=None)

        :param n_threads: int, number of threads used.

        Reference
        ----------
        .. [1] J. Stevens, M. Williams 'uBoost: A boosting method
            for producing uniform selection efficiencies from multivariate classifiers'
        """
        self.uniform_features = uniform_features
        self.uniform_label = uniform_label
        self.knn = n_neighbors
        self.efficiency_steps = efficiency_steps
        self.random_state = random_state
        self.n_estimators = n_estimators
        self.base_estimator = base_estimator
        self.subsample = subsample
        self.train_features = train_features
        self.smoothing = smoothing
        self.n_threads = n_threads
        self.algorithm = algorithm

    def _get_train_features(self, X):
        if self.train_features is not None:
            return X[self.train_features]
        else:
            return X

    def fit(self, X, y, sample_weight=None):
        """Build a boosted classifier from the training set.

        :param X: data, pandas.DatFrame of shape [n_samples, n_features]
        :param y: labels, array of shape [n_samples] with 0 and 1.
            The target values (integers that correspond to classes).
        :param sample_weight: array-like of shape [n_samples] with weights or None
        :return: self
        """
        if self.uniform_features is None:
            raise ValueError("Please set uniform variables")
        if len(self.uniform_features) == 0:
            raise ValueError("The set of uniform variables cannot be empty")
        assert np.in1d(y, [0, 1]).all(), \
            "only two-class classification is implemented"
        if self.base_estimator is None:
            self.base_estimator = DecisionTreeClassifier(max_depth=2)
        X, y, sample_weight = check_xyw(X, y, sample_weight=sample_weight, classification=True)
        data_train_features = self._get_train_features(X)

        if self.smoothing is None:
            self.smoothing = 10. / self.efficiency_steps

        neighbours_matrix = compute_knn_indices_of_same_class(
            X[self.uniform_features], y, n_neighbours=self.knn)
        self.target_efficiencies = np.linspace(0, 1, self.efficiency_steps + 2)[1:-1]
        self.classifiers = []

        for efficiency in self.target_efficiencies:
            classifier = uBoostBDT(
                uniform_features=self.uniform_features,
                uniform_label=self.uniform_label,
                train_features=None,
                target_efficiency=efficiency, n_neighbors=self.knn,
                n_estimators=self.n_estimators,
                base_estimator=self.base_estimator,
                random_state=self.random_state, subsample=self.subsample,
                smoothing=self.smoothing, algorithm=self.algorithm)
            self.classifiers.append(classifier)

        self.classifiers = map_on_cluster('threads-{}'.format(self.n_threads),
                                          _train_classifier,
                                          self.classifiers,
                                          self.efficiency_steps * [data_train_features],
                                          self.efficiency_steps * [y],
                                          self.efficiency_steps * [sample_weight],
                                          self.efficiency_steps * [neighbours_matrix])

        return self

    def predict(self, X):
        """Predict labels

        :param X: data, pandas.DataFrame of shape [n_samples, n_features]
        :return: numpy.array of shape [n_samples]
        """
        return self.predict_proba(X).argmax(axis=1)

    def predict_proba(self, X):
        """Predict probabilities

        :param X: data, pandas.DataFrame of shape [n_samples, n_features]
        :return: array of shape [n_samples, n_classes] with probabilities.
        """
        X = self._get_train_features(X)
        score = sum(clf._uboost_predict_score(X) for clf in self.classifiers)
        return commonutils.score_to_proba(score / self.efficiency_steps)

    def staged_predict_proba(self, X):
        """Predicted probabilities for each sample after each stage of boosting.

        :param X: data, pandas.DataFrame of shape [n_samples, n_features]
        :return: sequence of numpy.arrays of shape [n_samples, n_classes]
        """
        X = self._get_train_features(X)
        for scores in zip(*[clf._uboost_staged_predict_score(X) for clf in self.classifiers]):
            yield commonutils.score_to_proba(sum(scores) / self.efficiency_steps)


def _generate_subsample_mask(n_samples, subsample, random_generator):
    """
    :param float subsample: part of samples to be left
    :param random_generator: numpy.random.RandomState instance
    """
    assert 0 < subsample <= 1., 'subsample should be in range (0, 1]'
    if subsample == 1.0:
        mask = slice(None, None, None)
    else:
        mask = random_generator.uniform(size=n_samples) < subsample
    return mask