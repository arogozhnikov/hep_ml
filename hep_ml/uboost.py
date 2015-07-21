"""
The module contains an implementation of uBoost algorithm in sklearn-way

- ``uBoostBDT`` is a modified version of AdaBoost, that targets to
obtain efficiency uniformity at the specified level (global efficiency)
- ``uBoostClassifier`` - a combination of uBoostBDTs for different efficiencies
"""

# Authors:
# Alex Rogozhnikov <axelr@yandex-team.ru>
# Nikita Kazeev <kazeevn@yandex-team.ru>

from collections import defaultdict
from six.moves import zip

import numpy as np
from scipy.special import expit
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble.weight_boosting import ClassifierMixin
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.random import check_random_state

from .commonutils import sigmoid_function, map_on_cluster, \
    compute_knn_indices_of_same_class, compute_cut_for_efficiency, check_xyw
from hep_ml import commonutils
from .metrics_utils import compute_group_efficiencies_by_indices


__author__ = "Alex Rogozhnikov, Nikita Kazeev"

__all__ = ["uBoostBDT", "uBoostClassifier"]


class uBoostBDT(object):
    def __init__(self,
                 uniform_features,
                 target_efficiency=0.5,
                 n_neighbors=50,
                 bagging=True,
                 base_estimator=DecisionTreeClassifier(max_depth=1),
                 n_estimators=50,
                 learning_rate=1.,
                 uniforming_rate=1.,
                 train_features=None,
                 smoothing=0.0,
                 keep_debug_info=False,
                 random_state=None,
                 uniform_label=1,
                 algorithm="SAMME"):
        """
        uBoostBDT is AdaBoostClassifier, which is modified to have flat
        efficiency of signal (class=1) along some variables.
        Efficiency is only guaranteed at the cut,
        corresponding to global efficiency == target_efficiency.

        Can be used alone, without uBoost.

        Parameters
        ----------
        :param uniform_features: list of strings, names of variables, along which
         flatness is desired

        :param target_efficiency: float, the flatness is obtained at global BDT cut,
            corresponding to global efficiency

        :param n_neighbors: int, (default=50) the number of neighbours,
            which are used to compute local efficiency

        :param bagging: float or bool (default=True), bagging usually speeds up the
            convergence and prevents overfitting
            (see http://en.wikipedia.org/wiki/Bootstrap_aggregating)
            if True, usual bootstrap aggregating is used
            (sampling with replacement at each iteration, size=len(X))
            if float, used sampling with replacement, the size of generated set
             is bagging * len(X)
            if False, usual boosting is used

        :param base_estimator : classifier, optional (default=DecisionTreeClassifier)
            The base estimator from which the boosted ensemble is built.
            Support for sample weighting is required, as well as proper
            `classes_` and `n_classes_` attributes.

        :param n_estimators : integer, optional (default=50)
            The maximum number of estimators at which boosting is terminated.
            In case of perfect fit, the learning procedure is stopped early.

        :param learning_rate : float, optional (default=1.)
            Learning rate shrinks the contribution of each classifier by
            ``learning_rate``. There is a trade-off between ``learning_rate``
            and ``n_estimators``.

        :param uniforming_rate: float, optional (default=1.)
            how much do we take into account the uniformity of signal,
             there is a trade-off between uniforming_rate and the speed of
             uniforming, zero value corresponds to plain AdaBoost

        :param uniform_label: int, (default=1)
            label of class on which uniformity is desired

        :param train_features: list of strings, names of variables used in
           fit/predict. If None, all the variables are used
           (including uniform_variables)

        :param smoothing: float, default=(0.), used to smooth computing of local
           efficiencies, 0.0 corresponds to usual uBoost

        :param random_state: int, RandomState instance or None, (default=None),
           used to fix randomization.

        Attributes
        ----------
        `estimators_` : list of classifiers
            The collection of fitted sub-estimators.

        `estimator_weights_` : array of floats
            Weights for each estimator in the boosted ensemble.

        `estimator_errors_` : array of floats
            Classification error for each estimator in the boosted
            ensemble.

        Reference
        ----------
        .. [1] Justin Stevens, Mike Williams 'uBoost: A boosting method for
            producing uniform selection efficiencies from multivariate classifiers'
        """

        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.uniforming_rate = uniforming_rate
        self.uniform_features = uniform_features
        self.target_efficiency = target_efficiency
        self.n_neighbors = n_neighbors
        self.bagging = bagging
        self.train_features = train_features
        self.smoothing = smoothing
        self.uniform_label = uniform_label
        self.keep_debug_info = keep_debug_info
        self.random_state = random_state
        self.algorithm = algorithm

    def fit(self, X, y, sample_weight=None, neighbours_matrix=None):
        """Build a boosted classifier from the training set (X, y).
        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The training input samples.

        y : array-like of shape = [n_samples]
            The target values (integers that correspond to classes).

        sample_weight : array-like of shape = [n_samples], optional
            Sample weights. If None, the sample weights are initialized to
            ``1 / n_samples``.

        neighbours_matrix: array-like of shape [n_samples, n_neighbours],
            each row contains indices of signal neighbours
            (neighbours should be computed for background too),
            if None, this matrix is computed.

        Returns
        -------
        self : object
            Returns self.
        """
        if self.smoothing < 0:
            raise ValueError("Smoothing must be non-negative")
        if not isinstance(self.base_estimator, BaseEstimator):
            raise TypeError("estimator must be a subclass of BaseEstimator")
        if self.n_estimators <= 0:
            raise ValueError("n_estimators must be greater than zero.")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be greater than zero")

        # Check that algorithm is supported
        if self.algorithm not in ('SAMME', 'SAMME.R'):
            raise ValueError("algorithm %s is not supported"
                             % self.algorithm)
        if self.algorithm == 'SAMME.R':
            if not hasattr(self.base_estimator, 'predict_proba'):
                raise TypeError(
                    "uBoostBDT with algorithm='SAMME.R' requires "
                    "that the weak learner have a predict_proba method.\n"
                    "Please change the base estimator or set "
                    "algorithm='SAMME' instead.")

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

        X_train_features = self.get_train_features(X)
        X_train_features, y, sample_weight = check_xyw(X_train_features, y, sample_weight)

        # A dictionary to keep all intermediate weights, efficiencies and so on
        if self.keep_debug_info:
            self.debug_dict = defaultdict(list)

        self.random_generator = check_random_state(self.random_state)

        self._boost(X_train_features, y, sample_weight)

        self.score_cut = self.signed_uniform_label * compute_cut_for_efficiency(
            self.target_efficiency, y == self.uniform_label, self.predict_score(X) * self.signed_uniform_label)
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

    def compute_uboost_multipliers(self, sample_weight, score, y):
        """Returns uBoost multipliers to sample_weight
        and computed global cut"""
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

        if self.keep_debug_info:
            self.debug_dict['local_effs'].append(local_efficiencies.copy())

        return boost_weights, global_score_cut

    def _boost(self, X, y, sample_weight):
        """Implement a single boost using the SAMME or SAMME.R algorithm,
        which is modified in uBoost way"""
        cumulative_score = np.zeros(len(X))
        y_signed = 2 * y - 1
        for iteration in range(self.n_estimators):
            estimator = self._make_estimator()
            mask = generate_mask(len(X), self.bagging, self.random_generator)
            estimator.fit(X, y, sample_weight=sample_weight * mask)

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
                self.compute_uboost_multipliers(sample_weight, cumulative_score, y)
            sample_weight *= uboost_multipliers
            sample_weight = self._normalize_weight(y, sample_weight)

            self.score_cuts_.append(global_score_cut)
            self.estimators_.append(estimator)
            self.estimator_weights_.append(estimator_weight)

            if self.keep_debug_info:
                self.debug_dict['weights'].append(sample_weight.copy())

        if not self.keep_debug_info:
            self.knn_indices = None

    def get_train_features(self, X):
        """Gets the DataFrame and returns only columns
           that should be used in fitting / predictions"""
        if self.train_features is None:
            return X
        else:
            return X[self.train_features]

    def staged_predict_score(self, X):
        X = self.get_train_features(X)
        score = np.zeros(len(X))
        for classifier, weight in zip(self.estimators_, self.estimator_weights_):
            score += self._estimator_score(classifier, X) * weight
            yield score

    def predict_score(self, X):
        return commonutils.take_last(self.staged_predict_score(X))

    def predict(self, X):
        """ Predict returns labels of classes
        :param X:
        :return:
        """
        return np.array(self.predict_score(X) > self.score_cut, dtype=int)

    def predict_proba(self, X):
        return commonutils.score_to_proba(self.predict_score(X))

    def staged_predict_proba(self, X):
        for score in self.staged_predict_score(X):
            yield commonutils.score_to_proba(score)

    def _uboost_predict_score(self, X):
        """Method added specially for uBoostClassifier"""
        return sigmoid_function(self.predict_score(X) - self.score_cut,
                                self.smoothing)

    def _uboost_staged_predict_score(self, X):
        """Method added specially for uBoostClassifier"""
        for cut, score in zip(self.score_cuts_, self.staged_predict_score(X)):
            yield sigmoid_function(score - cut, self.smoothing)

    @property
    def feature_importances_(self):
        """Return the feature importances
           (the higher, the more important the feature) for train_features.
        Returns:
            array, shape = [n_features]
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
    def __init__(self, uniform_features=None,
                 uniform_label=1,
                 train_features=None,
                 n_neighbors=50,
                 efficiency_steps=20,
                 n_estimators=40,
                 base_estimator=DecisionTreeClassifier(max_depth=1),
                 bagging=True,
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

        :param uniform_label: int, (default=1)
            tha label of class for which uniformity is desired (default is signal)

        :param train_features: list of strings,
            names of variables used in fit/predict.
            if None, all the variables are used (including uniform_variables)

        :param n_neighbors: int, (default=50) the number of neighbours,
            which are used to compute local efficiency

        :param n_estimators : integer, optional (default=50)
            The maximum number of estimators at which boosting is terminated.
            In case of perfect fit, the learning procedure is stopped early.

        :param efficiency_steps: integer, optional (default=20),
            How many uBoostBDTs should be trained
            (each with its own target_efficiency)

        :param base_estimator : object, optional (default=DecisionTreeClassifier)
            The base estimator from which the boosted ensemble is built.
            Support for sample weighting is required,
            as well as proper `classes_` and `n_classes_` attributes.

        :param bagging: float or bool (default=True), bagging usually
            speeds up the convergence and prevents overfitting
            (see http://en.wikipedia.org/wiki/Bootstrap_aggregating)
            if True, usual bootstrap aggregating is used
            (sampling with replacement at each iteration, size=len(X))
            if float, used sampling with replacement, the size of generated
               set is bagging * len(X)
            if False, usual boosting is used

        :param smoothing: float, default=None, used to smooth computing of
            local efficiencies, 0.0 corresponds to usual uBoost,

        :param random_state: int, RandomState instance or None, (default=None)

        :param n_threads: int, number of threads used.

        Reference
        ----------
        .. [1] Justin Stevens, Mike Williams 'uBoost: A boosting method
            for producing uniform
            selection efficiencies from multivariate classifiers'
        """
        self.uniform_features = uniform_features
        self.uniform_label = uniform_label
        self.knn = n_neighbors
        self.efficiency_steps = efficiency_steps
        self.random_state = random_state
        self.n_estimators = n_estimators
        self.base_estimator = base_estimator
        self.bagging = bagging
        self.train_uniform_features = train_features
        self.smoothing = smoothing
        self.n_threads = n_threads
        self.algorithm = algorithm

    def get_train_vars(self, X):
        if self.train_uniform_features is not None:
            return X[self.train_uniform_features]
        else:
            return X

    def fit(self, X, y, sample_weight=None):
        if self.uniform_features is None:
            raise ValueError("Please set uniform variables")
        if len(self.uniform_features) == 0:
            raise ValueError("The set of uniform variables cannot be empty")
        assert np.in1d(y, [0, 1]).all(), \
            "only two-class classification is implemented"
        X, y, sample_weight = check_xyw(X, y, sample_weight=sample_weight, classification=True)
        X_train_vars = self.get_train_vars(X)

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
                random_state=self.random_state, bagging=self.bagging,
                smoothing=self.smoothing, algorithm=self.algorithm)
            self.classifiers.append(classifier)

        self.classifiers = map_on_cluster('threads-{}'.format(self.n_threads),
                                          _train_classifier,
                                          self.classifiers,
                                          self.efficiency_steps * [X_train_vars],
                                          self.efficiency_steps * [y],
                                          self.efficiency_steps * [sample_weight],
                                          self.efficiency_steps * [neighbours_matrix])

        return self

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)

    def predict_proba(self, X):
        X = self.get_train_vars(X)
        score = sum(clf._uboost_predict_score(X) for clf in self.classifiers)
        return commonutils.score_to_proba(score / self.efficiency_steps)

    def staged_predict_proba(self, X):
        X = self.get_train_vars(X)
        for scores in zip(*[clf._uboost_staged_predict_score(X) for clf in self.classifiers]):
            yield commonutils.score_to_proba(sum(scores) / self.efficiency_steps)


def generate_mask(n_samples, bagging=True, random_generator=np.random):
    """bagging: float or bool (default=True), bagging usually
        speeds up the convergence and prevents overfitting
        (see http://en.wikipedia.org/wiki/Bootstrap_aggregating)
        if True, usual bootstrap aggregating is used
           (sampling with replacement at each iteration, size=len(X))
        if float, used sampling without replacement, the size of generated
           set is bagging * len(X)
        if False, returns ones for all events."""
    if bagging is True:
        indices = random_generator.randint(0, n_samples, size=n_samples)
        mask = np.bincount(indices, minlength=n_samples)
    elif isinstance(bagging, float):
        mask = random_generator.uniform(size=n_samples) > 1. - bagging
    elif bagging is False:
        mask = slice(None, None, None)
    else:
        raise ValueError("something wrong was passed as bagging")
    return mask