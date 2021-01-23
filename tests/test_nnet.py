from __future__ import division, print_function

import numpy
# sklearn 0.22 deprecated sklearn.linear_model.logistic
# remain backwards compatible
try:
    from sklearn.linear_model import LogisticRegression
except ImportError as e:
    from sklearn.linear_model.logistic import LogisticRegression
from sklearn.metrics import roc_auc_score, mean_squared_error, log_loss
from sklearn.base import clone
from sklearn.datasets import make_blobs

from hep_ml import nnet
from hep_ml.commonutils import generate_sample
from hep_ml.nnet import MLPRegressor
from hep_ml.preprocessing import BinTransformer, IronTransformer

__author__ = 'Alex Rogozhnikov'

nn_types = [
    nnet.SimpleNeuralNetwork,
    nnet.MLPClassifier,
    nnet.SoftmaxNeuralNetwork,
    nnet.RBFNeuralNetwork,
    nnet.PairwiseNeuralNetwork,
    nnet.PairwiseSoftplusNeuralNetwork,
]


# TODO test pipelines, bagging and boosting

def check_single_classification_network(neural_network, n_samples=200, n_features=7, distance=0.8, retry_attempts=3):
    X, y = generate_sample(n_samples=n_samples, n_features=n_features, distance=distance)
    # each combination is tried 3 times. before raising exception

    for retry_attempt in range(retry_attempts):
        # to initial state
        neural_network = clone(neural_network)
        neural_network.set_params(random_state=42 + retry_attempt)
        print(neural_network)
        neural_network.fit(X, y)
        quality = roc_auc_score(y, neural_network.predict_proba(X)[:, 1])
        # checking that computations don't fail
        computed_loss = neural_network.compute_loss(X, y, sample_weight=y * 0 + 1)
        if quality > 0.8:
            break
        else:
            print('attempt {} : {}'.format(retry_attempt, quality))
            if retry_attempt == retry_attempts - 1:
                raise RuntimeError('quality of model is too low: {} {}'.format(quality, neural_network))


def test_classification_nnets():
    """
    checking combinations of losses, nn_types, trainers, most of them are used once during tests.
    """
    attempts = max(len(nnet.losses), len(nnet.trainers), len(nn_types))
    losses_shift = numpy.random.randint(10)
    trainers_shift = numpy.random.randint(10)
    for combination in range(attempts):
        loss = list(nnet.losses.keys())[(combination + losses_shift) % len(nnet.losses)]
        trainer = list(nnet.trainers.keys())[(combination + trainers_shift) % len(nnet.trainers)]

        nn_type = nn_types[combination % len(nn_types)]
        neural_network = nn_type(layers=[5], loss=loss, trainer=trainer, epochs=200)
        yield check_single_classification_network, neural_network


def test_regression_nnets():
    from sklearn.datasets import make_regression
    X, y = make_regression(n_samples=300, n_features=20, n_informative=10, bias=5)
    print(y[:20])

    original_mse = mean_squared_error(y, y * 0 + y.mean())
    for loss in ['mse_loss', 'smooth_huber_loss']:
        reg = MLPRegressor(layers=(5,), loss=loss)
        reg.fit(X, y)
        p = reg.predict(X)
        print(numpy.sort(abs(p))[-10:])
        mse = mean_squared_error(y, p)
        assert mse < original_mse * 0.3

    # fitting a constant
    y[:] = 100.
    for loss in ['mse_loss', 'smooth_huber_loss']:
        reg = MLPRegressor(layers=(1,), loss=loss, epochs=300)
        reg.fit(X, y)
        print(mean_squared_error(y, reg.predict(X)))
        assert mean_squared_error(y, reg.predict(X)) < 5., "doesn't fit constant"


def compare_nnets_quality(n_samples=200, n_features=7, distance=0.8):
    X, y = generate_sample(n_samples=n_samples, n_features=n_features, distance=distance)
    # checking all possible combinations
    for loss in ['log_loss']:  # nnet.losses:
        for NNType in nn_types:
            for trainer in nnet.trainers:
                nn = NNType(layers=[5], loss=loss, trainer=trainer, epochs=100, random_state=42)
                nn.fit(X, y)
                print(roc_auc_score(y, nn.predict_proba(X)[:, 1]), nn)

    lr = LogisticRegression().fit(X, y)
    print(roc_auc_score(y, lr.predict_proba(X)[:, 1]), lr)


def test_network_with_scaler(n_samples=200, n_features=15, distance=0.5):
    X, y = generate_sample(n_samples=n_samples, n_features=n_features, distance=distance)
    for scaler in [BinTransformer(max_bins=16), IronTransformer()]:
        clf = nnet.SimpleNeuralNetwork(scaler=scaler, epochs=300)
        clf.fit(X, y)

        p = clf.predict_proba(X)
        assert roc_auc_score(y, p[:, 1]) > 0.8, 'quality is too low for model: {}'.format(clf)


def test_adaptive_methods(n_samples=200, n_features=15, distance=0.5):
    X, y = generate_sample(n_samples=n_samples, n_features=n_features, distance=distance)
    for trainer in ['sgd', 'adadelta']:
        clf = nnet.SimpleNeuralNetwork(trainer=trainer, trainer_parameters={'batch': 1})
        clf.fit(X, y)
        assert roc_auc_score(y, clf.predict_proba(X)[:, 1]) > 0.8, 'quality is too low for model: {}'.format(clf)


def test_reproducibility(n_samples=200, n_features=15, distance=0.5):
    X, y = generate_sample(n_samples=n_samples, n_features=n_features, distance=distance)
    for trainer in nnet.trainers.keys():
        clf1 = nnet.MLPClassifier(trainer=trainer, random_state=42).fit(X, y)
        clf2 = nnet.MLPClassifier(trainer=trainer, random_state=42).fit(X, y)
        assert numpy.allclose(clf1.predict_proba(X), clf2.predict_proba(X))


def test_multiclassification(n_samples=200, n_features=10):
    for n_classes in [2, 3, 4]:
        X, y = make_blobs(n_samples=n_samples, centers=n_classes, n_features=n_features)
        losses = []
        for n_epochs in [1, 10, 100]:
            clf = nnet.MLPMultiClassifier(epochs=n_epochs).fit(X, y)
            loss1 = log_loss(y, clf.predict_proba(X))
            loss2 = clf.compute_loss(X, y)
            assert numpy.allclose(loss1, loss2), 'computed losses are different'
            losses.append(loss1)

        assert losses[0] > losses[-1], 'loss is not decreasing'
