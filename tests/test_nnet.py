from __future__ import division, print_function

import numpy
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.metrics import roc_auc_score, mean_squared_error
from sklearn.base import clone

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


def check_single_classification_network(neural_network, n_samples=200, n_features=7, distance=0.8, retry_attempts=3):
    X, y = generate_sample(n_samples=n_samples, n_features=n_features, distance=distance)
    # each combination is tried 3 times. before raising exception

    for retry_attempt in range(retry_attempts):
        # to initial state
        neural_network = clone(neural_network)
        neural_network.set_params(random_state=42 + retry_attempt)
        print(neural_network)
        neural_network.fit(X, y, epochs=200)
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
        neural_network = nn_type(layers=[5], loss=loss, trainer=trainer)
        yield check_single_classification_network, neural_network


def test_regression_nnets():
    from sklearn.datasets import make_regression
    X, y = make_regression(n_samples=500, n_features=20, n_informative=10, bias=5)
    print(y[:20])
    for loss in ['mse_loss', 'smooth_huber_loss']:
        reg = MLPRegressor(layers=(5,), loss=loss)
        reg.fit(X, y)
        mse = mean_squared_error(y, reg.predict(X))
        print(mse, reg)
    print('original', mean_squared_error(y, y * 0 + y.mean()))


def compare_nnets_quality(n_samples=200, n_features=7, distance=0.8):
    X, y = generate_sample(n_samples=n_samples, n_features=n_features, distance=distance)
    # checking all possible combinations
    for loss in ['log_loss']:  # nnet.losses:
        for NNType in nn_types:
            for trainer in nnet.trainers:
                nn = NNType(layers=[5], loss=loss, trainer=trainer, random_state=42)
                nn.fit(X, y, epochs=100)
                print(roc_auc_score(y, nn.predict_proba(X)[:, 1]), nn)

    lr = LogisticRegression().fit(X, y)
    print(roc_auc_score(y, lr.predict_proba(X)[:, 1]), lr)


def test_network_with_scaler(n_samples=200, n_features=15, distance=0.5):
    X, y = generate_sample(n_samples=n_samples, n_features=n_features, distance=distance)
    for scaler in [BinTransformer(max_bins=16), IronTransformer()]:
        clf = nnet.SimpleNeuralNetwork(scaler=scaler)
        clf.fit(X, y, epochs=300)

        p = clf.predict_proba(X)
        assert roc_auc_score(y, p[:, 1]) > 0.8, 'quality is too low for model: {}'.format(clf)


def test_reproducibility(n_samples=200, n_features=15, distance=0.5):
    X, y = generate_sample(n_samples=n_samples, n_features=n_features, distance=distance)
    for trainer in nnet.trainers.keys():
        clf1 = nnet.MLPClassifier(trainer=trainer, random_state=42).fit(X, y)
        clf2 = nnet.MLPClassifier(trainer=trainer, random_state=42).fit(X, y)
        assert numpy.allclose(clf1.predict_proba(X), clf2.predict_proba(X))
