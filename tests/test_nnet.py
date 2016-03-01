"""
Testing all the nnet library
"""
from __future__ import division, print_function

import numpy
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.base import clone

from hep_ml import nnet
from hep_ml.commonutils import generate_sample
from hep_ml.preprocessing import BinTransformer, IronTransformer

__author__ = 'Alex Rogozhnikov'


def check_single_network(neural_network, n_samples=200, n_features=7, distance=0.8, retry_attempts=3):
    X, y = generate_sample(n_samples=n_samples, n_features=n_features, distance=distance)
    for retry_attempt in range(retry_attempts):
        # to initial state
        neural_network = clone(neural_network)
        neural_network.set_params(random_state=42 + retry_attempt)
        print(neural_network)
        neural_network.fit(X, y, epochs=200)
        quality = roc_auc_score(y, neural_network.predict_proba(X)[:, 1])
        computed_loss = neural_network.compute_loss(X, y, sample_weight=y * 0 + 1)
        if quality > 0.8:
            break
        else:
            print('attempt {} : {}'.format(retry_attempt, quality))
            if retry_attempt == retry_attempts - 1:
                raise RuntimeError('quality of model is too low: {} {}'.format(quality, neural_network))


def test_nnet(n_samples=200, n_features=7, distance=0.8, complete=False):
    """
    :param complete: if True, all possible combinations will be checked, quality is printed
    """

    nn_types = [
        nnet.SimpleNeuralNetwork,
        nnet.MLPClassifier,
        nnet.SoftmaxNeuralNetwork,
        nnet.RBFNeuralNetwork,
        nnet.PairwiseNeuralNetwork,
        nnet.PairwiseSoftplusNeuralNetwork,
    ]

    if complete:
        X, y = generate_sample(n_samples=n_samples, n_features=n_features, distance=distance)
        # checking all possible combinations
        for loss in nnet.losses:
            for NNType in nn_types:
                for trainer in nnet.trainers:
                    nn = NNType(layers=[5], loss=loss, trainer=trainer, random_state=42)
                    nn.fit(X, y, epochs=100)
                    print(roc_auc_score(y, nn.predict_proba(X)[:, 1]), nn)

        lr = LogisticRegression().fit(X, y)
        print(lr, roc_auc_score(y, lr.predict_proba(X)[:, 1]))
    else:
        # checking combinations of losses, nn_types, trainers, most of them are used once during tests.
        attempts = max(len(nnet.losses), len(nnet.trainers), len(nn_types))
        losses_shift = numpy.random.randint(10)
        trainers_shift = numpy.random.randint(10)
        for attempt in range(attempts):
            # each combination is tried 3 times. before raising exception

            loss = list(nnet.losses.keys())[(attempt + losses_shift) % len(nnet.losses)]
            trainer = list(nnet.trainers.keys())[(attempt + trainers_shift) % len(nnet.trainers)]

            nn_type = nn_types[attempt % len(nn_types)]
            neural_network = nn_type(layers=[5], loss=loss, trainer=trainer)
            yield check_single_network, neural_network


def test_network_with_scaler(n_samples=200, n_features=15, distance=0.5):
    X, y = generate_sample(n_samples=n_samples, n_features=n_features, distance=distance)
    for scaler in [BinTransformer(max_bins=16), IronTransformer()]:
        clf = nnet.SimpleNeuralNetwork(scaler=scaler)
        clf.fit(X, y, epochs=300)

        p = clf.predict_proba(X)
        assert roc_auc_score(y, p[:, 1]) > 0.8, 'quality is too low for model: {}'.format(clf)
