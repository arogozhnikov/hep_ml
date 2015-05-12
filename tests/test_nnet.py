"""
Testing all the nnet library
"""
from __future__ import division, print_function

import numpy
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.datasets import make_blobs
from sklearn.metrics import roc_auc_score

from hep_ml import nnet


__author__ = 'Alex Rogozhnikov'


def test_nnet(n_samples=200, n_features=5, distance=0.5, complete=False):
    X, y = make_blobs(n_samples=n_samples, n_features=5,
                      centers=[numpy.ones(n_features) * distance, - numpy.ones(n_features) * distance])

    nn_types = [
        nnet.SimpleNeuralNetwork,
        nnet.MultiLayerNetwork,
        nnet.SoftmaxNeuralNetwork,
        nnet.RBFNeuralNetwork,
        nnet.PairwiseNeuralNetwork,
        nnet.PairwiseSoftplusNeuralNetwork,
    ]

    if complete:
        # checking all possible combinations
        for loss in nnet.losses:
            for NNType in nn_types:
                for trainer in nnet.trainers:
                    nn = NNType(layers=[5], loss=loss, trainer=trainer, random_state=42)
                    nn.fit(X, y, epochs=100)
                    print(roc_auc_score(y, nn.predict_proba(X)[:, 1]), nn)

        lr = LogisticRegression().fit(X, y)
        print(lr, roc_auc_score(y, lr.predict_proba(X)[:, 1]))

        assert 0 == 1, "Let's see and compare results"
    else:
        # checking combinations of losses, nn_types, trainers, most of them are playing once.
        attempts = max(len(nnet.losses), len(nnet.trainers), len(nn_types))
        losses_shift = numpy.random.randint(10)
        trainers_shift = numpy.random.randint(10)
        for attempt in range(attempts):
            loss = nnet.losses.keys()[(attempt + losses_shift) % len(nnet.losses)]
            trainer = nnet.trainers.keys()[(attempt + trainers_shift) % len(nnet.trainers)]

            nn_type = nn_types[attempt % len(nn_types)]

            nn = nn_type(layers=[5], loss=loss, trainer=trainer, random_state=42)
            print(nn)
            nn.fit(X, y, epochs=200)
            assert roc_auc_score(y, nn.predict_proba(X)[:, 1]) > 0.8, \
                'quality of model is too low: {}'.format(nn)

