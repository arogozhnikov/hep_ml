from __future__ import division, print_function, absolute_import

import numpy
from hep_ml import losses
from hep_ml.commonutils import generate_sample

__author__ = 'Alex Rogozhnikov'


def test_loss_functions(size=50, epsilon=1e-3):
    """
    Testing that Hessians and gradients of loss functions
    coincide with numerical approximations
    """
    X, y = generate_sample(size, n_features=10)
    rank_column = X.columns[2]
    X[rank_column] = numpy.random.randint(0, 3, size=size)
    sample_weight = numpy.random.exponential(size=size)
    tested_losses = [
        losses.BinomialDevianceLossFunction(),
        losses.AdaLossFunction(),
        losses.SimpleKnnLossFunction(X.columns[:1], knn=5),
        losses.CompositeLossFunction(),
        losses.RankBoostLossFunction(rank_column)
    ]
    pred = numpy.random.normal(size=size)

    for loss in tested_losses:
        loss.fit(X, y, sample_weight=sample_weight)
        # testing sign of gradient
        val = loss(pred)
        gradient = loss.negative_gradient(pred)
        hessian = loss.hessian(pred)

        numer_gradient = numpy.zeros(len(pred))
        numer_hessian = numpy.zeros(len(pred))
        for i in range(size):
            pred_plus = pred.copy()
            pred_plus[i] += epsilon
            val_plus = loss(pred_plus)

            pred_minus = pred.copy()
            pred_minus[i] -= epsilon
            val_minus = loss(pred_minus)

            numer_gradient[i] = - (val_plus - val_minus) / 2. / epsilon
            numer_hessian[i] = (val_plus + val_minus - 2 * val) / epsilon ** 2

        print(loss)
        assert numpy.allclose(gradient, numer_gradient), 'wrong computation of gradient'
        assert (gradient * (2 * y - 1) >= 0).all(), 'wrong signs of gradients'
        if isinstance(loss, losses.RankBoostLossFunction):
            assert numpy.allclose(hessian, numer_hessian, rtol=1e-1, atol=1e-2), 'wrong computation of hessian'
        else:
            assert numpy.allclose(hessian, numer_hessian, atol=1e-7), 'wrong computation of hessian'
