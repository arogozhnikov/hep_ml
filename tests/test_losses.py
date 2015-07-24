from __future__ import division, print_function, absolute_import

import numpy
from hep_ml import losses
from hep_ml.commonutils import generate_sample

__author__ = 'Alex Rogozhnikov'


def test_orders(size=40):
    effs1 = losses._compute_positions(numpy.arange(size), numpy.ones(size))
    p = numpy.random.permutation(size)
    effs2 = losses._compute_positions(numpy.arange(size)[p], numpy.ones(size))
    assert numpy.all(effs1[p] == effs2), 'Efficiencies are wrong'
    assert numpy.all(effs1 == numpy.sort(effs1)), 'sortings are wrong'


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
        losses.LogLossFunction(),
        losses.AdaLossFunction(),
        losses.KnnAdaLossFunction(X.columns[:1], knn=5),
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

        print(loss, numer_gradient, numer_hessian)
        assert numpy.allclose(gradient, numer_gradient), 'wrong computation of gradient'
        assert (gradient * (2 * y - 1) >= 0).all(), 'wrong signs of gradients'
        assert numpy.allclose(hessian, numer_hessian, atol=1e-7), 'wrong computation of hessian'


def test_step_optimality(n_samples=50):
    """
    testing that for single leaf function returns the optimal value
    """
    X, y = generate_sample(n_samples, n_features=10)
    rank_column = X.columns[2]
    X[rank_column] = numpy.random.randint(0, 3, size=n_samples)
    sample_weight = numpy.random.exponential(size=n_samples)

    tested_losses = [
        losses.LogLossFunction(),
        losses.AdaLossFunction(),
        losses.KnnAdaLossFunction(X.columns[:1], knn=5),
        losses.CompositeLossFunction(),
        losses.RankBoostLossFunction(rank_column)
    ]

    pred = numpy.random.normal(size=n_samples)

    for loss in tested_losses:
        loss.fit(X, y, sample_weight=sample_weight)
        leaf_value = numpy.random.normal()
        # Some basic optimization goes here:
        new_value = 0.
        for _ in range(4):
            ministep, = loss.prepare_new_leaves_values(
                terminal_regions=numpy.zeros(n_samples, dtype=int),
                leaf_values=[leaf_value], X=X, y=y, y_pred=pred + new_value, sample_weight=sample_weight,
                update_mask=None, residual=loss.negative_gradient(pred + new_value))
            new_value += ministep

        print(new_value)
        loss_values = []
        coeffs = [0.9, 1.0, 1.1]
        for coeff in coeffs:
            loss_values.append(loss(pred + coeff * new_value))
        print(loss, new_value, 'losses: ', loss_values)
        assert loss_values[1] <= loss_values[0] + 1e-7
        assert loss_values[1] <= loss_values[2] + 1e-7
