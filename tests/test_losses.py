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
    Testing that Hessians and gradients of loss functions coincide with numerical approximations
    """
    X, y = generate_sample(size, n_features=10)
    rank_column = X.columns[2]
    X[rank_column] = numpy.random.randint(0, 3, size=size)
    sample_weight = numpy.random.exponential(size=size)
    tested_losses = [
        losses.MSELossFunction(),
        losses.MAELossFunction(),
        losses.LogLossFunction(),
        losses.AdaLossFunction(),
        losses.KnnAdaLossFunction(X.columns[:1], uniform_label=1, knn=5),
        losses.CompositeLossFunction(),
        losses.RankBoostLossFunction(rank_column),
    ]
    pred = numpy.random.normal(size=size)
    # y = pred is a special point in i.e. MAELossFunction
    pred[numpy.abs(y - pred) < epsilon] = - 0.1
    print(sum(numpy.abs(y - pred) < epsilon))

    for loss in tested_losses:
        loss.fit(X, y, sample_weight=sample_weight)
        # testing sign of gradient
        val = loss(pred)
        gradient = loss.negative_gradient(pred)

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

        assert numpy.allclose(gradient, numer_gradient), 'wrong computation of gradient for {}'.format(loss)
        if not isinstance(loss, losses.MSELossFunction) and not isinstance(loss, losses.MAELossFunction):
            assert (gradient * (2 * y - 1) >= 0).all(), 'wrong signs of gradients'
        if isinstance(loss, losses.HessianLossFunction):
            hessian = loss.hessian(pred)
            assert numpy.allclose(hessian, numer_hessian, atol=1e-5), 'wrong computation of hessian for {}'.format(loss)


def test_step_optimality(n_samples=100):
    """
    testing that for single leaf function returns the optimal value
    """
    X, y = generate_sample(n_samples, n_features=10)
    sample_weight = numpy.random.exponential(size=n_samples)

    rank_column = X.columns[2]
    X[rank_column] = numpy.random.randint(0, 3, size=n_samples)

    tested_losses = [
        losses.LogLossFunction(),
        losses.AdaLossFunction(),
        losses.KnnAdaLossFunction(X.columns[:1], uniform_label=0, knn=5),
        losses.CompositeLossFunction(),
        losses.RankBoostLossFunction(rank_column),
        losses.MSELossFunction(),
    ]

    pred = numpy.random.normal(size=n_samples)

    for loss in tested_losses:
        loss.fit(X, y, sample_weight=sample_weight)

        # Test simple way to get optimal step
        leaf_value = numpy.random.normal()
        step = 0.
        for _ in range(4):
            ministep, = loss.prepare_new_leaves_values(terminal_regions=numpy.zeros(n_samples, dtype=int),
                                                       leaf_values=[leaf_value], y_pred=pred + step)
            step += ministep

        if isinstance(loss, losses.MAELossFunction):
            # checking that MAE is minimized with long process
            for iteration in range(1, 30):
                ministep, = loss.prepare_new_leaves_values(terminal_regions=numpy.zeros(n_samples, dtype=int),
                                                           leaf_values=[leaf_value], y_pred=pred + step)
                step += ministep * 1. / iteration

        loss_values = []
        coeffs = [0.9, 1.0, 1.1]
        for coeff in coeffs:
            loss_values.append(loss(pred + coeff * step))
        print(loss, step, 'losses: ', loss_values)
        assert loss_values[1] <= loss_values[0] + 1e-7
        assert loss_values[1] <= loss_values[2] + 1e-7

        # Test standard function
        opt_value = loss.compute_optimal_step(y_pred=pred)
        loss_values2 = []
        for coeff in coeffs:
            loss_values2.append(loss(pred + coeff * opt_value))
        print(loss, step, 'losses: ', loss_values)
        assert loss_values2[1] <= loss_values2[0] + 1e-7
        assert loss_values2[1] <= loss_values2[2] + 1e-7
