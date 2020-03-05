from __future__ import division, print_function, absolute_import
import copy
import numpy
from sklearn.metrics import mean_squared_error, roc_auc_score
from sklearn.base import clone
from hep_ml.commonutils import generate_sample
from hep_ml.losses import LogLossFunction, MSELossFunction, AdaLossFunction
from hep_ml import losses
from hep_ml.gradientboosting import UGradientBoostingClassifier, UGradientBoostingRegressor


def test_gb_with_ada_and_log(n_samples=1000, n_features=10, distance=0.6):
    """
    Testing with two main classification losses.
    Also testing copying
    """
    testX, testY = generate_sample(n_samples, n_features, distance=distance)
    trainX, trainY = generate_sample(n_samples, n_features, distance=distance)
    for loss in [LogLossFunction(), AdaLossFunction()]:
        clf = UGradientBoostingClassifier(loss=loss, min_samples_split=20, max_depth=5, learning_rate=.2,
                                          subsample=0.7, n_estimators=10, train_features=None)
        clf.fit(trainX, trainY)
        assert clf.n_features == n_features
        assert len(clf.feature_importances_) == n_features
        # checking that predict proba works
        for p in clf.staged_predict_proba(testX):
            assert p.shape == (n_samples, 2)
        assert numpy.all(p == clf.predict_proba(testX))
        assert roc_auc_score(testY, p[:, 1]) > 0.8, 'quality is too low'
        # checking clonability
        _ = clone(clf)
        clf_copy = copy.deepcopy(clf)
        assert numpy.all(clf.predict_proba(trainX) == clf_copy.predict_proba(trainX)), 'copied classifier is different'


def test_gradient_boosting(n_samples=1000, distance = 0.6):
    """
    Testing workability of GradientBoosting with different loss function
    """
    # Generating some samples correlated with first variable
    testX, testY = generate_sample(n_samples, 10, distance)
    trainX, trainY = generate_sample(n_samples, 10, distance)
    # We will try to get uniform distribution along this variable
    uniform_features = ['column0']

    loss1 = LogLossFunction()
    loss2 = AdaLossFunction()
    loss3 = losses.CompositeLossFunction()
    loss4 = losses.KnnAdaLossFunction(uniform_features=uniform_features, knn=5, uniform_label=1)
    loss5 = losses.KnnAdaLossFunction(uniform_features=uniform_features, knn=5, uniform_label=[0, 1])
    loss6bin = losses.BinFlatnessLossFunction(uniform_features, fl_coefficient=1., uniform_label=0)
    loss7bin = losses.BinFlatnessLossFunction(uniform_features, fl_coefficient=1., uniform_label=[0, 1])
    loss6knn = losses.KnnFlatnessLossFunction(uniform_features, fl_coefficient=1., uniform_label=1)
    loss7knn = losses.KnnFlatnessLossFunction(uniform_features, fl_coefficient=1., uniform_label=[0, 1])

    for loss in [loss1, loss2, loss3, loss4, loss5, loss6bin, loss7bin, loss6knn, loss7knn]:
        clf = UGradientBoostingClassifier(loss=loss, min_samples_split=20, max_depth=5, learning_rate=0.2,
                                          subsample=0.7, n_estimators=25, train_features=None)
        clf.fit(trainX[:n_samples], trainY[:n_samples])
        result = clf.score(testX, testY)
        assert result >= 0.7, "The quality is too poor: {} with loss: {}".format(result, loss)

    trainX['fake_request'] = numpy.random.randint(0, 4, size=len(trainX))
    testX['fake_request'] = numpy.random.randint(0, 4, size=len(testX))
    for loss in [losses.MSELossFunction(),
                 losses.MAELossFunction(),
                 losses.RankBoostLossFunction(request_column='fake_request')]:
        print(loss)
        clf = UGradientBoostingRegressor(loss=loss, max_depth=3, n_estimators=50, learning_rate=0.01, subsample=0.5,
                                         train_features=list(trainX.columns[1:]))
        clf.fit(trainX, trainY)
        roc_auc = roc_auc_score(testY, clf.predict(testX))
        assert roc_auc >= 0.7, "The quality is too poor: {} with loss: {}".format(roc_auc, loss)


def test_gb_regression(n_samples=1000):
    X, _ = generate_sample(n_samples, 10, distance=0.6)
    y = numpy.tanh(X.sum(axis=1))
    clf = UGradientBoostingRegressor(loss=MSELossFunction())
    clf.fit(X, y)
    y_pred = clf.predict(X)
    zeromse = 0.5 * mean_squared_error(y, y * 0.)
    assert mean_squared_error(y, y_pred) < zeromse, 'something wrong with regression quality'


def test_gb_ranking(n_samples=1000):
    """
    Testing RankingLossFunction
    """
    distance = 0.6
    testX, testY = generate_sample(n_samples, 10, distance)
    trainX, trainY = generate_sample(n_samples, 10, distance)

    rank_variable = 'column1'
    trainX[rank_variable] = numpy.random.randint(0, 3, size=len(trainX))
    testX[rank_variable] = numpy.random.randint(0, 3, size=len(testX))

    rank_loss1 = losses.RankBoostLossFunction(request_column=rank_variable, update_iterations=1)
    rank_loss2 = losses.RankBoostLossFunction(request_column=rank_variable, update_iterations=2)
    rank_loss3 = losses.RankBoostLossFunction(request_column=rank_variable, update_iterations=10)

    for loss in [rank_loss1, rank_loss2, rank_loss3]:
        clf = UGradientBoostingRegressor(loss=loss, min_samples_split=20, max_depth=5, learning_rate=0.2,
                                         subsample=0.7, n_estimators=25, train_features=None) \
            .fit(trainX[:n_samples], trainY[:n_samples])
        result = roc_auc_score(testY, clf.predict(testX))
        assert result >= 0.8, "The quality is too poor: {} with loss: {}".format(result, loss)


def test_constant_fitting(n_samples=1000, n_features=5):
    """
    Testing if initial constant fitted properly
    """
    X, y = generate_sample(n_samples=n_samples, n_features=n_features)
    y = y.astype(numpy.float) + 1000.
    for loss in [MSELossFunction(), losses.MAELossFunction()]:
        gb = UGradientBoostingRegressor(loss=loss, n_estimators=10)
        gb.fit(X, y)
        p = gb.predict(X)
        assert mean_squared_error(p, y) < 0.5


def test_weight_misbalance(n_samples=1000, n_features=10, distance=0.6):
    """
    Testing how classifiers work with highly misbalanced (in the terms of weights) datasets.
    """
    testX, testY = generate_sample(n_samples, n_features, distance=distance)
    trainX, trainY = generate_sample(n_samples, n_features, distance=distance)
    trainW = trainY * 10000 + 1
    testW = testY * 10000 + 1
    for loss in [LogLossFunction(), AdaLossFunction(), losses.CompositeLossFunction()]:
        clf = UGradientBoostingClassifier(loss=loss, min_samples_split=20, max_depth=5, learning_rate=.2,
                                          subsample=0.7, n_estimators=10, train_features=None)
        clf.fit(trainX, trainY, sample_weight=trainW)
        p = clf.predict_proba(testX)
        assert roc_auc_score(testY, p[:, 1], sample_weight=testW) > 0.8, 'quality is too low'
