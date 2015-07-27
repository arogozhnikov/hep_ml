from __future__ import division, print_function, absolute_import
import numpy
from sklearn.metrics import mean_squared_error, roc_auc_score
from hep_ml.commonutils import generate_sample
from hep_ml.losses import LogLossFunction, KnnAdaLossFunction, \
    BinFlatnessLossFunction, KnnFlatnessLossFunction, AdaLossFunction, \
    RankBoostLossFunction, CompositeLossFunction, MSELossFunction
from hep_ml.gradientboosting import UGradientBoostingClassifier, UGradientBoostingRegressor


def test_gb_with_ada(n_samples=1000, n_features=10, distance=0.6):
    testX, testY = generate_sample(n_samples, n_features, distance=distance)
    trainX, trainY = generate_sample(n_samples, n_features, distance=distance)
    loss = LogLossFunction()
    clf = UGradientBoostingClassifier(loss=loss, min_samples_split=20, max_depth=5, learning_rate=.2,
                                      subsample=0.7, n_estimators=10, train_features=None)
    clf.fit(trainX, trainY)
    assert clf.n_features == n_features
    assert len(clf.feature_importances_) == n_features
    # checking that predict proba works
    for p in clf.staged_predict_proba(testX):
        assert p.shape == (n_samples, 2)
    assert numpy.all(p == clf.predict_proba(testX))


def test_gradient_boosting(n_samples=1000):
    """
    Testing workability of GradientBoosting with different loss function
    """
    # Generating some samples correlated with first variable
    distance = 0.6
    testX, testY = generate_sample(n_samples, 10, distance)
    trainX, trainY = generate_sample(n_samples, 10, distance)
    # We will try to get uniform distribution along this variable
    uniform_features = ['column0']

    loss1 = LogLossFunction()
    loss2 = AdaLossFunction()
    loss3 = CompositeLossFunction()
    loss4 = KnnAdaLossFunction(uniform_features=uniform_features, uniform_label=1)
    loss5 = KnnAdaLossFunction(uniform_features=uniform_features, uniform_label=[0, 1])
    loss6bin = BinFlatnessLossFunction(uniform_features, fl_coefficient=2., uniform_label=0)
    loss7bin = BinFlatnessLossFunction(uniform_features, fl_coefficient=2., uniform_label=[0, 1])
    loss6knn = KnnFlatnessLossFunction(uniform_features, fl_coefficient=2., uniform_label=1)
    loss7knn = KnnFlatnessLossFunction(uniform_features, fl_coefficient=2., uniform_label=[0, 1])

    for loss in [loss1, loss2, loss3, loss4, loss5, loss6bin, loss7bin, loss6knn, loss7knn]:
        clf = UGradientBoostingClassifier(loss=loss, min_samples_split=20, max_depth=5, learning_rate=0.2,
                                          subsample=0.7, n_estimators=25, train_features=None) \
            .fit(trainX[:n_samples], trainY[:n_samples])
        result = clf.score(testX, testY)
        assert result >= 0.7, "The quality is too poor: {} with loss: {}".format(result, loss)


def test_gb_regression(n_samples=1000):
    X, _ = generate_sample(n_samples, 10, distance=0.6)
    y = numpy.tanh(X.sum(axis=1))
    clf = UGradientBoostingRegressor(loss=MSELossFunction())
    clf.fit(X, y)
    y_pred = clf.predict(X)
    assert mean_squared_error(y, y_pred) < 0.5 * mean_squared_error(y, y * 0.), 'something wrong with regression quality'


def test_gb_ranking(n_samples=1000):
    distance = 0.6
    testX, testY = generate_sample(n_samples, 10, distance)
    trainX, trainY = generate_sample(n_samples, 10, distance)

    rank_variable = 'column1'
    trainX[rank_variable] = numpy.random.randint(0, 3, size=len(trainX))
    testX[rank_variable] = numpy.random.randint(0, 3, size=len(testX))

    rank_loss1 = RankBoostLossFunction(request_column=rank_variable, update_iterations=1)
    rank_loss2 = RankBoostLossFunction(request_column=rank_variable, update_iterations=2)
    rank_loss3 = RankBoostLossFunction(request_column=rank_variable, update_iterations=10)

    for loss in [rank_loss1, rank_loss2, rank_loss3]:
        clf = UGradientBoostingRegressor(loss=loss, min_samples_split=20, max_depth=5, learning_rate=0.2,
                                         subsample=0.7, n_estimators=25, train_features=None) \
            .fit(trainX[:n_samples], trainY[:n_samples])
        result = roc_auc_score(testY, clf.predict(testX))
        assert result >= 0.8, "The quality is too poor: {} with loss: {}".format(result, loss)
