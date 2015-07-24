from __future__ import division, print_function, absolute_import
import numpy
from hep_ml.commonutils import generate_sample
from hep_ml.losses import LogLossFunction, KnnAdaLossFunction, \
    BinFlatnessLossFunction, KnnFlatnessLossFunction, AdaLossFunction, RankBoostLossFunction, CompositeLossFunction
from hep_ml.gradientboosting import GradientBoostingClassifier as uGradientBoostingClassifier


def test_gb_with_ada(n_samples=1000, n_features=10, distance=0.6):
    testX, testY = generate_sample(n_samples, n_features, distance=distance)
    trainX, trainY = generate_sample(n_samples, n_features, distance=distance)
    loss = LogLossFunction()
    clf = uGradientBoostingClassifier(loss=loss, min_samples_split=20, max_depth=5, learning_rate=.2,
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
    rank_variable = 'column1'
    trainX[rank_variable] = numpy.random.randint(0, 3, size=len(trainX))
    testX[rank_variable] = numpy.random.randint(0, 3, size=len(testX))

    loss1 = LogLossFunction()
    loss2 = AdaLossFunction()
    loss3 = CompositeLossFunction()
    loss4 = KnnAdaLossFunction(uniform_features=uniform_features)
    loss5 = RankBoostLossFunction(request_column=rank_variable)
    loss51 = RankBoostLossFunction(request_column=rank_variable, update_iterations=2)
    loss52 = RankBoostLossFunction(request_column=rank_variable, update_iterations=10)
    loss6bin = BinFlatnessLossFunction(uniform_features, ada_coefficient=0.5)
    loss7bin = BinFlatnessLossFunction(uniform_features, ada_coefficient=0.5, uniform_label=[0, 1])
    loss6knn = KnnFlatnessLossFunction(uniform_features, ada_coefficient=0.5)
    loss7knn = KnnFlatnessLossFunction(uniform_features, ada_coefficient=0.5, uniform_label=[0, 1])

    for loss in [loss5, loss51, loss52, loss1, loss2, loss3, loss4, loss5, loss6bin, loss7bin, loss6knn, loss7knn]:
        clf = uGradientBoostingClassifier(loss=loss, min_samples_split=20, max_depth=5, learning_rate=0.2,
                                          subsample=0.7, n_estimators=25, train_features=None) \
            .fit(trainX[:n_samples], trainY[:n_samples])
        result = clf.score(testX, testY)
        assert result >= 0.7, "The quality is too poor: {} with loss: {}".format(result, loss)

