from __future__ import division, print_function

import tempfile

import numpy
import pandas
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from hep_ml.rootutils import predict_rootfile_by_estimator, predict_by_estimator
from hep_ml.reweight import GBReweighter, ReweighterMixin


def check_estimator(estimator, n_dimensions=2, n_samples=200000):
    mean_original = numpy.random.normal(size=n_dimensions)
    cov_original = numpy.diag([1.] * n_dimensions)

    mean_target = numpy.random.mtrand.multivariate_normal(mean=mean_original, cov=cov_original)
    cov_target = cov_original * 0.4 + numpy.ones([n_dimensions, n_dimensions]) * 0.2

    original = numpy.random.mtrand.multivariate_normal(mean=mean_original, cov=cov_original, size=n_samples + 1)
    original_weight = numpy.ones(n_samples + 1)

    target = numpy.random.mtrand.multivariate_normal(mean=mean_target, cov=cov_target, size=n_samples)
    target_weight = numpy.ones(n_samples)

    if isinstance(estimator, ReweighterMixin):
        estimator.fit(original, target, original_weight=original_weight, target_weight=target_weight)
    else:
        data = numpy.vstack([original, target])
        labels = [0] * (n_samples + 1) + [1] * n_samples
        estimator.fit(data, labels)
    import root_numpy
    root_data = pandas.DataFrame(original, columns=['feature%d' % i for i in range(n_dimensions)])
    predictions = predict_by_estimator(root_data, estimator)
    with tempfile.NamedTemporaryFile(mode="w", suffix='.root', dir='.', delete=True) as rootfile:
        fname = rootfile.name
        root_numpy.array2root(root_data.to_records(index=False), fname, treename='tree', mode='recreate')
        predict_rootfile_by_estimator(fname, 'tree', len(predictions), estimator,
                                      ['feature%d' % i for i in range(n_dimensions)], training_selection=None,
                                      chunk=100, id_column_name='ID_VAR', id_column_dtype='i8', id_column_exist=False,
                                      estimator_column_name='BDT', estimator_column_dtype='f8')
        added_column = root_numpy.root2array(fname, 'tree', branches=['BDT'])['BDT']
        print(added_column)
        print(predictions)
        assert numpy.allclose(added_column, predictions), 'Predictions are different in the root file and initial ones'


def test_add_predictions_reweighter():
    reweighter = GBReweighter(max_depth=3, n_estimators=30, learning_rate=0.3, gb_args=dict(subsample=0.3))
    check_estimator(reweighter, n_dimensions=2, n_samples=200000)


def test_add_predictions_classifier():
    classifier = GradientBoostingClassifier()
    check_estimator(classifier, n_dimensions=2, n_samples=10000)


def test_add_predictions_regressor():
    regressor = GradientBoostingRegressor()
    check_estimator(regressor, n_dimensions=2, n_samples=10000)
