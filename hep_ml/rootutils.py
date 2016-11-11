from __future__ import print_function, division, absolute_import

import numpy
import pandas
from sklearn.base import ClassifierMixin, RegressorMixin
from .reweight import ReweighterMixin


def add_column_to_rootfile(column, filename, treename, dtype, column_name):
    """
    Add column to the existence root file.
    **This function works only if ROOT, rootpy, root_numpy are installed.**

    :param array column: column values
    :param str filename: root file name
    :param str treename: tree name
    :param str dtype: type
    :param str column_name: creating branch name for adding column
    """
    from rootpy.io import root_open
    import root_numpy
    with root_open(filename, mode='a') as rootfile:
        new_column = numpy.array(column, dtype=[(column_name, dtype)])
        root_numpy.array2tree(new_column, tree=rootfile[treename])
        rootfile.write()


def predict_by_estimator(data, estimator):
    """
    Predict samples by estimator: classifier, regressor or reweighter

    :param data: data which are necessary to predict
    :type data: pandas.DataFrame or array-like
    :param estimator: trained estimator by which data will be marked
    :type estimator: ClassifierMixin or RegressorMixin or ReweighterMixin

    :return: predictions
    """
    prediction = None
    if isinstance(estimator, ClassifierMixin):
        prediction = estimator.predict_proba(data)[:, 1]
    elif isinstance(estimator, RegressorMixin):
        prediction = estimator.predict(data)
    elif isinstance(estimator, ReweighterMixin):
        prediction = estimator.predict_weights(data)
    assert prediction is not None, 'estimator cannot predict samples'
    return prediction


def compute_indices_of_intersection(ids_sorted, searched_ids):
    """
    Compute positions of the searched ids in the full ids vector

    :param ids_sorted: sorted full ids vector
    :param searched_ids: searched ids (for them positions in the ids_sorted ara considered)
    :return: positions
    """
    positions = numpy.searchsorted(ids_sorted, searched_ids)
    match = searched_ids == ids_sorted[positions]
    return positions[match]


def predict_rootfile_by_estimator(filename, treename, filelength, estimator,
                                  branches, training_selection=None,
                                  chunk=10000, id_column_name='ID_column', id_column_dtype='i8', id_column_exist=False,
                                  estimator_column_name='BDT', estimator_column_dtype='f8'):
    """
    Predict the whole root file by estimator (classifier, regressor or reweighter) and add predictions into file.
    Identification value and prediction for each sample will be added.
    **This function works only if ROOT, rootpy, root_numpy are installed.**

    :param str ilename: root file name
    :param str treename: tree name
    :param int filelength: the length of tree
    :param estimator: trained estimator by which data will be marked
    :type estimator: ClassifierMixin or RegressorMixin or ReweighterMixin
    :param list branches: name of columns used by estimator in the same order as for estimator training
    :param training_selection: selection for training samples,
     if None then training selection is not used. (This is needed for folding estimators)
    :type training_selection: str or None
    :param int chunk: chunk of samples for reading from file
    :param str id_column_name: id column name
    :param str id_column_dtype: dtype of id column
    :param bool id_column_exist: exist or not id column. If not then it will be added.
    :param estimator_column_name:
    :param estimator_column_dtype:
    :return:
    """
    import root_numpy

    if not id_column_exist:
        ids = numpy.arange(filelength)
        add_column_to_rootfile(ids, filename, treename, id_column_dtype, column_name=id_column_name)

    ids = root_numpy.root2array(filename, treename=treename, branches=[id_column_name])[id_column_name]
    ids_sorted = numpy.sort(ids)
    indices_of_ids = numpy.argsort(ids)

    whole_predictions = numpy.zeros(filelength)

    remaining_length = filelength

    test_selection = ""
    if training_selection is not None:
        # predict training samples, this is necessary for folding scheme of training which predictions depend on samples
        training_data = pandas.DataFrame(root_numpy.root2array(filename, treename=treename,
                                                               branches=list(branches) + [id_column_name],
                                                               selection=training_selection))
        training_idices = compute_indices_of_intersection(ids_sorted, training_data[id_column_name].values)

        whole_predictions[indices_of_ids[training_idices]] = predict_by_estimator(training_data, estimator)
        remaining_length = filelength - len(training_data)
        test_selection = "!({})".format(training_selection)

    read_samples = 0
    for _ in range((remaining_length - 1) // chunk + 1):
        read_sample_last_index = read_samples + chunk
        if read_sample_last_index > remaining_length:
            read_sample_last_index = remaining_length
        if read_samples >= read_sample_last_index:
            break
        data = pandas.DataFrame(root_numpy.root2array(filename, treename=treename,
                                                      branches=list(branches) + [id_column_name],
                                                      selection= test_selection,
                                                      start=read_samples, stop=read_sample_last_index))

        indices = compute_indices_of_intersection(ids_sorted, data[id_column_name].values)
        whole_predictions[indices_of_ids[indices]] = predict_by_estimator(data[branches], estimator)
        read_samples += chunk
    add_column_to_rootfile(whole_predictions, filename, treename,
                           estimator_column_dtype, column_name=estimator_column_name)
