"""
**sPlot** is reweighting technique frequently used in HEP to reconstruct the distributions of features in mixture.
Initial information used is the probabilities obtained after fitting.

**hep_ml.splot** contains standalone python implementation of this technique.
This implementation is brilliantly simple and clear - just as it should be.

Example
-------

>>> from hep_ml.splot import compute_sweights
>>> p = pandas.DataFrame({'signal': p_signal, 'bkg', b_bkg})
>>> sWeights = compute_sweights(p)
>>> # plotting reconstructed distribution of some other variable
>>> plt.hist(other_var, weights=sWeights.signal)
>>> plt.hist(other_var, weights=sWeights.bkg)

For more examples and explanations, see notebooks/Splot in repository.

"""

from __future__ import division, print_function, absolute_import
import pandas
import numpy
from .commonutils import check_sample_weight

__author__ = 'Alex Rogozhnikov'


def compute_sweights(probabilities, sample_weight=None):
    """Computes sWeights based on probabilities obtained from distribution fit.

    :param probabilities: pandas.DataFrame with probabilities of shape [n_samples, n_classes].
        These probabilities are obtained after fit (typically, mass fit).
        Pay attention, that for each sample sum of probabilities should be equal to 1.
    :param sample_weight: optionally you can pass weights of events, numpy.array of shape [n_samples]
    :return: pandas.DataFrame with sWeights of shape [n_samples, n_classes]
    """
    # converting to pandas.DataFrame
    probabilities = pandas.DataFrame(probabilities)
    # checking sample_weight
    sample_weight = check_sample_weight(probabilities, sample_weight=sample_weight)
    # checking that all weights are positive
    assert numpy.all(sample_weight >= 0), 'sample weight are expected to be non-negative'

    p = numpy.array(probabilities)
    # checking that probabilities sum up to 1.
    assert numpy.allclose(p.sum(axis=1), 1, atol=1e-3), 'sum of probabilities is not equal to 1.'

    # computations
    initial_stats = (p * sample_weight[:, numpy.newaxis]).sum(axis=0)
    V_inv = p.T.dot(p * sample_weight[:, numpy.newaxis])
    V = numpy.linalg.inv(V_inv) * initial_stats[numpy.newaxis, :]

    # Final formula
    sweights = p.dot(V) * sample_weight[:, numpy.newaxis]
    return pandas.DataFrame(sweights, columns=probabilities.keys())
