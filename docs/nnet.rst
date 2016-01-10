
Neural networks
===============

.. automodule:: hep_ml.nnet
    :members:
    :show-inheritance:
    :undoc-members:

Loss functions
______________

loss functions are targets in optimization.
The following functions are available for classification.


.. autofunction:: hep_ml.nnet.log_loss

.. autofunction:: hep_ml.nnet.exp_loss

.. autofunction:: hep_ml.nnet.exp_log_loss

.. autofunction:: hep_ml.nnet.squared_loss

Trainers
________

trainers are implemented as functions with some standard parameters and some optional.
Their aim is to optimize the target loss function as some black box.


.. autofunction:: hep_ml.nnet.sgd_trainer

.. autofunction:: hep_ml.nnet.irprop_minus_trainer

.. autofunction:: hep_ml.nnet.irprop_plus_trainer

.. autofunction:: hep_ml.nnet.adadelta_trainer
