"""
**hep_ml.nnet** is minimalistic **theano**-powered version of feed-forward neural networks.
The neural networks from this library provide sklearn classifier's interface.

Definitions for loss functions, trainers of neural networks are defined in this file too.
Main point of this library: black-box stochastic optimization of any given loss function.
This gives ability to define any activation expression (at the cost of unavailability of pretraining).

In this file we have **examples** of neural networks,
user is encouraged to write his own specific architecture,
which can be much more complex than those used usually.

If you don't want to dive into details, use
:class:`hep_ml.nnet.MLPClassifier`, :class:`hep_ml.nnet.MLPRegressor`

This library should be preferred for different experiments with architectures.
Also **hep_ml.nnet** allows optimization of parameters in any differentiable decision function.

Being written in theano, these neural networks are able to make use of your GPU.

See also libraries: keras, mxnet, pytorch.

Examples
________

Training a neural network with two hidden layers using IRPROP- algorithm

>>> network = MLPClassifier(layers=[7, 7], loss='log_loss', trainer='irprop-', epochs=1000)
>>> network.fit(X, y)
>>> probability = network.predict_proba(X)

Training an AdaBoost over neural network with adadelta trainer. Trainer specific parameter was used
(size of minibatch)

>>> from sklearn.ensemble import AdaBoostClassifier
>>> base_network = MLPClassifier(layers=[10], trainer='adadelta', trainer_parameters={'batch': 600})
>>> classifier = AdaBoostClassifier(base_estimator=base_network, n_estimators=20)
>>> classifier.fit(X, y)

Using custom pretransformer and ExponentialLoss:

>>> from sklearn.preprocessing import PolynomialFeatures
>>> network = MLPClassifier(layers=[10], scaler=PolynomialFeatures(), loss='exp_loss')

To create custom neural network, see code of SimpleNeuralNetwork, which is good place to start.

Interface
_________

Below an interface for classifier and regressor is demonstrated.

.. autoclass :: AbstractNeuralNetworkClassifier
    :members: fit, predict, predict_proba

.. autoclass :: AbstractNeuralNetworkRegressor
    :members: fit, predict


Recommended networks
____________________

Multilayer Perceptron is well-known popular algorithm working well in most applications.

.. autoclass :: MLPClassifier

.. autoclass :: MLPRegressor

.. autoclass :: MLPMultiClassifier

Custom networks
_______________

Below some examples of custom networks are given.
Those are initial point for constructing own architectures.

.. autoclass :: SimpleNeuralNetwork

.. autoclass :: SoftmaxNeuralNetwork

.. autoclass :: RBFNeuralNetwork

.. autoclass :: PairwiseNeuralNetwork

.. autoclass :: PairwiseSoftplusNeuralNetwork


Loss functions
______________

The following loss functions are available for **classification**:


.. autofunction:: hep_ml.nnet.log_loss

.. autofunction:: hep_ml.nnet.exp_loss

.. autofunction:: hep_ml.nnet.exp_log_loss

.. autofunction:: hep_ml.nnet.squared_loss

The following loss functions are available for **regression**:

.. autofunction:: hep_ml.nnet.mse_loss

.. autofunction:: hep_ml.nnet.smooth_huber_loss


Trainers
________

The trainers are optimization algorithms used to minimize target loss function in neural networks.
The trainers are implemented as functions with some standard parameters and some optional.


.. autofunction:: hep_ml.nnet.sgd_trainer

.. autofunction:: hep_ml.nnet.irprop_minus_trainer

.. autofunction:: hep_ml.nnet.irprop_plus_trainer

.. autofunction:: hep_ml.nnet.adadelta_trainer


"""
from __future__ import print_function, division, absolute_import
from copy import deepcopy

import numpy
import theano
import theano.tensor as T
from theano.ifelse import ifelse
from theano.tensor.shared_randomstreams import RandomStreams

from sklearn.utils.validation import check_random_state
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, TransformerMixin, clone
from sklearn import preprocessing
from .commonutils import check_xyw, check_sample_weight, score_to_proba
from .preprocessing import IronTransformer

floatX = theano.config.floatX
DEFAULT_MINIBATCH = 30

__author__ = 'Alex Rogozhnikov'
__all__ = ['MLPClassifier',
           'MLPRegressor',
           'SimpleNeuralNetwork',
           'SoftmaxNeuralNetwork',
           'RBFNeuralNetwork',
           'PairwiseNeuralNetwork',
           'PairwiseSoftplusNeuralNetwork',
           ]


# region Loss functions


def squared_loss(y, pred, w):
    """ Squared loss for classification, not to be messed up with MSE"""
    return T.mean(w * (y - T.nnet.sigmoid(pred)) ** 2)


def log_loss(y, pred, w):
    """ Logistic loss for classification (aka cross-entropy, aka binomial deviance) """
    margin = pred * (1 - 2 * y)
    return T.mean(w * T.nnet.softplus(margin))


def exp_loss(y, pred, w):
    """ Exponential loss for classification (aka AdaLoss function) """
    margin = pred * (1 - 2 * y)
    return T.mean(w * T.exp(margin))


def exp_log_loss(y, pred, w):
    """ Classification loss function,
    combines logistic loss for signal and exponential loss for background """
    return 2 * log_loss(y, pred, w=w * y) + exp_loss(y, pred, w=w * (1 - y))


def mse_loss(y, pred, w):
    """ Regression loss function, mean squared error. """
    return T.mean(w * (y - pred) ** 2)


def smooth_huber_loss(y, pred, w):
    """ Regression loss function, smooth version of Huber loss function. """
    difference = abs(y - pred)
    # error = logaddexp(-difference, difference), but this is not implemented in theano.
    error = difference + T.log(1 + T.exp(-2 * difference))
    return T.mean(w * error)


classification_losses = {
    'exp_loss': exp_loss,
    'log_loss': log_loss,
    'exp_log_loss': exp_log_loss,
    'squared_loss': squared_loss,
}

regression_losses = {
    'mse_loss': mse_loss,
    'smooth_huber_loss': smooth_huber_loss,
}

losses = classification_losses.copy()
losses.update(regression_losses)


# endregion


# region Trainers
def get_batch(x, y, w, random_stream, batch_size=DEFAULT_MINIBATCH):
    """ Generates subset of training dataset, of size batch"""
    indices = random_stream.choice(a=T.shape(x)[0], size=(batch_size,))
    return x[indices], y[indices], w[indices]


def sgd_trainer(x, y, w, parameters, loss, random_stream, batch=DEFAULT_MINIBATCH,
                learning_rate=0.1, l2_penalty=0.001, momentum=0.9, ):
    """Stochastic gradient descent with momentum, trivial but very popular.

    :param int batch: size of minibatch, each time averaging gradient over minibatch.
    :param float learning_rate: size of step
    :param float l2_penalty: speed of weights' decay, l2 regularization prevents overfitting
    :param float momentum: momentum to stabilize learning process.
    """
    updates = []
    shareds = []
    xp, yp, wp = get_batch(x, y, w, batch_size=batch, random_stream=random_stream)
    for name, param in parameters.items():
        der = T.grad(loss(xp, yp, wp), param)
        momentum_ = theano.shared(param.get_value() * 0.)
        shareds.append(momentum_)
        updates.append([momentum_, momentum_ * momentum + (1. - momentum) * der])
        updates.append([param, param * (1. - learning_rate * l2_penalty) - learning_rate * momentum_])
    return shareds, updates


def irprop_minus_trainer(x, y, w, parameters, loss, random_stream,
                         positive_step=1.2, negative_step=0.5, max_step=1., min_step=1e-6):
    """IRPROP- is batch trainer, for details see http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.21.3428 .
    This is default trainer, very stable for classification.

    :param positive_step: factor, by which the step is increased when continuing going in the direction
    :param negative_step: factor, by which the step is increased when changing direction to opposite
    :param min_step: minimal change of weight during iteration
    :param max_step: maximal change of weight during iteration
    """
    shareds = []
    updates = []
    loss_value = loss(x, y, w)
    for name, param in parameters.items():
        old_derivative = theano.shared(param.get_value() * 0.)
        delta = theano.shared(param.get_value() * 0. + 1e-3)
        shareds.extend([old_derivative, delta])
        new_derivative = T.grad(loss_value, param)

        new_delta = T.where(new_derivative * old_derivative > 0, delta * positive_step, delta * negative_step)
        new_delta = T.clip(new_delta, min_step, max_step)

        updates.append([param, param - new_delta * T.sgn(new_derivative)])
        updates.append([delta, new_delta])

        new_old_derivative = T.where(new_derivative * old_derivative < 0, 0, new_derivative)
        updates.append([old_derivative, new_old_derivative])
    return shareds, updates


def irprop_plus_trainer(x, y, w, parameters, loss, random_stream,
                        positive_step=1.2, negative_step=0.5, max_step=1., min_step=1e-6):
    """IRPROP+ is batch trainer, for details see http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.21.3428

    :param positive_step: factor, by which the step is increased when continuing going in the direction
    :param negative_step: factor, by which the step is increased when changing direction to opposite
    :param min_step: minimal change of weight during iteration
    :param max_step: maximal change of weight during iteration
    """
    loss_value = loss(x, y, w)
    prev_loss_value = theano.shared(1e10)
    shareds = [prev_loss_value]
    updates = []
    for name, param in parameters.items():
        old_derivative = theano.shared(param.get_value() * 0.)
        delta = theano.shared(param.get_value() * 0. + 1e-3)
        new_derivative = T.grad(loss_value, param)

        shift_if_bad_step = T.where(new_derivative * old_derivative < 0, delta * T.sgn(old_derivative), 0)
        shift = ifelse(loss_value > prev_loss_value, shift_if_bad_step, 0. * param)
        # unfortunately we can't do it this way: param += shift

        new_delta = T.where(new_derivative * old_derivative > 0, delta * positive_step, delta * negative_step)
        new_delta = T.clip(new_delta, min_step, max_step)

        updates.append([param, param + shift - new_delta * T.sgn(new_derivative)])
        updates.append([delta, new_delta])

        new_old_derivative = T.where(new_derivative * old_derivative < 0, 0, new_derivative)
        updates.append([old_derivative, new_old_derivative])
        shareds.extend([old_derivative, delta])

    updates.append([prev_loss_value, loss_value])
    return shareds, updates


def adadelta_trainer(x, y, w, parameters, loss, random_stream, batch=DEFAULT_MINIBATCH,
                     learning_rate=0.1, half_life=1000, epsilon=1e-4, ):
    """AdaDelta is trainer with adaptive learning rate.

    :param half_life: momentum-like parameter. The estimated parameters are decreased by 2 after so many events.
        It is recommended for small datasets to put halflife = number of samples in dataset.
    :param learning_rate: size of step
    :param batch: size of minibatch
    :param epsilon: regularization
    """
    shareds = []
    updates = []
    decay_rate = pow(0.5, batch / float(half_life))

    xp, yp, wp = get_batch(x, y, w, batch_size=batch, random_stream=random_stream)
    for name, param in parameters.items():
        derivative = T.grad(loss(xp, yp, wp), param)
        cumulative_derivative = theano.shared(param.get_value() * 0.)
        cumulative_step = theano.shared(param.get_value() * 0.)
        shareds.extend([cumulative_derivative, cumulative_step])

        new_cumulative_derivative = cumulative_derivative * decay_rate + (1 - decay_rate) * derivative ** 2
        step = - derivative * T.sqrt((cumulative_step + epsilon) / (new_cumulative_derivative + epsilon))

        updates.append([cumulative_derivative, new_cumulative_derivative])
        updates.append([cumulative_step, cumulative_step * decay_rate + (1 - decay_rate) * step ** 2])
        updates.append([param, param + learning_rate * step])

    return shareds, updates


trainers = {'sgd': sgd_trainer,
            'irprop-': irprop_minus_trainer,
            'irprop+': irprop_plus_trainer,
            'adadelta': adadelta_trainer,
            }


# endregion


def _prepare_scaler(transform):
    """Returns new transformer used in neural network
    :param transform: str ot transformer
    :return: transformer, cloned or created.
    """
    if transform == 'standard':
        return preprocessing.StandardScaler()
    elif transform == 'minmax':
        return preprocessing.MinMaxScaler()
    elif transform == 'iron':
        return IronTransformer(symmetrize=True)
    else:
        assert isinstance(transform, TransformerMixin), 'provided transformer should be derived from TransformerMixin'
        return clone(transform)


class AbstractNeuralNetworkBase(BaseEstimator):
    """Base class for neural networks"""

    def __init__(self, layers, scaler, loss, trainer, epochs, trainer_parameters, random_state):
        """
        :param layers: list of int, e.g [9, 7] - the number of units in each *hidden* layer
        :param scaler: 'standard', 'minmax', 'iron' or some other Transformer used to transform features.
            Default is 'standard', which will apply StandardScaler.
            'Iron' is for heavy tails distributions.
        :param loss: loss function used.
            Options: {loss_options}
        :param trainer: string, name of optimization method used.
            Options: {trainer_options}
        :param epochs: number of times each sample takes part in training
        :param dict trainer_parameters: parameters passed to trainer function (learning_rate, etc., trainer-specific).
            See parameters in documentation.
        """
        self.scaler = scaler
        self.layers = layers
        self.loss = loss
        self.epochs = epochs
        self.parameters = {}
        self.trainer = trainer
        self.trainer_parameters = deepcopy(trainer_parameters)
        self.random_state = random_state

    def _create_matrix_parameter(self, name, n1, n2):
        """Creates and registers matrix parameter of neural network"""
        matrix = theano.shared(value=self.random_state_.normal(size=[n1, n2]).astype(floatX) * 0.01, name=name)
        self.parameters[name] = matrix
        return matrix

    def _create_scalar_parameter(self, name):
        """Creates and registers scalar parameter of neural network"""
        scalar_param = theano.shared(value=self.random_state_.normal() * 0.01, name=name)
        self.parameters[name] = scalar_param
        return scalar_param

    def prepare(self):
        """This method should provide activation function and set parameters.
        Each network overrides this function.

        :return: Activation function, f: X -> df,
            X of shape [n_events, n_outputs], df (df = decision function) of shape [n_events].
            For classification, df is arbitrary real, the greater df, the more signal-like the event.
            Probabilities are computed by applying logistic function to output of activation.
        """
        raise NotImplementedError()

    def _prepare(self, n_input_features):
        """This function is called once, it creates the activation function, it's gradient
        and initializes the weights

        :return: loss function as lambda (x, y, w) -> loss
        """
        self.random_state_ = check_random_state(self.random_state)
        self.layers_ = [n_input_features] + list(self.layers) + [1]
        self.parameters = {}
        loss_function = losses.get(self.loss, self.loss)

        x = T.matrix('X')
        y = T.vector('y')
        w = T.vector('w')
        activation_raw = self.prepare()
        self.Activation = theano.function([x], activation_raw(x).flatten(), allow_input_downcast=True)
        loss_ = lambda x, y, w: loss_function(y, activation_raw(x).flatten(), w)
        self.Loss = theano.function([x, y, w], loss_(x, y, w))
        return loss_

    def _transform(self, X, y=None, fit=False):
        """Apply selected scaler or transformer to dataset
        (also this method adds a column filled with ones).

        :param numpy.array X: of shape [n_samples, n_features], data
        :param numpy.array y: of shape [n_samples], labels
        :param bool fit: if True, fits transformer
        :return: transformed data, numpy.array of shape [n_samples, n_output_features]
        """
        # Fighting copy-bug of sklearn's transformers
        X = numpy.array(X, dtype=float)

        if fit:
            self.scaler_ = _prepare_scaler(self.scaler)
            self.scaler_.fit(X, y)

        result = self.scaler_.transform(X)
        result = numpy.hstack([result, numpy.ones([len(X), 1])]).astype('float32')

        return result

    def _prepare_inputs(self, X, y, sample_weight):
        X, y, sample_weight = check_xyw(X, y, sample_weight)
        X = self._transform(X, y, fit=True)
        return X, y, sample_weight

    def fit(self, X, y, sample_weight=None):
        """ Prepare the model by optimizing selected loss function with some trainer.

        :param X: numpy.array of shape [n_samples, n_features]
        :param y: numpy.array of shape [n_samples]
        :param sample_weight: numpy.array of shape [n_samples], leave None for array of 1's

        :return: self
        """
        X, y, sample_weight = self._prepare_inputs(X, y, sample_weight=sample_weight)

        loss_lambda = self._prepare(X.shape[1])

        trainer_function = trainers[self.trainer]
        parameters_ = self.trainer_parameters or {}

        x = theano.shared(X.astype(floatX))
        y = theano.shared(y)
        w = theano.shared(sample_weight.astype(floatX))

        shareds, updates = trainer_function(x, y, w, self.parameters, loss_lambda,
                                            RandomStreams(seed=self.random_state_.randint(0, 1000)), **parameters_)

        make_one_step = theano.function([], [], updates=updates, allow_input_downcast=True)

        # computing correct number of iterations in epoch is not simple:
        if 'batch' in parameters_:
            n_batches = len(X) // parameters_['batch'] + 1
        elif self.trainer in ['adadelta', 'sgd']:
            n_batches = len(X) // DEFAULT_MINIBATCH + 1
        else:
            n_batches = 1

        for i in range(self.epochs):
            for _ in range(n_batches):
                make_one_step()

        return self

    def decision_function(self, X):
        """
        Activates NN on particular dataset

        :param numpy.array X: of shape [n_samples, n_features]
        :return: numpy.array with results of shape [n_samples]
        """
        X = self._transform(X, fit=False)
        return self.Activation(X)

    def compute_loss(self, X, y, sample_weight=None):
        """Computes loss (that was used in training) on labeled dataset

        :param X: numpy.array of shape [n_samples, n_features]
        :param y: numpy.array with integer labels of shape [n_samples],
            in two-class classification 0 and 1 labels should be used
        :param sample_weight: optional, numpy.array of shape [n_samples].
        :return float, the loss vales computed"""
        sample_weight = check_sample_weight(y, sample_weight)
        X = self._transform(X, fit=False)
        return self.Loss(X, y, sample_weight)


class AbstractNeuralNetworkClassifier(AbstractNeuralNetworkBase, ClassifierMixin):
    """
    Base class for classification neural networks.
    Supports only binary classification, supports weights, which makes it usable in boosting.

    Works as usual sklearn classifier, can be used in pipelines, ensembles, pickled, etc.
    """

    def __init__(self, layers=(10,), scaler='standard', loss='log_loss', trainer='irprop-', epochs=100,
                 trainer_parameters=None, random_state=None):
        AbstractNeuralNetworkBase.__init__(self, layers=layers, scaler=scaler, loss=loss,
                                           trainer=trainer, epochs=epochs, trainer_parameters=trainer_parameters,
                                           random_state=random_state)

    __init__.__doc__ = AbstractNeuralNetworkBase.__init__.__doc__.format(
        loss_options=classification_losses.keys(), trainer_options=trainers.keys())

    def _prepare_inputs(self, X, y, sample_weight):
        X, y, sample_weight = check_xyw(X, y, sample_weight)
        sample_weight = check_sample_weight(y, sample_weight, normalize=True)
        X = self._transform(X, y, fit=True)
        self.classes_ = numpy.array([0, 1])
        assert (numpy.unique(y) == self.classes_).all(), 'only two-class classification supported, labels are 0 and 1'
        y = numpy.array(y, dtype=int)
        return X, y, sample_weight

    def predict_proba(self, X):
        """Computes probability of each event to belong to each class

        :param numpy.array X: of shape [n_samples, n_features]
        :return: numpy.array of shape [n_samples, n_classes]
        """
        return score_to_proba(self.decision_function(X))

    def predict(self, X):
        """ Predict the classes for new events (not recommended, use `predict_proba`).

        :param numpy.array X: of shape [n_samples, n_features]
        :return: numpy.array of shape [n_samples] with labels of predicted classes.
        """
        return self.predict_proba(X).argmax(axis=1)


class AbstractNeuralNetworkRegressor(AbstractNeuralNetworkBase, RegressorMixin):
    """
    Base class for regression neural networks. Supports weights.

    Works as usual sklearn classifier, can be used in pipelines, ensembles, pickled, etc.
    """

    def __init__(self, layers=(10,), scaler='standard', loss='mse_loss', trainer='irprop-', epochs=100,
                 trainer_parameters=None, random_state=None):
        AbstractNeuralNetworkBase.__init__(self, layers=layers, scaler=scaler, loss=loss,
                                           trainer=trainer, epochs=epochs, trainer_parameters=trainer_parameters,
                                           random_state=random_state)

    __init__.__doc__ = AbstractNeuralNetworkBase.__init__.__doc__.format(
        loss_options=regression_losses.keys(), trainer_options=trainers.keys())

    def predict(self, X):
        """ Compute predictions for new events.

        :param numpy.array X: of shape [n_samples, n_features]
        :return: numpy.array of shape [n_samples] with labels of predicted classes """
        return self.decision_function(X)


# region Neural networks


class SimpleNeuralNetwork(AbstractNeuralNetworkClassifier):
    """The most simple NN with one hidden layer (sigmoid activation), for example purposes.
    Supports only one hidden layer.

    See source code as an example of custom NN.
    """

    def prepare(self):
        n1, n2, n3 = self.layers_
        W1 = self._create_matrix_parameter('W1', n1, n2)
        W2 = self._create_matrix_parameter('W2', n2, n3)

        def activation(input):
            first = T.nnet.sigmoid(T.dot(input, W1))
            return T.dot(first, W2)

        return activation


class MLPBase(object):
    """
    MLPRegressor and MLPClassifier have identical implementation, but should be derived from different base classes.
    """

    def prepare(self):
        activation = lambda x: x
        for i, layer in list(enumerate(self.layers_))[1:]:
            W = self._create_matrix_parameter('W' + str(i), self.layers_[i - 1], self.layers_[i])
            # act=activation and W_=W are tricks to avoid lambda-capturing
            if i == 1:
                activation = lambda x, act=activation, W_=W: T.dot(act(x), W_)
            else:
                activation = lambda x, act=activation, W_=W: T.dot(T.tanh(act(x)), W_)

        return activation


class MLPClassifier(MLPBase, AbstractNeuralNetworkClassifier):
    """
    MLP (MultiLayerPerceptron) for binary classification.
    Supports arbitrary number of layers (tanh is used as activation).
    """
    pass


class MLPRegressor(MLPBase, AbstractNeuralNetworkRegressor):
    """
    MLP (MultiLayerPerceptron) regressor.
    Supports arbitrary number of layers (tanh is used as activation).
    """
    pass


class MLPMultiClassifier(MLPBase, AbstractNeuralNetworkClassifier):
    """
    MLP (MultiLayerPerceptron) for multi-class classification.
    Supports arbitrary number of layers (tanh is used as activation).
    """

    def __init__(self, layers=(10,), scaler='standard', trainer='irprop-', epochs=100,
                 trainer_parameters=None, random_state=None):
        AbstractNeuralNetworkBase.__init__(self, layers=layers, scaler=scaler, loss=False,
                                           trainer=trainer, epochs=epochs, trainer_parameters=trainer_parameters,
                                           random_state=random_state)

    def _prepare(self, n_input_features):
        """This function is called once, it creates the activation function, it's gradient
        and initializes the weights

        :return: loss function as lambda (x, y, w) -> loss
        """
        self.random_state_ = check_random_state(self.random_state)
        self.layers_ = [n_input_features] + list(self.layers) + [len(self.classes_)]
        self.parameters = {}

        x = T.matrix('X')
        y = T.vector('y', dtype='int64')
        w = T.vector('w')
        activation_raw = self.prepare()
        self.Activation = theano.function([x], activation_raw(x), allow_input_downcast=True)
        loss_ = lambda x, y, w: -T.mean(w * T.log(T.nnet.softmax(activation_raw(x))[T.arange(y.shape[0]), y]))
        self.Loss = theano.function([x, y, w], loss_(x, y, w))
        return loss_

    def _prepare_inputs(self, X, y, sample_weight):
        X, y, sample_weight = check_xyw(X, y, sample_weight)
        sample_weight = check_sample_weight(y, sample_weight, normalize=True)
        X = self._transform(X, y, fit=True)
        self.classes_, y = numpy.unique(y, return_inverse=True)
        y = y.astype('int32')
        assert numpy.allclose(self.classes_, range(len(self.classes_))), 'Classes should be 0, 1... n_classes - 1'
        return X, y, sample_weight

    def predict_proba(self, X):
        """Computes probability of each event to belong to each class

        :param numpy.array X: of shape [n_samples, n_features]
        :return: numpy.array of shape [n_samples, n_classes]
        """
        activations = self.decision_function(X)
        # taking softmax
        activations -= numpy.max(activations, axis=1, keepdims=True)
        probabilities = numpy.exp(activations)
        return probabilities / numpy.sum(probabilities, axis=1, keepdims=True)


class RBFNeuralNetwork(AbstractNeuralNetworkClassifier):
    """
    Neural network with one hidden layer with normalized RBF activation (Radial Basis Function).
    """

    def prepare(self):
        n1, n2, n3 = self.layers_
        W1 = self._create_matrix_parameter('W1', n2, n1)
        W2 = self._create_matrix_parameter('W2', n2, n3)
        # this parameter is responsible for scaling, it is optimised too
        G = theano.shared(value=0.1, name='G')
        self.parameters['G'] = G

        def activation(input):
            translation_vectors = W1.reshape((1, W1.shape[0], -1)) - input.reshape((input.shape[0], 1, -1))
            minkowski_distances = (abs(translation_vectors) ** 2).sum(2)
            first = T.nnet.softmax(- (0.001 + G * G) * minkowski_distances)
            return T.dot(first, W2)

        return activation


class SoftmaxNeuralNetwork(AbstractNeuralNetworkClassifier):
    """Neural network with one hidden layer, softmax activation function """

    def prepare(self):
        n1, n2, n3 = self.layers_
        W1 = self._create_matrix_parameter('W1', n1, n2)
        W2 = self._create_matrix_parameter('W2', n2, n3)

        def activation(input):
            first = T.nnet.softmax(T.dot(input, W1))
            return T.dot(first, W2)

        return activation


class PairwiseNeuralNetwork(AbstractNeuralNetworkClassifier):
    """The result is computed as :math:`h = sigmoid(Ax)`, :math:`output = \sum_{ij} B_{ij} h_i (1 - h_j)`,
    this is a brilliant example when easier to define activation
    function rather than trying to implement this inside some framework.
    """

    def prepare(self):
        n1, n2, n3 = self.layers_
        W1 = self._create_matrix_parameter('W1', n1, n2)
        W2 = self._create_matrix_parameter('W2', n2, n2)

        def activation(input):
            first = T.nnet.sigmoid(T.dot(input, W1))
            return T.batched_dot(T.dot(first, W2), 1 - first)

        return activation


class PairwiseSoftplusNeuralNetwork(AbstractNeuralNetworkClassifier):
    """The result is computed as :math:`h1 = softplus(A_1 x)`, :math:`h2 = sigmoid(A_2 x)`,
        :math:`output = \sum_{ij} B_{ij} h1_i h2_j)`
    """

    def prepare(self):
        n1, n2, n3 = self.layers_
        A1 = self._create_matrix_parameter('A1', n1, n2)
        A2 = self._create_matrix_parameter('A2', n1, n2)
        B = self._create_matrix_parameter('B', n2, n2)

        def activation(input):
            first1 = T.nnet.softplus(T.dot(input, A1))
            first2 = T.nnet.sigmoid(T.dot(input, A2))
            return T.batched_dot(T.dot(first1, B), first2)

        return activation

# endregion
