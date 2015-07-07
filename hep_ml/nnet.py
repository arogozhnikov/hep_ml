"""
hep_ml.nnet

Minimalistic version of feed-forward neural networks on theano.
The neural networks from this library provide sklearn classifier's interface.

Definitions for loss functions, trainers of neural networks are defined in this file too.
Main point of this library: black-box stochastic optimization of any given loss function.
This gives ability to define any activation expression at the cost of unavailability of pretraining.

In this file we have **examples** of neural networks,
 user is encouraged to write his own specific architecture,
 which can be much more complex than those used usually.

This library should be preferred for different experiments with architectures.
Also nnet allows optimization of parameters in any differentiable decision function.

"""
from __future__ import print_function, division, absolute_import
from copy import deepcopy

import numpy
import theano
import theano.tensor as T
from theano.ifelse import ifelse
from theano.tensor.shared_randomstreams import RandomStreams

from sklearn.utils.validation import check_random_state
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin, clone
from sklearn import preprocessing
from .commonutils import check_xyw, check_sample_weight
from scipy.special import expit


floatX = theano.config.floatX
__author__ = 'Alex Rogozhnikov'


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


# regression loss
def mse_loss(y, pred, w):
    """ Regression loss function, mean squared error. """
    return T.mean(w * (y - pred) ** 2)


def smooth_huber_loss(y, pred, w):
    """Regression loss function, smooth version of Huber loss function. """
    return T.mean(w * T.log(T.cosh(y - pred)))


losses = {'mse_loss': mse_loss,
          'exp_loss': exp_loss,
          'log_loss': log_loss,
          'squared_loss': squared_loss,
          'exp_log_loss': exp_log_loss,
          'smooth_huber_loss': smooth_huber_loss,
}

# endregion


# region Trainers
def get_batch(x, y, w, random_stream, batch_size=10):
    """ Generates subset of training dataset, of size batch"""
    indices = random_stream.choice(a=T.shape(x)[0], size=(batch_size,))
    return x[indices, :], y[indices], w[indices]


def sgd_trainer(x, y, w, parameters, loss, random_stream, batch=10, learning_rate=0.1,
                l2_penalty=0.001, momentum=0.9, ):
    """Stochastic gradient descent with momentum"""
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
    """ IRPROP- trainer, see http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.21.3428 """
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


def irprop_star_trainer(x, y, w, parameters, loss, random_stream,
                        positive_step=1.2, negative_step=0.5, max_step=1., min_step=1e-6):
    """ IRPROP* trainer (own experimental modification, not recommended for usage) """
    shareds = []
    updates = []
    loss_value = loss(x, y, w)

    for name, param in parameters.items():
        param_shape = param.get_value().shape
        n = numpy.prod(param_shape).astype(int)
        new_derivative_ = T.grad(loss_value, param).flatten()
        lnewder, rnewder = new_derivative_.reshape([n, 1]), new_derivative_.reshape([1, n])
        new_derivative_plus = lnewder + rnewder
        new_derivative_minus = lnewder - rnewder
        new_param = param
        for new_derivative in [new_derivative_plus, new_derivative_minus]:
            delta = theano.shared(numpy.zeros([n, n], dtype=floatX) + 1e-3)
            old_derivative = theano.shared(numpy.zeros([n, n], dtype=floatX))

            new_delta = T.where(new_derivative * old_derivative > 0, delta * positive_step, delta * negative_step)
            new_delta = T.clip(new_delta, min_step, max_step)

            updates.append([delta, new_delta])
            new_old_derivative = T.where(new_derivative * old_derivative < 0, 0, new_derivative)
            updates.append([old_derivative, new_old_derivative])
            new_param = new_param - (new_delta * T.sgn(new_derivative)).sum(axis=1).reshape(param.shape)
            shareds.extend([old_derivative, delta])

        updates.append([param, new_param])

    return shareds, updates


def irprop_plus_trainer(x, y, w, parameters, loss, random_stream,
                        positive_step=1.2, negative_step=0.5, max_step=1., min_step=1e-6):
    """IRPROP+ trainer, see http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.17.1332"""
    loss_value = loss(x, y, w)
    prev_loss_value = theano.shared(1e10)
    shareds = []
    updates = []
    for name, param in parameters.iteritems():
        old_derivative = theano.shared(param.get_value() * 0.)
        delta = theano.shared(param.get_value() * 0. + 1e-3)
        new_derivative = T.grad(loss_value, param)

        shift_if_bad_step = T.where(new_derivative * old_derivative < 0, delta * T.sgn(old_derivative), 0)
        # THIS doesn't work!
        shift = ifelse(loss_value > prev_loss_value, shift_if_bad_step, 0. * param)
        # unfortunately we can't do it this way: param += shift

        new_delta = T.where(new_derivative * old_derivative > 0, delta * positive_step, delta * negative_step)
        new_delta = T.clip(new_delta, min_step, max_step)

        updates.append([param, param + shift - new_delta * T.sgn(new_derivative)])
        updates.append([delta, new_delta])

        new_old_derivative = T.where(new_derivative * old_derivative < 0, 0, new_derivative)
        updates.append([old_derivative, new_old_derivative])
        shareds.extend([old_derivative, delta, prev_loss_value])

    updates.append([prev_loss_value, loss_value])
    return shareds, updates


def adadelta_trainer(x, y, w, parameters, loss, random_stream,
                     decay_rate=0.95, epsilon=1e-5, learning_rate=1., batch=1000):
    shareds = []
    updates = []

    xp, yp, wp = get_batch(x, y, w, batch_size=batch, random_stream=random_stream)
    for name, param in parameters.items():
        derivative = T.grad(loss(xp, yp, wp), param)
        cumulative_derivative = theano.shared(param.get_value() * 0.)
        cumulative_step = theano.shared(param.get_value() * 0.)
        shareds.extend([cumulative_derivative, cumulative_step])

        updates.append([cumulative_derivative, cumulative_derivative * decay_rate + (1 - decay_rate) * derivative ** 2])
        step = - derivative * T.sqrt((cumulative_step + epsilon) / (cumulative_derivative + epsilon))

        updates.append([cumulative_step, cumulative_step * decay_rate + (1 - decay_rate) * step ** 2])
        updates.append([param, param + learning_rate * step])

    return shareds, updates


trainers = {'sgd': sgd_trainer,
            'irprop-': irprop_minus_trainer,
            'irprop+': irprop_plus_trainer,
            'irprop*': irprop_star_trainer,
            'adadelta': adadelta_trainer,
}
# endregion


# TODO think of dropper and noises

def _prepare_scaler(transform):
    """Returns new transformer used in neural network
    :param transform: str ot transformer
    :return: transformer, cloned or created.
    """
    if transform == 'standard':
        return preprocessing.StandardScaler()
    elif transform == 'minmax':
        return preprocessing.MinMaxScaler()
    else:
        assert isinstance(transform, TransformerMixin), 'provided transformer should be derived from TransformerMixin'
        return clone(transform)


class AbstractNeuralNetworkClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, layers=None, scaler='standard', loss='log_loss', trainer='irprop-', epochs=100,
                 trainer_parameters=None, random_state=None):
        """
        Constructs the neural network based on Theano (for classification purposes).
        Supports only binary classification, supports weights, which makes it usable in boosting.

        Works in sklearn fit-predict way: X is [n_samples, n_features], y is [n_samples], sample_weight is [n_samples].
        Works as usual sklearn classifier, can be used in boosting, for instance, pickled, etc.
        :param layers: list of int, e.g [9, 7] - the number of units in each *hidden* layer
        :param scaler: 'standard' or 'minmax' or Transformer used to transform features
        :param loss: loss function used (log_loss by default), str ot function(y, pred, w) -> float
        :param trainer: string, name of optimization method used
        :param epochs: number of times each takes part in training
        :param dict trainer_parameters: parameters passed to trainer function (learning_rate, etc., trainer-specific).
        """
        self.scaler = scaler
        self.layers = layers
        self.loss = loss
        self.prepared = False
        self.epochs = epochs
        self.parameters = {}
        self.trainer = trainer
        self.trainer_parameters = deepcopy(trainer_parameters)
        self.random_state = random_state
        self.classes_ = numpy.array([0, 1])

    def _create_shared_matrix(self, name, n1, n2):
        """Creates a parameter of neural network, which is typically a matrix """
        matrix = theano.shared(value=self.random_state.normal(size=[n1, n2]).astype(floatX) * 0.01, name=name)
        self.parameters[name] = matrix
        return matrix

    def prepare(self):
        """This method should provide activation function and set parameters
        :return Activation function, f: X -> p,
        X of shape [n_events, n_outputs], p of shape [n_events].
        For classification, p is arbitrary real, the greater p, the more event
        looks like signal event (label 1).
        """
        raise NotImplementedError()

    def _prepare(self, n_input_features):
        """This function is called once, it creates the activation function, it's gradient
        and initializes the weights
        :return: loss function as lambda (x, y, w) -> loss"""
        if not self.prepared:
            self.random_state = check_random_state(self.random_state)
            self.layers_ = [n_input_features] + self.layers + [1]
            self.parameters = {}
            self.scaler_ = _prepare_scaler(self.scaler)
            self.prepared = True

        loss_function = losses.get(self.loss, self.loss)

        x = T.matrix('X')
        y = T.vector('y')
        w = T.vector('w')
        activation_raw = self.prepare()
        self.Activation = theano.function([x], activation_raw(x).flatten())
        loss_ = lambda x, y, w: loss_function(y, activation_raw(x).flatten(), w)
        self.Loss = theano.function([x, y, w], loss_(x, y, w))
        return loss_

    def activate(self, X):
        """ Activates NN on particular dataset
        :param numpy.array X: of shape [n_samples, n_features]
        :return: numpy.array with results of shape [n_samples] """
        return self.Activation(X)

    def predict_proba(self, X):
        """Computes probability of each event to belong to a particular class
        :param numpy.array X: of shape [n_samples, n_features]
        :return: numpy.array of shape [n_samples, n_classes]
        """
        result = numpy.zeros([len(X), 2])
        result[:, 1] = expit(self.activate(X))
        result[:, 0] = 1 - result[:, 1]
        return result

    def predict(self, X):
        """ Predict the classes for new events.
        :param numpy.array X: of shape [n_samples, n_features]
        :return: numpy.array of shape [n_samples] with labels of predicted classes """
        return self.predict_proba(X).argmax(axis=1)

    def compute_loss(self, X, y, sample_weight=None):
        """Computes loss (that was used in training) on labeled dataset
        :param X: numpy.array of shape [n_samples, n_features]
        :param y: numpy.array with integer labels of shape [n_samples],
            in two-class classification 0 and 1 labels should be used
        :param sample_weight: optional, numpy.array of shape [n_samples].
        :return float, the loss vales computed"""
        sample_weight = check_sample_weight(y, sample_weight, normalize=False)
        return self.Loss(X, y, sample_weight)

    def fit(self, X, y, sample_weight=None, trainer=None, epochs=None, **trainer_parameters):
        """ Prepare the model by optimizing selected loss function with some trainer.
        This method can (and should) be called several times, each time with new parameters
        :param X: numpy.array of shape [n_samples, n_features]
        :param y: numpy.array of shape [n_samples]
        :param sample_weight: numpy.array of shape [n_samples], leave None for array of 1's
        :param trainer: str, method used to minimize loss, overrides one in the ctor
        :param trainer_parameters: parameters for this method, override ones in ctor
        :return: self """
        X, y, sample_weight = check_xyw(X, y, sample_weight)
        sample_weight = check_sample_weight(y, sample_weight, normalize=True)
        self.classes_ = numpy.array([0, 1])
        assert (numpy.unique(y) == self.classes_).all(), 'only two-class classification supported, labels are 0 and 1'
        loss_lambda = self._prepare(X.shape[1])

        trainer = trainers[self.trainer if trainer is None else trainer]
        parameters_ = {} if self.trainer_parameters is None else self.trainer_parameters.copy()
        parameters_.update(trainer_parameters)

        x = theano.shared(numpy.array(X, dtype=floatX))
        y = theano.shared(numpy.array(y, dtype=floatX))
        w = theano.shared(numpy.array(sample_weight, dtype=floatX))

        shareds, updates = trainer(x, y, w, self.parameters, loss_lambda,
                                   RandomStreams(seed=self.random_state.randint(0, 1000)), **parameters_)

        make_one_step = theano.function([], [], updates=updates)

        # TODO epochs are computed wrongly at the moment if not passed batch_size
        n_batches = 1
        if parameters_.has_key('batch'):
            batch = parameters_['batch']
            n_batches = len(X) // batch + 1

        for i in range(epochs or self.epochs):
            for _ in range(n_batches):
                make_one_step()

        return self


# region Neural networks

class SimpleNeuralNetwork(AbstractNeuralNetworkClassifier):
    """The most simple NN with one hidden layer (sigmoid activation), for example purposes """

    def prepare(self):
        n1, n2, n3 = self.layers_
        W1 = self._create_shared_matrix('W1', n1, n2)
        W2 = self._create_shared_matrix('W2', n2, n3)

        def activation(input):
            first = T.nnet.sigmoid(T.dot(input, W1))
            return T.dot(first, W2)

        return activation


class MultiLayerNetwork(AbstractNeuralNetworkClassifier):
    """Supports arbitrary number of layers (sigmoid activation each).
    aka MLP (MultiLayerPerceptron)"""

    def prepare(self):
        activation = lambda x: x
        for i, layer in list(enumerate(self.layers_))[1:]:
            W = self._create_shared_matrix('W' + str(i), self.layers_[i - 1], self.layers_[i])
            # act=activation and W_=W are tricks to avoid lambda-capturing
            activation = lambda x, act=activation, W_=W: T.tanh(T.dot(act(x), W_))
        return activation


class RBFNeuralNetwork(AbstractNeuralNetworkClassifier):
    """One hidden layer with normalized RBF activation (Radial Basis Function)"""

    def prepare(self):
        n1, n2, n3 = self.layers_
        W1 = self._create_shared_matrix('W1', n1, n2)
        W2 = self._create_shared_matrix('W2', n2, n3)
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
    """One hidden layer, softmax activation function """

    def prepare(self):
        n1, n2, n3 = self.layers_
        W1 = self._create_shared_matrix('W1', n1, n2)
        W2 = self._create_shared_matrix('W2', n2, n3)

        def activation(input):
            first = T.nnet.softmax(T.dot(input, W1))
            return T.dot(first, W2)

        return activation


class PairwiseNeuralNetwork(AbstractNeuralNetworkClassifier):
    """The result is computed as h = sigmoid(Ax), output = sum_{ij} B_ij h_i (1 - h_j),
     this is a brilliant example when easier to define activation
     function rather than trying to implement this inside some framework."""

    def prepare(self):
        n1, n2, n3 = self.layers_
        W1 = self._create_shared_matrix('W1', n1, n2)
        W2 = self._create_shared_matrix('W2', n2, n2)

        def activation(input):
            first = T.nnet.sigmoid(T.dot(input, W1))
            return T.batched_dot(T.dot(first, W2), 1 - first)

        return activation


class PairwiseSoftplusNeuralNetwork(AbstractNeuralNetworkClassifier):
    """The result is computed as h = softplus(Ax), output = sum_{ij} B_ij h_i (1 - h_j) """

    def prepare(self):
        n1, n2, n3 = self.layers_
        W1 = self._create_shared_matrix('W1', n1, n2)
        W2 = self._create_shared_matrix('W2', n2, n2)

        def activation(input):
            z = T.dot(input, W1)
            first1 = T.nnet.softplus(z)
            first2 = T.nnet.softplus(-z)
            return T.batched_dot(T.dot(first1, W2), first2)

        return activation

# endregion