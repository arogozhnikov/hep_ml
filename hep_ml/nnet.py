"""
Minimalistic version of feed-forward neural networks on theano.
The neural networks from this library provide sklearn classifier's interface.

Definitions for loss functions, trainers of neural networks are defined in this file too.
"""
from __future__ import print_function, division, absolute_import
from copy import deepcopy

import numpy
import theano
import theano.tensor as T
from sklearn.utils.validation import check_random_state
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin, clone
from sklearn import preprocessing
from .commonutils import check_xyw, check_sample_weight
from scipy.special import expit


floatX = theano.config.floatX
__author__ = 'Alex Rogozhnikov'

# print every 1000000 stage, so the same as silent
SILENT = 1000000

# region Loss functions


def squared_loss(y, pred, w):
    """ Squared loss for classification, not to be messed up with MSE"""
    return T.mean(w * (y - T.nnet.sigmoid(pred)) ** 2)


def log_loss(y, pred, w):
    """ Logistic loss (aka cross-entropy, aka binomial deviance) """
    margin = pred * (1 - 2 * y)
    return T.mean(w * T.nnet.softplus(margin))


def exp_loss(y, pred, w):
    """ Exponential loss (aka AdaLoss function) """
    margin = pred * (1 - 2 * y)
    return T.mean(w * T.exp(margin))


def exp_log_loss(y, pred, w):
    """ Combines logistic loss for signal and exponential loss for background """
    return 2 * log_loss(y, pred, w=w * y) + exp_loss(y, pred, w=w * (1 - y))


# regression loss
def mse_loss(y, pred, w):
    """Regression loss function"""
    return T.mean(w * (y - pred) ** 2)


# Hinge-like loss function
def hinge_like_loss(y, pred, w):
    """Regression loss function"""
    return T.mean(w * T.log(T.cosh((y - pred) ** 2)))


losses = {'mse_loss': mse_loss,
          'exp_loss': exp_loss,
          'log_loss': log_loss,
          'squared_loss': squared_loss,
          'exp_log_loss': exp_log_loss,
          'hinge_like_loss': hinge_like_loss,
}

# endregion


# region Trainers
def get_batch(x, y, w, random, batch_size=10):
    """ Generates subset of training dataset, of size batch"""
    if len(y) > batch_size:
        indices = random.choice(len(x), size=batch_size)
        return x[indices, :], y[indices], w[indices]
    else:
        return x, y, w


def sgd_trainer(x, y, w, parameters, derivatives, loss,
                stages=1000, batch=10, learning_rate=0.1, l2_penalty=0.001, momentum=0.9,
                random=0, verbose=SILENT):
    """Stochastic gradient descent with momentum"""
    random = check_random_state(random)
    momenta = {name: 0. for name in parameters}

    for stage in range(stages):
        xp, yp, wp = get_batch(x, y, w, batch_size=batch, random=random)
        for name in parameters:
            der = derivatives[name](xp, yp, wp)
            momenta[name] *= momentum
            momenta[name] += (1 - momentum) * der
            val = parameters[name].get_value() * (1. - learning_rate * l2_penalty) - learning_rate * momenta[name]
            parameters[name].set_value(val)
        if (stage + 1) % verbose == 0:
            print(loss(x, y, w))


def irprop_minus_trainer(x, y, w, parameters, derivatives, loss,
                         stages=100, positive_step=1.2, negative_step=0.5, max_step=1., min_step=1e-6,
                         random=0, verbose=SILENT):
    """ IRPROP- trainer, see http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.21.3428 """
    deltas = {name: 1e-3 for name in parameters}
    prev_derivatives = {name: 0. for name in parameters}
    for stage in range(stages):
        for name in parameters:
            new_derivative = derivatives[name](x, y, w)
            old_derivative = prev_derivatives[name]
            delta = deltas[name]
            delta = numpy.where(new_derivative * old_derivative > 0, delta * positive_step, delta * negative_step)
            delta = numpy.clip(delta, min_step, max_step)
            deltas[name] = delta
            val = parameters[name].get_value()
            parameters[name].set_value(val - delta * numpy.sign(new_derivative))
            new_derivative[new_derivative * old_derivative < 0] = 0
            prev_derivatives[name] = new_derivative

        if (stage + 1) % verbose == 0:
            print(loss(x, y, w))


def irprop_star_trainer(x, y, w, parameters, derivatives, loss, stages=100,
                        positive_step=1.2, negative_step=0.5, max_step=1., min_step=1e-6,
                        random=0, verbose=SILENT):
    """ IRPROP* trainer (own experimental modification, not recommended for usage) """
    from collections import defaultdict

    deltas = defaultdict(lambda: 1e-3)
    prev_derivatives = defaultdict(lambda: 0)
    for stage in range(stages):
        for name in parameters:
            new_derivative_ = derivatives[name](x, y, w).flatten()
            new_derivative_plus = new_derivative_[:, numpy.newaxis] + new_derivative_[numpy.newaxis, :]
            new_derivative_minus = new_derivative_[:, numpy.newaxis] - new_derivative_[numpy.newaxis, :]
            for i, new_derivative in enumerate([new_derivative_plus, new_derivative_minus]):
                iname = '{}__{}'.format(i, name)
                old_derivative = prev_derivatives[iname]
                delta = deltas[iname]
                delta = numpy.where(new_derivative * old_derivative > 0, delta * positive_step, delta * negative_step)
                delta = numpy.clip(delta, min_step, max_step)
                deltas[iname] = delta
                val = parameters[name].get_value()
                parameters[name].set_value(val - (delta * numpy.sign(new_derivative)).sum(axis=1).reshape(val.shape))
                new_derivative[new_derivative * old_derivative < 0] = 0
                prev_derivatives[iname] = new_derivative
        if (stage + 1) % verbose == 0:
            print(loss(x, y, w))


def irprop_plus_trainer(x, y, w, parameters, derivatives, loss, stages=100,
                        positive_step=1.2, negative_step=0.5, max_step=1., min_step=1e-6,
                        random=0, verbose=SILENT):
    """IRPROP+ trainer, see http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.17.1332"""
    deltas = dict([(name, 1e-3) for name in parameters])
    prev_derivatives = dict([(name, 0.) for name in parameters])
    prev_loss_value = 1e10
    for stage in range(stages):
        loss_value = loss(x, y, w)
        for name in parameters:
            new_derivative = derivatives[name](x, y, w)
            old_derivative = prev_derivatives[name]
            val = parameters[name].get_value()
            delta = deltas[name]
            if loss_value > prev_loss_value:
                # step back on those variables where sign was changed
                val += numpy.where(new_derivative * old_derivative < 0, delta * numpy.sign(old_derivative), 0)
            delta = numpy.where(new_derivative * old_derivative > 0, delta * positive_step, delta * negative_step)
            delta = numpy.clip(delta, min_step, max_step)
            deltas[name] = delta
            val -= numpy.where(new_derivative * old_derivative >= 0, delta * numpy.sign(new_derivative), 0)
            parameters[name].set_value(val)
            new_derivative[new_derivative * old_derivative < 0] = 0
            prev_derivatives[name] = new_derivative
        prev_loss_value = loss_value
        if (stage + 1) % verbose == 0:
            print(loss(x, y, w))


def adadelta_trainer(x, y, w, parameters, derivatives, loss, stages=1000, decay_rate=0.95,
                     epsilon=1e-5, learning_rate=1., batch=1000, random=0, verbose=SILENT):
    random = check_random_state(random)
    cumulative_derivatives = {name: epsilon for name in parameters}
    cumulative_steps = {name: epsilon for name in parameters}

    for stage in range(stages):
        xp, yp, wp = get_batch(x, y, w, batch_size=batch, random=random)
        for name in parameters:
            derivative = derivatives[name](xp, yp, wp)
            cumulative_derivatives[name] *= decay_rate
            cumulative_derivatives[name] += (1 - decay_rate) * derivative ** 2

            step = - numpy.sqrt(
                (cumulative_steps[name] + epsilon) / (cumulative_derivatives[name] + epsilon)) * derivative
            cumulative_steps[name] *= decay_rate
            cumulative_steps[name] += (1 - decay_rate) * step ** 2

            val = parameters[name].get_value()
            parameters[name].set_value(val + learning_rate * step)

        if (stage + 1) % verbose == 0:
            print(loss(x, y, w))


trainers = {'sgd': sgd_trainer,
            'irprop-': irprop_minus_trainer,
            'irprop+': irprop_plus_trainer,
            'irprop*': irprop_star_trainer,
            'adadelta': adadelta_trainer,
}
# endregion


# TODO think of dropper and noises
# TODO add scaler
# TODO yield system


def _prepare_scaler(scaler):
    """Returns new transformer used in neural network

    :param scaler:
    :return:
    """
    if scaler == 'standard':
        return preprocessing.StandardScaler()
    elif scaler == 'minmax':
        return preprocessing.MinMaxScaler()
    else:
        assert isinstance(scaler, TransformerMixin), 'provided scaler should be derived from TransformerMixin'
        return clone(scaler)




class AbstractNeuralNetworkClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, layers=None, scaler='standard', loss='log_loss', trainer='irprop-', trainer_parameters=None, random_state=None):
        """
        Constructs the neural network based on Theano (for classification purposes).
        Supports only binary classification, supports weights, which makes it usable in boosting.

        Works in sklearn fit-predict way: X is [n_samples, n_features], y is [n_samples], sample_weight is [n_samples].
        Works as usual sklearn classifier, can be used in boosting, for instance, pickled, etc.
        :param layers: list of int, e.g [9, 7] - the number of units in each *hidden* layer
        :param scaler: Transformer used to transform features
        :param loss: loss function used (log_loss by default), str ot function(y, pred, w) -> float
        :param trainer: string, name of optimization method used
        :param dict trainer_parameters: parameters passed to trainer function (learning_rate, etc., trainer-specific).
        """
        self.scaler = scaler
        self.layers = layers
        self.loss = loss
        self.prepared = False
        self.parameters = {}
        self.derivatives = {}
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
        and initializes the weights"""
        self.random_state = check_random_state(self.random_state)
        self.layers_ = [n_input_features] + self.layers + [1]
        self.parameters = {}
        activation_raw = self.prepare()
        activation = lambda x: activation_raw(x).flatten()
        loss_function = losses.get(self.loss, self.loss)
        loss_ = lambda x, y, w: loss_function(y, activation(x), w)
        self.scaler_ = _prepare_scaler(self.scaler)
        x = T.matrix('X')
        y = T.vector('y')
        w = T.vector('w')
        self.Activation = theano.function([x], activation(x))
        self.Loss = theano.function([x, y, w], loss_(x, y, w))
        for name, param in self.parameters.iteritems():
            self.derivatives[name] = theano.function([x, y, w], T.grad(loss_(x, y, w), param))

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
        result[:, 1] = expit(self.Activation(X))
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
        :param sample_weight: optional, numpy.array of shape [n_samples],
            weights are normalized (so that 'mean' == 1).
        :return float, the loss vales computed"""
        sample_weight = check_sample_weight(y, sample_weight, normalize=True)
        return self.Loss(X, y, sample_weight)

    def fit(self, X, y, sample_weight=None, trainer=None, **trainer_parameters):
        """ Prepare the model by optimizing selected loss function with some trainer.
        This method can (and should) be called several times, each time with new parameters
        :param X: numpy.array of shape [n_samples, n_features]
        :param y: numpy.array of shape [n_samples]
        :param sample_weight: numpy.array of shape [n_samples], leave None for array of 1's
        :param trainer: str, method used to minimize loss, overrides one in the ctor
        :param trainer_parameters: parameters for this method, override ones in ctor
        :return: self """
        X, y, sample_weight = check_xyw(X, y, sample_weight)
        self.classes_ = numpy.array([0, 1])
        assert (numpy.unique(y) == self.classes_).all(), 'only two-class classification supported, labels are 0 and 1'
        if not self.prepared:
            self._prepare(X.shape[1])
            self.prepared = True
        sample_weight = check_sample_weight(y, sample_weight, normalize=True)

        trainer = trainers[self.trainer if trainer is None else trainer]
        parameters_ = {} if self.trainer_parameters is None else self.trainer_parameters.copy()
        parameters_.update(trainer_parameters)
        trainer(X, y, sample_weight, self.parameters, self.derivatives, self.Loss, **parameters_)
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
    """The result is computed as h = sigmoid(Ax), output = sum_{ij} B_ij h_i (1 - h_j) """

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


class ObliviousNeuralNetwork(AbstractNeuralNetworkClassifier):
    """ Uses idea of oblivious trees,
     but not strict cuts on features and not rectangular cuts, but linear conditions
     """
    # TODO: needs pretraining (i.e. by oblivious tree) first
    def prepare(self):
        n1, n2, n3 = self.layers_
        W1 = theano.shared(value=self.random_state.normal(size=[n1, n2]).astype(floatX), name='W1')
        W2 = theano.shared(value=self.random_state.normal(size=[2] * n2).astype(floatX), name='W2')
        self.parameters = {'W1': W1, 'W2': W2}

        def activation(input):
            x = T.nnet.sigmoid(T.dot(input, W1))
            first = T.swapaxes(T.stack(x, (1 - x)), 0, 1)
            result = first[:, :, 0]
            for axis in range(1, n2):
                result = T.batched_tensordot(result, first[:, :, axis], axes=[[], []])

            return T.tensordot(result, W2, axes=[range(1, n2 + 1), range(n2)])

        return activation

# endregion