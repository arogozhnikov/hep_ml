# Changelog


## [Unreleased]

### Major Changes
- Support for Python 3.9 up to 3.13.

### Breaking Changes
- Removed `hep_ml.nnet` module, which was based on Theano. Users are encouraged to use PyTorch, JAX, TensorFlow and similar for neural networks.
- Removed support for Python 3.8 and earlier versions.