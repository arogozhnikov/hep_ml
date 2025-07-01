# Changelog


## [Unreleased]

## 0.8.0

### Major Changes
- Support for Python 3.9 up to 3.13.

### Breaking Changes
- Require at least stable versions (>1.0) for scikit-learn, numpy, scipy
- Removed `hep_ml.nnet` module, which was based on Theano. Users are encouraged to use PyTorch, JAX, TensorFlow and similar for neural networks.
- Removed support for Python 3.8 and earlier versions.
