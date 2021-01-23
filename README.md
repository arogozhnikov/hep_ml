# hep_ml

**hep_ml** provides specific machine learning tools for purposes of high energy physics.

[![travis status](https://travis-ci.org/arogozhnikov/hep_ml.svg?branch=master)](https://travis-ci.org/arogozhnikov/hep_ml)
[![PyPI version](https://badge.fury.io/py/hep-ml.svg)](https://badge.fury.io/py/hep-ml)
[![Documentation](https://img.shields.io/badge/documentation-link-blue.svg)](https://arogozhnikov.github.io/hep_ml/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.1247391.svg)](https://doi.org/10.5281/zenodo.1247391)



![hep_ml, python library for high energy physics](https://github.com/arogozhnikov/hep_ml/blob/data/data_to_download/hep_ml_image.png)


### Main features

* uniform classifiers - the classifiers with low correlation of predictions and mass (or some other variable, or even set of variables)
  * __uBoost__ optimized implementation inside
  * __UGradientBoosting__ (with different losses, specially __FlatnessLoss__ is of high interest)
* measures of uniformity (see **hep_ml.metrics**)
* advanced losses for classification, regression and ranking for __UGradientBoosting__ (see **hep_ml.losses**).  
* **hep_ml.nnet** - theano-based flexible neural networks 
* **hep_ml.reweight** - reweighting multidimensional distributions <br />
  (_multi_ here means 2, 3, 5 and more dimensions - see GBReweighter!)
* **hep_ml.splot** - minimalistic sPlot-ting 
* **hep_ml.speedup** - building models for fast classification (Bonsai BDT)
* **sklearn**-compatibility of estimators.

### Installation

Basic installation:

```bash
pip install hep_ml
```

If you're new to python and never used `pip`, first install scikit-learn [with these instructions](http://scikit-learn.org/stable/install.html).

To use **latest development version**, clone it and install with `pip`:
```bash
git clone https://github.com/arogozhnikov/hep_ml.git
cd hep_ml
pip install .
```

Local testing: 
```bash
nosetests tests/
```

### Links

* [documentation](https://arogozhnikov.github.io/hep_ml/)
* [notebooks, code examples](https://github.com/arogozhnikov/hep_ml/tree/master/notebooks)
    - you may need to install `ROOT` and `root_numpy` to run those 
* [repository](https://github.com/arogozhnikov/hep_ml)
* [issue tracker](https://github.com/arogozhnikov/hep_ml/issues)

### Related projects 
Libraries you'll require to make your life easier and HEPpier.

* [IPython Notebook](http://ipython.org/notebook.html) &mdash; web-shell for python
* [scikit-learn](http://scikit-learn.org/)  &mdash; general-purpose library for machine learning in python
* [numpy](http://www.numpy.org/)  &mdash; 'MATLAB in python', vector operation in python. 
    Use it you need to perform any number crunching. 
* [theano](http://deeplearning.net/software/theano/)  &mdash; optimized vector analytical math engine in python
* [ROOT](https://root.cern.ch/)  &mdash; main data format in high energy physics 
* [root_numpy](http://rootpy.github.io/root_numpy/)  &mdash; python library to deal with ROOT files (without pain)


### License
Apache 2.0, `hep_ml` is an open-source library.

### Platforms 
Linux, Mac OS X and Windows are supported.

**hep_ml** supports both python 2 and python 3.
