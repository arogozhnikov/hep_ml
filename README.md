# hep_ml
**hep_ml** provides specific machine learning tools for purposes of high energy physics (written in python).

![travis status](https://travis-ci.org/arogozhnikov/hep_ml.svg?branch=master)
[![Build status](https://ci.appveyor.com/api/projects/status/kxatlw869t9ibbo3?svg=true)](https://ci.appveyor.com/project/arogozhnikov/hep-ml)
[![PyPI version](https://badge.fury.io/py/hep_ml.svg)](http://badge.fury.io/py/hep_ml)

![hep_ml, python library for high energy physics](https://github.com/arogozhnikov/hep_ml/blob/data/data_to_download/hep_ml_image.png)


### Main points
* uniform classifiers - the classifiers with low correlation of predictions and mass (or some other variable(s))
  * __uBoost__ optimized implementation inside
  * __UGradientBoosting__ (with different losses, specially __FlatnessLoss__ is very interesting)
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

If you're new to python and don't never used `pip`, first install scikit-learn [with these instructions](http://scikit-learn.org/stable/install.html).

To use latest development version, clone it and install with `pip`:
```bash
git clone https://github.com/arogozhnikov/hep_ml.git
cd hep_ml
sudo pip install .
```

### Links

* [documentation](https://arogozhnikov.github.io/hep_ml/)
* [notebooks, code examples](https://github.com/arogozhnikov/hep_ml/tree/master/notebooks)
* [repository](https://github.com/arogozhnikov/hep_ml)
* [issue tracker](https://github.com/arogozhnikov/hep_ml/issues)
* [old repository](https://github.com/anaderi/lhcb_trigger_ml)

### Related projects 
Libraries you'll require to make your life easier.

* [IPython Notebook](http://ipython.org/notebook.html) &mdash; web-shell for python
* [scikit-learn](http://scikit-learn.org/)  &mdash; general-purpose library for machine learning in python
* [yandex/REP](https://github.com/yandex/REP)  &mdash; python wrappers around different machine learning libraries 
    (including TMVA) + goodies, required to plot learning curves and reports after classification. Required to execute *howto*s from this repository
* [numpy](http://www.numpy.org/)  &mdash; 'MATLAB in python', vector operation in python. 
    Use it whenever you need to perform any number crunching. 
* [theano](http://deeplearning.net/software/theano/)  &mdash; optimized vector analytical math engine in python
* [ROOT](https://root.cern.ch/)  &mdash; main data format in high energy physics 
* [root_numpy](http://rootpy.github.io/root_numpy/)  &mdash; python library to deal with ROOT files (without pain)


### License
Apache 2.0, library is open-source.

### Platforms 
Linux, Mac OS X and Windows are supported.
