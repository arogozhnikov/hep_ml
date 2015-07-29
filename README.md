# hep_ml
**hep_ml** provides specific machine learning tools for purposes of high energy physics (written in python).



## Main points
* uniform classifiers - the classifiers with low correlation of predictions and mass (or some other variable(s))
  * __uBoost__ optimized implementation inside
  * __UGradientBoosting__ (with different losses, specially __FlatnessLoss__ is very interesting)
* measures of uniformity (see **hep_ml.metrics**)
* advanced losses for classification, regression and ranking for __UGradientBoosting__ (see **hep_ml.losses**).  
* **hep_ml.nnet** - theano-based flexible neural networks 
* **sklearn**-compatibility of estimators.

### Installation

```bash
pip install hep_ml
```

To use latest version, clone it and install with `pip`:
```bash
git clone https://github.com/arogozhnikov/hep_ml.git
cd hep_ml
sudo pip install .
```

### Links

* [documentation](https://arogozhnikov.github.io/hep_ml/)
* [notebook examples](https://github.com/arogozhnikov/hep_ml/tree/master/notebooks)
* [repository](https://github.com/arogozhnikov/hep_ml)
* [issue tracker](https://github.com/arogozhnikov/hep_ml/issues)
* [old repository](https://github.com/anaderi/lhcb_trigger_ml)

### Related projects 
Libraries you'll require to make your life easier.

* [IPython Notebook](http://ipython.org/notebook.html) &mdash; web-shell for python
* [scikit-learn](http://scikit-learn.org/)  &mdash; general-purpose library for machine learning in python
* [REP](https://github.com/yandex/REP)  &mdash; python wrappers around different machine learning libraries 
    (including TMVA) + goodies, required to plot learning curves and reports after classification
* [numpy](http://www.numpy.org/)  &mdash; 'MATLAB in python', vector operation in python. 
    Use it you need to perform any number crunching. 
* [theano](http://deeplearning.net/software/theano/)  &mdash; optimized vector analytical math engine in python
* [ROOT](https://root.cern.ch/)  &mdash; main data format in high energy physics 
* [root_numpy](http://rootpy.github.io/root_numpy/)  &mdash; python library to deal with ROOT files (without pain)


### License
Apache 2.0, library is open-source.