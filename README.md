# hep_ml
**hep_ml** provides specific ml-tools for purposes of high energy physics (written in python).


## Main points
* uniform classifiers - the classifiers with low correlation of predictions and mass (or some other variable(s))
  * __uBoost__ optimized implementation inside
  * __UGradientBoosting__ (with different losses, specially __FlatnessLoss__ is very interesting)
* measures of uniformity (`SDE`, `Theil`, `CvM`, `KS`)
* __UGradientBoosting__ with advanced losses for classification, regression and ranking.  
* **hep_ml.nnet** - theano-based neural networks 


## Installation
To use the repository, clone it and install with `pip`:
```bash
git clone https://github.com/iamfullofspam/hep_ml.git
cd hep_ml
sudo pip install .
```

## Links
* [documentation](iamfullofspam.github.io/hep_ml/)
* [notebook examples](https://github.com/iamfullofspam/hep_ml/tree/master/notebooks)
* [issue tracker](https://github.com/iamfullofspam/hep_ml/issues)

## Related projects you should know about

* [IPython Notebook](http://ipython.org/notebook.html) -- web-shell for ipython
* [ROOT](https://root.cern.ch/) -- main data format in HEP 
* [scikit-learn](http://scikit-learn.org/) -- general-purpose library for machine learning in python
* [REP](https://github.com/yandex/REP) -- python wrappers around different machine learning libraries (including TMVA) + goodies
* [numpy](http://www.numpy.org/) -- 'MATLAB in python', vector operation in python. Don't ever try doing 
* [root_numpy](http://rootpy.github.io/root_numpy/) -- python library to deal with ROOT files (without pain).
* [theano](http://deeplearning.net/software/theano/) -- optimized vector analytical math engine in python.

