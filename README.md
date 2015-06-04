# hep_ml
This library provides specific ml-tools for purposes of high energy physics (written in python).


# Main points
* working on uniform classifiers - the classifiers with low correlation of predictions and mass (or some other variable(s))
  * __uBoost__ optimized implementation inside
  * __uGradientBoosting__ (with different losses, specially __FlatnessLoss__ is very interesting)
  * measures of uniformity (`SDE`, `Theil`, `CvM`, `KS`)
* classifiers and tools have `scikit-learn` compatible interface (and uses many things from `sklearn`) <br />
  Classifiers are compatible with [REP](https://github.com/yandex/rep) and moreover provided inside the REP docker image.
* there is also procedure to generate toy Monte-Carlo in `toymc` module <br />
  (generates new set of events based on the set of events we already have with same distribution) 
  and special notebook 'ToyMonteCarlo' to demonstrate and analyze its results. 

### Installation
To use the repository, clone it and install with `pip`:
<pre>
git clone https://github.com/iamfullofspam/hep_ml.git
cd hep_ml
sudo pip install .
</pre>

### Getting started
To run most of the notebooks, only IPython and some python libraries are needed.
Those should be installed by `pip` as well. 


In order to work with `.root` files, you need CERN ROOT (make sure you have it by typing `root` in the console) 
with `pyROOT` package.
