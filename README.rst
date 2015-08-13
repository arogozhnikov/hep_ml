hep\_ml
=======

**hep\_ml** provides specific machine learning tools for purposes of
high energy physics (written in python).

Main points
-----------

-  uniform classifiers - the classifiers with low correlation of
   predictions and mass (or some other variable(s))
-  **uBoost** optimized implementation inside
-  **UGradientBoosting** (with different losses, specially
   **FlatnessLoss** is very interesting)
-  measures of uniformity (see **hep\_ml.metrics**)
-  advanced losses for classification, regression and ranking for
   **UGradientBoosting** (see **hep\_ml.losses**).
-  **hep\_ml.nnet** - theano-based flexible neural networks
-  **hep\_ml.reweight** - reweighting multidimensional distributions
   (*multi* here means 2, 3, 5 and more dimensions - see GBReweighter!)
-  **sklearn**-compatibility of estimators.

Installation
------------

.. code:: bash

    pip install hep_ml

To use latest version, clone it and install with ``pip``:

.. code:: bash

    git clone https://github.com/arogozhnikov/hep_ml.git
    cd hep_ml
    sudo pip install .

Links
-----

-  `documentation <https://arogozhnikov.github.io/hep_ml/>`__
-  `notebook
   examples <https://github.com/arogozhnikov/hep_ml/tree/master/notebooks>`__
-  `repository <https://github.com/arogozhnikov/hep_ml>`__
-  `issue tracker <https://github.com/arogozhnikov/hep_ml/issues>`__
-  `old repository <https://github.com/anaderi/lhcb_trigger_ml>`__

Related projects
----------------

Libraries you'll require to make your life easier.

-  `IPython Notebook <http://ipython.org/notebook.html>`__ — web-shell
   for python
-  `scikit-learn <http://scikit-learn.org/>`__ — general-purpose library
   for machine learning in python
-  `REP <https://github.com/yandex/REP>`__ — python wrappers around
   different machine learning libraries (including TMVA) + goodies,
   required to plot learning curves and reports after classification.
   Required to execute *howto*\ s from this repository
-  `numpy <http://www.numpy.org/>`__ — 'MATLAB in python', vector
   operation in python. Use it you need to perform any number crunching.
-  `theano <http://deeplearning.net/software/theano/>`__ — optimized
   vector analytical math engine in python
-  `ROOT <https://root.cern.ch/>`__ — main data format in high energy
   physics
-  `root\_numpy <http://rootpy.github.io/root_numpy/>`__ — python
   library to deal with ROOT files (without pain)

License
-------

Apache 2.0, library is open-source.
