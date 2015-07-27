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
-  **hep\_ml.nnet** - theano-based neural networks

Installation
~~~~~~~~~~~~

To use the repository, clone it and install with ``pip``:

.. code:: bash

    git clone https://github.com/iamfullofspam/hep_ml.git
    cd hep_ml
    sudo pip install .

Links
~~~~~

-  `documentation <https://iamfullofspam.github.io/hep_ml/>`__
-  `notebook
   examples <https://github.com/iamfullofspam/hep_ml/tree/master/notebooks>`__
-  `repository <https://github.com/iamfullofspam/hep_ml>`__
-  `issue tracker <https://github.com/iamfullofspam/hep_ml/issues>`__

Related projects
~~~~~~~~~~~~~~~~

Libraries you'll require to make your life easier.

-  `IPython Notebook <http://ipython.org/notebook.html>`__ — web-shell
   for python
-  `scikit-learn <http://scikit-learn.org/>`__ — general-purpose library
   for machine learning in python
-  `REP <https://github.com/yandex/REP>`__ — python wrappers around
   different machine learning libraries (including TMVA) + goodies,
   required to plot learning curves reports after classification
-  `numpy <http://www.numpy.org/>`__ — 'MATLAB in python', vector
   operation in python. Use it you need to perform any number crunching.
-  `theano <http://deeplearning.net/software/theano/>`__ — optimized
   vector analytical math engine in python
-  `ROOT <https://root.cern.ch/>`__ — main data format in high energy
   physics
-  `root\_numpy <http://rootpy.github.io/root_numpy/>`__ — python
   library to deal with ROOT files (without pain)
