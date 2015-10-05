.. hep_ml documentation master file, created by
   sphinx-quickstart on Wed Jul 22 20:45:24 2015.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

hep_ml documentation
====================

**hep_ml** is machine learning library filled with methods used in high energy physics.
Classifiers and transformers from **hep_ml** are **sklearn**-compatible.

.. raw:: html

    <div style='height: 36px;'>
    <iframe src="https://ghbtns.com/github-btn.html?user=arogozhnikov&repo=hep_ml&type=star&count=true" frameborder="0" scrolling="0" width="120px" height="27px"></iframe>
    <iframe src="https://ghbtns.com/github-btn.html?user=arogozhnikov&repo=hep_ml&type=watch&count=true&v=2" frameborder="0" scrolling="0" width="120px" height="27px"></iframe>
    <div style='clear: both' ></div>
    </div>

Installation
____________

To install **hep_ml**, use **pip**. Type in terminal:

.. code:: bash

    pip install hep_ml


Installation for developers
___________________________

After cloning repository type in bash:

.. code:: bash

    cd hep_ml
    pip install -e . -r requirements.txt



Contents:
_________

.. toctree::
   :maxdepth: 2

   self
   gb
   losses
   uboost
   metrics
   nnet
   preprocessing
   reweight
   speedup
   splot
   notebooks


Links
_____

-  `documentation <https://arogozhnikov.github.io/hep_ml/>`__
-  `notebook
   examples <https://github.com/arogozhnikov/hep_ml/tree/master/notebooks>`__
-  `repository <https://github.com/arogozhnikov/hep_ml>`__
-  `issue tracker <https://github.com/arogozhnikov/hep_ml/issues>`__

Related projects
________________

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

