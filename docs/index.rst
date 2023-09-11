.. arfpy documentation master file, created by
   sphinx-quickstart on Mon Apr 17 15:07:35 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to arfpy's documentation!
=================================

This is a python implementation of adversarial random forests (ARFs) for density estimation and generative modelling. Adversarial random forests (ARFs) recursively partition data into fully factorized leaves, where features are jointly independent. The procedure is iterative, with alternating rounds of generation and discrimination. Data become increasingly realistic at each round, until original and synthetic samples can no longer be reliably distinguished. This is useful for several unsupervised learning tasks, such as density estimation and data synthesis. Methods for both are implemented in this package. ARFs naturally handle unstructured data with mixed continuous and categorical covariates. They inherit many of the benefits of RFs, including speed, flexibility, and solid performance with default parameters. 


.. toctree::
   :maxdepth: 2
   :caption: Module Documentation:

   modules

   


Installation
------------------

The ``arf`` package is available on `PyPI`_. 


.. _PyPI:  https://pypi.org/

.. highlight:: console

::

   $ pip install arfpy

To install the development version from GitHub, run:

::

   $ git clone https://bips-hb.github.io/arfpy/
   $ python setup.py install


Usage
------------------
Using Fisher's iris dataset, we train an ARF, estimate distribution parameters and generate new data:

.. highlight:: python

::
   
   from sklearn.datasets import load_iris
   from arfpy import arf
   import pandas as pd
   
   # Load data
   iris = load_iris() 
   df = pd.DataFrame(iris['data'], columns=iris['feature_names'])

   # Train the ARF
   my_arf = arf.arf(x = df)

   # Get density estimates
   my_arf.forde()

   # Generate data
   my_arf.forge(n = 10)



------------------

Here are some examples that illustrate how ARF works.

.. toctree::
   :maxdepth: 2
   :caption: Data Examples:

   examples


Other distributions
--------------------

An R implementation of ARF is available on `CRAN`_.

.. _CRAN:  https://cran.r-project.org/web/packages/arf/index.html 

The development version of this package is on `GitHub`_. 

.. _GitHub: https://github.com/bips-hb/arfpy/

References
------------------

Watson, D. S., Blesch, K., Kapar, J. & Wright, M. N. (2022). Adversarial random forests for density estimation and generative modeling. To appear in *Proceedings of the 26th International Conference on Artificial Intelligence and Statistics*. Preprint: https://arxiv.org/abs/2205.09435.


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
