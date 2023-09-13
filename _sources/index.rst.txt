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

The ``arfpy`` package is available on `PyPI`_. 


.. _PyPI:  https://pypi.org/

.. highlight:: console

::

   $ pip install arfpy

To install the development version from GitHub, run:

::

   $ git clone https://github.com/bips-hb/arfpy
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


.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Examples

   examples


Examples
------------------

Generative modeling with `twomoons data`_:

.. _twomoons data: https://bips-hb.github.io/arfpy/examples/twomoons.html

.. image:: examples/examples_twomoons_11_0.png

Generative modeling with `digit zero data`_:

.. _digit zero data: https://bips-hb.github.io/arfpy/examples/digits.html

.. image:: examples/examples_digits_9_0.png
.. image:: examples/examples_digits_9_1.png

Generative modeling and density estimation with `multivariate normal data`_:

.. _multivariate normal data: https://bips-hb.github.io/arfpy/examples/mvnorm.html

.. image:: examples/examples_mvnorm_11_0.png


Other distributions
--------------------

An R implementation of ARF is available on `CRAN`_.

.. _CRAN:  https://cran.r-project.org/web/packages/arf/index.html 

The development version of this package is on `GitHub`_. 

.. _GitHub: https://github.com/bips-hb/arfpy/

References
------------------

Watson, D. S., Blesch, K., Kapar, J. & Wright, M. N. (2023). Adversarial random forests for density estimation and generative modeling. In *Proceedings of the 26th International Conference on Artificial Intelligence and Statistics*. Link here_.

.. _here: https://proceedings.mlr.press/v206/watson23a.html

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
