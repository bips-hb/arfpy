# arfpy: Adversarial random forests <a href='https://bips-hb.github.io/arfpy/'><img src='docs/figures/logo.png' align="right" height="139" /></a>


## Introduction
This is a python implementation of adversarial random forests (ARFs) for density estimation and generative modelling. Adversarial random forests (ARFs) recursively partition data into fully factorized leaves, where features are jointly independent. The procedure is iterative, with alternating rounds of generation and discrimination. Data become increasingly realistic at each round, until original and synthetic samples can no longer be reliably distinguished. This is useful for several unsupervised learning tasks, such as density estimation and data synthesis. Methods for both are implemented in this package. ARFs naturally handle unstructured data with mixed continuous and categorical covariates. They inherit many of the benefits of RFs, including speed, flexibility, and solid performance with default parameters. 


## Installation
The `arf` package is available on [PyPI](https://pypi.org/):
```bash
$ pip install arfpy
```
To install the development version from GitHub, run:
```bash
git clone https://bips-hb.github.io/arfpy/
python setup.py install
```

## Usage
Using Fisher's iris dataset, we train an ARF, estimate distribution parameters and generate new data:

```python

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

```

## Other distributions
An R implementation of ARF is available on [CRAN](https://cran.r-project.org/web/packages/arf/index.html) and the development version  [here](https://github.com/bips-hb/arf/).

## References
* Watson, D. S., Blesch, K., Kapar, J. & Wright, M. N. (2022). Adversarial random forests for density estimation and generative modeling. To appear in *Proceedings of the 26th International Conference on Artificial Intelligence and Statistics*. Preprint: https://arxiv.org/abs/2205.09435.