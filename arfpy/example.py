# this is an example
from sklearn.datasets import load_iris
import pandas as pd
import arf as arf
import utils

# load data
iris = load_iris()
df = pd.DataFrame(iris['data'], columns=iris['feature_names'])

# define ARF object
arf = arf.arf(x = df)

# estimate density
FORDE = arf.forde()
FORDE['cnt']
FORDE['cat']
FORDE['forest']
FORDE['meta']

# generate data
arf.forge(n =10)