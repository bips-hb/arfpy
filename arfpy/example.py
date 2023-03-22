# this is an example
from sklearn.datasets import *
import arf as arf
import utils
import pandas as pd

# load data
iris =  load_iris() # try also load_digits() 
df = pd.DataFrame(iris['data'], columns=iris['feature_names'])

# define ARF object
my_arf = arf.arf(x = df, delta=0)

# estimate density
FORDE = my_arf.forde()

# generate data
my_arf.forge(n =10)