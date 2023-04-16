# this is an example
from sklearn.datasets import *
import arf as arf
import utils
import pandas as pd

# load data
iris = load_iris() # try also load_digits() 
df = pd.DataFrame(iris['data'], columns=iris['feature_names'])

# define ARF object
my_arf = arf.arf(x = df)

# estimate density
FORDE = my_arf.forde()

# generate data
my_arf.forge(n =10)


# try with mixed data - add categorical target, 3 levels
iris = load_iris()
df = pd.concat([pd.DataFrame(iris['data']), pd.Series(iris['target'])], axis = 1)#, columns=iris['feature_names'])
colnames = iris['feature_names']
colnames.append('target')
df.columns = colnames
df['target'] = df['target'].astype('category')
# new arf
my_arf = arf.arf(x = df, delta=0, min_node_size =5)
# estimate density
FORDE = my_arf.forde(alpha = 0.5)

# generate data
my_arf.forge(n =10)