# this is an example that illustrates data generation with twomoons data
from sklearn.datasets import make_moons
from arfpy import arf
import pandas as pd
from matplotlib import pyplot as plt
from numpy import random
random.seed(seed=2022)

# load data
n_train = 2000
n_test = 1000
moons_X, moons_y = make_moons(n_samples = n_train+n_test,  noise=0.1)  
df = pd.DataFrame({"dim_1" : moons_X[:,0], "dim_2" : moons_X[:,1], "target" : moons_y})
df['target'] = df['target'].astype('category')
# define ARF object
my_arf = arf.arf(x = df[:n_train], min_node_size=2, num_trees=20, max_features = 2)

# estimate density
FORDE = my_arf.forde()

# generate data
df_syn = my_arf.forge(n = n_test)

# plot results
plt.subplot(2, 2, 1)
df_test = df[:n_train].sample(n=n_test)
plt.scatter(df_test['dim_1'], df_test['dim_2'], c = df_test['target'], alpha = 0.5)
plt.title('Original Data')

plt.subplot(2, 2, 2)
plt.scatter(df_syn['dim_1'], df_syn['dim_2'], c = df_syn['target'], alpha = 0.5)
plt.title('Synthesized Data')


plt.show()