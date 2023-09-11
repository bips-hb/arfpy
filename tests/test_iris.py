
import unittest
from sklearn.datasets import load_iris
from arfpy import arf
import pandas as pd
from test import TestClass

# Testing arfpy on iris data set
# classification task 
# both continuous and categorical variables

class TestIris(TestClass, unittest.TestCase):
    iris = load_iris()
    df = pd.DataFrame(iris['data'], columns=iris['feature_names'])
    df = pd.concat([pd.DataFrame(iris['data']), pd.Series(iris['target'])], axis = 1)#, columns=iris['feature_names'])
    colnames = iris['feature_names']
    colnames.append('target')
    df.columns = colnames
    df['target'] = df['target'].astype('category')
    my_arf = arf.arf(x = df)
    FORDE = my_arf.forde()
    n = df.shape[0]
    gen_df = my_arf.forge(n = n)


if __name__ == '__main__':
    unittest.main()