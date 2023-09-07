import unittest
from sklearn.datasets import load_diabetes
from arfpy import arf
import pandas as pd
from test import TestClass

# Testing arfpy on diabetes data set
# regression task 
# continuous variables only

class TestDiabetes(TestClass, unittest.TestCase):
    diabetes = load_diabetes() 
    df = pd.DataFrame(diabetes['data'], columns=diabetes['feature_names'])
    my_arf = arf.arf(x = df)
    FORDE = my_arf.forde()
    n = df.shape[0]
    gen_df = my_arf.forge(n = n)
    
if __name__ == '__main__':
    unittest.main()