import numpy as np
import pandas as pd
from arfpy import arf
import sklearn
import unittest

# this script provides automated tests for arfpy

# run this script from command line if you want to test arfpy on multiple datasets
# if you want to test arfpy on individual datasets, you can run the scripts imported here directly, e.g. test_iris.py for only iris data 

class TestClass:
    def __init__(self, *args,**kwargs):
        super().__init__(*args,**kwargs)

    # test FORDE output for type and names
    def test_forde_type(self):
        self.assertIsInstance(self.FORDE, dict,  'FORDE output is not a dict')
        assert self.FORDE.keys() == {'cnt', 'cat', 'forest', 'meta'}
        self.assertIsInstance(self.FORDE['cnt'], pd.core.frame.DataFrame, 'FORDE output of continuous variables is not a data frame')
        self.assertIsInstance(self.FORDE['cat'], pd.core.frame.DataFrame,  'FORDE output of categorical variables is not a data frame')
        self.assertIsInstance(self.FORDE['forest'], sklearn.ensemble._forest.RandomForestClassifier,  'FORDE output of forest is not a sklearn forest')
        self.assertIsInstance(self.FORDE['meta'], pd.core.frame.DataFrame,  'FORDE output of meta data is not data frame')
    
    # test type of FORGE output
    def test_forge_type(self):
        self.assertIsInstance(self.gen_df, pd.core.frame.DataFrame)

    # number of generated instances = number of instances that should be generated?
    def test_instance_generation(self):
        self.assertEqual(self.gen_df.shape[0], self.n, 'Not as many instances generated as specified')
    
    # number of columns in origina data = number of columns in generated data?
    def test_column_generation(self):
        self.assertEqual(self.gen_df.shape[1], self.df.shape[1], 'Not as many columns generated as specified')
    
    # data types in original data = data types in generated data?
    def test_datatypes(self):
        self.assertCountEqual(self.gen_df.dtypes, self.df.dtypes, 'Mismatch in original and generated dtypes')

    # column names in original data = column names in generated data? 
    def test_colummnames(self):
        self.assertCountEqual(self.gen_df.columns, self.df.columns, 'Mismatch in original and generated column names')

    # test whether probabilities for categorical variables sum to 1
    def test_probs_sum_to_one(self):
        if self.FORDE['cat'].shape[0] > 0: # if any categoricals
            self.assertTrue(all(np.isclose(self.FORDE['cat'].groupby(['f_idx', 'variable']).sum('prob')['prob'],1)), 'probabilities of categoricals do not sum to one')

    # test whether num_trees argument gets parsed correctly to sklearn.RandomForest
    def test_num_trees(self):
        tmp_num_trees = 40
        tmp_arf = arf.arf(x = self.my_arf.x_real, num_trees=tmp_num_trees) # refit arf with new specifications
        self.assertEqual(tmp_arf.clf.n_estimators, tmp_num_trees, "number of trees in arf do not correspond to parameter")

    # test whether min_node_size argument gets parsed correctly to sklearn.RandomForest
    def test_min_node_size(self):
        tmp_min_node_size = 20
        tmp_arf = arf.arf(x = self.my_arf.x_real, min_node_size=tmp_min_node_size) # refit arf with new specifications
        self.assertEqual(tmp_arf.clf.min_samples_leaf, tmp_min_node_size, "minimum node size in arf does not correspond to parameter")
    
    # test whether assertion errors are raised
    def test_assertion_errors(self):
        with self.assertRaises(ValueError):
            self.my_arf.forde(dist = "misspelleddistribution")
        with self.assertRaises(AssertionError) as msg:
            arf.arf(x = pd.concat([self.my_arf.x_real,self.my_arf.x_real], axis =1))
        self.assertEqual(str(msg.exception), "every column must have a unique column name")
        with self.assertRaises(AssertionError) as msg:
            arf.arf(x = np.array(self.my_arf.x_real))
        self.assertEqual(str(msg.exception), f"expected pandas DataFrame as input, got:{type(np.array(self.my_arf.x_real))}")
        with self.assertRaises(AssertionError) as msg:
            arf.arf(x = self.my_arf.x_real, min_node_size=-1)
        self.assertEqual(str(msg.exception), "minimum number of samples in terminal nodes (parameter min_node_size) must be greater than zero")
        with self.assertRaises(AssertionError) as msg:
            arf.arf(x = self.my_arf.x_real, num_trees=-1)
        self.assertEqual(str(msg.exception), "number of trees in the random forest (parameter num_trees) must be greater than zero")
        with self.assertRaises(AssertionError) as msg:
            arf.arf(x = self.my_arf.x_real, delta=-1)
        self.assertEqual(str(msg.exception), "parameter delta must be in range 0 <= delta <= 0.5")
        with self.assertRaises(AssertionError) as msg:
            arf.arf(x = self.my_arf.x_real, max_iters=-1)
        self.assertEqual(str(msg.exception), "negative number of iterations is not allowed: parameter max_iters must be >= 0")
        
# run all tests
if __name__ == '__main__':
    from test_iris import TestIris
    from test_diabetes import TestDiabetes
    unittest.main()