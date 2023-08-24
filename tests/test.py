import numpy as np
import unittest

# this script provides automated tests for arfpy

# run this script if you want to test arfpy on multiple datasets
# if you want to test arfpy on individual datasets, you can run the scripts imported here directly, e.g. test_iris.py for only iris data 

class TestClass:
    def __init__(self, *args,**kwargs):
        super().__init__(*args,**kwargs)

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

    # summary statistics of original and generated data  -- do they look similar? (rough measure of performance)
    def test_summary_statistics(self):
        with np.printoptions(threshold=np.inf):
            print('\nComparing the data distributions:')
            print(f'Summary statistics for original data \n{self.df.describe()}')
            print(f'Summary statistics for generated data \n{self.gen_df.describe()} ')


# run all tests
if __name__ == '__main__':
    from test_iris import TestIris
    from test_diabetes import TestDiabetes
    unittest.main()