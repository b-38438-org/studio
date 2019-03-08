import unittest
import numpy as np
import numpy.testing as npt
from brightics.common.datasets import load_iris
from brightics.function.extraction import add_shift


class TestAddShift(unittest.TestCase):
    def setUp(self):
        print("*** Add Shift UnitTest Start ***")
        self.input_dataframe = load_iris()

    def tearDown(self):
        print("*** Add Shift UnitTest End ***")

    def test_add_shift(self):
        res = add_shift(self.input_dataframe, input_col='petal_width', shift_list=[1, 2, 3])
        table = res['out_table'].values.tolist()
        npt.assert_array_equal(table[0], [5.1, 3.5, 1.4, 0.2, 'setosa', np.NaN, np.NaN, np.NaN])
        npt.assert_array_equal(table[1], [4.9, 3.0, 1.4, 0.2, 'setosa', 0.2, np.NaN, np.NaN])
        npt.assert_array_equal(table[2], [4.7, 3.2, 1.3, 0.2, 'setosa', 0.2, 0.2, np.NaN])
        npt.assert_array_equal(table[3], [4.6, 3.1, 1.5, 0.2, 'setosa', 0.2, 0.2, 0.2])
        npt.assert_array_equal(table[4], [5.0, 3.6, 1.4, 0.2, 'setosa', 0.2, 0.2, 0.2])


if __name__ == '__main__':
    unittest.main()
