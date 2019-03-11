import unittest
import os.path
import os
from brightics.function.io.unload import unload
from brightics.common.datasets import load_iris


class TestUnload(unittest.TestCase):
    def setUp(self):
        print("*** Unload UnitTest Start ***")
        self.input_dataframe = load_iris()

    def tearDown(self):
        print("*** Unload UnitTest End ***")

    def test_plot_roc_pr_curve(self):
        partial_path = ['/brightics@samsung.com/upload/unload_iris.csv']
        unload(self.input_dataframe, partial_path=partial_path)
        res = os.path.exists(partial_path[0])
        self.assertEqual(res, True)
        if os.path.exists(partial_path[0]):
            os.remove(partial_path[0])


if __name__ == '__main__':
    unittest.main()
