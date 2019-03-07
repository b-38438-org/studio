import unittest
from brightics.common.datasets import load_iris
from brightics.function.classification import logistic_regression_train, logistic_regression_predict


class TestLogisticRegression(unittest.TestCase):
    def setUp(self):
        print("*** Logistic Regression UnitTest Start ***")
        self.iris = load_iris()

    def tearDown(self):
        print("*** Logistic Regression UnitTest End ***")

    def test_train_predict(self):
        input_dataframe = self.iris
        res_train = logistic_regression_train(input_dataframe,
                                              feature_cols=['sepal_length', 'sepal_width',
                                                            'petal_length', 'petal_width'],
                                              label_col='species',
                                              random_state=12345)
        res_predict = logistic_regression_predict(input_dataframe,
                                                  res_train['model'],
                                                  prediction_col='prediction')
        table = res_predict['out_table'].values.tolist()
        self.assertEqual(table[0], [5.1, 3.5, 1.4, 0.2, 'setosa', 'setosa', 0.8796816489561853, 0.1203075379065891, 1.0813137225508066e-05])
        self.assertEqual(table[1], [4.9, 3.0, 1.4, 0.2, 'setosa', 'setosa', 0.7997063251281568, 0.2002632923353134, 3.0382536530016292e-05])
        self.assertEqual(table[2], [4.7, 3.2, 1.3, 0.2, 'setosa', 'setosa', 0.853796794849413, 0.14617730202211324, 2.590312847381948e-05])
        self.assertEqual(table[3], [4.6, 3.1, 1.5, 0.2, 'setosa', 'setosa', 0.8253831268363401, 0.17455893749671547, 5.793566694451843e-05])
        self.assertEqual(table[4], [5.0, 3.6, 1.4, 0.2, 'setosa', 'setosa', 0.8973236276177116, 0.10266516737872604, 1.1205003562481428e-05])


if __name__ == '__main__':
    unittest.main()
