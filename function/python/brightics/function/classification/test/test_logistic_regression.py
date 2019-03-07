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
        self.assertEqual(table[0], [5.1, 3.5, 1.4, 0.2, 'setosa', 'setosa', 0.8780303050242847, 0.12195890005075813, 1.0794924957147882e-05])
        self.assertEqual(table[1], [4.9, 3.0, 1.4, 0.2, 'setosa', 'setosa', 0.797058291879083, 0.20291141319673614, 3.0294924180971997e-05])
        self.assertEqual(table[2], [4.7, 3.2, 1.3, 0.2, 'setosa', 'setosa', 0.8519976652686149, 0.14797647964561186, 2.585508577322853e-05])
        self.assertEqual(table[3], [4.6, 3.1, 1.5, 0.2, 'setosa', 'setosa', 0.8234060190878218, 0.17653615914179305, 5.782177038521865e-05])
        self.assertEqual(table[4], [5.0, 3.6, 1.4, 0.2, 'setosa', 'setosa', 0.8960349729152124, 0.10395383635092102, 1.119073386671762e-05])


if __name__ == '__main__':
    unittest.main()
