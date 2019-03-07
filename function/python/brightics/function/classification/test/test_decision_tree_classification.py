import unittest
from brightics.common.datasets import load_iris
from brightics.function.classification import decision_tree_classification_train, decision_tree_classification_predict


class TestDecisionTreeClassification(unittest.TestCase):
    def setUp(self):
        print("*** Decision Tree Classification UnitTest Start ***")
        self.iris = load_iris()

    def tearDown(self):
        print("*** Decision Tree Classification UnitTest End ***")

    def test_train_predict(self):
        input_dataframe = self.iris
        res_train = decision_tree_classification_train(input_dataframe,
                                                       feature_cols=['sepal_length', 'sepal_width',
                                                                     'petal_length', 'petal_width'],
                                                       label_col='species',
                                                       random_state=12345)
        res_predict = decision_tree_classification_predict(input_dataframe,
                                                           res_train['model'],
                                                           prediction_col='prediction')
        table = res_predict['out_table'].values.tolist()
        self.assertEqual(table[0], [5.1, 3.5, 1.4, 0.2, 'setosa', 'setosa'])
        self.assertEqual(table[1], [4.9, 3.0, 1.4, 0.2, 'setosa', 'setosa'])
        self.assertEqual(table[2], [4.7, 3.2, 1.3, 0.2, 'setosa', 'setosa'])
        self.assertEqual(table[3], [4.6, 3.1, 1.5, 0.2, 'setosa', 'setosa'])
        self.assertEqual(table[4], [5.0, 3.6, 1.4, 0.2, 'setosa', 'setosa'])


if __name__ == '__main__':
    unittest.main()
