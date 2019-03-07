import unittest
from brightics.common.datasets import load_iris
from brightics.function.classification import svm_classification_train, svm_classification_predict


class TestSVMClassification(unittest.TestCase):
    def setUp(self):
        print("*** SVM Classification UnitTest Start ***")
        self.iris = load_iris()

    def tearDown(self):
        print("*** SVM Classification UnitTest End ***")

    def test_train_predict(self):
        input_dataframe = self.iris
        res_train = svm_classification_train(input_dataframe,
                                             feature_cols=['sepal_length', 'sepal_width',
                                                           'petal_length', 'petal_width'],
                                             label_col='species',
                                             random_state=12345)
        res_predict = svm_classification_predict(input_dataframe,
                                                 res_train['model'],
                                                 prediction_col='prediction')
        table = res_predict['out_table'].values.tolist()
        self.assertEqual(table[0], [5.1, 3.5, 1.4, 0.2, 'setosa', 'setosa', 0.9723426469898122, 0.01392386466156433, 0.013733488348623488])
        self.assertEqual(table[1], [4.9, 3.0, 1.4, 0.2, 'setosa', 'setosa', 0.9688158193430956, 0.016554394697797053, 0.01462978595910726])
        self.assertEqual(table[2], [4.7, 3.2, 1.3, 0.2, 'setosa', 'setosa', 0.9710317808668876, 0.013516701219045182, 0.015451517914067197])
        self.assertEqual(table[3], [4.6, 3.1, 1.5, 0.2, 'setosa', 'setosa', 0.9652322609331151, 0.01774583823981569, 0.01702190082706917])
        self.assertEqual(table[4], [5.0, 3.6, 1.4, 0.2, 'setosa', 'setosa', 0.972109737865417, 0.013482431853932072, 0.014407830280651181])


if __name__ == '__main__':
    unittest.main()
