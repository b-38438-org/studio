import unittest
from brightics.common.datasets import load_iris
from brightics.function.clustering import kmeans_silhouette_train_predict, kmeans_predict


class TestKmeans(unittest.TestCase):
    def setUp(self):
        print("*** Kmeans UnitTest Start ***")
        self.iris = load_iris()

    def tearDown(self):
        print("*** Kmeans UnitTest End ***")

    def test_kmeans_silhouette_train_predict(self):
        input_dataframe = self.iris
        res_silhouette = kmeans_silhouette_train_predict(input_dataframe,
                                                         input_cols=['sepal_length', 'sepal_width',
                                                                     'petal_length', 'petal_width'],
                                                         seed=12345)
        table = res_silhouette['out_table'].values.tolist()
        self.assertEqual(table[0], [5.1, 3.5, 1.4, 0.2, 'setosa', 0])
        self.assertEqual(table[1], [4.9, 3.0, 1.4, 0.2, 'setosa', 0])
        self.assertEqual(table[2], [4.7, 3.2, 1.3, 0.2, 'setosa', 0])
        self.assertEqual(table[3], [4.6, 3.1, 1.5, 0.2, 'setosa', 0])
        self.assertEqual(table[4], [5.0, 3.6, 1.4, 0.2, 'setosa', 0])

        res_predict = kmeans_predict(input_dataframe, res_silhouette['model'])
        table = res_predict['out_table'].values.tolist()
        self.assertEqual(table[0], [5.1, 3.5, 1.4, 0.2, 'setosa', 0])
        self.assertEqual(table[1], [4.9, 3.0, 1.4, 0.2, 'setosa', 0])
        self.assertEqual(table[2], [4.7, 3.2, 1.3, 0.2, 'setosa', 0])
        self.assertEqual(table[3], [4.6, 3.1, 1.5, 0.2, 'setosa', 0])
        self.assertEqual(table[4], [5.0, 3.6, 1.4, 0.2, 'setosa', 0])


if __name__ == '__main__':
    unittest.main()
