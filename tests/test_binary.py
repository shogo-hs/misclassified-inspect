import unittest
from unittest.mock import Mock

import pandas as pd

from misinspect.analysis.binary import MisClassifiedTxnAnalyzer


class TestMisClassifiedTxnAnalyzer(unittest.TestCase):
    def setUp(self):
        # テスト用データセットの準備
        self.test_data = pd.DataFrame(
            {
                "user_id": [1, 2, 3, 4, 5],
                "price": [100, 200, 300, 400, 500],
                "use_dt": pd.to_datetime(
                    [
                        "2021-01-01",
                        "2021-01-02",
                        "2021-01-03",
                        "2021-01-04",
                        "2021-01-05",
                    ]
                ),
                "probability": [0.1, 0.2, 0.8, 0.9, 0.5],
                "label": [0, 0, 1, 1, 0],
            }
        )

        self.analyzer = MisClassifiedTxnAnalyzer(
            dataset=self.test_data,
            user_id_col="user_id",
            price_col="price",
            datetime_col="use_dt",
            prob_col="probability",
            label_col="label",
            threshold=0.5,
        )

    def test_get_misclassified_data(self):
        # 誤分類されたデータの取得テスト
        self.analyzer.get_misclassified_data()
        misclassified_data = self.analyzer.misclassified_data

        self.assertEqual(len(misclassified_data), 1)  # 期待される誤分類されたデータの数

    def test_create_confusion_matrix(self):
        # 混同行列の作成テスト
        self.analyzer.get_misclassified_data()
        confusion_matrix_df = self.analyzer.create_confusion_matrix()

        self.assertIn("TP", confusion_matrix_df["Item"].values)
        self.assertIn("FP", confusion_matrix_df["Item"].values)
        self.assertIn("TN", confusion_matrix_df["Item"].values)
        self.assertIn("FN", confusion_matrix_df["Item"].values)

    # 他のメソッドのテストも同様に記述します。


if __name__ == "__main__":
    unittest.main()
