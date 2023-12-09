import unittest

import pandas as pd

from misinspect.analysis.classification_utils import (
    calculate_classification_types,
    get_classification_data_by_type,
)


class TestClassificationUtils(unittest.TestCase):
    def setUp(self):
        # テスト用データセットの準備
        self.test_data = pd.DataFrame(
            {"probability": [0.1, 0.6, 0.2, 0.8, 0.9], "label": [0, 0, 1, 1, 0]}
        )

    def test_calculate_classification_types(self):
        # 分類結果の計算テスト
        result = calculate_classification_types(
            self.test_data, "probability", "label", 0.5
        )
        self.assertTrue("classification_type" in result.columns)
        self.assertEqual(len(result[result["classification_type"] == "TP"]), 1)
        self.assertEqual(len(result[result["classification_type"] == "FP"]), 2)

    def test_get_classification_data_by_type(self):
        # 分類タイプによるデータ抽出テスト
        result = calculate_classification_types(
            self.test_data, "probability", "label", 0.5
        )
        tp_data = get_classification_data_by_type(result, "classification_type", "TP")
        self.assertEqual(len(tp_data), 1)

    # Spark DataFrameを用いた関数のテストも同様に記述することができますが、
    # それにはPySparkのテスト環境の設定が必要になります。


if __name__ == "__main__":
    unittest.main()
