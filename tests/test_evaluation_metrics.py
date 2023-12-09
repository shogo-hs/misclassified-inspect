import unittest

import pandas as pd

from misinspect.analysis.evaluation_metrics import calculate_precision_recall


class TestEvaluationMetrics(unittest.TestCase):
    def setUp(self):
        # テスト用データセットの準備
        self.test_data = pd.DataFrame(
            {"probability": [0.1, 0.6, 0.2, 0.8, 0.9], "label": [0, 0, 1, 1, 0]}
        )

    def test_calculate_precision_recall(self):
        # Pandas DataFrame用の精度と再現率の計算テスト
        result = calculate_precision_recall(self.test_data, "label", "probability")
        self.assertTrue("precision" in result.columns and "recall" in result.columns)
        self.assertEqual(len(result), len(self.test_data))

    # PySpark DataFrameを用いた関数のテストも同様に記述することができますが、
    # それにはPySparkのテスト環境の設定が必要になります。


if __name__ == "__main__":
    unittest.main()
