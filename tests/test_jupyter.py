import unittest
from unittest.mock import Mock

import pandas as pd

from misinspect.gui.jupyter import MisClassifiedTxnVisualizer


class TestMisClassifiedTxnVisualizer(unittest.TestCase):
    def setUp(self):
        # テスト用の MisClassifiedTxnAnalyzer モックを作成
        self.analyzer = Mock()
        self.analyzer.user_id_col = "user_id"
        self.analyzer.dataset = pd.DataFrame(
            {"user_id": [1, 2, 3], "classification_type": ["FP", "FN", "TP"]}
        )

        self.visualizer = MisClassifiedTxnVisualizer(self.analyzer)

    def test_initialization(self):
        # 初期化のテスト
        self.assertIsNotNone(self.visualizer.analyzer)
        self.assertEqual(self.visualizer.target_user_ids, [])

    def test_update_misclassified_data(self):
        # 誤分類データ更新メソッドのテスト
        self.visualizer.update_misclassified_data()
        self.assertIsNotNone(self.visualizer.fp_user_ids)
        self.assertIsNotNone(self.visualizer.fn_user_ids)

    # その他、インタラクティブな動作やUIのテストも追加します。
    # これには特別なフレームワークやツールが必要になる場合があります。


if __name__ == "__main__":
    unittest.main()
