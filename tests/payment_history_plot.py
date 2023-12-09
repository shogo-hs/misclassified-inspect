import unittest
from datetime import datetime
from unittest.mock import patch

import pandas as pd

from misinspect.visualization.payment_history_plot import plot_payment_history


class TestPlotPaymentHistory(unittest.TestCase):
    def setUp(self):
        # テスト用データセットの準備
        self.test_data = pd.DataFrame(
            {
                "use_dt": pd.date_range(start="2021-01-01", periods=5, freq="D"),
                "price": [100, 200, 300, 400, 500],
                "label": [0, 1, 0, 1, 0],
            }
        )

    @patch("matplotlib.pyplot.show")
    def test_plot_payment_history(self, mock_show):
        # plot_payment_history 関数のテスト
        plot_payment_history(
            user_data=self.test_data,
            datetime_col="use_dt",
            price_col="price",
            label_col="label",
        )
        # 描画関数が呼び出されたことを確認
        mock_show.assert_called_once()

        # さらに詳細なグラフの特性（色、ラベル、軸の範囲など）を検証するテストを追加します。
        # これには matplotlib のオブジェクトやプロパティを直接検証する方法を取り入れる必要があります。


if __name__ == "__main__":
    unittest.main()
