import unittest
from datetime import datetime

from misinspect.datasets.payment_transaction import generate_transaction_data


class TestGenerateTransactionData(unittest.TestCase):
    def test_generate_transaction_data(self):
        # データ生成関数のテスト
        num_entries = 100
        user_id_range = (1, 1000)
        merchant_id_range = (1, 500)
        date_range = (datetime(2021, 1, 1), datetime(2021, 12, 31))
        payment_methods = ["cash", "card", "online"]
        fraud_percentage = 0.05
        seed = 123

        df = generate_transaction_data(
            num_entries,
            user_id_range,
            merchant_id_range,
            date_range,
            payment_methods,
            fraud_percentage,
            seed,
        )

        # 検証: 生成されたデータの行数が期待値と一致するか
        self.assertEqual(len(df), num_entries)

        # 検証: 不正取引の割合が期待値に近いか
        frauds = df[df["label"] == 1]
        actual_fraud_percentage = len(frauds) / num_entries
        self.assertAlmostEqual(actual_fraud_percentage, fraud_percentage, delta=0.02)

        # その他、必要に応じてデータの範囲、フォーマット、型などを検証するテストを追加します。


if __name__ == "__main__":
    unittest.main()
