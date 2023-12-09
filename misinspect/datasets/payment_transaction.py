import random
from datetime import datetime
from typing import List, Tuple

import pandas as pd


def generate_transaction_data(
    num_entries: int,
    user_id_range: Tuple[int, int],
    merchant_id_range: Tuple[int, int],
    date_range: Tuple[datetime, datetime],
    payment_methods: List[str],
    fraud_percentage: float = 0.01,
    seed: int = 123,
) -> pd.DataFrame:
    """
    決済トランザクションの疑似データを生成し、不正利用を含むデータセットを作成します。

    Args:
        num_entries (int): 生成するデータエントリの数。
        user_id_range (Tuple[int, int]): ユーザIDの範囲（最小値、最大値）。
        merchant_id_range (Tuple[int, int]): 加盟店IDの範囲（最小値、最大値）。
        date_range (Tuple[datetime, datetime]): 取引日時の範囲（開始日、終了日）。
        payment_methods (List[str]): 利用可能な支払い方法のリスト。
        fraud_percentage (float): 全取引における不正利用の割合。デフォルトは0.01（1%）。
        seed (int): 乱数のシード。デフォルトは123。

    Returns:
        pd.DataFrame: 生成された決済トランザクション、不正利用ラベル、確率を含むデータセット。
    """
    random.seed(seed)

    data = []
    num_frauds = int(num_entries * fraud_percentage)

    for i in range(num_entries):
        user_id = str(random.randint(*user_id_range))
        merchant_id = str(random.randint(*merchant_id_range))
        date_time = date_range[0] + (date_range[1] - date_range[0]) * random.random()
        amount = round(random.uniform(10, 500), 0)
        discount = round(amount * random.uniform(0, 0.3), 0)
        payment_method = random.choice(payment_methods)

        is_fraud = i < num_frauds
        if is_fraud:
            amount = round(random.uniform(300, 500), 0)
            probability = round(random.uniform(0.4, 1.0), 3)
        else:
            probability = round(random.uniform(0.0, 0.6), 3)

        label = int(is_fraud)
        data.append(
            [
                user_id,
                merchant_id,
                date_time,
                amount,
                discount,
                payment_method,
                label,
                probability,
            ]
        )

    random.shuffle(data)

    return pd.DataFrame(
        data,
        columns=[
            "user_id",
            "shop_id",
            "use_dt",
            "price",
            "discount",
            "pay_method",
            "label",
            "probability",
        ],
    )


# この関数を使用してデータを生成する
# df = generate_transaction_data(...)
