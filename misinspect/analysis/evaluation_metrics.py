import pandas as pd
from sklearn.metrics import precision_recall_curve
from pyspark.sql import DataFrame as SparkDataFrame
from pyspark.sql import functions as F
from pyspark.mllib.evaluation import BinaryClassificationMetrics

# Pandas DataFrame用の精度と再現率の計算関数
def calculate_precision_recall(
    df: pd.DataFrame, label_col: str, probability_col: str, decimal_places: int = 3
) -> pd.DataFrame:
    """
    Pandas DataFrameを用いて精度（Precision）と再現率（Recall）を計算します。

    Args:
        df (pd.DataFrame): 真のラベルと予測確率を含むデータフレーム。
        label_col (str): 真のラベルを含むカラムの名前。
        probability_col (str): 予測確率を含むカラムの名前。
        decimal_places (int): 予測確率を丸める小数点以下の桁数。デフォルトは3。

    Returns:
        pd.DataFrame: 再現率と精度を含むデータフレーム。
    """
    # 真のラベルと予測確率を取得
    y_true = df[label_col]
    y_scores = df[probability_col].round(decimal_places)

    # scikit-learnのprecision_recall_curve関数を使用して精度と再現率を計算
    precision, recall, _ = precision_recall_curve(y_true, y_scores)

    # 結果をデータフレームに変換
    pr_df = pd.DataFrame({
        'precision': precision,
        'recall': recall
    })

    return pr_df

# PySpark DataFrame用の精度と再現率の計算関数
def calculate_precision_recall_spark(df: SparkDataFrame, label_col: str, probability_col: str) -> pd.DataFrame:
    """
    PySpark DataFrameを用いて精度（Precision）と再現率（Recall）を計算します。

    Args:
        df (SparkDataFrame): 真のラベルと予測確率を含むPySpark DataFrame。
        label_col (str): 真のラベルを含むカラムの名前。
        probability_col (str): 予測確率を含むカラムの名前。

    Returns:
        pd.DataFrame: 精度、再現率、閾値を含むDataFrame。
    """
    # 真のラベルと予測確率を取得
    score_and_labels = df.select(col(probability_col), col(label_col)).rdd

    # BinaryClassificationMetricsを使用して精度と再現率を計算
    metrics = BinaryClassificationMetrics(score_and_labels)

    # 精度、再現率、閾値を含むリストを作成
    pr_thresholds = [
        (threshold, precision, recall)
        for threshold, precision in metrics.precisionByThreshold().collect()
        for _, recall in metrics.recallByThreshold().filter(lambda x: x[0] == threshold).collect()
    ]

    # Pandas DataFrameに変換
    pr_df = pd.DataFrame(pr_thresholds, columns=['threshold', 'precision', 'recall'])

    return pr_df
