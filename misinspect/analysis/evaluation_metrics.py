import pandas as pd
from pyspark.sql import DataFrame as SparkDataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window


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

    # 予測確率を指定された小数点以下の桁数で丸める
    df["probability_round"] = df[probability_col].round(decimal_places)

    # 正の事例（陽性）の合計数を計算
    P = df[label_col].sum()

    # 予測確率に基づいてデータを降順に並び替え
    df_sorted = df.sort_values(by="probability_round", ascending=False).reset_index(drop=True)

    # 累積的な真の陽性（TP）と偽の陽性（FP）の数を計算
    df_sorted["TP_cumulative"] = df_sorted[label_col].cumsum()  # 真の陽性の累積数
    df_sorted["FP_cumulative"] = (df_sorted.index + 1) - df_sorted["TP_cumulative"]  # 偽の陽性の累積数

    # 各データポイントにおける精度と再現率を計算
    df_sorted["precision"] = df_sorted["TP_cumulative"] / (df_sorted.index + 1)  # 精度 = 真の陽性の累積数 / (インデックス + 1)
    df_sorted["recall"] = df_sorted["TP_cumulative"] / P  # 再現率 = 真の陽性の累積数 / 正の事例の合計数

    # 重複する行を削除して一意のデータフレームを返す
    df_sorted = df_sorted[["recall", "precision"]].drop_duplicates()
    return df_sorted.sort_values(by="recall", ascending=True).reset_index(drop=True)



# PySpark DataFrame用の精度と再現率の計算関数
def calculate_precision_recall_spark(
    df, label_col: str, probability_col: str, decimal_places: int = 3
) -> pd.DataFrame:
    """
    PySpark DataFrameを用いて精度（Precision）と再現率（Recall）を計算します。

    Args:
        df: 真のラベルと予測確率を含むPySpark DataFrame。
        label_col (str): 真のラベルを含むカラムの名前。
        probability_col (str): 予測確率を含むカラムの名前。
        decimal_places (int): 予測確率を丸める小数点以下の桁数。デフォルトは3。

    Returns:
        pd.DataFrame: 再現率と精度を含むPandas DataFrame。
    """
    # 予測確率を指定された小数点以下の桁数で丸める
    df = df.withColumn("probability_round", F.round(probability_col, decimal_places))

    # 真の陽性の合計数を計算
    P = df.select(F.sum(label_col)).collect()[0][0]

    # 予測確率でデータを降順にソートするためのWindowSpecificationを定義
    windowSpec = Window.orderBy(F.col("probability_round").desc())

    # 累積的な真の陽性（TP）と偽の陽性（FP）の数を計算
    df = df.withColumn("TP_cumulative", F.sum(label_col).over(windowSpec))
    df = df.withColumn("FP", (F.lit(1) - F.col(label_col)))
    df = df.withColumn("FP_cumulative", F.sum("FP").over(windowSpec))

    # 精度と再現率を計算
    df = df.withColumn(
        "precision",
        F.col("TP_cumulative") / (F.col("TP_cumulative") + F.col("FP_cumulative"))
    )
    df = df.withColumn("recall", F.col("TP_cumulative") / F.lit(P))

    # 重複する行を削除して一意のデータフレームを作成
    df_unique = df.select("recall", "precision").dropDuplicates()

    # Pandas DataFrameに変換して返す
    pdf = df_unique.toPandas()
    return pdf.sort_values(by="recall", ascending=True).reset_index(drop=True)
