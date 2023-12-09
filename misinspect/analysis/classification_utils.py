import pandas as pd
from pyspark.sql import DataFrame as SparkDataFrame
from pyspark.sql import functions as F


def calculate_classification_types(
    dataset: pd.DataFrame, prob_col: str, label_col: str, threshold: float
) -> pd.DataFrame:
    """
    与えられたデータセットについて、予測確率の閾値を基にした分類結果（TP、TN、FP、FN）を計算します。

    Args:
        dataset (pd.DataFrame): 分類結果を計算するためのデータセット。
        prob_col (str): 予測確率を含むカラムの名前。
        label_col (str): 実際のラベルを含むカラムの名前。
        threshold (float): 分類の閾値。

    Returns:
        pd.DataFrame: 分類結果（TP、TN、FP、FN）を含むデータセット。
    """
    dataset["predicted_label"] = (dataset[prob_col] >= threshold).astype(int)
    dataset.loc[
        (dataset[label_col] == 1) & (dataset["predicted_label"] == 1),
        "classification_type",
    ] = "TP"
    dataset.loc[
        (dataset[label_col] == 0) & (dataset["predicted_label"] == 0),
        "classification_type",
    ] = "TN"
    dataset.loc[
        (dataset[label_col] == 0) & (dataset["predicted_label"] == 1),
        "classification_type",
    ] = "FP"
    dataset.loc[
        (dataset[label_col] == 1) & (dataset["predicted_label"] == 0),
        "classification_type",
    ] = "FN"

    return dataset


def calculate_classification_types_spark(
    dataset: SparkDataFrame, prob_col: str, label_col: str, threshold: float
) -> SparkDataFrame:
    """
    Spark DataFrameを用いて、予測確率の閾値を基にした分類結果（TP、TN、FP、FN）を計算します。

    Args:
        dataset (SparkDataFrame): 分類結果を計算するためのデータセット。
        prob_col (str): 予測確率を含むカラムの名前。
        label_col (str): 実際のラベルを含むカラムの名前。
        threshold (float): 分類の閾値。

    Returns:
        SparkDataFrame: 分類結果（TP、TN、FP、FN）を含むデータセット。
    """
    # モデルの確率閾値に基づいて予測ラベルを計算
    dataset = dataset.withColumn(
        "predicted_label", (F.col(prob_col) >= threshold).cast("int")
    )

    # 分類のタイプを計算
    dataset = dataset.withColumn(
        "classification_type",
        F.when((F.col(label_col) == 1) & (F.col("predicted_label") == 1), "TP")
        .when((F.col(label_col) == 0) & (F.col("predicted_label") == 0), "TN")
        .when((F.col(label_col) == 0) & (F.col("predicted_label") == 1), "FP")
        .when((F.col(label_col) == 1) & (F.col("predicted_label") == 0), "FN"),
    )
    return dataset


def get_classification_data_by_type(
    df: pd.DataFrame, type_col: str, extract_type: str
) -> pd.DataFrame:
    """
    指定された分類タイプに該当するデータをデータセットから抽出します。

    Args:
        df (pd.DataFrame): 抽出対象のデータセット。
        type_col (str): 分類タイプを含むカラムの名前。
        extract_type (str): 抽出する分類タイプ。

    Returns:
        pd.DataFrame: 指定された分類タイプに該当するデータを含むデータセット。
    """

    return df[df[type_col] == extract_type].reset_index(drop=True)


def get_classification_data_by_type_spark(
    df: SparkDataFrame, type_col: str, extract_type: str
) -> SparkDataFrame:
    """
    Spark DataFrameを用いて、指定された分類タイプに該当するデータを抽出します。

    Args:
        df (SparkDataFrame): 抽出対象のデータセット。
        type_col (str): 分類タイプを含むカラムの名前。
        extract_type (str): 抽出する分類タイプ。

    Returns:
        SparkDataFrame: 指定された分類タイプに該当するデータを含むデータセット。
    """
    return df.filter(F.col(type_col) == extract_type)
