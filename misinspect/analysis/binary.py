from typing import List, Union

import matplotlib.pyplot as plt
import pandas as pd
from pandas.plotting import register_matplotlib_converters
from pyspark.sql import DataFrame as SparkDataFrame
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from misinspect.analysis.classification_utils import (
    calculate_classification_types,
    calculate_classification_types_spark,
    get_classification_data_by_type,
    get_classification_data_by_type_spark,
)
from misinspect.analysis.evaluation_metrics import (
    calculate_precision_recall,
    calculate_precision_recall_spark,
)

register_matplotlib_converters()


class MisClassifiedTxnAnalyzer:
    """
    誤分類されたトランザクションを分析するためのクラス。

    このクラスは、指定されたデータセットに対して誤分類分析を行い、
    混同行列や精度・再現率の計算、特定ユーザーの取引データの可視化などを提供します。

    Attributes:
        dataset (Union[pd.DataFrame, SparkDataFrame]): 分析対象のデータセット。
        user_id_col (str): ユーザーIDを表すカラムの名前。
        price_col (str): 取引額を表すカラムの名前。デフォルトは 'price'。
        datetime_col (str): 取引日時を表すカラムの名前。デフォルトは 'use_dt'。
        prob_col (str): 予測確率を表すカラムの名前。デフォルトは 'probability'。
        label_col (str): 実際のラベルを表すカラムの名前。デフォルトは 'label'。
        threshold (float): 予測ラベルの閾値。デフォルトは 0.5。
        spark (SparkSession): Sparkセッション。デフォルトは None。
    """

    def __init__(
        self,
        dataset: Union[pd.DataFrame, SparkDataFrame],
        user_id_col: str,
        price_col: str = "price",
        datetime_col: str = "use_dt",
        prob_col: str = "probability",
        label_col: str = "label",
        threshold: float = 0.5,
        spark: SparkSession = None,
    ) -> None:
        """
        インスタンスの初期化。

        Args:
            dataset (Union[pd.DataFrame, SparkDataFrame]): 分析対象のデータセット。
            user_id_col (str): ユーザーIDを表すカラムの名前。
            price_col (str): 取引額を表すカラムの名前。デフォルトは 'price'。
            datetime_col (str): 取引日時を表すカラムの名前。デフォルトは 'use_dt'。
            prob_col (str): 予測確率を表すカラムの名前。デフォルトは 'probability'。
            label_col (str): 実際のラベルを表すカラムの名前。デフォルトは 'label'。
            threshold (float): 予測ラベルの閾値。デフォルトは 0.5。
            spark (SparkSession): Sparkセッション。デフォルトは None。
        """

        # Sparkセッションが提供されている場合
        if spark is not None:
            if not isinstance(dataset, SparkDataFrame):
                raise ValueError(
                    "dataset must be a SparkDataFrame when a SparkSession is provided"
                )

        # Sparkセッションが提供されていない場合
        else:
            if not isinstance(dataset, pd.DataFrame):
                raise ValueError(
                    "dataset must be a pandas DataFrame when no SparkSession is provided"
                )

        # 必要なカラムがデータセットに存在するか確認
        required_columns = [user_id_col, price_col, datetime_col, prob_col, label_col]
        missing_columns = [
            col for col in required_columns if col not in dataset.columns
        ]
        if missing_columns:
            raise ValueError(
                f"Missing required columns in dataset: {', '.join(missing_columns)}"
            )

        # インスタンス変数の初期化
        self.dataset = dataset
        self.user_id_col = user_id_col
        self.price_col = price_col
        self.datetime_col = datetime_col
        self.prob_col = prob_col
        self.label_col = label_col
        self.threshold = threshold
        self.misclassified_data = None
        self.spark = spark
        self.columns = None
        self.user_data = None

    def get_misclassified_data(self) -> None:
        """
        誤分類されたデータを特定し、インスタンスのデータセットに追加します。
        """
        if self.spark is None:
            self.dataset = calculate_classification_types(
                self.dataset, self.prob_col, self.label_col, self.threshold
            )
            self.misclassified_data = self.dataset[
                self.dataset["classification_type"].isin(["FP", "FN"])
            ]
        else:
            self.dataset = calculate_classification_types_spark(
                self.dataset, self.prob_col, self.label_col, self.threshold
            )

            self.misclassified_data = self.dataset.filter(
                F.col("classification_type").isin(["FP", "FN"])
            )
        self.columns = list(self.dataset.columns)  # 更新

    def get_misclassified_data_by_type(
        self, type: str
    ) -> Union[pd.DataFrame, SparkDataFrame]:
        """
        指定された誤分類タイプ（FPやFNなど）に該当するデータを返します。

        Args:
            type (str): 誤分類タイプを指定する文字列。

        Returns:
            Union[pd.DataFrame, SparkDataFrame]: 指定された誤分類タイプに該当するデータ。
        """
        if self.spark is None:
            return get_classification_data_by_type(
                self.misclassified_data, "classification_type", type
            )

        else:
            return get_classification_data_by_type_spark(
                self.misclassified_data, "classification_type", "FP"
            )

    def get_selected_user_data(self, user_id: str, df: pd.DataFrame) -> pd.DataFrame:
        """
        指定されたユーザーIDに対応する取引データを返します。

        Args:
            user_id (str): ユーザーID。
            df (pd.DataFrame): 検索対象のデータフレーム。

        Returns:
            pd.DataFrame: 指定されたユーザーIDに対応するデータ。
        """
        if self.spark is None:
            return df[df[self.user_id_col] == user_id].sort_values(self.datetime_col)
        else:
            return (
                df.filter(F.col(self.user_id_col) == user_id)
                .orderBy(self.datetime_col)
                .toPandas()
            )

    def create_confusion_matrix(self) -> pd.DataFrame:
        """
        混同行列を作成し、それをデータフレームとして返します。

        Returns:
            pd.DataFrame: 混同行列を含むデータフレーム。
        """

        # 各値のカウント
        if self.spark is None:
            value_counts = self.dataset["classification_type"].value_counts()
        else:
            value_counts_df = self.dataset.groupBy("classification_type").count()

            # PySpark DataFrameをPandas DataFrameに変換
            value_counts_pandas = value_counts_df.toPandas()

            # Pandas DataFrameをシリーズに変換し、'classification_type' をインデックスに設定
            value_counts = value_counts_pandas.set_index("classification_type")["count"]

        # 新しいデータフレームの作成
        count_df = pd.DataFrame(value_counts)
        count_df.reset_index(inplace=True)
        count_df.columns = ["Item", "Value"]

        # カウント結果のデータフレームを指定の順序で並べ替え
        ordered_values = ["TP", "FP", "TN", "FN"]
        count_df_ordered = (
            count_df.set_index("Item").reindex(ordered_values).reset_index()
        )
        count_df_ordered.fillna(0, inplace=True)  # 0の値を持つ行を0で埋める

        # Recall と Precision の計算
        TP = count_df_ordered[count_df_ordered["Item"] == "TP"]["Value"].values[0]
        FP = count_df_ordered[count_df_ordered["Item"] == "FP"]["Value"].values[0]
        FN = count_df_ordered[count_df_ordered["Item"] == "FN"]["Value"].values[0]

        # 0で割ることを避けるためのチェック
        recall = TP / (TP + FN) if (TP + FN) != 0 else 0
        precision = TP / (TP + FP) if (TP + FP) != 0 else 0

        # Recall と Precision をデータフレームに追加
        additional_rows = pd.DataFrame(
            {
                "Item": ["Recall", "Precision"],
                "Value": ["{:.4f}".format(recall), "{:.4f}".format(precision)],
            }
        )

        return pd.concat([count_df_ordered, additional_rows], ignore_index=True)

    def get_confusion_matrix_lists(self) -> tuple:
        """
        混同行列のデータとヘッダーをリスト形式で返します。

        Returns:
            tuple: 混同行列のデータとヘッダーを含むタプル。
        """
        confusion_matrix_df = self.create_confusion_matrix()
        confusion_matrix_data = confusion_matrix_df.values.tolist()
        confusion_matrix_headers = confusion_matrix_df.columns.tolist()
        return confusion_matrix_data, confusion_matrix_headers

    def plot_pr_auc(self):
        """
        精度-再現率曲線（Precision-Recall Curve）をプロットします。

        Returns:
            matplotlib.figure.Figure: 生成されたグラフのFigureオブジェクト。
        """
        if self.spark is None:
            pr_df = calculate_precision_recall(
                self.dataset, self.label_col, self.prob_col
            )
        else:
            pr_df = calculate_precision_recall_spark(
                self.dataset, self.label_col, self.prob_col
            )

        # グラフサイズは変更せずに、ここでfigsizeをそのままにします。
        fig, ax = plt.subplots(figsize=(2, 2))  # このサイズは小さすぎる可能性がありますが、要求に応じてそのままにします。

        # グラフのデータをプロット
        ax.plot(pr_df["recall"], pr_df["precision"])

        # 軸ラベルのフォントサイズをさらに小さくします。
        ax.set_xlabel("Recall", fontsize=7)
        ax.set_ylabel("Precision", fontsize=7)

        # タイトルのフォントサイズを小さくし、余白を調整します。
        ax.set_title("Precision-Recall Curve", fontsize=8)
        plt.subplots_adjust(left=0.2, right=0.8, top=0.8, bottom=0.2)

        # ティックのフォントサイズを調整
        ax.tick_params(axis="both", which="major", labelsize=6)

        return fig

    def get_unique_user_ids(self, data) -> List[str]:
        """
        データセット内の一意のユーザーIDをリストとして返します。

        Args:
            data (Union[pd.DataFrame, SparkDataFrame]): ユーザーIDを抽出するデータセット。

        Returns:
            List[str]: 一意のユーザーIDのリスト。
        """
        if self.spark is None:
            return sorted([str(uid) for uid in data[self.user_id_col].unique()])
        else:
            unique_user_ids = data.select(self.user_id_col).distinct().collect()
            return sorted([str(row[self.user_id_col]) for row in unique_user_ids])
