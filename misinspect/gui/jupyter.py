import pandas as pd

pd.set_option("display.max_rows", None)
import ipywidgets as widgets
from IPython.display import display

from misinspect.analysis.binary import MisClassifiedTxnAnalyzer
from misinspect.visualization.payment_history_plot import plot_payment_history

output = widgets.Output(layout={"border": "1px solid black"})


class MisClassifiedTxnVisualizer:
    """
    誤分類されたトランザクションの可視化を行うクラス。

    このクラスは、誤分類された取引（FPとFN）を分析し、ユーザー単位での取引履歴を可視化するための
    インターフェースを提供します。

    Attributes:
        analyzer (MisClassifiedTxnAnalyzer): 誤分類分析を行うAnalyzerクラスのインスタンス。
        target_user_ids (List[str]): 現在選択されているFPまたはFNのユーザーIDリスト。
        fp_user_ids (List[str]): FPのユーザーIDリスト。
        fn_user_ids (List[str]): FNのユーザーIDリスト。
        user_data (pd.DataFrame): 現在選択されているユーザーのデータ。
    """

    def __init__(self, analyzer: MisClassifiedTxnAnalyzer) -> None:
        """
        MisClassifiedTxnVisualizerのインスタンスを初期化します。

        Args:
            analyzer (MisClassifiedTxnAnalyzer): 誤分類分析を行うAnalyzerクラスのインスタンス。
        """
        self.analyzer = analyzer

        self.target_type = "FP"
        self.target_user_ids = []
        self.fp_user_ids = None
        self.fn_user_ids = None
        self.user_data = None

        # self.update_misclassified_data()

    def show(self) -> None:
        """ウィジェットの可視化を実行します。"""
        # ウィジェットの作成
        self.fpfn_select = self.create_fpfn_select()
        self.thredhold_dropdown = self.create_threshold_dropdown()
        self.user_dropdown = self.create_user_dropdown()
        display_user_data_button = widgets.Button(description="display user data")
        plot_history_button = widgets.Button(description="plot payment history")

        # ボタンを横並びに配置するためのHBoxを作成
        buttons_box = widgets.HBox([display_user_data_button, plot_history_button])

        # イベントハンドラの設定
        self.thredhold_dropdown.observe(self.on_thredhold_dropdown, "value")
        self.fpfn_select.observe(self.on_select_fpfn, "value")
        self.user_dropdown.observe(self.on_user_dropdown, "value")

        plot_history_button.on_click(self.on_click_plot_history_button_callback)
        display_user_data_button.on_click(self.on_click_display_user_data_callback)

        # ウィジェットの表示
        display(
            self.thredhold_dropdown,
            self.fpfn_select,
            self.user_dropdown,
            buttons_box,
            output,
        )

    # 関数が呼ばれる度に出力をクリアする
    @output.capture(clear_output=True)
    def on_click_callback(self, b: widgets.Button) -> None:
        """ボタンクリック時のコールバック関数。"""

        print("threshold: ", self.analyzer.threshold)
        print("target_user_ids: ", self.target_user_ids)

    def on_click_plot_history_button_callback(self, b: widgets.Button) -> None:
        """「plot payment history」ボタンクリック時のコールバック関数。"""
        with output:
            output.clear_output()
            if self.user_data is not None and not self.user_data.empty:
                plot_payment_history(
                    self.user_data,
                    self.analyzer.datetime_col,
                    self.analyzer.price_col,
                    self.analyzer.label_col,
                )
            else:
                print("データが見つかりません。")

    def on_click_display_user_data_callback(self, b: widgets.Button) -> None:
        """「display user data」ボタンクリック時のコールバック関数。"""

        def pdf_styler(df: pd.DataFrame):
            """
            データフレームにスタイリングを適用する関数。

            Args:
            df (pd.DataFrame): スタイリングを適用するデータフレーム

            Returns:
            スタイリングが適用されたデータフレーム
            """
            # データフレームのスタイルを設定
            styler = df.style

            # label=1 の行のスタイルを設定
            idx_pos = df[df[self.analyzer.label_col] == 1].index
            styler = styler.apply(
                lambda x: ["font-weight: bold" if x.name in idx_pos else "" for _ in x],
                axis=1,
            )

            # FP の行のスタイルを設定
            idx_fp = df[df["classification_type"] == "FP"].index
            styler = styler.apply(
                lambda x: [
                    "background-color: green" if x.name in idx_fp else "" for _ in x
                ],
                axis=1,
            )

            # FN の行のスタイルを設定
            idx_fn = df[df["classification_type"] == "FN"].index
            styler = styler.apply(
                lambda x: [
                    "background-color: #75A9FF" if x.name in idx_fn else "" for _ in x
                ],
                axis=1,
            )

            return styler

        # 出力ウィジェットにデータを表示
        with output:
            output.clear_output()
            if self.user_data is not None and not self.user_data.empty:
                display(pdf_styler(self.user_data))
                # display(self.user_data)
            else:
                print("データが見つかりません。")

    def create_fpfn_select(self) -> widgets.Select:
        """FPとFNの選択画面を設定します。"""
        return widgets.Select(
            options=["FP", "FN"],
            value=self.target_type,
            description="Select misclassification type: ",
            disabled=False,
        )

    def on_select_fpfn(self, change) -> None:
        """FPまたはFNの選択が変更された時のイベントハンドラ。"""
        # 選択された値（FPまたはFN）を取得
        self.target_type = change["new"]

        self.update_user_dropdown_options()

    def create_threshold_dropdown(self) -> widgets.Dropdown:
        """閾値のドロップダウンを設定します。"""

        return widgets.Dropdown(
            options=["{:.2f}".format(v / 100) for v in range(50, 100, 5)],
            value="{:.2f}".format(self.analyzer.threshold),
            description="Select threshold: ",
            disabled=False,
        )

    def on_thredhold_dropdown(self, change) -> None:
        """閾値ドロップダウンの選択が変更された時のイベントハンドラ。"""
        # 新しい閾値を取得
        new_threshold = float(change["new"])

        # analyzerインスタンスの閾値を更新
        self.analyzer.threshold = new_threshold

        self.analyzer.get_misclassified_data()

        # FPFNドロップダウンのオプションを更新
        self.update_fpfn_dropdown_options()

        # userドロップダウンのオプションを更新        
        self.update_user_dropdown_options()

    def create_user_dropdown(self) -> widgets.Dropdown:
        """対象ユーザーのドロップダウンを設定します。"""

        return widgets.Dropdown(
            options=self.target_user_ids,
            value=None,
            description="Select User Id: ",
            disabled=False,
        )

    def on_user_dropdown(self, change):
        """ユーザードロップダウンの選択が変更された時のイベントハンドラ。"""
        selected_user_id = change["new"]

        # ユーザーデータの取得
        self.user_data = self.analyzer.get_selected_user_data(
            selected_user_id, self.analyzer.dataset
        ).reset_index(drop=True)

    def update_fpfn_dropdown_options(self) -> None:
        """FPFNドロップダウンのオプションを更新します。"""

        # FPとFNのデータを更新
        fp_data = self.analyzer.get_misclassified_data_by_type("FP")
        fn_data = self.analyzer.get_misclassified_data_by_type("FN")
        # ユーザーIDリストを更新
        self.fp_user_ids = self.analyzer.get_unique_user_ids(fp_data)
        self.fn_user_ids = self.analyzer.get_unique_user_ids(fn_data)

    def update_user_dropdown_options(self) -> None:
        """userドロップダウンのオプションを更新します。"""

        # 対象ユーザーIDリストを更新
        self.target_user_ids = (
            self.fp_user_ids if self.target_type == "FP" else self.fn_user_ids
        )

        self.user_dropdown.options = self.target_user_ids
