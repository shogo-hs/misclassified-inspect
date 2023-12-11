# misinspect

misinspect は、誤分類された取引を分析し、可視化するためのライブラリです。
このライブラリは、Jupyter ノートブック内でインタラクティブなウィジェットを使用して、誤分類された取引（FPとFN）の分析結果を表示し、特定のユーザーの取引履歴をグラフとして可視化します。

## 特徴

- Jupyter ノートブックでのインタラクティブなデータ可視化。
- 誤分類された取引（FPとFN）の分析。
- 特定ユーザーの取引履歴のグラフ表示。

## インストール方法

GitHub リポジトリから直接インストールできます：
```
git clone https://github.com/shogo-hs/misclassified-inspect.git
```
```
cd misclassified-inspect
```
```
pip install .
```

## 使用方法

misinspect の基本的な使用方法は以下のとおりです：

```python
from misinspect.analysis.binary import MisClassifiedTxnAnalyzer
from misinspect.gui.jupyter import MisClassifiedTxnVisualizer
import pandas as pd

# データセットの読み込み
data = pd.read_csv('your_data.csv')

# 分析器の初期化
analyzer = MisClassifiedTxnAnalyzer(
    dataset=data,            #if you want to analyze spark dataframe you set spark dataframe
    user_id_col='user_id',
    price_col='price',
    datetime_col='use_dt',
    prob_col='probability',
    label_col='label',
    threshold=0.8,           #default0.8
    spark=None,              #if you use spark dataframe you set sparksession
)

# 可視化クラスの初期化
visualizer = MisClassifiedTxnVisualizer(analyzer)

# 可視化の表示
visualizer.show()
```
![image](https://github.com/shogo-hs/misclassified-inspect/assets/71430108/459de54a-370e-4e01-b4e5-f57455409c72)

