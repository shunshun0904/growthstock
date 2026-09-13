"""
会計フローグラフによる決算後リターン予測（accounting graph）。

このパッケージは既存の research/ とは独立した研究ラインで、
生データ（research/_data/*.parquet）だけを共有する。

  schema.py   会計フローグラフのノード・エッジ定義
  panel.py    決算開示を「その時点で見えていた値」に組み直す（as-of）
  labels.py   翌営業日始値を起点とした超過リターンと3クラスラベル
  build.py    グラフ系列データセットの構築
  splits.py   Purged / Embargo つき時系列分割
  baselines.py ロジスティック回帰・LightGBM・MLP
  backtest.py 取引コスト控除後の損益
  evaluate.py 評価の入口（CLI）
  leakage.py  リーク検査
"""
