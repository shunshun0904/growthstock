# LightGBM固定+10%分類の理論株価特徴量比較

研究用。本番・既存ランキング出力は変更していません。
前回と同じ2評価窓、7群、各30候補×内側3窓、最終3種・200本。
固定と再探索を同じCode/Date/ラベル/共通行で比較しました。

- `predeclared-plan.json`：評価前の対象・予算・指標。
- `label-validation.json`：固定+10%のラベル契約・元ラベル一致・全体分布。
- `*-results.csv`：窓別・集合別・群別のAP/ROC-AUC/窓内上位10%適合率・20日収益。
- `*-summary.csv`：2窓の単純平均（全行をプールしたAPではありません）。
- `*-deltas.csv`、`*-ablation-differences.csv`：Aとの差、各対照との差。
- `samples.csv`：行数・銘柄数・日数・正例数・正例率。
- `retuned-cv-summary.csv`：全420候補の3内側スコア・平均・選択結果。
- `*-selection_frozen.json`、`*-plan.json`：外側採点前に確定した設定・列・ラベル・予算。
- `*-completed.json`：実行完了、窓、論理適合数、選択SHA256。
- `*-inputs.json`：入力・コード・ライブラリの指紋。
- `verification.json`：全予測出力からの指標再計算、同一行・ラベル・共通行、CV最良設定、本番23ファイル・前回ランキング62出力の不変。
- `retuned-importance-share.csv`、`retuned-feature_overlap.csv`：gain比率と情報重複の診断。因果効果ではありません。

`primary`はAverage Precision（PR-AUC）、`top_decile_precision`は0〜1の比率、
`top_decile_close_return_pct`は%単位です。適合率の上位10%は各窓全体のスコア順位で、日次順位ではありません。
クラス重み付きスコアは校正済み確率とみなしません。
`*-preflight.json`の`backtest_completed=false`は学習前の記録で、完了は`*-completed.json`で確認します。

[仕様・結果](../../../MODEL_VALUATION_GAP_CLASS10_EXPERIMENT.md)を参照してください。
公開Gitに生の財務原本・個別予測スコアは追加していません。
