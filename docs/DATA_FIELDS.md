# J-Quants /fins/summary の項目一覧（実測）

`research/probe_fins_fields.py` の出力。**API を実際に叩いた結果のみ**を記載する。

モデリングの前にデータ探索をする、という方針に沿って、
「何が取れるか」を推測せずに実測した。

- 対象: 18日分の開示（決算集中期を選定）
- 取得行数: 10,799
- 開示種別の内訳: 1Q: 2,918 / 2Q: 1,952 / 3Q: 1,919 / 4Q: 2 / FY: 4,008

## 探している指標が取れるか

| 指標 | 該当する項目（実測） |
| --- | --- |
| EPS(1株利益) | `DEPS`, `EPS`, `FEPS`, `FEPS2Q`, `FNCEPS`, `FNCEPS2Q`, `NCEPS`, `NxFEPS`, `NxFEPS2Q`, `NxFNCEPS`, `NxFNCEPS2Q` |
| BPS(1株純資産) | `BPS`, `NCBPS` |
| PER | **該当なし** |
| PBR | **該当なし** |
| ROE | `NCROE`, `ROE` |
| ROA | **該当なし** |
| 総資産 | `NCTA`, `TA` |
| 自己資本 | `Eq`, `EqAR`, `NCEq`, `NCEqAR`, `NCShEq`, `ShEq` |
| 株数 | `ShOutFY` |

## 現在 FIN_COLS で捨てている項目

`research/jq_bulk.py` の `FIN_COLS` はホワイトリストで、
ここに無い項目は取得時点で捨てている。

現在 `FIN_COLS = None`（絞らず全項目を保持）なので、捨てている項目は無い。

## 全項目の充足率

| 項目 | 全体 | 開示種別ごと |
| --- | ---: | --- |
| `Code` | 100.0% | 1Q: 100.0% / 2Q: 100.0% / 3Q: 100.0% / 4Q: 100.0% / FY: 100.0% |
| `CurFYEn` | 100.0% | 1Q: 100.0% / 2Q: 100.0% / 3Q: 100.0% / 4Q: 100.0% / FY: 100.0% |
| `CurFYSt` | 100.0% | 1Q: 100.0% / 2Q: 100.0% / 3Q: 100.0% / 4Q: 100.0% / FY: 100.0% |
| `CurPerEn` | 100.0% | 1Q: 100.0% / 2Q: 100.0% / 3Q: 100.0% / 4Q: 100.0% / FY: 100.0% |
| `CurPerSt` | 100.0% | 1Q: 100.0% / 2Q: 100.0% / 3Q: 100.0% / 4Q: 100.0% / FY: 100.0% |
| `CurPerType` | 100.0% | 1Q: 100.0% / 2Q: 100.0% / 3Q: 100.0% / 4Q: 100.0% / FY: 100.0% |
| `DiscDate` | 100.0% | 1Q: 100.0% / 2Q: 100.0% / 3Q: 100.0% / 4Q: 100.0% / FY: 100.0% |
| `DiscNo` | 100.0% | 1Q: 100.0% / 2Q: 100.0% / 3Q: 100.0% / 4Q: 100.0% / FY: 100.0% |
| `DiscTime` | 100.0% | 1Q: 100.0% / 2Q: 100.0% / 3Q: 100.0% / 4Q: 100.0% / FY: 100.0% |
| `DocType` | 100.0% | 1Q: 100.0% / 2Q: 100.0% / 3Q: 100.0% / 4Q: 100.0% / FY: 100.0% |
| `ChgByASRev` | 94.9% | 1Q: 99.7% / 2Q: 95.1% / 3Q: 99.9% / 4Q: 0.0% / FY: 88.8% |
| `ChgNoASRev` | 94.9% | 1Q: 99.7% / 2Q: 95.1% / 3Q: 99.9% / 4Q: 0.0% / FY: 88.8% |
| `Eq` | 94.9% | 1Q: 99.7% / 2Q: 95.1% / 3Q: 99.9% / 4Q: 100.0% / FY: 88.8% |
| `ShOutFY` | 94.9% | 1Q: 99.7% / 2Q: 95.1% / 3Q: 99.9% / 4Q: 0.0% / FY: 88.8% |
| `TA` | 94.9% | 1Q: 99.7% / 2Q: 95.1% / 3Q: 99.9% / 4Q: 100.0% / FY: 88.8% |
| `AvgSh` | 94.8% | 1Q: 99.7% / 2Q: 95.1% / 3Q: 99.9% / 4Q: 0.0% / FY: 88.7% |
| `EPS` | 94.8% | 1Q: 99.7% / 2Q: 95.1% / 3Q: 99.9% / 4Q: 0.0% / FY: 88.7% |
| `EqAR` | 94.8% | 1Q: 99.7% / 2Q: 95.1% / 3Q: 99.9% / 4Q: 0.0% / FY: 88.8% |
| `NP` | 94.8% | 1Q: 99.7% / 2Q: 95.1% / 3Q: 99.9% / 4Q: 100.0% / FY: 88.7% |
| `Sales` | 94.7% | 1Q: 99.6% / 2Q: 95.1% / 3Q: 99.8% / 4Q: 100.0% / FY: 88.4% |
| `ShEq` | 94.7% | 1Q: 99.7% / 2Q: 95.1% / 3Q: 99.9% / 4Q: 0.0% / FY: 88.5% |
| `ChgAcEst` | 94.2% | 1Q: 99.6% / 2Q: 91.9% / 3Q: 99.9% / 4Q: 0.0% / FY: 88.7% |
| `OP` | 92.3% | 1Q: 97.4% / 2Q: 91.9% / 3Q: 98.0% / 4Q: 100.0% / FY: 86.1% |
| `OdP` | 88.9% | 1Q: 93.1% / 2Q: 89.7% / 3Q: 93.4% / 4Q: 100.0% / FY: 83.2% |
| `TrShFY` | 88.6% | 1Q: 94.2% / 2Q: 90.1% / 3Q: 91.3% / 4Q: 0.0% / FY: 82.6% |
| `RetroRst` | 88.3% | 1Q: 93.1% / 2Q: 86.5% / 3Q: 93.4% / 4Q: 0.0% / FY: 83.2% |
| `Div2Q` | 63.6% | 1Q: 0.0% / 2Q: 89.0% / 3Q: 94.2% / 4Q: 0.0% / FY: 82.8% |
| `FNP` | 60.3% | 1Q: 93.4% / 2Q: 89.2% / 3Q: 91.8% / 4Q: 0.0% / FY: 7.0% |
| `FSales` | 60.2% | 1Q: 92.9% / 2Q: 88.8% / 3Q: 92.3% / 4Q: 0.0% / FY: 7.1% |
| `FEPS` | 59.8% | 1Q: 92.8% / 2Q: 88.6% / 3Q: 90.8% / 4Q: 0.0% / FY: 6.9% |
| `FOP` | 59.0% | 1Q: 91.3% / 2Q: 86.8% / 3Q: 90.8% / 4Q: 0.0% / FY: 6.8% |
| `FDivFY` | 58.3% | 1Q: 91.3% / 2Q: 87.3% / 3Q: 92.4% / 4Q: 0.0% / FY: 3.9% |
| `MatChgSub` | 58.1% | 1Q: 52.4% / 2Q: 58.0% / 3Q: 60.4% / 4Q: 0.0% / FY: 61.1% |
| `FDivAnn` | 57.8% | 1Q: 90.7% / 2Q: 86.5% / 3Q: 91.5% / 4Q: 0.0% / FY: 3.9% |
| `FOdP` | 56.9% | 1Q: 88.0% / 2Q: 84.7% / 3Q: 86.6% / 4Q: 0.0% / FY: 6.6% |
| `CashEq` | 51.3% | 1Q: 11.1% / 2Q: 76.2% / 3Q: 8.9% / 4Q: 100.0% / FY: 88.8% |
| `CFI` | 50.8% | 1Q: 9.9% / 2Q: 75.9% / 3Q: 8.2% / 4Q: 100.0% / FY: 88.7% |
| `CFO` | 50.8% | 1Q: 9.9% / 2Q: 76.0% / 3Q: 8.2% / 4Q: 100.0% / FY: 88.8% |
| `CFF` | 50.6% | 1Q: 9.9% / 2Q: 75.2% / 3Q: 8.2% / 4Q: 100.0% / FY: 88.4% |
| `BPS` | 46.7% | 1Q: 23.9% / 2Q: 20.3% / 3Q: 20.5% / 4Q: 0.0% / FY: 88.7% |
| `NxtFYEn` | 33.0% | 1Q: 0.0% / 2Q: 0.0% / 3Q: 0.0% / 4Q: 0.0% / FY: 88.9% |
| `NxtFYSt` | 33.0% | 1Q: 0.0% / 2Q: 0.0% / 3Q: 0.0% / 4Q: 0.0% / FY: 88.9% |
| `DivFY` | 32.7% | 1Q: 0.0% / 2Q: 0.0% / 3Q: 0.0% / 4Q: 0.0% / FY: 88.2% |
| `DivAnn` | 32.4% | 1Q: 0.0% / 2Q: 0.0% / 3Q: 0.0% / 4Q: 0.0% / FY: 87.4% |
| `ROE` | 32.4% | 1Q: 0.0% / 2Q: 0.0% / 3Q: 0.0% / 4Q: 0.0% / FY: 87.4% |
| `DEPS` | 30.8% | 1Q: 30.3% / 2Q: 31.9% / 3Q: 35.7% / 4Q: 0.0% / FY: 28.2% |
| `NxFNp` | 29.6% | 1Q: 0.0% / 2Q: 0.0% / 3Q: 0.0% / 4Q: 0.0% / FY: 79.8% |
| `NxFEPS` | 29.4% | 1Q: 0.0% / 2Q: 0.0% / 3Q: 0.0% / 4Q: 0.0% / FY: 79.3% |
| `NxFDivFY` | 29.3% | 1Q: 0.0% / 2Q: 0.0% / 3Q: 0.0% / 4Q: 0.0% / FY: 78.8% |
| `NxFSales` | 29.3% | 1Q: 0.0% / 2Q: 0.0% / 3Q: 0.0% / 4Q: 0.0% / FY: 79.0% |
| `NxFDivAnn` | 29.2% | 1Q: 0.0% / 2Q: 0.0% / 3Q: 0.0% / 4Q: 0.0% / FY: 78.7% |
| `NxFOP` | 28.8% | 1Q: 0.0% / 2Q: 0.0% / 3Q: 0.0% / 4Q: 0.0% / FY: 77.7% |
| `NxFDiv2Q` | 28.1% | 1Q: 0.0% / 2Q: 0.0% / 3Q: 0.0% / 4Q: 0.0% / FY: 75.8% |
| `NxFOdP` | 27.9% | 1Q: 0.0% / 2Q: 0.0% / 3Q: 0.0% / 4Q: 0.0% / FY: 75.1% |
| `DivTotalAnn` | 25.0% | 1Q: 0.0% / 2Q: 0.0% / 3Q: 0.0% / 4Q: 0.0% / FY: 67.3% |
| `SigChgInC` | 24.8% | 1Q: 35.6% / 2Q: 23.8% / 3Q: 26.1% / 4Q: 0.0% / FY: 16.7% |
| `NCEPS` | 23.7% | 1Q: 0.0% / 2Q: 2.6% / 3Q: 0.0% / 4Q: 0.0% / FY: 62.6% |
| `NCEq` | 23.7% | 1Q: 0.0% / 2Q: 2.6% / 3Q: 0.0% / 4Q: 0.0% / FY: 62.5% |
| `NCEqAR` | 23.7% | 1Q: 0.0% / 2Q: 2.6% / 3Q: 0.0% / 4Q: 0.0% / FY: 62.5% |
| `NCNP` | 23.7% | 1Q: 0.0% / 2Q: 2.6% / 3Q: 0.0% / 4Q: 0.0% / FY: 62.6% |
| `NCOdP` | 23.7% | 1Q: 0.0% / 2Q: 2.6% / 3Q: 0.0% / 4Q: 0.0% / FY: 62.6% |
| `NCSales` | 23.7% | 1Q: 0.0% / 2Q: 2.6% / 3Q: 0.0% / 4Q: 0.0% / FY: 62.5% |
| `NCShEq` | 23.7% | 1Q: 0.0% / 2Q: 2.6% / 3Q: 0.0% / 4Q: 0.0% / FY: 62.5% |
| `NCTA` | 23.7% | 1Q: 0.0% / 2Q: 2.6% / 3Q: 0.0% / 4Q: 0.0% / FY: 62.5% |
| `FDiv2Q` | 23.6% | 1Q: 85.4% / 2Q: 0.1% / 3Q: 0.0% / 4Q: 0.0% / FY: 1.4% |
| `NCBPS` | 23.2% | 1Q: 0.0% / 2Q: 0.2% / 3Q: 0.0% / 4Q: 0.0% / FY: 62.5% |
| `NCOP` | 22.8% | 1Q: 0.0% / 2Q: 0.6% / 3Q: 0.0% / 4Q: 0.0% / FY: 61.2% |
| `PayoutRatioAnn` | 22.7% | 1Q: 0.0% / 2Q: 0.0% / 3Q: 0.0% / 4Q: 0.0% / FY: 61.3% |
| `NxFPayoutRatioAnn` | 20.8% | 1Q: 0.0% / 2Q: 0.0% / 3Q: 0.0% / 4Q: 0.0% / FY: 56.1% |
| `NxFEPS2Q` | 14.5% | 1Q: 0.0% / 2Q: 0.0% / 3Q: 0.0% / 4Q: 0.0% / FY: 39.1% |
| `NxFNp2Q` | 14.5% | 1Q: 0.0% / 2Q: 0.0% / 3Q: 0.0% / 4Q: 0.0% / FY: 39.1% |
| `NxFSales2Q` | 14.2% | 1Q: 0.0% / 2Q: 0.0% / 3Q: 0.0% / 4Q: 0.0% / FY: 38.3% |
| `NxFOP2Q` | 14.0% | 1Q: 0.0% / 2Q: 0.0% / 3Q: 0.0% / 4Q: 0.0% / FY: 37.8% |
| `NxFOdP2Q` | 14.0% | 1Q: 0.0% / 2Q: 0.0% / 3Q: 0.0% / 4Q: 0.0% / FY: 37.6% |
| `FNP2Q` | 13.3% | 1Q: 45.9% / 2Q: 3.9% / 3Q: 0.0% / 4Q: 0.0% / FY: 0.4% |
| `FEPS2Q` | 13.2% | 1Q: 45.8% / 2Q: 3.9% / 3Q: 0.0% / 4Q: 0.0% / FY: 0.4% |
| `FSales2Q` | 13.0% | 1Q: 45.1% / 2Q: 3.9% / 3Q: 0.0% / 4Q: 0.0% / FY: 0.4% |
| `FOP2Q` | 12.8% | 1Q: 44.2% / 2Q: 3.8% / 3Q: 0.0% / 4Q: 0.0% / FY: 0.4% |
| `FOdP2Q` | 12.7% | 1Q: 43.9% / 2Q: 3.8% / 3Q: 0.0% / 4Q: 0.0% / FY: 0.4% |
| `NxFNCNP` | 4.2% | 1Q: 0.0% / 2Q: 0.0% / 3Q: 0.0% / 4Q: 0.0% / FY: 11.2% |
| `NxFNCOdP` | 4.2% | 1Q: 0.0% / 2Q: 0.0% / 3Q: 0.0% / 4Q: 0.0% / FY: 11.2% |
| `NxFNCEPS` | 4.1% | 1Q: 0.0% / 2Q: 0.0% / 3Q: 0.0% / 4Q: 0.0% / FY: 11.2% |
| `NxFNCSales` | 4.0% | 1Q: 0.0% / 2Q: 0.0% / 3Q: 0.0% / 4Q: 0.0% / FY: 10.8% |
| `NxFNCEPS2Q` | 3.0% | 1Q: 0.0% / 2Q: 0.0% / 3Q: 0.0% / 4Q: 0.0% / FY: 8.2% |
| `NxFNCNP2Q` | 3.0% | 1Q: 0.0% / 2Q: 0.0% / 3Q: 0.0% / 4Q: 0.0% / FY: 8.2% |
| `NxFNCOdP2Q` | 3.0% | 1Q: 0.0% / 2Q: 0.0% / 3Q: 0.0% / 4Q: 0.0% / FY: 8.2% |
| `NxFNCSales2Q` | 2.9% | 1Q: 0.0% / 2Q: 0.0% / 3Q: 0.0% / 4Q: 0.0% / FY: 7.8% |
| `FNCEPS` | 1.9% | 1Q: 0.0% / 2Q: 2.7% / 3Q: 0.0% / 4Q: 0.0% / FY: 3.9% |
| `FNCNP` | 1.9% | 1Q: 0.0% / 2Q: 2.7% / 3Q: 0.0% / 4Q: 0.0% / FY: 3.9% |
| `FNCOdP` | 1.9% | 1Q: 0.0% / 2Q: 2.7% / 3Q: 0.0% / 4Q: 0.0% / FY: 3.9% |
| `FNCSales` | 1.8% | 1Q: 0.0% / 2Q: 1.8% / 3Q: 0.0% / 4Q: 0.0% / FY: 3.9% |
| `NxFNCOP` | 1.4% | 1Q: 0.0% / 2Q: 0.0% / 3Q: 0.0% / 4Q: 0.0% / FY: 3.7% |
| `Div1Q` | 1.0% | 1Q: 0.7% / 2Q: 1.4% / 3Q: 1.4% / 4Q: 0.0% / FY: 0.7% |
| `FNCOP` | 1.0% | 1Q: 0.0% / 2Q: 0.6% / 3Q: 0.0% / 4Q: 0.0% / FY: 2.4% |
| `NxFNCOP2Q` | 0.8% | 1Q: 0.0% / 2Q: 0.0% / 3Q: 0.0% / 4Q: 0.0% / FY: 2.2% |
| `Div3Q` | 0.6% | 1Q: 0.0% / 2Q: 0.0% / 3Q: 1.6% / 4Q: 0.0% / FY: 0.8% |
| `FDiv3Q` | 0.4% | 1Q: 0.3% / 2Q: 1.1% / 3Q: 0.0% / 4Q: 0.0% / FY: 0.2% |
| `FNCEPS2Q` | 0.4% | 1Q: 0.0% / 2Q: 1.9% / 3Q: 0.0% / 4Q: 0.0% / FY: 0.2% |
| `FNCNP2Q` | 0.4% | 1Q: 0.0% / 2Q: 1.9% / 3Q: 0.0% / 4Q: 0.0% / FY: 0.2% |
| `FNCOdP2Q` | 0.4% | 1Q: 0.0% / 2Q: 1.9% / 3Q: 0.0% / 4Q: 0.0% / FY: 0.2% |
| `FNCSales2Q` | 0.4% | 1Q: 0.0% / 2Q: 1.9% / 3Q: 0.0% / 4Q: 0.0% / FY: 0.2% |
| `FNCOP2Q` | 0.2% | 1Q: 0.0% / 2Q: 1.0% / 3Q: 0.0% / 4Q: 0.0% / FY: 0.1% |
| `NxFDiv1Q` | 0.2% | 1Q: 0.0% / 2Q: 0.0% / 3Q: 0.0% / 4Q: 0.0% / FY: 0.5% |
| `NxFDiv3Q` | 0.2% | 1Q: 0.0% / 2Q: 0.0% / 3Q: 0.0% / 4Q: 0.0% / FY: 0.5% |
| `FDiv1Q` | 0.1% | 1Q: 0.0% / 2Q: 0.0% / 3Q: 0.0% / 4Q: 0.0% / FY: 0.2% |
| `DivUnit` | 0.0% | 1Q: 0.0% / 2Q: 0.0% / 3Q: 0.0% / 4Q: 0.0% / FY: 0.1% |
| `FDivTotalAnn` | 0.0% | 1Q: 0.0% / 2Q: 0.0% / 3Q: 0.0% / 4Q: 0.0% / FY: 0.0% |
| `FDivUnit` | 0.0% | 1Q: 0.0% / 2Q: 0.0% / 3Q: 0.0% / 4Q: 0.0% / FY: 0.0% |
| `FPayoutRatioAnn` | 0.0% | 1Q: 0.0% / 2Q: 0.0% / 3Q: 0.0% / 4Q: 0.0% / FY: 0.0% |
| `NCROE` | 0.0% | 1Q: 0.0% / 2Q: 0.0% / 3Q: 0.0% / 4Q: 0.0% / FY: 0.0% |
| `NxFDivUnit` | 0.0% | 1Q: 0.0% / 2Q: 0.0% / 3Q: 0.0% / 4Q: 0.0% / FY: 0.1% |

## エンドポイントの疎通（実測）

叩いて確かめた結果。存在しない・権限が無いものは NG になる。

| エンドポイント | 結果 | 件数 | 備考 |
| --- | :---: | ---: | --- |
| `/fins/summary` | OK | 612 | 111項目 |
| `/fins/details` | NG | — | HTTP 403 https://api.jquants.com/v2/fins/details?date=2024-05-15 : {"message": "This API is not available on your subscr |
| `/fins/statements` | NG | — | HTTP 403 https://api.jquants.com/v2/fins/statements?date=2024-05-15 : {"message": "The requested endpoint does not exist |
| `/fins/dividend` | NG | — | HTTP 403 https://api.jquants.com/v2/fins/dividend?date=2024-05-15 : {"message": "This API is not available on your subsc |
| `/fins/fs_details` | NG | — | HTTP 403 https://api.jquants.com/v2/fins/fs_details?date=2024-05-15 : {"message": "The requested endpoint does not exist |
| `/equities/master` | OK | 4450 | 14項目 |
| `/equities/bars/daily` | OK | 4359 | 18項目 |
| `/markets/margin-interest` | OK | 0 | 0項目 |
| `/markets/short-selling` | NG | — | HTTP 403 https://api.jquants.com/v2/markets/short-selling?date=2024-05-15 : {"message": "The requested endpoint does not |
| `/markets/breakdown` | NG | — | HTTP 403 https://api.jquants.com/v2/markets/breakdown?date=2024-05-15 : {"message": "This API is not available on your s |
| `/markets/trades-spec` | NG | — | HTTP 403 https://api.jquants.com/v2/markets/trades-spec : {"message": "The requested endpoint does not exist. Please che |
| `/indices/topix` | NG | — | HTTP 403 https://api.jquants.com/v2/indices/topix?from=2024-05-01&to=2024-05-15 : {"message": "The requested endpoint do |
| `/indices/prices` | NG | — | HTTP 403 https://api.jquants.com/v2/indices/prices?date=2024-05-15 : {"message": "The requested endpoint does not exist. |
| `/fins/announcement` | NG | — | HTTP 403 https://api.jquants.com/v2/fins/announcement : {"message": "The requested endpoint does not exist. Please check |
| `/fins/announcements` | NG | — | HTTP 403 https://api.jquants.com/v2/fins/announcements : {"message": "The requested endpoint does not exist. Please chec |
| `/fins/disclosure` | NG | — | HTTP 403 https://api.jquants.com/v2/fins/disclosure?date=2024-05-15 : {"message": "The requested endpoint does not exist |
| `/disclosure/timely` | NG | — | HTTP 403 https://api.jquants.com/v2/disclosure/timely?date=2024-05-15 : {"message": "The requested endpoint does not exi |
| `/fins/forecast` | NG | — | HTTP 403 https://api.jquants.com/v2/fins/forecast?date=2024-05-15 : {"message": "The requested endpoint does not exist.  |
| `/fins/consensus` | NG | — | HTTP 403 https://api.jquants.com/v2/fins/consensus?date=2024-05-15 : {"message": "The requested endpoint does not exist. |
| `/equities/shareholders` | NG | — | HTTP 403 https://api.jquants.com/v2/equities/shareholders?code=72030 : {"message": "The requested endpoint does not exis |
| `/equities/ownership` | NG | — | HTTP 403 https://api.jquants.com/v2/equities/ownership?code=72030 : {"message": "The requested endpoint does not exist.  |
| `/markets/ownership` | NG | — | HTTP 403 https://api.jquants.com/v2/markets/ownership?date=2024-05-15 : {"message": "The requested endpoint does not exi |
| `/` | NG | — | HTTP 403 https://api.jquants.com/v2/ : {"message": "The requested endpoint does not exist. Please check the URL, HTTP me |


### 2つの NG は意味が違う（2026-09-22）

| メッセージ | 意味 |
|---|---|
| `This API is not available on your subscription plan` | **エンドポイントは在る。** 契約が足りないだけ。プレミアムで開く |
| `The requested endpoint does not exist` | そのパスには何も無い。プランを上げても開かない |

前者は `/fins/details`・`/fins/dividend`・`/markets/breakdown` の3本。
つまり **BS/PL/CF明細・配当明細・売買内訳はプレミアムで取れる**。

### 適時開示・アナリスト予想・株主構成は取れるか（2026-09-22）

運用者の問いに答えるために候補10本を叩いた。**全部 NG、しかも全部が
「does not exist」側**で、「プランを上げれば開く」側は1本も無かった。

| 分野 | 叩いたパス | 結果 |
|---|---|---|
| 適時開示 | `/fins/announcement` `/fins/announcements` `/fins/disclosure` `/disclosure/timely` | 4本とも does not exist |
| アナリスト予想 | `/fins/forecast` `/fins/consensus` | 2本とも does not exist |
| 株主構成 | `/equities/shareholders` `/equities/ownership` `/markets/ownership` | 3本とも does not exist |
| 一覧の取得 | `/`（API に自分で言わせる試み） | does not exist |

**この結果は「機能が無い」証明ではない。** 名前は当てずっぽうで、正しい
パス名を知らないまま叩いている。公式のエンドポイント一覧
（jpx-jquants.com）はこの作業環境から到達できない（egress ブロック）。

**正しいパス名が分かれば、この probe に足して1回で確かめられる。**

### それでも適時開示は半分取れている

`/fins/summary` の `DocType` に **33種類**ある（全期間 180,671行）。
決算短信だけではない:

| DocType | 件数 |
|---|---:|
| `EarnForecastRevision`（業績予想の修正） | 24,293 |
| `DividendForecastRevision`（配当予想の修正） | 4,174 |
| `REITEarnForecastRevision` | 612 |
| `REITDividendForecastRevision` | 38 |

本文は入らないが、**「いつ・どの銘柄が・業績予想を修正したか」は取れている**。
`jq_bulk.py` の取得列に `DocType` が入っており保存済みで、
**`build_dataset.py` では未使用**。追加のAPI呼び出しゼロで特徴量にできる。

既存の `guidance_revision`（FOP の前回開示比）とは別物。あちらは修正の
**幅**、こちらは修正**イベントの発生とタイミング**。
`days_since_disc` も実績（Sales か NP が入る開示）だけを数えているので、
修正イベントは勘定に入っていない。

### FOP が通期決算で空になる（2026-09-22 に気づいた穴）

`FOP`（会社予想営業利益）の充足を DocType 別に見ると:

| DocType | 行数 | FOP 充足 |
|---|---:|---:|
| 1QFinancialStatements_Consolidated_JP | 2,768 | 91.5% |
| 2QFinancialStatements_Consolidated_JP | 983 | 93.2% |
| 3QFinancialStatements_Consolidated_JP | 2,490 | 92.2% |
| **FYFinancialStatements_Consolidated_JP** | **2,918** | **0.0%** |
| EarnForecastRevision | 1,246 | 71.3% |

通期発表時の翌期予想は `FOP` ではなく別フィールド（`NxF*` 系）に入って
いるとみられる。`guidance_op_growth` の充足が 50% 止まりなのはこれが理由。
この指標は両側スクリーニングで**下位10%が z = −3.68（11窓中10窓で悪い）**と
測った中で最も強かったので、穴を塞ぐ価値がある。

### エンドポイント一覧を取りに行ってから叩いた結果（2026-09-22）

Claude の作業環境からは公式ドキュメントに到達できない（egress ブロック）ので、
**GitHub Actions のランナーで一覧の取得ごとやる**（`research/probe_endpoints.py` / `Probe Endpoints`）。
思いついた名前を並べるのではなく、取ってきた一覧を全部叩いている。

#### 一覧の取得元

| 取得元 | HTTP | 本文 | 拾えたパス |
|---|---:|---:|---:|
| `https://raw.githubusercontent.com/J-Quants/jquants-api-client-python/main/jquantsapi/client.py` | 404 | 0B | 0 |
| `https://raw.githubusercontent.com/J-Quants/jquants-api-client-python/master/jquantsapi/client.py` | 404 | 0B | 0 |
| `https://raw.githubusercontent.com/J-Quants/jquants-api-client-python/main/jquantsapi/constants.py` | 200 | 15,041B | 0 |
| `https://raw.githubusercontent.com/J-Quants/jquants-api-client-R/main/R/api.R` | 404 | 0B | 0 |
| `https://api.github.com/repos/J-Quants/jquants-api-client-python/git/trees/main?recursive=1` | 200 | 10,497B | 7 |
| `https://api.github.com/orgs/J-Quants/repos?per_page=100` | 200 | 54,575B | 0 |
| `https://jpx.gitbook.io/j-quants-ja/llms.txt` | 200 | 11,154B | 0 |
| `https://jpx.gitbook.io/j-quants-ja/llms-full.txt` | 200 | 11,154B | 0 |
| `https://jpx-jquants.com/llms.txt` | 403 | 0B | 0 |
| `https://api.jquants.com/v2/openapi.json` | 403 | 0B | 0 |
| `https://api.jquants.com/v1/openapi.json` | 403 | 0B | 0 |
| `https://jpx-jquants.com/` | 403 | 0B | 0 |
| `https://jpx.gitbook.io/j-quants-ja/api-reference` | 200 | 11,154B | 0 |

叩いたパス **25本**（OK 4 / v1でOK 0 / 契約不足 3 / 存在しない 18 / 引数不足 0）

#### 使える（OK）

| パス | 引数 | 件数 | 主な項目 |
|---|---|---:|---|
| `/equities/bars/daily` | date | 4359 | `AdjC`, `AdjFactor`, `AdjH`, `AdjL`, `AdjO`, `AdjVo`, `C`, `Code` |
| `/equities/master` | なし | 4450 | `CoName`, `CoNameEn`, `Code`, `Date`, `Mkt`, `MktNm`, `Mrgn`, `MrgnNm` |
| `/fins/summary` | date | 612 | `AvgSh`, `BPS`, `CFF`, `CFI`, `CFO`, `CashEq`, `ChgAcEst`, `ChgByASRev` |
| `/markets/margin-interest` | date | 0 |  |

#### 在るが契約が足りない（プレミアムで開く）

| パス | メッセージ |
|---|---|
| `/fins/details` | This API is not available on your subscription.  If you want more data, please check other plans:https://jpx-j |
| `/fins/dividend` | This API is not available on your subscription.  If you want more data, please check other plans:https://jpx-j |
| `/markets/breakdown` | This API is not available on your subscription.  If you want more data, please check other plans:https://jpx-j |

#### そのパスには何も無い

`/bulk`、`/derivatives`、`/edinet`、`/equities`、`/equities/ownership`、`/equities/shareholders`、`/fins`、`/fins/announcement`、`/fins/consensus`、`/fins/disclosure`、`/fins/forecast`、`/indices`、`/indices/prices`、`/indices/topix`、`/markets`、`/markets/ownership`、`/markets/short-selling`、`/markets/trades-spec`

