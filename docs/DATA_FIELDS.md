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

### FOP が通期決算で空になっていた（2026-09-22 に塞いだ）

`FOP`（会社予想営業利益）の充足を DocType 別に見ると:

| DocType | 行数 | FOP | NxFOP |
|---|---:|---:|---:|
| 1QFinancialStatements_Consolidated_JP | 2,768 | 91.5% | 0.0% |
| 2QFinancialStatements_Consolidated_JP | 983 | 93.2% | 0.0% |
| 3QFinancialStatements_Consolidated_JP | 2,490 | 92.2% | 0.0% |
| **FYFinancialStatements_Consolidated_JP** | 2,918 | **0.0%** | **87.8%** |
| EarnForecastRevision | 1,246 | 71.3% | 0.0% |

**通期では翌期予想が `NxFOP` に入る。** `guidance_op_growth` は `FOP` だけを
見ていたので、通期行で必ず欠測になっていた。

#### 繋いでよいと確かめた

通期の `NxFOP` と、その次の1Q の `FOP` を 13,185組つき合わせた:

| | |
|---|---:|
| 完全一致 | **92.7%** |
| 1%以内 | 92.8% |
| 10%以内 | 94.9% |
| 誤差の中央値 | **0.00%** |

残りは1Qで会社が予想を修正した分で説明がつく。同じ事業年度の予想を
指しているので繋いでよい。

#### 分母も変えた

`FOP` は「進行中の事業年度」の予想なので、前年は `shift(4)` した4期和。
`NxFOP` は「次の事業年度」の予想なので、比べる相手は**いま締めた事業年度**
＝ shift しない4期和。ここを揃えないと、**通期行だけ2年ぶんの伸び**を
見ることになる。どちらを使ったかは `guidance_basis` に残す。

| | 開示単位 | サンプル単位 |
|---|---:|---:|
| これまで（FOP のみ） | 63.8% | 約50% |
| 直したあと | **75.6%** | **67.8%** |
| うち通期 | 21.6% → 55.6% | — |

この指標は両側スクリーニングで**下位10%が z = −3.68（11窓中10窓で悪い）**と、
測った中でいちばん強い（実験37）。充足が上がるぶん効きやすくなる。

**`guidance_revision` は `NxFOP` で埋めていない。** 通期行の `NxFOP` は
次の年度の最初の予想で、同じ `CurFYSt` の前回（3Q）とは別の年度を指す。
埋めると年度をまたいだ差を「修正」として出してしまう。新しい年度の最初の
予想に「修正」は定義できないので、欠測が正しい。

### 予想修正イベント（DocType）を特徴量にした（2026-09-22）

`/fins/summary` の `DocType` は33種類あり、決算短信だけではない
（全期間 180,671行）:

| DocType | 件数 |
|---|---:|
| `EarnForecastRevision` | 24,293 |
| `DividendForecastRevision` | 4,174 |
| `REITEarnForecastRevision` | 612 |
| `REITDividendForecastRevision` | 38 |

取り込みには元から入っていたが、特徴量として一度も使っていなかった。
**追加の取得はいらない。** `build_dataset.forecast_revisions()` で8列を作る。

| 列 | 充足 | 中身 |
|---|---:|---|
| `days_since_rev` | 84.3% | 直近の業績予想修正からの日数（上限400） |
| `rev_pct` | 44% | その修正の幅（%）。同じ事業年度の直前の予想と比べる |
| `rev_up` | 47.3% | 上方なら1、下方なら0。前の予想が無ければ欠測 |
| `rev_n_60` / `rev_n_250` | 100% | 過去60/250暦日の修正回数 |
| `rev_up_n_250` / `rev_dn_n_250` | 100% | 上方/下方の回数 |
| `days_since_divrev` | 36% | 直近の配当予想修正からの日数 |

既存の `guidance_revision`（FOP の前回開示比）は修正の**幅**、こちらは
**発生とタイミング**。`days_since_disc` は実績（Sales か NP）のある開示
だけを数えているので、修正だけの開示は勘定に入っていない。

採否は一括評価（実験39）で決める。それまで `ALL_GROUPS` には入れない。


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
| `J-Quants/JQuantsR/LICENSE.md` | 200 | 1,088B | 0 |
| `J-Quants/JQuantsR/R/authorize.R` | 200 | 2,978B | 2 |
| `J-Quants/JQuantsR/R/constants.R` | 200 | 42B | 0 |
| `J-Quants/JQuantsR/R/fetch.R` | 200 | 3,024B | 2 |
| `J-Quants/JQuantsR/R/wrapper.R` | 200 | 11,940B | 17 |
| `J-Quants/JQuantsR/README.md` | 200 | 3,540B | 0 |
| `J-Quants/JQuantsR/tests/testthat.R` | 200 | 74B | 0 |
| `J-Quants/JQuantsR/tests/testthat/test_all.R` | 200 | 6,740B | 0 |
| `J-Quants/jquants-api-client-python/.github/ISSUE_TEMPLATE/bug_report.yaml` | 200 | 2,704B | 0 |
| `J-Quants/jquants-api-client-python/.github/ISSUE_TEMPLATE/feature-request.yaml` | 200 | 1,550B | 0 |
| `J-Quants/jquants-api-client-python/.github/ISSUE_TEMPLATE/question.yaml` | 200 | 1,375B | 0 |
| `J-Quants/jquants-api-client-python/.github/PULL_REQUEST_TEMPLATE.md` | 200 | 74B | 0 |
| `J-Quants/jquants-api-client-python/.github/workflows/build.yml` | 200 | 2,648B | 0 |
| `J-Quants/jquants-api-client-python/.github/workflows/publish.yml` | 200 | 668B | 0 |
| `J-Quants/jquants-api-client-python/CONTRIBUTING.md` | 200 | 1,122B | 0 |
| `J-Quants/jquants-api-client-python/README.md` | 200 | 5,897B | 0 |
| `J-Quants/jquants-api-client-python/examples/20260119-007-jquants-api-v2-starter.ipynb` | 200 | 11,834B | 1 |
| `J-Quants/jquants-api-client-python/examples/README.md` | 200 | 1,479B | 0 |
| `J-Quants/jquants-api-client-python/jquantsapi/__init__.py` | 200 | 172B | 0 |
| `J-Quants/jquants-api-client-python/jquantsapi/apis/__init__.py` | 200 | 130B | 0 |
| `J-Quants/jquants-api-client-python/jquantsapi/apis/base.py` | 200 | 866B | 0 |
| `J-Quants/jquants-api-client-python/jquantsapi/apis/v2/__init__.py` | 200 | 21B | 0 |
| `J-Quants/jquants-api-client-python/jquantsapi/apis/v2/bulk.py` | 200 | 3,427B | 1 |
| `J-Quants/jquants-api-client-python/jquantsapi/apis/v2/derivatives.py` | 200 | 3,892B | 2 |
| `J-Quants/jquants-api-client-python/jquantsapi/apis/v2/edinet.py` | 200 | 3,704B | 3 |
| `J-Quants/jquants-api-client-python/jquantsapi/apis/v2/equities.py` | 200 | 10,195B | 7 |
| `J-Quants/jquants-api-client-python/jquantsapi/apis/v2/fins.py` | 200 | 7,072B | 2 |
| `J-Quants/jquants-api-client-python/jquantsapi/apis/v2/indices.py` | 200 | 2,496B | 2 |
| `J-Quants/jquants-api-client-python/jquantsapi/apis/v2/markets.py` | 200 | 8,548B | 6 |
| `J-Quants/jquants-api-client-python/jquantsapi/apis/v2/td.py` | 200 | 4,274B | 0 |
| `J-Quants/jquants-api-client-python/jquantsapi/client_v2.py` | 200 | 58,430B | 1 |
| `J-Quants/jquants-api-client-python/jquantsapi/constants.py` | 200 | 15,041B | 0 |
| `J-Quants/jquants-api-client-python/jquantsapi/enums.py` | 200 | 1,884B | 20 |
| `J-Quants/jquants-api-client-python/tests/__init__.py` | 200 | 0B | 0 |
| `J-Quants/jquants-api-client-python/tests/test_client_v2.py` | 200 | 44,170B | 8 |
| `J-Quants/jquants-api-jvm/.circleci/config.yml` | 200 | 376B | 0 |
| `J-Quants/jquants-api-jvm/BUILD.md` | 200 | 738B | 0 |
| `J-Quants/jquants-api-jvm/CHANGELOG.md` | 200 | 776B | 0 |
| `J-Quants/jquants-api-jvm/README.md` | 200 | 11,051B | 0 |
| `J-Quants/jquants-api-jvm/examples/JavaSample/src/main/java/info/hellonico/jquantsapi/JQuantsApiSample.java` | 200 | 4,429B | 0 |
| `J-Quants/jquants-api-jvm/examples/JavaSample/src/test/java/info/hellonico/jquantsapi/AppTest.java` | 200 | 297B | 0 |
| `J-Quants/jquants-api-jvm/examples/charting-with-oz/CHANGELOG.md` | 200 | 786B | 0 |
| `J-Quants/jquants-api-jvm/examples/charting-with-oz/README.md` | 200 | 1,447B | 0 |
| `J-Quants/jquants-api-jvm/examples/charting-with-oz/doc/intro.md` | 200 | 124B | 0 |
| `J-Quants/jquants-api-jvm/examples/jupyter-clj/README.md` | 200 | 555B | 0 |
| `J-Quants/jquants-api-jvm/examples/jupyter-clj/jupyter-jquants-jvm.ipynb` | 200 | 25,432B | 0 |
| `J-Quants/jquants-api-jvm/examples/someml/CHANGELOG.md` | 200 | 766B | 0 |
| `J-Quants/jquants-api-jvm/examples/someml/README.md` | 200 | 1,923B | 0 |
| `J-Quants/jquants-api-jvm/examples/someml/doc/intro.md` | 200 | 104B | 0 |
| `J-Quants/jquants-api-jvm/test/daily_86970_20201001.json` | 200 | 28B | 0 |
| `J-Quants/jquants-api-jvm/test/daily_86970_20220118.json` | 200 | 433B | 0 |
| `J-Quants/jquants-api-jvm/test/listed_info_86970.json` | 200 | 288B | 0 |
| `J-Quants/jquants-api-jvm/test/listed_info_empty.json` | 200 | 20B | 0 |
| `J-Quants/jquants-api-jvm/test/listed_sections.json` | 200 | 2,624B | 0 |
| `J-Quants/jquants-api-jvm/test/statements_6digitscode.json` | 200 | 65B | 0 |
| `J-Quants/jquants-api-jvm/test/statements_86970_20220118.json` | 200 | 2,035B | 0 |
| `J-Quants/jquants-api-jvm/test/statements_empty.json` | 200 | 24B | 0 |
| `J-Quants/jquants-api-go/.circleci/config.yml` | 200 | 434B | 1 |
| `J-Quants/jquants-api-go/README.md` | 200 | 570B | 0 |
| `J-Quants/jquants-api-go/example/jquants.go` | 200 | 1,005B | 0 |
| `J-Quants/jquants-api-go/example2/hello.go` | 200 | 17B | 0 |
| `J-Quants/jquants-api-go/helper.go` | 200 | 4,816B | 0 |
| `J-Quants/jquants-api-go/helper_test.go` | 200 | 814B | 0 |
| `J-Quants/jquants-api-go/vendor/olympos.io/encoding/edn/.travis.yml` | 200 | 37B | 0 |
| `J-Quants/jquants-api-go/vendor/olympos.io/encoding/edn/README.md` | 200 | 3,307B | 0 |
| `J-Quants/jquants-api-go/vendor/olympos.io/encoding/edn/compact.go` | 200 | 2,066B | 0 |
| `J-Quants/jquants-api-go/vendor/olympos.io/encoding/edn/decode.go` | 200 | 41,608B | 0 |
| `J-Quants/jquants-api-go/vendor/olympos.io/encoding/edn/edn_tags.go` | 200 | 4,625B | 2 |
| `J-Quants/jquants-api-go/vendor/olympos.io/encoding/edn/encode.go` | 200 | 35,431B | 0 |
| `J-Quants/jquants-api-go/vendor/olympos.io/encoding/edn/extras.go` | 200 | 3,632B | 0 |
| `J-Quants/jquants-api-go/vendor/olympos.io/encoding/edn/fold.go` | 200 | 3,467B | 0 |
| `J-Quants/jquants-api-go/vendor/olympos.io/encoding/edn/lexer.go` | 200 | 13,463B | 0 |
| `J-Quants/jquants-api-go/vendor/olympos.io/encoding/edn/pprint.go` | 200 | 6,733B | 0 |
| `J-Quants/jquants-api-go/vendor/olympos.io/encoding/edn/tags.go` | 200 | 1,075B | 0 |
| `J-Quants/jquants-api-go/vendor/olympos.io/encoding/edn/types.go` | 200 | 3,223B | 0 |
| `J-Quants/JPXTokyoStockExchangePrediction/README.md` | 200 | 3,519B | 0 |
| `J-Quants/JPXTokyoStockExchangePrediction/winner-models/10th/ModelSummary.md` | 200 | 1,768B | 0 |
| `J-Quants/JPXTokyoStockExchangePrediction/winner-models/10th/notebooks/simulations/simulation_aggregation.ipynb` | 200 | 362,450B | 0 |
| `J-Quants/JPXTokyoStockExchangePrediction/winner-models/10th/notebooks/submission/submission-notebook.ipynb` | 200 | 10,953B | 1 |
| `J-Quants/JPXTokyoStockExchangePrediction/winner-models/10th/readme.md` | 200 | 1,310B | 0 |

叩いたパス **55本**（OK 17 / v1でOK 0 / 契約不足 7 / 存在しない 31 / 引数不足 0）

#### 使える（OK）

| パス | 引数 | 件数 | 主な項目 |
|---|---|---:|---|
| `/edinet/cross-shareholdings` | date | 2 | `Code`, `DocId`, `DocTypeCode`, `EdinetCode`, `FilerName`, `FilerNameEn`, `Largest`, `PerEn` |
| `/edinet/large-volume-shareholders` | date | 47 | `ChgRsn`, `Code`, `DocId`, `DocTitle`, `DocTypeCode`, `EdinetCode`, `Hldrs`, `IsrName` |
| `/edinet/major-shareholders` | date | 68 | `Code`, `CurPerEn`, `CurPerSt`, `DocId`, `DocTypeCode`, `EdinetCode`, `FilerName`, `FilerNameEn` |
| `/equities/bars/daily` | date | 4359 | `AdjC`, `AdjFactor`, `AdjH`, `AdjL`, `AdjO`, `AdjVo`, `C`, `Code` |
| `/equities/earnings-calendar` | なし | 1 | `CoName`, `Code`, `Date`, `FQ`, `FY`, `Section`, `SectorNm` |
| `/equities/investor-types` | なし | 2377 | `BankBal`, `BankBuy`, `BankSell`, `BankTot`, `BrkBal`, `BrkBuy`, `BrkSell`, `BrkTot` |
| `/equities/master` | なし | 4450 | `CoName`, `CoNameEn`, `Code`, `Date`, `Mkt`, `MktNm`, `Mrgn`, `MrgnNm` |
| `/equities/valuation` | date | 4359 | `BPS`, `Code`, `Date`, `EPS`, `FwdEPS`, `FwdPER`, `FwdROE`, `MktCap` |
| `/fins/earnings-date` | date | 21 | `CoName`, `CoNameEn`, `Code`, `FQName`, `FYE`, `PubDate`, `SchDate` |
| `/fins/summary` | date | 612 | `AvgSh`, `BPS`, `CFF`, `CFI`, `CFO`, `CashEq`, `ChgAcEst`, `ChgByASRev` |
| `/indices/bars/daily` | date | 79 | `C`, `Code`, `Date`, `H`, `L`, `O` |
| `/indices/bars/daily/topix` | なし | 2441 | `C`, `Date`, `H`, `L`, `O` |
| `/markets/calendar` | なし | 4118 | `Date`, `HolDiv` |
| `/markets/margin-alert` | date | 194 | `AppDate`, `Code`, `LongNegOut`, `LongNegOutChg`, `LongOut`, `LongOutChg`, `LongOutRatio`, `LongStdOut` |
| `/markets/margin-interest` | code | 507 | `Code`, `Date`, `IssType`, `LongNegVol`, `LongStdVol`, `LongVol`, `ShrtNegVol`, `ShrtStdVol` |
| `/markets/short-ratio` | date | 34 | `Date`, `S33`, `SellExShortVa`, `ShrtNoResVa`, `ShrtWithResVa` |
| `/markets/short-sale-report` | code | 22 | `CalcDate`, `Code`, `DICAddr`, `DICName`, `DiscDate`, `FundName`, `Notes`, `PrevRptDate` |

#### 在るが契約が足りない（プレミアムで開く）

| パス | メッセージ |
|---|---|
| `/derivatives/bars/daily/futures` | This API is not available on your subscription.  If you want more data, please check other plans:https://jpx-j |
| `/derivatives/bars/daily/options` | This API is not available on your subscription.  If you want more data, please check other plans:https://jpx-j |
| `/equities/bars/daily/am` | This API is not available on your subscription.  If you want more data, please check other plans:https://jpx-j |
| `/equities/bars/minute` | This API is not available on your subscription.  If you want more data, please check other plans:https://jpx-j |
| `/fins/details` | This API is not available on your subscription.  If you want more data, please check other plans:https://jpx-j |
| `/fins/dividend` | This API is not available on your subscription.  If you want more data, please check other plans:https://jpx-j |
| `/markets/breakdown` | This API is not available on your subscription.  If you want more data, please check other plans:https://jpx-j |

#### そのパスには何も無い

`/bulk`、`/content/drive`、`/derivatives/futures`、`/derivatives/options`、`/equities/ownership`、`/equities/shareholders`、`/equities/trades`、`/examples`、`/fins/announcement`、`/fins/consensus`、`/fins/disclosure`、`/fins/forecast`、`/fins/fs_details`、`/fins/statements`、`/go/pkg/mod`、`/indices/prices`、`/indices/topix`、`/kaggle/input`、`/listed/info`、`/markets/ownership`、`/markets/short-selling`、`/markets/short_selling`、`/markets/short_selling_positions`、`/markets/trades-spec`、`/markets/trades_spec`、`/markets/trading_calendar`、`/markets/weekly_margin_interest`、`/prices/daily_quotes`、`/prices/prices_am`、`/token/auth_refresh`、`/token/auth_user`

### 運用者の3つの問いへの答え（2026-09-22、4回目で確定）

| 問い | 答え | エンドポイント |
|---|---|---|
| **株主構成** | **取れる** | `/edinet/major-shareholders`（大株主）、`/edinet/large-volume-shareholders`（大量保有報告書）、`/edinet/cross-shareholdings`（政策保有）、`/equities/investor-types`（投資部門別売買） |
| **アナリスト予想** | **予想ベースの指標は取れる** | `/equities/valuation` の `FwdEPS` / `FwdPER` / `FwdROE`。ただし**コンセンサスかは未確認**（会社予想由来の可能性がある。中身を突き合わせて確かめること） |
| **適時開示の本文** | **取れない** | `/fins/earnings-date`（発表**予定日**）と `/equities/earnings-calendar` はある。本文そのものは依然として無い |

**前の節で「J-Quants に株主構成は無い」と書いたのは誤り。** `/edinet/` という
グループが在り、大株主も大量保有報告書も入っていた。名前を当てずっぽうで
叩いていたので見つけられなかっただけ。

### 使えるのに使っていないもの（優先度つき）

| 優先 | パス | 件数 | 中身 | なぜ効きそうか |
|---|---|---:|---|---|
| **1** | `/markets/calendar` | 4,118 | `Date` `HolDiv` | **営業日カレンダー。** 2026-09-22 に予測が落ちたのは、鮮度チェックが祝日を知らず連休を「取り込み障害」と誤判定したため。これを取り込めば根本的に直る |
| **2** | `/equities/valuation` | 4,359/日 | `BPS` `EPS` **`FwdEPS`** `MktCap` `PBR` `PER` **`FwdPER`** `ROE` **`FwdROE`** | PER/PBR/ROE は今すべて自前計算。置き換えれば欠測が減る。`Fwd*` 3本は**まったく新しい情報** |
| **3** | `/edinet/large-volume-shareholders` | 47/日 | `TotalShsRatio` **`TotalShsRatioLast`** `TotalShsHeld` `ChgRsn` `RptOblgDate` | 大量保有報告書。前回比があるので「**誰かが5%超を買い増した**」がイベントとして取れる。高値更新との関係は仮説として筋が良い |
| **4** | `/equities/investor-types` | 2,377 | `Frgn*`（外国人）`InvTr*`（投信）`Ind*`（個人）`Prop*`（自己）`Bank*` `InsCo*` … 各 Buy/Sell/Bal/Tot | 投資部門別売買。地合い11列は指数のリターンだけなので、**主体別の需給**は新しい軸 |
| 5 | `/markets/short-sale-report` | 22/銘柄 | `ShrtPosShares` `ShrtPosToSO` `FundName` `PrevRptRatio` | 空売り残高（ファンド名まで）。`credit_ratio` より直接的 |
| 6 | `/markets/short-ratio` | 34 | `S33` `SellExShortVa` `ShrtWithResVa` `ShrtNoResVa` | 業種別の空売り比率 |
| 7 | `/markets/margin-alert` | 194 | `TSEMrgnRegCls` `PubReason` `SLRatio` … | 信用規制。規制がかかった銘柄は値動きが変わる |
| 8 | `/fins/earnings-date` | 21/日 | `SchDate`（予定）`PubDate`（実績） | **次の決算までの日数**が作れる。いまは `days_since_disc`（前回からの日数）だけ |
| 9 | `/edinet/major-shareholders` | 68/日 | `Hldrs`（保有者）`FilerName` `PerSt`/`PerEn` | 大株主（有報ベース、年1回） |
| 10 | `/indices/bars/daily/topix` | 2,441 | `O` `H` `L` `C` | TOPIX。いまは ETF から代用している |

### 探し方の記録（同じ失敗を3回した）

| 回 | やり方 | 結果 |
|---|---|---|
| 1 | エンドポイント名を思いついて並べる | 候補10本すべて「存在しない」。何の証明にもならず |
| 2 | 公式ドキュメント（GitBook）から拾う | 全URLが同じ 11,154バイト。本文を JavaScript で描くので中身が無い |
| 3 | 公式クライアントの**置き場所を決め打って**取りに行く | `client.py` は 404。ファイル一覧だけ取れて、最上位のグループ名7本 |
| 4 | org → リポジトリ一覧 → ファイル一覧 → **ソース本体**と辿る | **46本発見、17本が OK** |

1・3 はどちらも「名前や場所を当てる」やり方で、外れても外れたことが分からない。
**当てずに辿る**形にして初めて出た。`research/probe_endpoints.py` はこの形で
書いてある。再実行は `Probe Endpoints`（手動起動）。

