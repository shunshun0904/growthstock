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

| 項目 | 充足率 |
| --- | ---: |
| `CurFYEn` | 100.0% |
| `CurPerSt` | 100.0% |
| `DiscNo` | 100.0% |
| `ChgByASRev` | 94.9% |
| `ChgNoASRev` | 94.9% |
| `AvgSh` | 94.8% |
| `EqAR` | 94.8% |
| `ShEq` | 94.7% |
| `ChgAcEst` | 94.2% |
| `OdP` | 88.9% |
| `RetroRst` | 88.3% |
| `Div2Q` | 63.6% |
| `FDivFY` | 58.3% |
| `MatChgSub` | 58.1% |
| `FDivAnn` | 57.8% |
| `FOdP` | 56.9% |
| `CashEq` | 51.3% |
| `CFI` | 50.8% |
| `CFO` | 50.8% |
| `CFF` | 50.6% |
| `BPS` | 46.7% |
| `NxtFYEn` | 33.0% |
| `NxtFYSt` | 33.0% |
| `DivFY` | 32.7% |
| `DivAnn` | 32.4% |
| `DEPS` | 30.8% |
| `NxFNp` | 29.6% |
| `NxFEPS` | 29.4% |
| `NxFDivFY` | 29.3% |
| `NxFSales` | 29.3% |
| `NxFDivAnn` | 29.2% |
| `NxFOP` | 28.8% |
| `NxFDiv2Q` | 28.1% |
| `NxFOdP` | 27.9% |
| `DivTotalAnn` | 25.0% |
| `SigChgInC` | 24.8% |
| `NCEPS` | 23.7% |
| `NCEq` | 23.7% |
| `NCEqAR` | 23.7% |
| `NCNP` | 23.7% |
| `NCOdP` | 23.7% |
| `NCSales` | 23.7% |
| `NCShEq` | 23.7% |
| `NCTA` | 23.7% |
| `FDiv2Q` | 23.6% |
| `NCBPS` | 23.2% |
| `NCOP` | 22.8% |
| `PayoutRatioAnn` | 22.7% |
| `NxFPayoutRatioAnn` | 20.8% |
| `NxFEPS2Q` | 14.5% |
| `NxFNp2Q` | 14.5% |
| `NxFSales2Q` | 14.2% |
| `NxFOP2Q` | 14.0% |
| `NxFOdP2Q` | 14.0% |
| `FNP2Q` | 13.3% |
| `FEPS2Q` | 13.2% |
| `FSales2Q` | 13.0% |
| `FOP2Q` | 12.8% |
| `FOdP2Q` | 12.7% |
| `NxFNCNP` | 4.2% |
| `NxFNCOdP` | 4.2% |
| `NxFNCEPS` | 4.1% |
| `NxFNCSales` | 4.0% |
| `NxFNCEPS2Q` | 3.0% |
| `NxFNCNP2Q` | 3.0% |
| `NxFNCOdP2Q` | 3.0% |
| `NxFNCSales2Q` | 2.9% |
| `FNCEPS` | 1.9% |
| `FNCNP` | 1.9% |
| `FNCOdP` | 1.9% |
| `FNCSales` | 1.8% |
| `NxFNCOP` | 1.4% |
| `Div1Q` | 1.0% |
| `FNCOP` | 1.0% |
| `NxFNCOP2Q` | 0.8% |
| `Div3Q` | 0.6% |
| `FDiv3Q` | 0.4% |
| `FNCEPS2Q` | 0.4% |
| `FNCNP2Q` | 0.4% |
| `FNCOdP2Q` | 0.4% |
| `FNCSales2Q` | 0.4% |
| `FNCOP2Q` | 0.2% |
| `NxFDiv1Q` | 0.2% |
| `NxFDiv3Q` | 0.2% |
| `FDiv1Q` | 0.1% |
| `DivUnit` | 0.0% |
| `FDivTotalAnn` | 0.0% |
| `FDivUnit` | 0.0% |
| `FPayoutRatioAnn` | 0.0% |
| `NCROE` | 0.0% |
| `NxFDivUnit` | 0.0% |

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
| `/equities/master` | OK | 4441 | 14項目 |
| `/equities/bars/daily` | OK | 4359 | 18項目 |
| `/markets/margin-interest` | OK | 0 | 0項目 |
| `/markets/short-selling` | NG | — | HTTP 403 https://api.jquants.com/v2/markets/short-selling?date=2024-05-15 : {"message": "The requested endpoint does not |
| `/markets/breakdown` | NG | — | HTTP 403 https://api.jquants.com/v2/markets/breakdown?date=2024-05-15 : {"message": "This API is not available on your s |
| `/markets/trades-spec` | NG | — | HTTP 403 https://api.jquants.com/v2/markets/trades-spec : {"message": "The requested endpoint does not exist. Please che |
| `/indices/topix` | NG | — | HTTP 403 https://api.jquants.com/v2/indices/topix?from=2024-05-01&to=2024-05-15 : {"message": "The requested endpoint do |
| `/indices/prices` | NG | — | HTTP 403 https://api.jquants.com/v2/indices/prices?date=2024-05-15 : {"message": "The requested endpoint does not exist. |

