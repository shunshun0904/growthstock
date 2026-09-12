# 株価分足の取得可否（実測）

`research/probe_minute_bars.py` の出力。**API を実際に叩いた結果のみ**を記載する。

- 実測日時: 2026-09-12T02:17:02.385663+00:00
- エンドポイント: `/equities/bars/minute`（`https://api.jquants.com/v2/equities/bars/minute`）
- 測定に使った銘柄: `72030` / `90820`

## 1. キー自体は通るか（日足で確認）

`/equities/bars/daily` は **OK**（1行 / 0.59秒）。
以降で分足が失敗した場合、原因はキーではなくアドオンの契約範囲である。

## 2. 分足は引けるか

**引けない**。返ってきたエラーをそのまま載せる。

```
HTTP 403 https://api.jquants.com/v2/equities/bars/minute?code=72030&date=2026-09-11 : {"message": "This API is not available on your subscription.  If you want more data, please check other plans:https://jpx-jquants.com/#dataset"}
```

日足は通っているので、キーの失効ではない。分足・Tick アドオンが契約に含まれていない状態だと読める。

この結果は失敗ではなく測定結果である。契約すれば何が変わるかは、契約後に本プローブを再実行すれば分かる。

---

以降の測定（列・立会の形・遡及範囲・スループット）は、分足が引けないため実行していない。
