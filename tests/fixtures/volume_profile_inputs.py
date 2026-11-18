"""Write representative synthetic J-Quants CSV/GZ files for the import smoke test."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd


def write_inputs(output):
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    calendar = pd.date_range("2025-10-01", "2025-11-18")
    holidays = {"2025-10-13", "2025-11-03"}
    daily, minute, tick, cal = [], [], [], []
    for index, current in enumerate(calendar):
        day = current.date().isoformat()
        trading = current.weekday() < 5 and day not in holidays
        cal.append((day, int(trading)))
        if not trading:
            continue
        for code, base in [("123A0", 1340), ("99990", 1810)]:
            # Concentrate trades near two observed levels, with a small daily drift.
            anchor = base + int(21 * np.sin(index / 7)) + index
            prices = [anchor + offset for offset in (-18, -13, -6, 0, 2, 7, 11, 18)]
            shares = [190, 390, 680, 2050, 1770, 770, 420, 270]
            values = [p * q for p, q in zip(prices, shares)]
            daily.append((day, code, prices[0], prices[-1], prices[4], sum(shares), sum(values), 1., 0))
            for minute_index in range(4):
                section = slice(minute_index * 2, minute_index * 2 + 2)
                pair = prices[section]
                time = f"09:0{minute_index}"
                minute.append((day, code, time, pair[0], pair[-1], min(pair), max(pair),
                               sum(shares[section]), sum(values[section])))
                if code == "123A0":
                    for i in range(section.start, section.stop):
                        tick.append((day, code, f"{time}:{i:02}.000000", "01", prices[i], shares[i], f"{i+1:012}"))
    pd.DataFrame(cal, columns=["Date", "HolDiv"]).to_csv(root / "calendar.csv", index=False)
    pd.DataFrame(daily, columns=["Date", "Code", "L", "H", "C", "Vo", "Va", "AdjFactor", "ExRT"]).to_csv(
        root / "daily.csv", index=False)
    pd.DataFrame(minute, columns=["Date", "Code", "Time", "O", "C", "L", "H", "Vo", "Va"]).to_csv(
        root / "minute.csv.gz", index=False)
    pd.DataFrame(tick, columns=["Date", "Code", "Time", "SessionDistinction", "Price", "TradingVolume", "TransactionId"]).to_csv(
        root / "tick.csv.gz", index=False)


if __name__ == "__main__":
    write_inputs(sys.argv[1])
