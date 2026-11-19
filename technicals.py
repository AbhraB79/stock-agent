"""
technicals.py — technical-analysis maths, using plain pandas.

Everything here works on a DataFrame with Date, Open, High, Low, Close, Volume
columns (one row per bar). No internet access, so it's easy to test.

Indicators (the standard textbook settings):
  SMA20/50/200  simple moving averages of the close
  Bollinger     20-bar SMA ± 2 standard deviations
  RSI14         relative strength index, Wilder's smoothing (0-100)
  MACD 12/26/9  EMA12 - EMA26, its 9-bar EMA "signal", and the difference
"""

import numpy as np
import pandas as pd


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of df with indicator columns added."""
    out = df.copy()
    close = out["Close"]

    for n in (20, 50, 200):
        out[f"SMA{n}"] = close.rolling(n).mean()

    std20 = close.rolling(20).std(ddof=0)
    out["BB_upper"] = out["SMA20"] + 2 * std20
    out["BB_lower"] = out["SMA20"] - 2 * std20

    out["RSI14"] = rsi(close, 14)

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    out["MACD"] = ema12 - ema26
    out["MACD_signal"] = out["MACD"].ewm(span=9, adjust=False).mean()
    out["MACD_hist"] = out["MACD"] - out["MACD_signal"]
    return out


def _wilder(values: np.ndarray, n: int) -> np.ndarray:
    """Wilder's smoothing: start with the plain average of the first n values,
    then avg = (previous avg * (n-1) + new value) / n."""
    out = np.full(len(values), np.nan)
    if len(values) <= n:
        return out
    out[n] = values[1 : n + 1].mean()  # values[0] is the NaN from diff()
    for i in range(n + 1, len(values)):
        out[i] = (out[i - 1] * (n - 1) + values[i]) / n
    return out


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    """Wilder's RSI: 100 = only gains lately, 0 = only losses, 50 = flat."""
    delta = close.diff().to_numpy(dtype=float)
    avg_gain = _wilder(np.clip(delta, 0, None), n)
    avg_loss = _wilder(np.clip(-delta, 0, None), n)
    with np.errstate(divide="ignore", invalid="ignore"):
        values = 100 - 100 / (1 + avg_gain / avg_loss)
    # No losses at all -> 100; no movement at all -> 50
    values = np.where(avg_loss == 0, 100.0, values)
    values = np.where((avg_gain == 0) & (avg_loss == 0), 50.0, values)
    values = np.where(np.isnan(avg_gain), np.nan, values)
    return pd.Series(values, index=close.index)


def find_levels(df: pd.DataFrame, window: int = 5, n: int = 2) -> dict:
    """Support/resistance from swing points.

    A swing low is a bar whose Low is the lowest within `window` bars on each
    side (swing high: same with High). Swing prices within 1.5% of each other
    are merged into one level. Returns the `n` levels nearest to the last close
    on each side, with how many swings ("touches") formed each level.
    """
    if len(df) < 2 * window + 1:
        return {"support": [], "resistance": []}
    span = 2 * window + 1
    lows = df["Low"][df["Low"] == df["Low"].rolling(span, center=True).min()]
    highs = df["High"][df["High"] == df["High"].rolling(span, center=True).max()]
    pivots = sorted(pd.concat([lows, highs]).dropna().astype(float))

    clusters: list[list[float]] = []
    for price in pivots:
        if clusters and price <= np.mean(clusters[-1]) * 1.015:
            clusters[-1].append(price)
        else:
            clusters.append([price])
    levels = [(round(float(np.mean(c)), 2), len(c)) for c in clusters]

    last = float(df["Close"].iloc[-1])
    below = sorted((lv for lv in levels if lv[0] < last), key=lambda lv: last - lv[0])
    above = sorted((lv for lv in levels if lv[0] >= last), key=lambda lv: lv[0] - last)
    return {
        "support": [{"price": p, "touches": t} for p, t in below[:n]],
        "resistance": [{"price": p, "touches": t} for p, t in above[:n]],
    }


def line_value_at(line: dict, when: pd.Timestamp) -> float:
    """Price of a user line at a moment (trendlines extend past their end)."""
    if line["type"] == "horizontal":
        return float(line["price"])
    t1, t2 = pd.Timestamp(line["start_date"]), pd.Timestamp(line["end_date"])
    p1, p2 = float(line["start_price"]), float(line["end_price"])
    if t1 == t2:
        return p2
    return p1 + (p2 - p1) * (when - t1) / (t2 - t1)


def describe_lines(lines: list[dict], last_close: float, now: pd.Timestamp) -> list[dict]:
    """Where the price stands against each user line, for Claude to discuss."""
    described = []
    for line in lines:
        value = line_value_at(line, now)
        info = {**line, "line_value_now": _r(value)}
        distance = (last_close - value) / value * 100 if value else 0.0
        info["price_is"] = "at" if abs(distance) < 0.05 else "above" if distance > 0 else "below"
        info["distance_pct"] = _r(distance)
        described.append(info)
    return described


def _last_cross(diff: pd.Series, dates: pd.Series):
    """Date and direction of the last time `diff` changed sign, or None."""
    sign = np.sign(diff.dropna())
    changes = sign[sign != sign.shift()].iloc[1:]  # first value isn't a cross
    changes = changes[changes != 0]
    if changes.empty:
        return None
    idx = changes.index[-1]
    return {
        "date": dates.loc[idx].strftime("%Y-%m-%d"),
        "direction": "bullish" if changes.iloc[-1] > 0 else "bearish",
    }


def _r(x):
    """Round to 2 dp, or None if missing."""
    return None if x is None or pd.isna(x) else round(float(x), 2)


def summarize(df: pd.DataFrame, levels: dict) -> dict:
    """Compact facts about the latest bar for Claude to interpret."""
    last = df.iloc[-1]
    close = float(last["Close"])
    out: dict = {
        "last_close": _r(close),
        "range_high": _r(df["High"].max()),
        "range_low": _r(df["Low"].min()),
        "range_change_pct": _r((close - df["Close"].iloc[0]) / df["Close"].iloc[0] * 100),
    }

    out["moving_averages"] = {
        name: (
            {"value": _r(last[name]), "price_is": "above" if close > last[name] else "below"}
            if pd.notna(last[name]) else "not enough data"
        )
        for name in ("SMA20", "SMA50", "SMA200")
    }
    if pd.notna(last["SMA50"]) and pd.notna(last["SMA200"]):
        cross = _last_cross(df["SMA50"] - df["SMA200"], df["Date"])
        out["sma50_vs_sma200"] = {
            "sma50_is": "above" if last["SMA50"] > last["SMA200"] else "below",
            "last_cross_in_range": (
                {**cross, "type": "golden cross" if cross["direction"] == "bullish" else "death cross"}
                if cross else None
            ),
        }

    if pd.notna(last["RSI14"]):
        value = float(last["RSI14"])
        zone = "overbought" if value > 70 else "oversold" if value < 30 else "neutral"
        out["rsi14"] = {"value": _r(value), "zone": zone}

    out["macd"] = {
        "macd": _r(last["MACD"]),
        "signal": _r(last["MACD_signal"]),
        "histogram": _r(last["MACD_hist"]),
        "macd_is": "above signal" if last["MACD_hist"] > 0 else "below signal",
        "last_crossover": _last_cross(df["MACD_hist"], df["Date"]),
    }

    if pd.notna(last["BB_upper"]) and last["BB_upper"] != last["BB_lower"]:
        pct_b = (close - last["BB_lower"]) / (last["BB_upper"] - last["BB_lower"])
        out["bollinger"] = {
            "upper": _r(last["BB_upper"]),
            "lower": _r(last["BB_lower"]),
            "position_pct": _r(pct_b * 100),  # 0 = lower band, 100 = upper band
        }

    avg_vol = df["Volume"].tail(21).iloc[:-1].mean()
    if avg_vol and avg_vol > 0:
        out["volume_vs_20bar_avg"] = _r(last["Volume"] / avg_vol)

    # Near a new high (or low) there may be no swing level on that side yet;
    # the range high/low is then the nearest meaningful level.
    out["support"] = levels["support"] or (
        [{"price": out["range_low"], "touches": 1, "note": "range low"}]
        if out["range_low"] < close else []
    )
    out["resistance"] = levels["resistance"] or (
        [{"price": out["range_high"], "touches": 1, "note": "range high"}]
        if out["range_high"] > close else []
    )
    return out
