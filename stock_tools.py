"""
stock_tools.py — the "hands" of the agent.

Claude can't browse the market by itself. Instead, we give it TOOLS:
plain Python functions plus a JSON description of each one. Claude reads
the descriptions, decides which tool to call, and we run the function
and hand the result back.

Data source: yfinance (free Yahoo Finance data, no API key needed).
Prices can be delayed ~15 minutes — fine for learning, not for trading.
"""

import json

import pandas as pd
import yfinance as yf

from technicals import add_indicators, find_levels, summarize

# Index symbols on Yahoo Finance
INDEXES = {
    "S&P 500": "^GSPC",
    "NASDAQ Composite": "^IXIC",
    "Nasdaq-100": "^NDX",
}


def _pct(change, base):
    return round(change / base * 100, 2) if base else None


def parse_date(text: str) -> pd.Timestamp:
    """Strict YYYY-MM-DD (pandas alone would read 'April' as the year 1)."""
    try:
        date = pd.to_datetime(str(text).strip(), format="%Y-%m-%d")
    except (ValueError, TypeError):
        date = pd.NaT
    if pd.isna(date):  # also catches "" which pandas turns into NaT
        raise ValueError(f"dates must be YYYY-MM-DD, got {text!r}")
    return date


# ---------------------------------------------------------------------------
# Tool 1: a single quote
# ---------------------------------------------------------------------------
def get_quote(ticker: str) -> dict:
    """Latest price and day change for one stock or index symbol."""
    symbol = ticker.strip().upper()
    info = yf.Ticker(symbol).fast_info
    price = info.last_price
    prev = info.previous_close
    if price is None:
        return {"error": f"No price data found for '{symbol}'. Check the ticker."}
    change = price - prev if prev else None
    return {
        "ticker": symbol,
        "price": round(price, 2),
        "previous_close": round(prev, 2) if prev else None,
        "change": round(change, 2) if change is not None else None,
        "change_pct": _pct(change, prev) if change is not None else None,
        "currency": info.currency,
    }


# ---------------------------------------------------------------------------
# Tool 2: market overview (the indexes)
# ---------------------------------------------------------------------------
def get_market_overview() -> dict:
    """Current level and day change for the S&P 500, NASDAQ Composite and Nasdaq-100."""
    overview = {}
    for name, symbol in INDEXES.items():
        try:
            overview[name] = get_quote(symbol)
        except Exception as exc:  # one failing index shouldn't break the rest
            overview[name] = {"error": str(exc)}
    return overview


# ---------------------------------------------------------------------------
# Tool 3: price history
# ---------------------------------------------------------------------------
VALID_PERIODS = ["5d", "1mo", "3mo", "6mo", "1y", "5y"]


def get_price_history(
    ticker: str,
    period: str = "1mo",
    start_date: str | None = None,
    end_date: str | None = None,
) -> dict:
    """Closing prices over a period (or a date window), plus a short summary."""
    symbol = ticker.strip().upper()
    if start_date:
        # A specific window, e.g. "April": always daily bars so highs/lows are exact
        try:
            start = parse_date(start_date)
            end = parse_date(end_date) if end_date else pd.Timestamp.now().normalize()
        except ValueError as exc:
            return {"error": str(exc)}
        interval = "1d"
        df = yf.Ticker(symbol).history(
            start=start, end=end + pd.Timedelta(days=1), interval=interval
        )
        period = f"{start:%Y-%m-%d} to {end:%Y-%m-%d}"
    else:
        if period not in VALID_PERIODS:
            return {"error": f"period must be one of {VALID_PERIODS}"}
        # Daily bars for short periods, weekly for long ones (keeps the result small)
        interval = "1d" if period in ("5d", "1mo", "3mo") else "1wk"
        df = yf.Ticker(symbol).history(period=period, interval=interval)
    if df.empty:
        return {"error": f"No history found for '{symbol}'."}

    closes = df["Close"]
    first, last = float(closes.iloc[0]), float(closes.iloc[-1])
    return {
        "ticker": symbol,
        "period": period,
        "interval": interval,
        "start_close": round(first, 2),
        "end_close": round(last, 2),
        "change_pct": _pct(last - first, first),
        "high": round(float(df["High"].max()), 2),
        "high_date": df["High"].idxmax().strftime("%Y-%m-%d"),
        "low": round(float(df["Low"].min()), 2),
        "low_date": df["Low"].idxmin().strftime("%Y-%m-%d"),
        # Last 30 points only — enough for Claude to describe the trend
        "closes": [
            {"date": idx.strftime("%Y-%m-%d"), "close": round(float(val), 2)}
            for idx, val in closes.tail(30).items()
        ],
    }


# ---------------------------------------------------------------------------
# Tool 4: price chart (web app only — the terminal can't draw charts)
# ---------------------------------------------------------------------------
# period -> (button label, bar size). Bar sizes keep every chart detailed but small:
# 5 daily bars would be a 5-point line, so 5d uses 30-minute bars.
CHART_PERIODS = {
    "5d": ("5D", "30m"),
    "1mo": ("1M", "1d"),
    "1y": ("1Y", "1d"),
    "5y": ("5Y", "1wk"),
    "max": ("Max", "1mo"),
}


def _fetch_bars(symbol: str, period: str, interval: str) -> pd.DataFrame:
    """OHLCV bars as a DataFrame with a Date column (exchange time, no timezone)."""
    df = yf.Ticker(symbol).history(period=period, interval=interval)
    if not df.empty:
        df = df[["Open", "High", "Low", "Close", "Volume"]].dropna(subset=["Close"])
    # Checked after dropna: rows with no Close are as good as no data
    if df.empty:
        raise ValueError(f"No price data found for '{symbol}'. Check the ticker.")
    # Drop the timezone so the chart shows exchange (New York) time, not UTC
    df.index = df.index.tz_localize(None)
    return df.rename_axis("Date").reset_index()


def get_chart_data(ticker: str, period: str) -> pd.DataFrame:
    """Price bars for a chart: Date, Open, High, Low, Close, Volume."""
    if period not in CHART_PERIODS:
        raise ValueError(f"period must be one of {list(CHART_PERIODS)}")
    _, interval = CHART_PERIODS[period]
    return _fetch_bars(ticker.strip().upper(), period, interval)


def show_price_chart(ticker: str, period: str = "1y") -> dict:
    """Tool: the app draws the chart; Claude gets a short summary to talk about."""
    try:
        data = get_chart_data(ticker, period)
    except ValueError as exc:
        return {"error": str(exc)}
    closes = data["Close"]
    start, end = float(closes.iloc[0]), float(closes.iloc[-1])
    return {
        "ticker": ticker.strip().upper(),
        "period": period,
        "chart": "displayed to the user below your reply",
        "start_date": data["Date"].iloc[0].strftime("%Y-%m-%d"),
        "start_close": round(start, 2),
        "end_close": round(end, 2),
        "change_pct": _pct(end - start, start),
        "high": round(float(closes.max()), 2),
        "low": round(float(closes.min()), 2),
    }


# ---------------------------------------------------------------------------
# Tool 5: technical analysis (web app only — the chart shows the indicators)
# ---------------------------------------------------------------------------
# Long averages (SMA200) need history from BEFORE the visible range, so fetch a
# longer period with the same bar size, calculate, then trim to the range.
_LOOKBACK = {"5d": "1mo", "1mo": "1y", "1y": "2y", "5y": "10y", "max": "max"}
_RANGE = {"1mo": pd.DateOffset(months=1), "1y": pd.DateOffset(years=1), "5y": pd.DateOffset(years=5)}


def get_analysis_data(ticker: str, period: str) -> pd.DataFrame:
    """Chart bars for `period` plus indicator columns (see technicals.py)."""
    if period not in CHART_PERIODS:
        raise ValueError(f"period must be one of {list(CHART_PERIODS)}")
    _, interval = CHART_PERIODS[period]
    df = add_indicators(_fetch_bars(ticker.strip().upper(), _LOOKBACK[period], interval))
    if period == "5d":  # the last 5 trading days
        days = df["Date"].dt.normalize()
        df = df[days >= sorted(days.unique())[-5:][0]]
    elif period in _RANGE:
        df = df[df["Date"] > df["Date"].iloc[-1] - _RANGE[period]]
    return df.reset_index(drop=True)


def get_technical_analysis(ticker: str, period: str = "1y") -> dict:
    """Tool: indicator readings and support/resistance for Claude to explain."""
    try:
        df = get_analysis_data(ticker, period)
    except ValueError as exc:
        return {"error": str(exc)}
    label, interval = CHART_PERIODS[period]
    return {
        "ticker": ticker.strip().upper(),
        "period": period,
        "bar_size": interval,  # SMA20 = 20 of these bars
        "chart": "shown with indicators below your reply",
        **summarize(df, find_levels(df)),
    }


# ---------------------------------------------------------------------------
# Tool definitions Claude sees (JSON Schema). Good descriptions = good tool use.
# ---------------------------------------------------------------------------
TOOLS = [
    {
        "name": "get_quote",
        "description": (
            "Get the latest price and today's change for one US stock or index. "
            "Use a ticker like AAPL, MSFT, NVDA, or an index symbol like ^GSPC."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string", "description": "Ticker symbol, e.g. AAPL"}
            },
            "required": ["ticker"],
        },
    },
    {
        "name": "get_market_overview",
        "description": (
            "Get today's level and change for the S&P 500, NASDAQ Composite and "
            "Nasdaq-100. Use when the user asks how the market is doing."
        ),
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "get_price_history",
        "description": (
            "Get closing prices for a stock or index over a period, with start/end, "
            "high/low (and the dates they happened) and % change. Use for trend or "
            "performance questions. For a specific window such as a month (e.g. to "
            "find 'the April low'), pass start_date/end_date to get daily prices."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string", "description": "Ticker symbol, e.g. NVDA"},
                "period": {
                    "type": "string",
                    "enum": VALID_PERIODS,
                    "description": "How far back to look (ignored if start_date is given)",
                },
                "start_date": {
                    "type": "string",
                    "description": "Optional window start, YYYY-MM-DD",
                },
                "end_date": {
                    "type": "string",
                    "description": "Optional window end, YYYY-MM-DD (default today)",
                },
            },
            "required": ["ticker"],
        },
    },
]

# Only the web app (app.py) offers this tool, because only it can draw charts
CHART_TOOL = {
    "name": "show_price_chart",
    "description": (
        "Show the user a price chart for a stock or index. Use whenever the user "
        "asks for a chart, graph or plot. The chart appears below your reply and "
        "the user can switch between 5 day, 1 month, 1 year, 5 year and max ranges. "
        "Returns a short summary of the chosen range."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "ticker": {"type": "string", "description": "Ticker symbol, e.g. AAPL or ^GSPC"},
            "period": {
                "type": "string",
                "enum": list(CHART_PERIODS),
                "description": "Range to show first (default 1y)",
            },
        },
        "required": ["ticker"],
    },
}

ANALYSIS_TOOL = {
    "name": "get_technical_analysis",
    "description": (
        "Technical analysis of a stock or index: SMA20/50/200, golden/death cross, "
        "RSI14, MACD 12/26/9, Bollinger bands, support/resistance levels and volume. "
        "Use when the user asks for technical analysis, indicators, trend, momentum, "
        "overbought/oversold, or support/resistance. The chart with these indicators "
        "is shown below your reply."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "ticker": {"type": "string", "description": "Ticker symbol, e.g. AAPL"},
            "period": {
                "type": "string",
                "enum": list(CHART_PERIODS),
                "description": "Range to analyse (default 1y; bars: 5d=30min, 1mo/1y=daily, 5y=weekly, max=monthly)",
            },
        },
        "required": ["ticker"],
    },
}

_LINE_SCHEMA = {
    "type": "object",
    "properties": {
        "type": {"type": "string", "enum": ["horizontal", "trendline"]},
        "label": {"type": "string", "description": "Short name, e.g. 'support' or 'uptrend'"},
        "price": {"type": "number", "description": "horizontal only: the price level"},
        "start_date": {"type": "string", "description": "trendline only: YYYY-MM-DD"},
        "start_price": {
            "type": "number",
            "description": "trendline only, optional: omit to use that day's close",
        },
        "end_date": {"type": "string", "description": "trendline only: YYYY-MM-DD"},
        "end_price": {
            "type": "number",
            "description": "trendline only, optional: omit to use that day's close",
        },
    },
    "required": ["type"],
}

DRAW_LINES_TOOL = {
    "name": "draw_chart_lines",
    "description": (
        "Save the user's own lines on a ticker's chart and show the chart: horizontal "
        "lines (support/resistance levels) and trendlines between two dates. Lines stay "
        "on every chart of that ticker until cleared. Returns ALL saved lines for the "
        "ticker with where the price is relative to each, so you can discuss them. "
        "To find a date like 'the April low', call get_price_history first. "
        "Call with an empty lines list to just review the saved lines."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "ticker": {"type": "string", "description": "Ticker symbol, e.g. MSFT"},
            "lines": {"type": "array", "items": _LINE_SCHEMA},
            "period": {
                "type": "string",
                "enum": list(CHART_PERIODS),
                "description": "Chart range to show (default 1y)",
            },
        },
        "required": ["ticker", "lines"],
    },
}

CLEAR_LINES_TOOL = {
    "name": "clear_chart_lines",
    "description": "Remove all of the user's saved lines from a ticker's charts.",
    "input_schema": {
        "type": "object",
        "properties": {"ticker": {"type": "string", "description": "Ticker symbol"}},
        "required": ["ticker"],
    },
}


def close_on(ticker: str, date: str) -> tuple[str, float]:
    """The closing price on `date` (YYYY-MM-DD), or the nearest trading day's."""
    target = parse_date(date)
    df = yf.Ticker(ticker.strip().upper()).history(
        start=target - pd.Timedelta(days=7), end=target + pd.Timedelta(days=8), interval="1d"
    )
    if df.empty:
        raise ValueError(f"No price data for {ticker.upper()} around {date}.")
    days = df.index.tz_localize(None).normalize()
    i = int(abs(days - target).argmin())
    return days[i].strftime("%Y-%m-%d"), round(float(df["Close"].iloc[i]), 2)


# Maps a tool name to the Python function that runs it. The two line tools are
# not here: they change the web page's state, so app.py runs them itself.
_FUNCTIONS = {
    "get_quote": get_quote,
    "get_market_overview": get_market_overview,
    "get_price_history": get_price_history,
    "show_price_chart": show_price_chart,
    "get_technical_analysis": get_technical_analysis,
}


def run_tool(name: str, tool_input: dict) -> str:
    """Run the tool Claude asked for and return the result as a JSON string."""
    func = _FUNCTIONS.get(name)
    if func is None:
        raise ValueError(f"Unknown tool: {name}")
    return json.dumps(func(**tool_input))
