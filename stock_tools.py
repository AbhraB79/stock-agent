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
import yfinance as yf

# Index symbols on Yahoo Finance
INDEXES = {
    "S&P 500": "^GSPC",
    "NASDAQ Composite": "^IXIC",
    "Nasdaq-100": "^NDX",
}


def _pct(change, base):
    return round(change / base * 100, 2) if base else None


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


def get_price_history(ticker: str, period: str = "1mo") -> dict:
    """Closing prices over a period, plus a short summary."""
    if period not in VALID_PERIODS:
        return {"error": f"period must be one of {VALID_PERIODS}"}
    symbol = ticker.strip().upper()
    # Daily bars for short periods, weekly for long ones (keeps the result small)
    interval = "1d" if period in ("5d", "1mo", "3mo") else "1wk"
    df = yf.Ticker(symbol).history(period=period, interval=interval)
    if df.empty:
        return {"error": f"No history found for '{symbol}'."}

    closes = df["Close"]
    start, end = float(closes.iloc[0]), float(closes.iloc[-1])
    return {
        "ticker": symbol,
        "period": period,
        "interval": interval,
        "start_close": round(start, 2),
        "end_close": round(end, 2),
        "change_pct": _pct(end - start, start),
        "high": round(float(df["High"].max()), 2),
        "low": round(float(df["Low"].min()), 2),
        # Last 30 points only — enough for Claude to describe the trend
        "closes": [
            {"date": idx.strftime("%Y-%m-%d"), "close": round(float(val), 2)}
            for idx, val in closes.tail(30).items()
        ],
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
            "high/low and % change. Use for trend or performance questions."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string", "description": "Ticker symbol, e.g. NVDA"},
                "period": {
                    "type": "string",
                    "enum": VALID_PERIODS,
                    "description": "How far back to look",
                },
            },
            "required": ["ticker"],
        },
    },
]

# Maps a tool name to the Python function that runs it
_FUNCTIONS = {
    "get_quote": get_quote,
    "get_market_overview": get_market_overview,
    "get_price_history": get_price_history,
}


def run_tool(name: str, tool_input: dict) -> str:
    """Run the tool Claude asked for and return the result as a JSON string."""
    func = _FUNCTIONS.get(name)
    if func is None:
        raise ValueError(f"Unknown tool: {name}")
    return json.dumps(func(**tool_input))
