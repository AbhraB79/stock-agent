# Stock Agent

A command-line AI agent built on the Claude API that tracks US equity prices
(S&P 500, NASDAQ and individual stocks) using tool calling.

## Setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    Mac/Linux: source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env    # then paste your API key into .env
python agent.py
```

## Try asking
- How is the market doing today?
- What's NVDA trading at?
- Compare AAPL and MSFT over the last 3 months.

## How it works
- `stock_tools.py` — Python functions (quote, market overview, history) + tool schemas
- `agent.py` — the agent loop: Claude picks tools, we run them, Claude answers
- `app.py` — the same agent as a web chat (`streamlit run app.py`), with a sidebar of
  index prices and price charts (5D / 1M / 1Y / 5Y / Max) when you ask for one,
  e.g. "Show me a chart of NVDA"
  - Technical analysis: "Do a technical analysis of AAPL" (SMA 20/50/200, RSI, MACD,
    Bollinger bands, support/resistance), shown on the chart and explained
  - Your own analysis: draw on charts with the mouse (temporary), or ask
    "Draw a support line at 400 on MSFT" to save lines the agent can discuss
- `technicals.py` — indicator maths (pandas); `charts.py` — builds the Plotly chart

Data from Yahoo Finance via `yfinance`; may be delayed. Not investment advice.
