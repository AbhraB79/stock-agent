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

Data from Yahoo Finance via `yfinance`; may be delayed. Not investment advice.
