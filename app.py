"""
app.py — a web chatbot for the stock agent, built with Streamlit.

Run it:   streamlit run app.py
It opens in your browser at http://localhost:8501  (Ctrl+C in the terminal stops it)

How Streamlit works (the one idea to understand):
  Every time you click or send a message, Streamlit re-runs this WHOLE file
  from top to bottom. Anything that must survive a re-run (like the chat
  history) is kept in st.session_state, a dictionary that persists.
"""

import json
import os
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st
from anthropic import Anthropic
from dotenv import load_dotenv

from charts import DRAW_CONFIG, build_figure
from stock_tools import (
    ANALYSIS_TOOL,
    CHART_PERIODS,
    CHART_TOOL,
    CLEAR_LINES_TOOL,
    DRAW_LINES_TOOL,
    TOOLS,
    close_on,
    get_analysis_data,
    get_chart_data,
    get_market_overview,
    get_quote,
    parse_date,
    run_tool,
)
from technicals import describe_lines, find_levels

# The web app can draw charts and lines; the terminal agent (agent.py) can't
APP_TOOLS = TOOLS + [CHART_TOOL, ANALYSIS_TOOL, DRAW_LINES_TOOL, CLEAR_LINES_TOOL]

load_dotenv()  # on your PC: reads ANTHROPIC_API_KEY (and optional CLAUDE_MODEL) from .env


def get_setting(name: str, default=None):
    """Read a setting from .env (your PC) or from Streamlit Secrets (the cloud)."""
    value = os.getenv(name)
    if value:
        return value
    try:
        return st.secrets.get(name, default)
    except Exception:  # no secrets file on your PC — that's fine
        return default


API_KEY = get_setting("ANTHROPIC_API_KEY")
MODEL = get_setting("CLAUDE_MODEL", "claude-sonnet-5-5")

# Cloud servers run on UTC time, so ask for New York time explicitly
NY = ZoneInfo("America/New_York")

# The system prompt must stay EXACTLY the same for the whole chat — Claude's
# earlier replies are tied to it. So no clock time here: the current time is
# added to each question instead (see time_stamped below).
SYSTEM_PROMPT = """You are a helpful US stock market assistant in a web chat.
Each user message starts with the current US Eastern time in [brackets].
US markets are open 9:30 AM-4:00 PM ET on weekdays; after 4 PM the latest price is the closing price.
- Always use your tools to get prices; never guess numbers from memory.
- Report prices with the change and % change, and say data may be ~15 min delayed.
- Keep answers short and clear. Markdown tables and bold are fine.
- When the user asks for a chart, graph or plot, call show_price_chart. The chart
  appears below your reply, so just add a 1-2 sentence summary; don't list prices.
- For technical analysis (indicators, trend, momentum, RSI, MACD, moving averages,
  overbought/oversold, support/resistance) call get_technical_analysis. Explain each
  signal in plain words (what it is, what the reading suggests), note where signals
  disagree, and finish with a one-line overall read. Indicators describe the past;
  they don't predict the future.
- When the user wants to draw or mark lines (support, resistance, trendlines,
  price levels), call draw_chart_lines; to remove them, clear_chart_lines. For a
  trendline from a described point (e.g. "the April low"), first find the date
  with get_price_history. To discuss the user's own lines, call draw_chart_lines
  with an empty list: it returns where the price is versus each saved line.
- You give information, not investment advice."""

EXAMPLES = {
    "Market & prices": [
        "How is the market doing today?",
        "What's NVDA trading at?",
        "Show me a chart of NVDA",
        "Compare AAPL and MSFT over the last 3 months",
    ],
    "Agent technical analysis": [
        "Do a technical analysis of AAPL",
        "Is NVDA overbought? Check RSI and MACD",
        "What are the support and resistance levels for TSLA?",
        "Analyse the S&P 500 trend over 5 years",
    ],
    "Your own analysis": [
        "Show me a chart of MSFT with indicators",
        "Draw a support line at 400 on MSFT",
        "Draw a trendline on AAPL from its April low to today",
        "Is AAPL above my trendline?",
        "Clear my lines on MSFT",
    ],
}

# Which tools put a chart under the reply, and whether it shows indicators
CHART_TOOLS = {
    "show_price_chart": False,
    "draw_chart_lines": False,
    "get_technical_analysis": True,
}

st.set_page_config(page_title="Stock Agent", page_icon="📈", layout="wide")


# ---------------------------------------------------------------------------
# Setup helpers
# ---------------------------------------------------------------------------
@st.cache_resource  # create the API client once, not on every re-run
def get_client():
    return Anthropic(api_key=API_KEY)


@st.cache_data(ttl=60)  # re-fetch index prices at most once a minute
def cached_overview():
    return get_market_overview()


@st.cache_data(ttl=300)  # switching ranges or replaying the chat reuses the data
def cached_chart(ticker: str, period: str):
    return get_chart_data(ticker, period)


@st.cache_data(ttl=300)
def cached_analysis(ticker: str, period: str):
    return get_analysis_data(ticker, period)


def time_stamped(question: str) -> str:
    """Put the current New York time in front of the question Claude receives."""
    return f"[Current time: {datetime.now(NY):%A, %B %d, %Y, %I:%M %p} ET]\n{question}"


def show(text: str):
    """Render Claude's text. Escape $ because Streamlit treats $...$ as math."""
    st.markdown(text.replace("$", r"\$"))


# ---------------------------------------------------------------------------
# The user's saved lines (kept in session_state so they survive re-runs)
# ---------------------------------------------------------------------------
def _clean_line(ticker: str, raw: dict) -> dict:
    """Validate one line from Claude; fill in trendline prices from closes."""
    kind = raw.get("type")
    label = str(raw.get("label") or "").strip()[:40]
    if kind == "horizontal":
        price = raw.get("price")
        if not isinstance(price, (int, float)) or price <= 0:
            raise ValueError(f"horizontal line needs a positive price, got {price!r}")
        return {"type": "horizontal", "price": round(float(price), 2), "label": label or "level"}
    if kind == "trendline":
        points = []
        for end in ("start", "end"):
            date = raw.get(f"{end}_date")
            if not date:
                raise ValueError(f"trendline needs {end}_date (YYYY-MM-DD)")
            price = raw.get(f"{end}_price")
            if price is None:  # no price given: use that day's close
                date, price = close_on(ticker, date)
            else:
                date = parse_date(date).strftime("%Y-%m-%d")
            points.append((date, round(float(price), 2)))
        points.sort()  # make start the earlier date
        (d1, p1), (d2, p2) = points
        return {
            "type": "trendline", "label": label or "trendline",
            "start_date": d1, "start_price": p1, "end_date": d2, "end_price": p2,
        }
    raise ValueError(f"line type must be 'horizontal' or 'trendline', got {kind!r}")


def handle_line_tool(name: str, tool_input: dict) -> dict:
    """Run draw_chart_lines / clear_chart_lines, which change this page's state."""
    ticker = tool_input["ticker"].strip().upper()
    saved = st.session_state.user_lines

    if name == "clear_chart_lines":
        removed = len(saved.pop(ticker, []))
        return {"ticker": ticker, "removed_lines": removed}

    errors = []
    for raw in tool_input.get("lines") or []:
        try:
            saved.setdefault(ticker, []).append(_clean_line(ticker, raw))
        except Exception as exc:
            errors.append({"line": raw, "error": str(exc)})

    quote = get_quote(ticker)
    if "error" in quote:
        return quote
    # Compare at today's date (trendline points are whole days), not the clock time
    now = pd.Timestamp(datetime.now(NY)).tz_localize(None).normalize()
    result = {
        "ticker": ticker,
        "last_close": quote["price"],
        "saved_lines": describe_lines(saved.get(ticker, []), quote["price"], now),
        "chart": "shown below your reply with the saved lines",
    }
    if errors:
        result["errors"] = errors
    return result


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------
def render_chart(spec: dict, key: str):
    """Price chart with range buttons, an Indicators toggle and drawing tools."""
    ticker = spec["ticker"].strip().upper()
    left, right = st.columns([4, 1], vertical_alignment="center")
    with left:
        period = st.segmented_control(
            f"{ticker} range",
            options=list(CHART_PERIODS),
            format_func=lambda p: CHART_PERIODS[p][0],
            default=spec["period"],
            key=key,
            label_visibility="collapsed",
        ) or spec["period"]  # None if the user clicks the selected range again
    with right:
        indicators = st.toggle(
            "Indicators", value=spec.get("analysis", False), key=f"{key}-ind",
            help="Moving averages, Bollinger band, support/resistance, RSI and MACD",
        )

    try:
        data = cached_analysis(ticker, period) if indicators else cached_chart(ticker, period)
    except Exception as exc:
        st.warning(f"Couldn't load the {ticker} chart: {exc}")
        return

    start, end = data["Close"].iloc[0], data["Close"].iloc[-1]
    change = (end - start) / start * 100 if start else 0.0
    color = "green" if change >= 0 else "red"
    st.markdown(
        f"**{ticker}** · {CHART_PERIODS[period][0]} · "
        f":{color}[{change:+.2f}%] · last close {end:,.2f}"
    )
    fig = build_figure(
        data, ticker, period,
        indicators=indicators,
        levels=find_levels(data) if indicators else None,
        user_lines=st.session_state.user_lines.get(ticker, []),
    )
    st.plotly_chart(fig, config=DRAW_CONFIG, key=f"{key}-fig")
    st.caption(
        "✏️ Drag on the chart to draw a line; more tools (freehand, box, circle, "
        "eraser) are in the chart's toolbar. Mouse drawings are temporary. Ask me "
        "to draw lines you want to keep."
    )


def show_message(msg: dict, index: int):
    """Render one chat message: its text, then any charts Claude asked for."""
    show(msg["content"])
    for i, spec in enumerate(msg.get("charts", [])):
        render_chart(spec, key=f"chart-{index}-{i}")


def add_chart(charts: list, ticker: str, period: str, analysis: bool):
    """One chart per ticker per answer; an analysis turns the indicators on."""
    ticker = ticker.strip().upper()
    for spec in charts:
        if spec["ticker"] == ticker:
            spec["analysis"] = spec["analysis"] or analysis
            if analysis:
                spec["period"] = period
            return
    charts.append({"ticker": ticker, "period": period, "analysis": analysis})


# ---------------------------------------------------------------------------
# The agent loop — same idea as agent.py, but it reports tool calls to the page
# ---------------------------------------------------------------------------
def ask_claude(api_messages: list, status) -> tuple[str, list[dict]]:
    """Run the agent loop. Returns Claude's answer and the charts it asked for."""
    client = get_client()
    charts = []
    while True:
        response = client.messages.create(
            model=MODEL,
            max_tokens=2000,
            system=SYSTEM_PROMPT,
            tools=APP_TOOLS,
            messages=api_messages,
        )
        api_messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason != "tool_use":
            answer = "".join(b.text for b in response.content if b.type == "text")
            return answer, charts

        tool_results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            status.write(f"🔧 `{block.name}` {block.input or ''}")
            try:
                if block.name in ("draw_chart_lines", "clear_chart_lines"):
                    result = json.dumps(handle_line_tool(block.name, block.input))
                else:
                    result = run_tool(block.name, block.input)
                tool_results.append(
                    {"type": "tool_result", "tool_use_id": block.id, "content": result}
                )
                if block.name in CHART_TOOLS and "error" not in json.loads(result):
                    add_chart(
                        charts,
                        block.input["ticker"],
                        block.input.get("period", "1y"),
                        CHART_TOOLS[block.name],
                    )
            except Exception as exc:
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": f"Error: {exc}",
                        "is_error": True,
                    }
                )
        api_messages.append({"role": "user", "content": tool_results})


# ---------------------------------------------------------------------------
# Memory that survives re-runs
# ---------------------------------------------------------------------------
if "chat" not in st.session_state:
    st.session_state.chat = []          # what we SHOW: {"role", "content", "charts"?}
if "api_messages" not in st.session_state:
    st.session_state.api_messages = []  # what we SEND to Claude, incl. tool calls
if "user_lines" not in st.session_state:
    st.session_state.user_lines = {}    # {ticker: [line, ...]} drawn via chat


# ---------------------------------------------------------------------------
# Sidebar: live index prices, example questions, saved lines, clear button
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("📈 Market now")
    try:
        for name, q in cached_overview().items():
            if "error" in q or q.get("price") is None:
                st.metric(name, "n/a")
            else:
                st.metric(
                    name,
                    f"{q['price']:,.2f}",
                    f"{q['change']:+,.2f} ({q['change_pct']:+.2f}%)",
                )
    except Exception as exc:
        st.warning(f"Couldn't load prices: {exc}")
    st.caption("Yahoo Finance data, may be ~15 min delayed. Refreshes every minute.")

    st.subheader("Try asking")
    for group, examples in EXAMPLES.items():
        st.caption(group)
        for example in examples:
            if st.button(example, use_container_width=True):
                st.session_state.pending = example  # picked up below

    with st.expander("✏️ How to draw on charts"):
        st.markdown(
            "**With the mouse:** drag on any chart to draw a line. The chart's "
            "toolbar (top-right, on hover) also has freehand, rectangle, circle and "
            "eraser tools. These drawings are temporary and disappear when the "
            "page updates.\n\n"
            "**By asking:** lines you ask for are saved, appear on every chart of "
            "that stock, and the agent can check the price against them:\n"
            "- *Draw a resistance line at 250 on TSLA*\n"
            "- *Draw a trendline on NVDA from 2026-04-07 to 2026-08-01*\n"
            "- *How far is MSFT from my support line?*\n"
            "- *Clear my lines on TSLA*\n\n"
            "Turn on **Indicators** above a chart to see moving averages, "
            "Bollinger band, support/resistance, RSI and MACD."
        )

    if st.session_state.user_lines:
        st.subheader("Your saved lines")
        for ticker, lines in st.session_state.user_lines.items():
            for line in lines:
                if line["type"] == "horizontal":
                    st.caption(f"**{ticker}** · {line['label']} at {line['price']:,.2f}")
                else:
                    st.caption(
                        f"**{ticker}** · {line['label']}: {line['start_date']} "
                        f"({line['start_price']:,.2f}) → {line['end_date']} ({line['end_price']:,.2f})"
                    )

    st.divider()
    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.chat = []
        st.session_state.api_messages = []
        st.session_state.user_lines = {}
        st.rerun()


# ---------------------------------------------------------------------------
# Main chat area
# ---------------------------------------------------------------------------
st.title("Stock Agent")
st.caption(f"Ask about S&P 500 and NASDAQ stocks · model: {MODEL} · not investment advice")

if not API_KEY:
    st.error("ANTHROPIC_API_KEY not found. Add it to .env (your PC) or to Secrets (Streamlit Cloud).")
    st.stop()

# Replay the conversation so far (Streamlit redraws everything on each re-run)
for i, msg in enumerate(st.session_state.chat):
    with st.chat_message(msg["role"]):
        show_message(msg, i)

# A new question comes from the input box OR an example button
prompt = st.chat_input("Ask about a stock or the market…")
if "pending" in st.session_state:
    prompt = st.session_state.pop("pending")

if prompt:
    st.session_state.chat.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        show(prompt)

    api = st.session_state.api_messages
    start = len(api)  # remember where this turn began, in case it fails
    api.append({"role": "user", "content": time_stamped(prompt)})  # Claude sees the time; you don't

    with st.chat_message("assistant"):
        with st.status("Checking the market…") as status:
            try:
                answer, charts = ask_claude(api, status)
                status.update(label="Done", state="complete")
            except Exception as exc:
                del api[start:]  # drop the failed turn so the next one starts clean
                answer, charts = f"Sorry, something went wrong: `{exc}`", []
                status.update(label="Error", state="error")
        reply = {"role": "assistant", "content": answer, "charts": charts}
        # Same index (and so the same widget keys) it will have when replayed
        show_message(reply, len(st.session_state.chat))

    st.session_state.chat.append(reply)
