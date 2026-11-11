"""
app.py — a web chatbot for the stock agent, built with Streamlit.

Run it:   streamlit run app.py
It opens in your browser at http://localhost:8501  (Ctrl+C in the terminal stops it)

How Streamlit works (the one idea to understand):
  Every time you click or send a message, Streamlit re-runs this WHOLE file
  from top to bottom. Anything that must survive a re-run (like the chat
  history) is kept in st.session_state, a dictionary that persists.
"""

import os
from datetime import datetime

import streamlit as st
from anthropic import Anthropic
from dotenv import load_dotenv

from stock_tools import TOOLS, get_market_overview, run_tool

load_dotenv()  # reads ANTHROPIC_API_KEY (and optional CLAUDE_MODEL) from .env
MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-5-5")

# In the browser we CAN show tables and bold text, so this prompt allows markdown
SYSTEM_PROMPT = f"""You are a helpful US stock market assistant in a web chat.
Right now it is {datetime.now():%A, %B %d, %Y, %I:%M %p} US Eastern time.
US markets are open 9:30 AM-4:00 PM ET on weekdays; after 4 PM the latest price is the closing price.
- Always use your tools to get prices; never guess numbers from memory.
- Report prices with the change and % change, and say data may be ~15 min delayed.
- Keep answers short and clear. Markdown tables and bold are fine.
- You give information, not investment advice."""

EXAMPLES = [
    "How is the market doing today?",
    "What's NVDA trading at?",
    "Compare AAPL and MSFT over the last 3 months",
    "How has the Nasdaq-100 done this year?",
]

st.set_page_config(page_title="Stock Agent", page_icon="📈", layout="wide")


# ---------------------------------------------------------------------------
# Setup helpers
# ---------------------------------------------------------------------------
@st.cache_resource  # create the API client once, not on every re-run
def get_client():
    return Anthropic()


@st.cache_data(ttl=60)  # re-fetch index prices at most once a minute
def cached_overview():
    return get_market_overview()


def show(text: str):
    """Render Claude's text. Escape $ because Streamlit treats $...$ as math."""
    st.markdown(text.replace("$", r"\$"))


# ---------------------------------------------------------------------------
# The agent loop — same idea as agent.py, but it reports tool calls to the page
# ---------------------------------------------------------------------------
def ask_claude(api_messages: list, status) -> str:
    client = get_client()
    while True:
        response = client.messages.create(
            model=MODEL,
            max_tokens=1500,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=api_messages,
        )
        api_messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason != "tool_use":
            return "".join(b.text for b in response.content if b.type == "text")

        tool_results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            status.write(f"🔧 `{block.name}` {block.input or ''}")
            try:
                result = run_tool(block.name, block.input)
                tool_results.append(
                    {"type": "tool_result", "tool_use_id": block.id, "content": result}
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
    st.session_state.chat = []          # what we SHOW: {"role", "content"} text only
if "api_messages" not in st.session_state:
    st.session_state.api_messages = []  # what we SEND to Claude, incl. tool calls


# ---------------------------------------------------------------------------
# Sidebar: live index prices, example questions, clear button
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
    for example in EXAMPLES:
        if st.button(example, use_container_width=True):
            st.session_state.pending = example  # picked up below

    st.divider()
    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.chat = []
        st.session_state.api_messages = []
        st.rerun()


# ---------------------------------------------------------------------------
# Main chat area
# ---------------------------------------------------------------------------
st.title("Stock Agent")
st.caption(f"Ask about S&P 500 and NASDAQ stocks · model: {MODEL} · not investment advice")

if not os.getenv("ANTHROPIC_API_KEY"):
    st.error("ANTHROPIC_API_KEY not found. Check that your .env file is in this folder.")
    st.stop()

# Replay the conversation so far (Streamlit redraws everything on each re-run)
for msg in st.session_state.chat:
    with st.chat_message(msg["role"]):
        show(msg["content"])

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
    api.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.status("Checking the market…") as status:
            try:
                answer = ask_claude(api, status)
                status.update(label="Done", state="complete")
            except Exception as exc:
                del api[start:]  # drop the failed turn so the next one starts clean
                answer = f"Sorry, something went wrong: `{exc}`"
                status.update(label="Error", state="error")
        show(answer)

    st.session_state.chat.append({"role": "assistant", "content": answer})