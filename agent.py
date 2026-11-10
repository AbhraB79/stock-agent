"""
agent.py — the "brain" loop.

How an agent works (the whole idea in 4 steps):
  1. Send the conversation + the list of tools to Claude.
  2. If Claude replies with stop_reason == "tool_use", it wants a tool run.
  3. We run the tool(s) and send the results back as a "tool_result".
  4. Repeat until Claude gives a normal text answer.

Run it:  python agent.py
"""

import os
from datetime import datetime

from anthropic import Anthropic
from dotenv import load_dotenv

from stock_tools import TOOLS, run_tool

load_dotenv()  # reads ANTHROPIC_API_KEY (and optional CLAUDE_MODEL) from .env

MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-5-5")
client = Anthropic()  # picks up ANTHROPIC_API_KEY automatically

SYSTEM_PROMPT = f"""You are a helpful US stock market assistant.
Today is {datetime.now():%A, %B %d, %Y}.
- Always use your tools to get prices; never guess numbers from memory.
- Report prices with the change and % change, and say data may be ~15 min delayed.
- Keep answers short and clear. You give information, not investment advice.
- Output plain text only (no markdown tables or ** bold), since this runs in a terminal."""

def ask_claude(messages: list) -> str:
    """Run the agent loop until Claude produces a final text answer."""
    while True:
        response = client.messages.create(
            model=MODEL,
            max_tokens=1500,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=messages,
        )
        # Always keep Claude's full reply (including tool requests) in history
        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason != "tool_use":
            # Final answer: join the text blocks
            return "".join(b.text for b in response.content if b.type == "text")

        # Claude asked for one or more tools — run each one
        tool_results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            print(f"  [tool] {block.name}({block.input})")
            try:
                result = run_tool(block.name, block.input)
                tool_results.append(
                    {"type": "tool_result", "tool_use_id": block.id, "content": result}
                )
            except Exception as exc:
                # Tell Claude the tool failed so it can recover or explain
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": f"Error: {exc}",
                        "is_error": True,
                    }
                )
        # Tool results go back to Claude as a "user" message
        messages.append({"role": "user", "content": tool_results})


def main():
    print(f"Stock Agent ({MODEL}) — ask about S&P 500 / NASDAQ stocks. Type 'quit' to exit.\n")
    messages = []
    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if question.lower() in ("quit", "exit", "q"):
            break
        if not question:
            continue
        messages.append({"role": "user", "content": question})
        answer = ask_claude(messages)
        print(f"\nAgent: {answer}\n")
    print("Bye!")


if __name__ == "__main__":
    main()
