"""
charts.py — builds the Plotly price chart (no Streamlit code, so it's testable).

The figure has:
  - candlesticks (a line for the Max range, where monthly candles are hard to read)
  - optional indicators: SMA20/50/200, Bollinger band, support/resistance,
    plus RSI and MACD panels underneath
  - the user's saved lines in purple
  - a drawing toolbar so the user can sketch their own lines with the mouse
"""

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from technicals import line_value_at

USER_COLOR = "#8e44ad"  # purple: the user's own lines and mouse drawings
SMA_COLORS = {"SMA20": "#f39c12", "SMA50": "#2980b9", "SMA200": "#7f8c8d"}

# Passed to st.plotly_chart: adds drawing buttons to the chart's toolbar
DRAW_CONFIG = {
    "modeBarButtonsToAdd": ["drawline", "drawopenpath", "drawrect", "drawcircle", "eraseshape"],
    "displaylogo": False,
    "scrollZoom": False,
}


def build_figure(
    data: pd.DataFrame,
    ticker: str,
    period: str,
    indicators: bool = False,
    levels: dict | None = None,
    user_lines: list[dict] | None = None,
) -> go.Figure:
    """Price chart for one ticker and range. `data` needs indicator columns if indicators=True."""
    rows = 3 if indicators else 1
    fig = make_subplots(
        rows=rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.6, 0.2, 0.2] if indicators else [1.0],
        subplot_titles=(None, "RSI 14", "MACD 12/26/9") if indicators else None,
    )
    dates = data["Date"]

    # --- price ------------------------------------------------------------------
    if period == "max":
        fig.add_trace(
            go.Scatter(x=dates, y=data["Close"], name=ticker, line=dict(color="#2c3e50")),
            row=1, col=1,
        )
    else:
        fig.add_trace(
            go.Candlestick(
                x=dates, open=data["Open"], high=data["High"], low=data["Low"],
                close=data["Close"], name=ticker,
            ),
            row=1, col=1,
        )

    # --- agent indicators -------------------------------------------------------
    if indicators:
        _add_indicators(fig, data)
        for level in (levels or {}).get("support", []):
            _hline(fig, level["price"], "#27ae60", "dash", f"support {level['price']:,.2f}")
        for level in (levels or {}).get("resistance", []):
            _hline(fig, level["price"], "#c0392b", "dash", f"resistance {level['price']:,.2f}")

    # --- the user's saved lines -------------------------------------------------
    for line in user_lines or []:
        _add_user_line(fig, line, dates.iloc[0], dates.iloc[-1])

    # --- layout -----------------------------------------------------------------
    breaks = []
    if period in ("5d", "1mo", "1y"):
        breaks.append(dict(bounds=["sat", "mon"]))  # hide weekends
    if period == "5d":
        breaks.append(dict(bounds=[16, 9.5], pattern="hour"))  # hide nights
    fig.update_xaxes(rangebreaks=breaks, range=[dates.iloc[0], dates.iloc[-1]])
    fig.update_layout(
        height=640 if indicators else 420,
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
        legend=dict(orientation="h", y=1.04, x=0, yanchor="bottom"),
        dragmode="drawline",  # dragging on the chart draws a line straight away
        newshape=dict(line=dict(color=USER_COLOR, width=2)),
    )
    return fig


def _hline(fig, price, color, dash, text):
    fig.add_hline(
        y=price, line=dict(color=color, width=1, dash=dash),
        annotation_text=text, annotation_position="top left",
        annotation_font=dict(color=color, size=11), row=1, col=1,
    )


def _add_indicators(fig: go.Figure, data: pd.DataFrame):
    dates = data["Date"]
    # Bollinger band: upper line, then lower line filled up to it
    fig.add_trace(
        go.Scatter(x=dates, y=data["BB_upper"], line=dict(width=0), hoverinfo="skip",
                   showlegend=False),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(x=dates, y=data["BB_lower"], line=dict(width=0), fill="tonexty",
                   fillcolor="rgba(52,152,219,0.10)", name="Bollinger 20,2", hoverinfo="skip"),
        row=1, col=1,
    )
    for name, color in SMA_COLORS.items():
        if data[name].notna().any():
            fig.add_trace(
                go.Scatter(x=dates, y=data[name], name=name, line=dict(color=color, width=1.3)),
                row=1, col=1,
            )

    fig.add_trace(
        go.Scatter(x=dates, y=data["RSI14"], name="RSI14", line=dict(color="#8e44ad", width=1.3),
                   showlegend=False),
        row=2, col=1,
    )
    for y in (70, 30):
        fig.add_hline(y=y, line=dict(color="#95a5a6", width=1, dash="dot"), row=2, col=1)
    fig.update_yaxes(range=[0, 100], row=2, col=1)

    colors = ["#27ae60" if v >= 0 else "#c0392b" for v in data["MACD_hist"].fillna(0)]
    fig.add_trace(
        go.Bar(x=dates, y=data["MACD_hist"], marker_color=colors, name="MACD hist",
               showlegend=False),
        row=3, col=1,
    )
    fig.add_trace(
        go.Scatter(x=dates, y=data["MACD"], name="MACD", line=dict(color="#2980b9", width=1.3),
                   showlegend=False),
        row=3, col=1,
    )
    fig.add_trace(
        go.Scatter(x=dates, y=data["MACD_signal"], name="Signal",
                   line=dict(color="#f39c12", width=1.3), showlegend=False),
        row=3, col=1,
    )


def _add_user_line(fig: go.Figure, line: dict, first: pd.Timestamp, last: pd.Timestamp):
    label = line.get("label") or ("level" if line["type"] == "horizontal" else "trendline")
    if line["type"] == "horizontal":
        _hline(fig, line["price"], USER_COLOR, "solid", f"{label} {line['price']:,.2f}")
        return

    start, end = pd.Timestamp(line["start_date"]), pd.Timestamp(line["end_date"])
    if start > last:
        return  # the line starts after this chart ends
    # Solid between the two chosen points, dotted where it's extended to today.
    # Clip to the visible range so old trendlines don't squash the chart.
    x0 = max(start, first)
    x1 = min(max(end, x0), last)
    fig.add_trace(
        go.Scatter(
            x=[x0, x1], y=[line_value_at(line, x0), line_value_at(line, x1)],
            mode="lines", name=label, line=dict(color=USER_COLOR, width=2),
        ),
        row=1, col=1,
    )
    if x1 < last:
        fig.add_trace(
            go.Scatter(
                x=[x1, last], y=[line_value_at(line, x1), line_value_at(line, last)],
                mode="lines", showlegend=False, hoverinfo="skip",
                line=dict(color=USER_COLOR, width=2, dash="dot"),
            ),
            row=1, col=1,
        )
