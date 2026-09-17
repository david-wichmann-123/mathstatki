"""Interaktive App zu Folgen und Partialsummen."""

import re
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st
import streamlit.components.v1 as components
from plotly.subplots import make_subplots
from sympy import E, Float, Function, Integer, Rational, Symbol, cos, exp, lambdify, latex, log, pi, sin, sqrt, tan
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)

_APP_DIR = Path(__file__).resolve().parent
_LOGO_PATH = _APP_DIR / "logo.png"

st.set_page_config(
    page_title="Folgen und Reihen",
    page_icon=str(_LOGO_PATH) if _LOGO_PATH.is_file() else "📈",
    layout="wide",
)

N_INDEX = Symbol("n")
A_NM1 = Symbol("a_nm1")
A_NM2 = Symbol("a_nm2")
A_NM3 = Symbol("a_nm3")
TRANSFORMATIONS = standard_transformations + (convert_xor, implicit_multiplication_application)
N_TERMS = 100
N_X_MIN = 0
N_X_MAX = 100

TOKEN_A_NM1 = "a_{n-1}"
TOKEN_A_NM2 = "a_{n-2}"
TOKEN_A_NM3 = "a_{n-3}"
TOKEN_SQRT = "√("


def normalize_recurrence_tokens(formula: str) -> str:
    text = formula
    text = re.sub(r"a_\{n-1\}", "a_nm1", text)
    text = re.sub(r"a_\{n-2\}", "a_nm2", text)
    text = re.sub(r"a_\{n-3\}", "a_nm3", text)
    text = re.sub(r"a\(n-1\)", "a_nm1", text)
    text = re.sub(r"a\(n-2\)", "a_nm2", text)
    text = re.sub(r"a\(n-3\)", "a_nm3", text)
    return text


def preprocess_for_parse(formula: str) -> str:
    text = formula.replace(",", ".")
    text = text.replace(TOKEN_SQRT, "sqrt(")
    text = text.replace("√", "sqrt(")
    return normalize_recurrence_tokens(text)


def max_predecessor_lag(formula: str) -> int:
    normalized = preprocess_for_parse(formula)
    lag = 0
    if "a_nm1" in normalized:
        lag = max(lag, 1)
    if "a_nm2" in normalized:
        lag = max(lag, 2)
    if "a_nm3" in normalized:
        lag = max(lag, 3)
    return lag


def formula_uses_predecessor(formula: str) -> bool:
    return max_predecessor_lag(formula) > 0


def parse_sequence_formula(formula: str):
    normalized = preprocess_for_parse(formula)
    allowed_names = {
        "n": N_INDEX,
        "a_nm1": A_NM1,
        "a_nm2": A_NM2,
        "a_nm3": A_NM3,
        "sin": sin,
        "cos": cos,
        "tan": tan,
        "exp": exp,
        "log": log,
        "ln": log,
        "sqrt": sqrt,
        "pi": pi,
        "e": E,
    }
    names_in_formula = re.findall(r"[A-Za-z_]\w*", normalized)
    if any(name not in allowed_names for name in names_in_formula):
        raise ValueError("Unbekannte Funktion oder Variable")

    return parse_expr(
        normalized,
        local_dict=allowed_names,
        global_dict={
            "__builtins__": {},
            "Integer": Integer,
            "Float": Float,
            "Rational": Rational,
            "Symbol": Symbol,
            "Function": Function,
        },
        transformations=TRANSFORMATIONS,
        evaluate=True,
    )


def latex_for_plot(expression) -> str:
    return latex(
        expression,
        symbol_names={
            A_NM1: r"a_{n-1}",
            A_NM2: r"a_{n-2}",
            A_NM3: r"a_{n-3}",
            N_INDEX: "n",
        },
    )


def compute_explicit(expression) -> np.ndarray:
    numeric = lambdify(N_INDEX, expression, modules=["numpy"])
    indices = np.arange(1, N_TERMS + 1, dtype=float)
    with np.errstate(all="ignore"):
        values = np.asarray(numeric(indices), dtype=float)
    if values.ndim == 0:
        values = np.full(N_TERMS, float(values))
    values[~np.isfinite(values)] = np.nan
    return values


def compute_recursive(expression, formula: str, a1: float, a2: float, a3: float) -> np.ndarray:
    max_lag = max_predecessor_lag(formula)
    fn = lambdify((A_NM1, A_NM2, A_NM3, N_INDEX), expression, modules=["numpy"])
    initials = [a1, a2, a3]

    values = np.full(N_TERMS, np.nan, dtype=float)
    for index in range(max_lag):
        values[index] = initials[index]

    for n in range(max_lag + 1, N_TERMS + 1):
        prev1 = values[n - 2]
        prev2 = values[n - 3] if n - 3 >= 0 else 0.0
        prev3 = values[n - 4] if n - 4 >= 0 else 0.0
        try:
            values[n - 1] = float(fn(prev1, prev2, prev3, n))
        except (TypeError, ValueError):
            values[n - 1] = np.nan
        if not np.isfinite(values[n - 1]):
            values[n - 1] = np.nan

    return values


def compute_sequence(expression, formula: str, a1: float, a2: float, a3: float) -> np.ndarray:
    if formula_uses_predecessor(formula):
        return compute_recursive(expression, formula, a1, a2, a3)
    return compute_explicit(expression)


def partial_sums(sequence: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore"):
        return np.cumsum(sequence)


def padded_y_range(*arrays: np.ndarray) -> list[float]:
    finite = np.concatenate([a[np.isfinite(a)] for a in arrays if a.size])
    if finite.size == 0:
        return [-1.0, 1.0]
    low = float(np.min(finite))
    high = float(np.max(finite))
    if low == high:
        low -= 1.0
        high += 1.0
    padding = (high - low) * 0.08
    return [low - padding, high + padding]


def format_value(value: float) -> str:
    if not np.isfinite(value):
        return "—"
    return f"{value:.6g}"


def horizontal_value_table(sequence: np.ndarray, partials: np.ndarray) -> pd.DataFrame:
    columns = [str(n) for n in range(1, N_TERMS + 1)]
    table = pd.DataFrame(
        [
            {col: format_value(sequence[i]) for i, col in enumerate(columns)},
            {col: format_value(partials[i]) for i, col in enumerate(columns)},
        ],
        index=[r"aₙ", r"Sₙ"],
    )
    table.index.name = "n"
    return table


def append_to_formula(token: str) -> None:
    st.session_state.formula = st.session_state.get("formula", "") + token


X_CLAMP_SCRIPT = f"""
const graph = document.getElementById('{{plot_id}}');
const N_MIN = {N_X_MIN};
const N_MAX = {N_X_MAX};
let internalRelayout = false;

function clampXRange(range) {{
  let left = Math.min(Number(range[0]), Number(range[1]));
  let right = Math.max(Number(range[0]), Number(range[1]));
  let width = right - left;
  if (!Number.isFinite(width) || width <= 0) {{
    return [N_MIN, N_MAX];
  }}
  if (width >= N_MAX - N_MIN) {{
    return [N_MIN, N_MAX];
  }}
  if (left < N_MIN) {{
    left = N_MIN;
    right = left + width;
  }}
  if (right > N_MAX) {{
    right = N_MAX;
    left = right - width;
  }}
  if (left < N_MIN) {{
    left = N_MIN;
  }}
  return [left, right];
}}

graph.on('plotly_relayout', event => {{
  if (internalRelayout) return;

  const fromTop = event['xaxis.range[0]'] !== undefined && event['xaxis.range[1]'] !== undefined;
  const fromBottom = event['xaxis2.range[0]'] !== undefined && event['xaxis2.range[1]'] !== undefined;
  if (!fromTop && !fromBottom) return;

  const raw = fromTop
    ? [event['xaxis.range[0]'], event['xaxis.range[1]']]
    : [event['xaxis2.range[0]'], event['xaxis2.range[1]']];
  const clamped = clampXRange(raw);
  if (Math.abs(clamped[0] - Number(raw[0])) < 1e-9 && Math.abs(clamped[1] - Number(raw[1])) < 1e-9) {{
    return;
  }}

  internalRelayout = true;
  Plotly.relayout(graph, {{
    'xaxis.range': clamped,
    'xaxis2.range': clamped,
  }}).finally(() => {{
    internalRelayout = false;
  }});
}});
"""


if "formula" not in st.session_state:
    st.session_state.formula = "0.5^n"

st.title("Folgen und Reihen")
st.caption("Es werden immer die ersten 100 Folgenglieder gezeichnet (Start bei n = 1).")

sidebar_formula = st.session_state.formula.replace(",", ".")
predecessor_lag = max_predecessor_lag(sidebar_formula)
show_start_values = predecessor_lag > 0

with st.sidebar:
    st.markdown(
        """
        <style>
        section[data-testid="stSidebar"] div[data-testid="InputInstructions"] {
            display: none;
        }
        section[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] {
            padding-bottom: 6.75rem !important;
        }
        section[data-testid="stSidebar"] .folgen-sidebar-footer {
            position: fixed;
            left: 0;
            bottom: 0;
            z-index: 999;
            box-sizing: border-box;
            width: var(--sidebar-width, 21rem);
            max-width: 100%;
            padding: 0.75rem 1rem 1rem;
            background-color: var(--secondary-background-color);
            border-top: 1px solid rgba(49, 51, 63, 0.12);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.subheader("Folge definieren")

    label_col, input_col = st.columns([1, 3], gap="small", vertical_alignment="top")
    with label_col:
        st.markdown("**a_n** =")
    with input_col:
        st.text_input("Folge", key="formula", label_visibility="collapsed")
        st.caption("Enter drücken, um die Formel zu übernehmen.")

    with st.expander("Funktionen", expanded=False):
        calc_rows = [
            [("n", "n"), ("e", "e")],
            [("^", "^"), ("√", TOKEN_SQRT)],
            [("sin", "sin("), ("cos", "cos(")],
            [("tan", "tan("), ("(", "(")],
            [(")", ")")],
        ]
        for row_index, row in enumerate(calc_rows):
            cols = st.columns(2)
            for col_index, item in enumerate(row):
                label, token = item
                with cols[col_index]:
                    st.button(
                        label,
                        key=f"calc_{row_index}_{col_index}_{token}",
                        on_click=append_to_formula,
                        args=(token,),
                        width="stretch",
                    )

        st.markdown("**Vorgänger**")
        rec_cols = st.columns(3)
        with rec_cols[0]:
            st.button("aₙ₋₁", key="rec_nm1", on_click=append_to_formula, args=(TOKEN_A_NM1,), width="stretch")
        with rec_cols[1]:
            st.button("aₙ₋₂", key="rec_nm2", on_click=append_to_formula, args=(TOKEN_A_NM2,), width="stretch")
        with rec_cols[2]:
            st.button("aₙ₋₃", key="rec_nm3", on_click=append_to_formula, args=(TOKEN_A_NM3,), width="stretch")

    a1 = 1.0
    a2 = 0.0
    a3 = 0.0
    if show_start_values:
        st.markdown("**Startwerte**")
        a1 = st.number_input(r"\(a_1\)", value=1.0, format="%.6g")
        if predecessor_lag >= 2:
            a2 = st.number_input(r"\(a_2\)", value=0.0, format="%.6g")
        if predecessor_lag >= 3:
            a3 = st.number_input(r"\(a_3\)", value=0.0, format="%.6g")

    st.caption(r"Beispiele: `0.5^n`, `1/n`, `sin(n)`, `a_{n-1}+a_{n-2}`.")

    st.info("Mit dem Mausrad zoomen, ziehen zum Verschieben. Plotly-Toolbar oben rechts im Diagramm.")

    st.markdown(
        '<div class="folgen-sidebar-footer">'
        '<p style="color:#002C5C;font-weight:600;font-size:0.95rem;line-height:1.45;text-align:left;margin:0;">'
        "Mathematik 1 für Wirtschaftsingenieure<br>"
        "App zur Vorlesung<br>"
        "Hochschule Esslingen<br>"
        "Prof. Dr. David Wichmann"
        "</p></div>",
        unsafe_allow_html=True,
    )

formula = st.session_state.formula.replace(",", ".")

try:
    expression = parse_sequence_formula(formula)
    sequence = compute_sequence(expression, formula, float(a1), float(a2), float(a3))
    seq_latex = latex_for_plot(expression)
    title_seq = rf"$\text{{Folge: }}\ a_n = {seq_latex}$"
    title_partial = r"$\text{Partialsummen: }\ S_n = \sum_{k=1}^{n} a_k$"

    partials = partial_sums(sequence)
    n_values = np.arange(1, N_TERMS + 1)
    y_seq = padded_y_range(sequence)
    y_part = padded_y_range(partials)

    figure = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.5, 0.5],
        vertical_spacing=0.26,
        subplot_titles=(title_seq, title_partial),
    )
    marker_style = {"cliponaxis": True}
    figure.add_trace(
        go.Scatter(
            x=n_values,
            y=sequence,
            mode="markers",
            marker={"size": 9, "color": "#2563eb", "line": {"width": 1, "color": "#1d4ed8"}},
            name=r"a_n",
            **marker_style,
        ),
        row=1,
        col=1,
    )
    figure.add_trace(
        go.Scatter(
            x=n_values,
            y=partials,
            mode="markers",
            marker={"size": 9, "color": "#16a34a", "line": {"width": 1, "color": "#15803d"}},
            name=r"S_n",
            **marker_style,
        ),
        row=2,
        col=1,
    )
    figure.update_layout(
        height=720,
        margin={"l": 50, "r": 20, "t": 95, "b": 45},
        showlegend=False,
        dragmode="pan",
    )
    figure.update_annotations(font_size=20, yanchor="bottom", xref="paper", x=0.5, xanchor="center")
    if len(figure.layout.annotations) > 0:
        y1_top = figure.layout.yaxis.domain[1]
        figure.layout.annotations[0].update(y=y1_top, yref="paper", yshift=10)
    if len(figure.layout.annotations) > 1:
        y2_top = figure.layout.yaxis2.domain[1]
        figure.layout.annotations[1].update(y=y2_top, yref="paper", yshift=10)
    x_axis_style = {
        "title_text": "n",
        "range": [N_X_MIN, N_X_MAX],
        "dtick": 10,
        "showticklabels": True,
        "ticks": "outside",
        "showline": False,
        "zeroline": False,
    }
    figure.update_xaxes(**x_axis_style, title_standoff=6, row=1, col=1)
    figure.update_xaxes(**x_axis_style, title_standoff=6, row=2, col=1)
    axis_frame = {"showline": False, "mirror": False, "zeroline": False}
    figure.update_yaxes(title_text="$a_n$", range=y_seq, **axis_frame, row=1, col=1)
    figure.update_yaxes(title_text="$S_n$", range=y_part, **axis_frame, row=2, col=1)

    table_df = horizontal_value_table(sequence, partials)

    st.subheader("Wertetabelle")
    st.caption("Horizontal scrollen, um alle 100 Folgenglieder zu sehen.")
    st.dataframe(table_df, width="stretch", hide_index=False)

    st.subheader("Graph")
    plot_config = {
        "scrollZoom": True,
        "displaylogo": False,
        "responsive": True,
        "modeBarButtonsToAdd": ["zoomIn2d", "zoomOut2d"],
    }
    chart_html = pio.to_html(
        figure,
        include_plotlyjs=True,
        include_mathjax="cdn",
        full_html=False,
        post_script=X_CLAMP_SCRIPT,
        config=plot_config,
    )
    components.html(chart_html, height=740, scrolling=False)

except Exception:
    st.error("Diese Eingabe konnte ich nicht auswerten.")
    st.info(
        r"Formel in \(n\) oder mit Vorgängern, z. B. `0.5^n`, `sin(n)`, `a_{n-1}+a_{n-2}` "
        r"(Buttons aₙ₋₁ / aₙ₋₂ / aₙ₋₃). Wurzel: Zeichen √."
    )
