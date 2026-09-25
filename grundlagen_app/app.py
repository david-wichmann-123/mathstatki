"""Übungs-App zu Mathematik-Grundlagen (Mathe 1 WNB)."""

import json
import random
import re
from pathlib import Path

import streamlit as st

_APP_DIR = Path(__file__).resolve().parent
_DATA_PATH = _APP_DIR / "data" / "aufgaben.json"


def _aufgaben_file_version() -> float:
    return _DATA_PATH.stat().st_mtime


@st.cache_data
def load_aufgaben(file_version: float) -> dict:
    with _DATA_PATH.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    for index, task in enumerate(payload["tasks"], start=1):
        task.setdefault("nummer", index)
    return payload


def topic_map(data: dict) -> dict[str, str]:
    return {topic["id"]: topic["title"] for topic in data["topics"]}


def filter_tasks(all_tasks: list[dict], topic_id: str | None) -> list[dict]:
    if topic_id and topic_id != "alle":
        return [task for task in all_tasks if task["topic_id"] == topic_id]
    return list(all_tasks)


def task_by_nummer(all_tasks: list[dict], nummer: int) -> dict | None:
    for task in all_tasks:
        if task.get("nummer") == nummer:
            return task
    return None


def index_in_filtered(filtered: list[dict], nummer: int) -> int | None:
    for index, task in enumerate(filtered):
        if task.get("nummer") == nummer:
            return index
    return None


def latex_json_to_markdown(text: str) -> str:
    normalized = text.strip()
    if not normalized:
        return ""

    def display_math(match: re.Match[str]) -> str:
        body = match.group(1).strip()
        return f"\n\n$${body}$$\n\n"

    def inline_math(match: re.Match[str]) -> str:
        body = match.group(1).strip()
        return f"${body}$"

    normalized = re.sub(r"\\\[(.*?)\\\]", display_math, normalized, flags=re.DOTALL)
    normalized = re.sub(r"\\\((.*?)\\\)", inline_math, normalized, flags=re.DOTALL)
    return normalized


def render_math_content(text: str) -> None:
    markdown = latex_json_to_markdown(text)
    if markdown:
        st.markdown(markdown)


def select_topic(topic_id: str) -> None:
    st.session_state.selected_topic = topic_id
    st.session_state.show_solution = False
    filtered = filter_tasks(st.session_state.all_tasks, topic_id)
    if filtered:
        if index_in_filtered(filtered, st.session_state.current_nummer) is None:
            st.session_state.current_nummer = filtered[0]["nummer"]


def go_to_nummer(nummer: int, switch_topic: bool = False) -> bool:
    task = task_by_nummer(st.session_state.all_tasks, nummer)
    if task is None:
        return False
    st.session_state.current_nummer = nummer
    st.session_state.show_solution = False
    if switch_topic:
        st.session_state.selected_topic = task["topic_id"]
    return True


def go_next_task(filtered: list[dict]) -> None:
    if not filtered:
        return
    if st.session_state.random_draw:
        choices = [t["nummer"] for t in filtered]
        if len(choices) == 1:
            st.session_state.current_nummer = choices[0]
        else:
            other = [n for n in choices if n != st.session_state.current_nummer]
            st.session_state.current_nummer = random.choice(other or choices)
    else:
        current_index = index_in_filtered(filtered, st.session_state.current_nummer)
        if current_index is None:
            st.session_state.current_nummer = filtered[0]["nummer"]
        elif current_index < len(filtered) - 1:
            st.session_state.current_nummer = filtered[current_index + 1]["nummer"]
    st.session_state.show_solution = False


def go_prev_task(filtered: list[dict]) -> None:
    if not filtered or st.session_state.random_draw:
        return
    current_index = index_in_filtered(filtered, st.session_state.current_nummer)
    if current_index is None:
        st.session_state.current_nummer = filtered[0]["nummer"]
    elif current_index > 0:
        st.session_state.current_nummer = filtered[current_index - 1]["nummer"]
    st.session_state.show_solution = False


st.set_page_config(
    page_title="Mathe 1 – Übungen",
    page_icon="📝",
    layout="wide",
)

data = load_aufgaben(_aufgaben_file_version())
topics = data["topics"]
titles = topic_map(data)
all_tasks = data["tasks"]
max_nummer = max(task.get("nummer", index + 1) for index, task in enumerate(all_tasks))

if "all_tasks" not in st.session_state:
    st.session_state.all_tasks = all_tasks
else:
    st.session_state.all_tasks = all_tasks

if "current_nummer" not in st.session_state:
    st.session_state.current_nummer = 1
if "show_solution" not in st.session_state:
    st.session_state.show_solution = False
if "selected_topic" not in st.session_state:
    st.session_state.selected_topic = "alle"
if "random_draw" not in st.session_state:
    st.session_state.random_draw = False

topic_titles = {"alle": "Grundlagen – Alle Themen"}
topic_titles.update(titles)

with st.sidebar:
    st.markdown(
        """
        <style>
        section[data-testid="stSidebar"] button[kind="primary"] {
            background-color: #002C5C !important;
            border-color: #002C5C !important;
            color: #ffffff !important;
        }
        section[data-testid="stSidebar"] button[kind="secondary"] {
            border: 1px solid rgba(0, 44, 92, 0.35) !important;
        }
        section[data-testid="stSidebar"] button {
            min-height: 3.1rem !important;
            height: auto !important;
            white-space: normal !important;
            line-height: 1.3 !important;
            padding-top: 0.55rem !important;
            padding-bottom: 0.55rem !important;
        }
        section[data-testid="stSidebar"] button p {
            white-space: normal !important;
            line-height: 1.3 !important;
        }
        section[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] {
            padding-bottom: 8.5rem !important;
        }
        section[data-testid="stSidebar"] .grundlagen-sidebar-footer {
            position: fixed;
            left: 0;
            bottom: 0;
            z-index: 999;
            box-sizing: border-box;
            width: var(--sidebar-width, 21rem);
            max-width: 100%;
            padding: 0.75rem 1rem 1rem;
            background: #f0f2f6 !important;
            background-color: #f0f2f6 !important;
            border-top: 1px solid rgba(49, 51, 63, 0.12);
            box-shadow: 0 -0.35rem 0.75rem rgba(49, 51, 63, 0.06);
        }
        [data-theme="dark"] section[data-testid="stSidebar"] .grundlagen-sidebar-footer {
            background: #262730 !important;
            background-color: #262730 !important;
            border-top-color: rgba(250, 250, 250, 0.12);
        }
        [data-testid="column"]:has(div[data-testid="stNumberInput"]) {
            flex: 0 0 auto !important;
            width: auto !important;
            min-width: unset !important;
        }
        div[data-testid="stNumberInput"] {
            max-width: 4.5rem;
        }
        div[data-testid="stNumberInput"] input {
            max-width: 4.5rem;
            padding-left: 0.5rem;
            padding-right: 0.5rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("Thema")

    tile_topics = [{"id": "alle", "title": "Alle Themen"}]
    tile_topics.extend(topics)

    for topic in tile_topics:
        is_selected = st.session_state.selected_topic == topic["id"]
        if st.button(
            topic["title"],
            key=f"topic_tile_{topic['id']}",
            type="primary" if is_selected else "secondary",
            width="stretch",
        ):
            select_topic(topic["id"])
            st.rerun()

    filtered = filter_tasks(all_tasks, st.session_state.selected_topic)
    st.caption(f"{len(filtered)} Aufgabe(n) in dieser Auswahl.")

    st.markdown(
        '<div class="grundlagen-sidebar-footer">'
        '<p style="color:#002C5C;font-weight:600;font-size:0.95rem;line-height:1.45;text-align:left;margin:0;">'
        "App zur Vorlesung Mathematik 1 für Wirtschaftsingenieure<br>"
        "Hochschule Esslingen<br>"
        "Prof. Dr. David Wichmann"
        "</p></div>",
        unsafe_allow_html=True,
    )

st.title(topic_titles.get(st.session_state.selected_topic, "Grundlagen"))
st.caption("Aufgabe wählen, rechnen, dann Lösung anzeigen.")

if not filtered:
    st.warning("Für dieses Thema sind noch keine Aufgaben hinterlegt.")
    st.stop()

if index_in_filtered(filtered, st.session_state.current_nummer) is None:
    st.session_state.current_nummer = filtered[0]["nummer"]

task = task_by_nummer(all_tasks, st.session_state.current_nummer)
if task is None:
    st.session_state.current_nummer = 1
    task = task_by_nummer(all_tasks, 1)

topic_title = titles.get(task["topic_id"], task["topic_id"])
position = (index_in_filtered(filtered, task["nummer"]) or 0) + 1

jump_col, random_col = st.columns([0.35, 4], gap="small")
with jump_col:
    jump_num = st.number_input(
        "Nr.",
        min_value=1,
        max_value=max_nummer,
        value=int(st.session_state.current_nummer),
        step=1,
        help="Aufgabennummer eingeben und mit Enter bestätigen.",
    )
    if int(jump_num) != int(st.session_state.current_nummer):
        if go_to_nummer(int(jump_num), switch_topic=True):
            st.rerun()
        else:
            st.error(f"Es gibt keine Aufgabe {int(jump_num)}.")

with random_col:
    st.markdown("<div style='height:1.75rem;'></div>", unsafe_allow_html=True)
    st.session_state.random_draw = st.checkbox(
        "Aufgabe zufällig ziehen",
        value=st.session_state.random_draw,
        help="Wenn aktiv, führt „Nächste →“ zu einer zufälligen Aufgabe im gewählten Thema.",
    )

st.markdown(
    f"**Aufgabe {task['nummer']}** "
    f"({position} von {len(filtered)} in dieser Auswahl)"
)
st.markdown(f"*{topic_title}*")

nav_left, nav_spacer, nav_right = st.columns([1, 2, 1])
with nav_left:
    if st.button("← Vorherige", disabled=st.session_state.random_draw or position <= 1):
        go_prev_task(filtered)
        st.rerun()
with nav_right:
    next_disabled = len(filtered) <= 1 or (
        not st.session_state.random_draw and position >= len(filtered)
    )
    if st.button("Nächste →", disabled=next_disabled):
        go_next_task(filtered)
        st.rerun()

st.markdown("<div style='height:0.35rem;'></div>", unsafe_allow_html=True)

with st.container(border=True):
    st.markdown(
        f'<p style="margin:0 0 0.75rem 0;font-size:1.15rem;font-weight:700;color:#002C5C;">'
        f"Aufgabe {task['nummer']}</p>",
        unsafe_allow_html=True,
    )
    render_math_content(task["aufgabe"])

st.markdown("<div style='height:1.75rem;'></div>", unsafe_allow_html=True)
st.divider()
st.markdown("<div style='height:0.5rem;'></div>", unsafe_allow_html=True)

if st.button("Lösung anzeigen", type="primary", width="stretch"):
    st.session_state.show_solution = True

if st.session_state.show_solution:
    st.markdown("<div style='height:0.75rem;'></div>", unsafe_allow_html=True)
    with st.container(border=True):
        st.markdown(
            '<p style="margin:0 0 0.75rem 0;font-size:1.05rem;font-weight:700;color:#002C5C;">Lösungsweg</p>',
            unsafe_allow_html=True,
        )
        for step_index, step in enumerate(task["loesung_schritte"], start=1):
            st.markdown(f"**Schritt {step_index}**")
            render_math_content(step)
        st.markdown("**Ergebnis**")
        render_math_content(task["loesung_kurz"])
