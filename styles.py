"""Shared visual styles for the TARDD Streamlit console."""

import streamlit as st

OCEAN_CSS = """
<style>
    /* Keep the page canvas light; message bubbles and inspector provide their own contrast. */
    .stApp {
        background: #f4f4f5 !important;
        color: #18181b;
    }
    .stApp header,
    [data-testid="stHeader"],
    [data-testid="stToolbar"] {
        background: #f4f4f5 !important;
    }
    .block-container {
        position: relative;
        z-index: 1;
    }
    h1, h2, h3, .relay-title {
        color: #18181b;
    }
    .relay-title { color: #18181b !important; }
    .user-context { color: #71717a !important; }
    [data-testid="stChatInput"],
    [data-testid="stChatInput"] > div,
    [data-testid="stChatInput"] textarea {
        background: #ffffff !important;
    }
    [data-testid="stChatInput"] textarea {
        color: #000000 !important;
        caret-color: #000000 !important;
    }
    [data-testid="stChatMessage"] {
        background-color: #1e293b !important;
        border: 1px solid #334155 !important;
        border-radius: 8px !important;
        padding: 12px !important;
    }
    div[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) :is(p, li, div, span, strong, b),
    div[data-testid="stChatMessage"]:has([aria-label="Chat avatar for user"]) :is(p, li, div, span, strong, b),
    div[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) :is(p, li, div, span, strong, b),
    div[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) :is(p, li, div, span, strong, b),
    div[data-testid="stChatMessage"]:has([aria-label="Chat avatar for assistant"]) :is(p, li, div, span, strong, b),
    div[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) :is(p, li, div, span, strong, b) {
        color: #ffffff !important;
    }
    [data-testid="stChatMessage"] code {
        background-color: #0f172a !important;
        color: #4ade80 !important;
        border: 1px solid #1e293b !important;
    }
    .stButton button,
    [data-testid="baseButton-secondary"] {
        background-color: #f1f5f9 !important;
        border: 1px solid #cbd5e1 !important;
        color: #0f172a !important;
        font-weight: 700 !important;
    }
    .stButton button:hover {
        background-color: #e2e8f0 !important;
        color: #000000 !important;
    }
    .stButton button p,
    .stButton button div,
    .stButton button span {
        color: #0f172a !important;
        font-weight: 700 !important;
    }
    [data-testid="stChatMessage"] button {
        background-color: #334155 !important;
        color: #ffffff !important;
        border-radius: 6px !important;
    }
    [data-testid="stChatMessage"] button p {
        color: #ffffff !important;
    }
    .stAlert,
    [data-testid="stAlert"] {
        background-color: #ffebe9 !important;
        color: #d9381e !important;
        font-weight: 600 !important;
        border: 1px solid #ffc0cb !important;
    }
    .stAlert p,
    [data-testid="stAlert"] p {
        color: #b30000 !important;
    }
    .inspector-container,
    .inspector-container p,
    .inspector-container h4,
    .inspector-container .metric-copy,
    .inspector-container .badge {
        color: #e0e6ed !important;
    }
    .inspector-container small,
    .inspector-container .caption,
    .inspector-container [data-testid="stCaptionContainer"] {
        color: #a0aec0 !important;
    }
    [data-testid="stSidebar"] {
        color: #18181b;
    }
</style>
"""


def apply_custom_styles() -> None:
    """Inject the shared app stylesheet into the Streamlit page."""
    st.markdown(OCEAN_CSS, unsafe_allow_html=True)