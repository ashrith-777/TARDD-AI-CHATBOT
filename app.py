"""TARDD ENTERPRICES — Streamlit console for live chat and Hindsight memory."""

from __future__ import annotations

import os
from dotenv import load_dotenv

# ----------------------------------------------------------------------
# Environment sanitization & key management (Milestone 1)
# ----------------------------------------------------------------------

# Read .env (and any process environment overrides) once at import
load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
hindsight_api_key = os.getenv("HINDSIGHT_API_KEY")

if not groq_api_key or not hindsight_api_key:
    raise ValueError("Missing API keys! Ensure GROQ_API_KEY and HINDSIGHT_API_KEY are set in your local .env file.")

import html
import json
import re
import uuid

import streamlit as st

import backend
from styles import apply_custom_styles

ACTIVE_USER_ID = "user_8492"
ACTIVE_USER_LABEL = "User:james hari ali - Lead Architect"

# Words and phrases used by the local, rule-based frustration estimate.
FRUSTRATION_MARKERS = (
    "angry",
    "furious",
    "outraged",
    "unacceptable",
    "hate",
    "worst",
    "broken",
    "down again",
    "still down",
    "escalate",
    "lawsuit",
    "refund",
    "incompetent",
    "ridiculous",
    "frustrated",
    "frustration",
    "urgent",
    "asap",
    "immediately",
    "outage",
)

PAGE_CSS = """
<style>
    /* Keep the chat area light and give the memory inspector a contrasting dark theme. */
    .stApp {
        background-color: #f4f4f5;
    }
    .stApp header {
        background-color: #f4f4f5;
        border-bottom: 1px solid #e4e4e7;
    }
    .block-container {
        padding-top: 0.85rem;
        padding-bottom: 1.35rem;
        padding-left: 1.35rem;
        padding-right: 1.35rem;
        max-width: 1480px;
    }
    [data-testid="stHeader"] {
        background: #f4f4f5;
    }
    [data-testid="stToolbar"] {
        background: transparent;
    }
    h1, h2, h3, .stMarkdown h1, .stMarkdown h2, .stSubheader {
        letter-spacing: -0.02em;
        color: #18181b !important;
    }
    .stSubheader {
        font-size: 0.92rem !important;
        font-weight: 650 !important;
        margin-bottom: 0.55rem !important;
    }
    .relay-title {
        font-size: 1.42rem;
        font-weight: 700;
        letter-spacing: -0.03em;
        color: #18181b;
        margin: 0;
        line-height: 1.15;
    }
    .enterprise-pill {
        display: inline-block;
        margin-left: 0.5rem;
        padding: 0.14rem 0.48rem;
        border-radius: 4px;
        background: #ffffff;
        border: 1px solid #e4e4e7;
        color: #2563eb;
        font-size: 0.65rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        vertical-align: middle;
    }
    .user-context {
        margin-top: 0.28rem;
        color: #71717a;
        font-size: 0.8rem;
        letter-spacing: 0.01em;
    }
    .chat-container {
        background: #ffffff;
        border: 1px solid #e4e4e7;
        border-radius: 8px;
        padding: 0.75rem 0.85rem 0.9rem 0.85rem;
    }
    div[data-testid="stChatMessage"] {
        background: #ffffff;
        border: 1px solid #e4e4e7;
        border-radius: 8px;
        padding: 0.55rem 0.7rem;
        margin-bottom: 0.45rem;
        gap: 0.55rem;
    }
    div[data-testid="stChatMessage"] p {
        margin-bottom: 0.2rem;
        font-size: 0.9rem;
        line-height: 1.45;
    }
    div[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]),
    div[data-testid="stChatMessage"]:has([aria-label="Chat avatar for user"]),
    div[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
        background: #2563eb;
        border-color: #1d4ed8;
    }
    div[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) p,
    div[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) span,
    div[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) div,
    div[data-testid="stChatMessage"]:has([aria-label="Chat avatar for user"]) p,
    div[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) p {
        color: #ffffff !important;
    }
    div[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]),
    div[data-testid="stChatMessage"]:has([aria-label="Chat avatar for assistant"]),
    div[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
        background: #f4f4f5;
        border-color: #e4e4e7;
    }
    div[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) p,
    div[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) span,
    div[data-testid="stChatMessage"]:has([aria-label="Chat avatar for assistant"]) p,
    div[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) p {
        color: #18181b !important;
    }
    [data-testid="stChatInput"] {
        background: #ffffff !important;
        border: 1px solid #e4e4e7;
        border-radius: 8px;
    }
    [data-testid="stChatInput"] textarea {
        background: #ffffff !important;
        color: #18181b;
    }
    div[data-testid="stChatInput"] > div {
        background: #ffffff !important;
    }
    .stButton > button {
        background: #ffffff;
        color: #18181b;
        border: 1px solid #e4e4e7;
        border-radius: 6px;
        font-size: 0.78rem;
        font-weight: 650;
        padding: 0.38rem 0.55rem;
        letter-spacing: 0.01em;
    }
    .stButton > button:hover {
        background: #f4f4f5;
        border-color: #d4d4d8;
        color: #18181b;
    }
    /* Memory inspector panel and its compact data cards. */
    .inspector-container {
        background: #111827;
        border: 1px solid #1f2937;
        border-radius: 10px;
        padding: 1.25rem;
        min-height: 72vh;
    }
    .inspector-heading {
        color: #f3f4f6;
        font-size: 0.84rem;
        font-weight: 650;
        margin: 0 0 0.7rem 0;
        letter-spacing: 0.03em;
        display: flex;
        align-items: center;
        gap: 0.45rem;
    }
    .slate-card {
        background: #1f2937;
        border: 1px solid #374151;
        border-radius: 8px;
        padding: 0.7rem 0.75rem;
        margin-bottom: 0.6rem;
    }
    .slate-card h4 {
        margin: 0 0 0.5rem 0;
        color: #d1d5db;
        font-size: 0.7rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.07em;
    }
    .badge-row {
        display: flex;
        flex-wrap: wrap;
        gap: 0.35rem;
        margin-bottom: 0.4rem;
    }
    .badge {
        display: inline-flex;
        align-items: center;
        font-weight: 700;
        font-size: 0.75rem;
        padding: 3px 8px;
        border-radius: 4px;
        letter-spacing: 0.02em;
        border: 1px solid transparent;
    }
    .badge-red {
        font-weight: 700;
        font-size: 0.75rem;
        padding: 3px 8px;
        border-radius: 4px;
        background: #7f1d1d;
        color: #fecaca;
        border: 1px solid #dc2626;
    }
    .badge-amber {
        font-weight: 700;
        font-size: 0.75rem;
        padding: 3px 8px;
        border-radius: 4px;
        background: #78350f;
        color: #fde68a;
        border: 1px solid #d97706;
    }
    .badge-emerald {
        font-weight: 700;
        font-size: 0.75rem;
        padding: 3px 8px;
        border-radius: 4px;
        background: #064e3b;
        color: #a7f3d0;
        border: 1px solid #10b981;
    }
    .badge-zinc {
        font-weight: 700;
        font-size: 0.75rem;
        padding: 3px 8px;
        border-radius: 4px;
        background: #27272a;
        color: #e4e4e7;
        border: 1px solid #3f3f46;
    }
    .metric-copy {
        color: #d1d5db;
        font-size: 0.78rem;
        line-height: 1.4;
        margin: 0;
    }
    pre,
    .mono-block,
    .inspector-container pre {
        font-family: monospace;
        font-size: 0.78rem;
        color: #f3f4f6;
        background: #0b1220;
        border: 1px solid #374151;
        border-radius: 6px;
        line-height: 1.45;
        white-space: pre-wrap;
        word-break: break-word;
        padding: 0.55rem 0.6rem;
        max-height: 220px;
        overflow-y: auto;
        margin: 0;
    }
    .empty-hint {
        color: #9ca3af;
        font-style: italic;
        font-family: monospace;
        font-size: 0.78rem;
    }
    /* Tighten spacing in Streamlit's generated column and section layouts. */
    [data-testid="stHorizontalBlock"] {
        gap: 0.75rem;
    }
    [data-testid="stVerticalBlock"] > div {
        gap: 0.45rem;
    }
    hr {
        border: none;
        border-top: 1px solid #e4e4e7;
        margin: 0.45rem 0;
    }
</style>
"""


def init_session_state() -> None:
    """Create the persistent UI values used across Streamlit reruns."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "show_inspector" not in st.session_state:
        st.session_state.show_inspector = True
    if "last_recalled_context" not in st.session_state:
        st.session_state.last_recalled_context = ""
    if "last_write_log" not in st.session_state:
        st.session_state.last_write_log = ""
    if "last_escalation" not in st.session_state:
        st.session_state.last_escalation = {"escalate": False}
    if "last_dispatch" not in st.session_state:
        st.session_state.last_dispatch = None
    if "last_ticket" not in st.session_state:
        st.session_state.last_ticket = None
    if "csat_ratings" not in st.session_state:
        st.session_state.csat_ratings = {}
    if "tts_audio" not in st.session_state:
        st.session_state.tts_audio = {}
    # Backfill stable IDs so existing assistant messages can own persistent widget state.
    for message in st.session_state.messages:
        if message.get("role") == "assistant" and not message.get("message_id"):
            message["message_id"] = uuid.uuid4().hex


def frustration_profile(messages: list[dict[str, str]]) -> dict[str, str | int]:
    """Estimate customer risk from user wording and recent write telemetry."""
    user_turns = [m["content"] for m in messages if m.get("role") == "user"]
    if not user_turns:
        # Show a neutral baseline until the customer has sent a message.
        return {
            "score": 8,
            "score_label": "Standby",
            "score_class": "badge-zinc",
            "churn": "Low",
            "churn_class": "badge-emerald",
            "summary": "No live turns yet. Baseline account risk remains low until sentiment is observed.",
        }

    hits = 0
    for text in user_turns:
        lowered = text.lower()
        hits += sum(1 for marker in FRUSTRATION_MARKERS if marker in lowered)
        hits += len(re.findall(r"[!]{2,}", text))

    # Repeated user turns raise the score; backend sentiment telemetry can adjust it further.
    raw = min(98, 12 + hits * 18 + max(0, len(user_turns) - 1) * 6)
    write_log = str(st.session_state.get("last_write_log") or "")
    if "sentiment=angry" in write_log:
        raw = min(98, max(raw, 74))
    elif "sentiment=positive" in write_log:
        raw = max(6, raw - 28)

    if raw >= 70:
        return {
            "score": raw,
            "score_label": "Elevated",
            "score_class": "badge-red",
            "churn": "High",
            "churn_class": "badge-red",
            "summary": "Repeated incident language detected. Prioritize ECS/Postgres continuity and skip soft openers.",
        }
    if raw >= 40:
        return {
            "score": raw,
            "score_label": "Watch",
            "score_class": "badge-amber",
            "churn": "Medium",
            "churn_class": "badge-amber",
            "summary": "Tone is tightening. Keep answers concrete and confirm ownership of the last ECS/RDS incident.",
        }
    return {
        "score": raw,
        "score_label": "Stable",
        "score_class": "badge-emerald",
        "churn": "Low",
        "churn_class": "badge-emerald",
        "summary": "Conversation is operational. Continue with direct technical guidance.",
    }


def _frustration_score_value(value: object) -> int:
    """Extract a numeric score from backend frustration telemetry."""
    if isinstance(value, (int, float)):
        return int(value)
    # Backend telemetry is formatted like "90/100 - Elevated".
    match = re.search(r"\b(\d+(?:\.\d+)?)\s*/\s*100\b", str(value or ""))
    return int(float(match.group(1))) if match else 0


def render_header() -> None:
    """Show the product identity and the control for the memory inspector."""
    st.sidebar.selectbox(
        "🌐 System Language",
        ["English (en)", "Spanish (es)", "French (fr)", "German (de)"],
        key="system_language",
    )
    left_col, right_col = st.columns([0.75, 0.25])
    with left_col:
        st.markdown(
            '<p class="relay-title">TARDD'
            '<span class="enterprise-pill">ENTERPRICES</span></p>'
            f'<p class="user-context">{html.escape(ACTIVE_USER_LABEL)}</p>',
            unsafe_allow_html=True,
        )
    with right_col:
        inspector_on = bool(st.session_state.show_inspector)
        toggle_label = (
            "🟢 Memory Inspector [ON]" if inspector_on else "⚪ Memory Inspector [OFF]"
        )
        if st.button(toggle_label, use_container_width=True, key="inspector_toggle"):
            # Streamlit reruns the script after this state change to redraw the layout.
            st.session_state.show_inspector = not inspector_on
            st.rerun()


def render_chat_panel() -> None:
    """Render stored user and assistant messages in the chat column."""
    st.subheader("Live Customer Chat")
    for message in st.session_state.messages:
        role = message.get("role", "assistant")
        if role not in ("user", "assistant"):
            role = "assistant"
        with st.chat_message(role):
            st.markdown(message.get("content", ""))
            if role != "assistant":
                continue

            message_id = str(message["message_id"])
            # Key each rating to its reply so reruns never mix feedback between messages.
            selected_rating = st.session_state.csat_ratings.get(message_id)
            if selected_rating:
                st.caption(f"Customer feedback: {selected_rating}/5 stars")
            else:
                rating_columns = st.columns(5)
                for rating, column in enumerate(rating_columns, start=1):
                    with column:
                        if st.button(
                            f"⭐ {rating}",
                            key=f"csat_{message_id}_{rating}",
                            use_container_width=True,
                        ):
                            feedback = backend.store_csat_feedback(
                                ACTIVE_USER_ID,
                                message_id,
                                rating,
                            )
                            if feedback.get("ok"):
                                st.session_state.csat_ratings[message_id] = rating
                                feedback_log = str(feedback.get("log") or "")
                                retention_log = str(feedback.get("write_log") or "")
                                st.session_state.last_write_log = " | ".join(
                                    part for part in (feedback_log, retention_log) if part
                                )
                                st.toast("Thank you for your feedback!")
                            else:
                                st.error(str(feedback.get("error") or "Feedback could not be saved."))
                            st.rerun()

            if st.button("🔊 Listen", key=f"listen_{message_id}"):
                language_code = st.session_state.system_language.rsplit("(", 1)[-1].rstrip(")")
                st.session_state.tts_audio[message_id] = backend.text_to_speech_bytes(
                    str(message.get("content", "")),
                    lang=language_code,
                )
                st.rerun()
            audio_bytes = st.session_state.tts_audio.get(message_id)
            if audio_bytes:
                st.audio(audio_bytes, format="audio/mp3")


def render_inspector_panel() -> None:
    """Display the current risk estimate and latest memory-service details."""
    profile = frustration_profile(st.session_state.messages)
    recalled = str(st.session_state.last_recalled_context or "").strip()
    write_log = str(st.session_state.last_write_log or "").strip()

    # Escape service-provided text before inserting it into the HTML inspector.
    recalled_html = (
        html.escape(recalled)
        if recalled
        else '<span class="empty-hint">No recall payload yet. Send a message to hydrate Hindsight context.</span>'
    )
    write_html = (
        html.escape(write_log)
        if write_log
        else '<span class="empty-hint">Awaiting retain event…</span>'
    )

    st.markdown(
        f"""
        <div class="inspector-container">
            <p class="inspector-heading">Hindsight Memory Inspector <span class="badge badge-emerald">Active memory</span></p>
            <div class="slate-card">
                <h4>Risk Detection</h4>
                <div class="badge-row">
                    <span class="badge {profile['score_class']}">Frustration {profile['score']}/100 · {html.escape(str(profile['score_label']))}</span>
                    <span class="badge {profile['churn_class']}">Churn risk · {html.escape(str(profile['churn']))}</span>
                </div>
                <p class="metric-copy">{html.escape(str(profile['summary']))}</p>
            </div>
            <div class="slate-card">
                <h4>Active Recall</h4>
                <pre class="mono-block">{recalled_html}</pre>
            </div>
            <div class="slate-card">
                <h4>Live Write Log</h4>
                <pre class="mono-block">{write_html}</pre>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    escalation = st.session_state.last_escalation or {}
    if escalation.get("escalate"):
        briefing = escalation.get("warm_handoff_briefing") or {}
        with st.expander("Human Agent Briefing Dashboard", expanded=True):
            st.markdown("**Executive Problem Summary**")
            st.write(briefing.get("executive_problem_summary", "No summary available."))
            st.markdown("**Sentiment State**")
            st.write(briefing.get("sentiment_state", "Unknown"))
            st.markdown("**Actions Attempted**")
            st.write(briefing.get("actions_attempted", []))
            st.markdown("**Hindsight Memory Context**")
            st.write(briefing.get("hindsight_memory_context", recalled))

    if st.button("📋 Generate Ticket & Root Cause Report", key="generate_ticket"):
        try:
            st.session_state.last_ticket = backend.generate_ticket_summary(
                ACTIVE_USER_ID,
                st.session_state.messages,
                recalled,
            )
        except Exception as exc:
            st.session_state.last_ticket = {"error": str(exc)}
        st.rerun()

    ticket = st.session_state.last_ticket
    if ticket:
        st.markdown("### Support Ticket Report")
        if ticket.get("error"):
            st.warning(f"Automated summary unavailable; showing the fallback report. {ticket['error']}")
        if ticket.get("ticket_id") or ticket.get("root_cause_analysis"):
            st.markdown(f"**Ticket:** {ticket.get('ticket_id', 'Unavailable')}")
            st.markdown("**Root Cause Analysis**")
            st.write(ticket.get("root_cause_analysis", "Not available."))
            st.markdown("**Resolution Steps Taken**")
            for step in ticket.get("resolution_steps_taken", []):
                st.markdown(f"- {step}")
            st.markdown("**Recommended Action Items**")
            for item in ticket.get("recommended_action_items", []):
                st.markdown(f"- {item}")
            st.download_button(
                "⬇️ Download ticket JSON",
                data=json.dumps(ticket, indent=2, ensure_ascii=False),
                file_name=f"{ticket.get('ticket_id', 'support-ticket')}.json",
                mime="application/json",
                key="download_ticket_json",
            )


def handle_prompt(user_input: str) -> None:
    """Send a customer message to the backend and save the complete UI result."""
    st.session_state.messages.append({"role": "user", "content": user_input})
    # Clear prior dispatch state before evaluating the new turn.
    st.session_state.last_dispatch = None
    try:
        chat_history = st.session_state.messages[:-1]
        result = backend.generate_agent_response(
            ACTIVE_USER_ID,
            user_input,
            chat_history=chat_history,
        )
        if not isinstance(result, dict):
            result = {}
        assistant_text = str(result.get("response") or "I could not generate a reply just now.")
        st.session_state.last_recalled_context = str(result.get("recalled_context") or "")
        st.session_state.last_write_log = str(result.get("write_log") or "")
        escalation_result = result.get("escalation")
        st.session_state.last_escalation = (
            escalation_result if isinstance(escalation_result, dict) else {"escalate": False}
        )
        frustration_score = _frustration_score_value(result.get("frustration_score"))
        if frustration_score >= 85:
            # Persist the dispatch summary because Streamlit reruns after this handler.
            st.session_state.last_dispatch = {
                "frustration_score": frustration_score,
                "incident_id": "INC-8492",
            }
    except Exception as exc:
        # Keep the chat usable and surface the failure in the inspector's write log.
        assistant_text = "The support agent hit an unexpected error. Please retry."
        st.session_state.last_write_log = f"generate_agent_response failed: {exc}"
    assistant_message = {
        "role": "assistant",
        "content": assistant_text,
        "message_id": uuid.uuid4().hex,
    }
    st.session_state.messages.append(assistant_message)
    try:
        selected_language = st.session_state.get("system_language", "English (en)")
        language_code = selected_language.rsplit("(", 1)[-1].rstrip(")")
        audio_bytes = backend.generate_tts_audio(assistant_text, lang=language_code)
        if audio_bytes:
            st.session_state.tts_audio[assistant_message["message_id"]] = audio_bytes
    except Exception as exc:
        st.session_state.last_write_log = f"TTS generation failed: {exc}"
    st.rerun()


def render_layout() -> None:
    """Choose a two-column view or a full-width chat based on the toggle."""
    dispatch = st.session_state.last_dispatch
    if dispatch:
        st.error("🚨 **Warm Hand-off Triggered: Live Senior Engineer Escalated**")
        # This status card reports UI state; external notification requires a configured integration.
        with st.status("📡 Routing session to Tier-2 Support Queue...", expanded=True) as status:
            st.write(f"✓ Incident Ticket **#{dispatch['incident_id']}** generated")
            st.write("✓ Telemetry logs & Active Recall history packaged")
            st.write("✓ On-call Senior Engineer notification prepared for PagerDuty/Slack")
            st.caption("Dispatch panel is UI-only; no PagerDuty/Slack integration is configured.")
            status.update(
                label="🟢 Human Agent Assigned: Sarah Chen (Senior Site Reliability Engineer)",
                state="complete",
            )
    elif (st.session_state.last_escalation or {}).get("escalate"):
        st.error("🚨 **Warm Hand-off Triggered: Live Senior Support Briefed**")
    if st.session_state.show_inspector:
        col_chat, col_inspector = st.columns([0.55, 0.45])
        with col_chat:
            render_chat_panel()
        with col_inspector:
            render_inspector_panel()
    else:
        render_chat_panel()


# Streamlit page setup must happen before rendering any UI elements.
st.set_page_config(
    layout="wide",
    page_title="TARDD ENTERPRICES",
    page_icon="⚡",
)
init_session_state()
st.markdown(PAGE_CSS, unsafe_allow_html=True)
try:
    apply_custom_styles()
except Exception as exc:
    print(f"CSS Load Warning: {exc}")
render_header()
render_layout()

user_input = st.chat_input("Type your response...")
if user_input:
    handle_prompt(user_input)
