"""
AI Video Assistant — Streamlit UI
Wraps the existing pipeline (transcription -> summary -> extraction -> RAG chat)
in a dark, aurora/glassmorphic UI inspired by cosmoq.framer.website
"""

import os
import tempfile

import streamlit as st
from dotenv import load_dotenv

from core.rag_engine import build_rag_chain, ask_question
from utils.audio_preprocessor import process_input
from core.transcriber import transcribe_all
from core.summarizer import summarize, generate_title
from core.extractor import extract_action_items, extract_key_decisions, extract_questions

load_dotenv()

st.set_page_config(
    page_title="Aiden — AI Video Assistant",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# --------------------------------------------------------------------------------------
# STYLE
# --------------------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Inter:wght@400;500;600&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    h1, h2, h3, h4, .hero-title, .navbar .logo, .tile-title { font-family: 'Space Grotesk', sans-serif; }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header[data-testid="stHeader"] { background: transparent; }
    div[data-testid="stToolbar"] { visibility: hidden; }
    div[data-testid="stDecoration"] { display: none; }

    .stApp {
        background: #050508;
        color: #f2f2f5;
    }

    .block-container { padding-top: 1.2rem; max-width: 1180px; }

    /* ================= NAVBAR ================= */
    .navbar {
        position: sticky;
        top: 14px;
        z-index: 999;
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 12px 12px 12px 22px;
        border-radius: 999px;
        background: rgba(12, 12, 16, 0.65);
        border: 1px solid rgba(255, 255, 255, 0.09);
        backdrop-filter: blur(18px);
        margin-bottom: 34px;
    }
    .navbar .logo {
        font-weight: 700;
        font-size: 1.05rem;
        letter-spacing: 0.01em;
        color: #ffffff;
    }
    .navbar .nav-links {
        display: flex;
        gap: 30px;
        font-size: 0.85rem;
        color: #b3b3bd;
    }
    .navbar .nav-links a { color: #b3b3bd; text-decoration: none; }
    .navbar .nav-links a:hover { color: #ffffff; }
    .navbar .nav-cta {
        background: linear-gradient(90deg, #fb923c, #3b82f6);
        color: #ffffff;
        padding: 9px 20px;
        border-radius: 999px;
        font-size: 0.85rem;
        font-weight: 600;
        text-decoration: none;
        white-space: nowrap;
    }
    @media (max-width: 800px) { .navbar .nav-links { display: none; } }

    /* ================= HERO ================= */
    .hero-shell {
        position: relative;
        overflow: hidden;
        border-radius: 32px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        padding: 88px 40px 130px;
        text-align: center;
        margin-bottom: 46px;
        background: #06060a;
    }
    .hero-glow-1 {
        position: absolute; top: -120px; left: -100px;
        width: 480px; height: 480px; border-radius: 50%;
        background: radial-gradient(circle, rgba(251,146,60,0.45) 0%, transparent 70%);
        filter: blur(30px); pointer-events: none;
    }
    .hero-glow-2 {
        position: absolute; top: -160px; right: -140px;
        width: 560px; height: 560px; border-radius: 50%;
        background: radial-gradient(circle, rgba(59,130,246,0.42) 0%, transparent 70%);
        filter: blur(30px); pointer-events: none;
    }
    .hero-horizon {
        position: absolute; left: 50%; bottom: -260px;
        transform: translateX(-50%);
        width: 900px; height: 420px; border-radius: 50%;
        background: radial-gradient(ellipse at center, rgba(96,165,250,0.55) 0%, rgba(96,165,250,0.15) 45%, transparent 72%);
        filter: blur(10px); pointer-events: none;
    }
    .hero-rays {
        position: absolute; inset: 0;
        background: repeating-linear-gradient(
            100deg,
            transparent 0px, transparent 60px,
            rgba(251,146,60,0.06) 60px, rgba(251,146,60,0.06) 90px,
            transparent 150px, transparent 210px,
            rgba(59,130,246,0.07) 210px, rgba(59,130,246,0.07) 240px,
            transparent 300px
        );
        filter: blur(6px);
        pointer-events: none;
    }
    .hero-content { position: relative; z-index: 2; }
    .badge-pill {
        display: inline-block;
        padding: 7px 18px;
        border-radius: 999px;
        border: 1px solid rgba(255, 255, 255, 0.16);
        background: rgba(255, 255, 255, 0.05);
        color: #d8d8e0;
        font-size: 0.78rem;
        margin-bottom: 30px;
    }
    .hero-title {
        font-size: 4rem;
        font-weight: 700;
        line-height: 1.08;
        letter-spacing: -0.02em;
        color: #ffffff;
        margin-bottom: 22px;
    }
    .hero-title .grad {
        background: linear-gradient(90deg, #fdba74, #93c5fd);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .hero-sub {
        font-size: 1.08rem;
        color: #a3a3ad;
        max-width: 560px;
        margin: 0 auto 34px;
        line-height: 1.65;
    }
    .hero-cta-row { display: flex; gap: 14px; justify-content: center; flex-wrap: wrap; }
    .btn-primary {
        background: linear-gradient(90deg, #fb923c, #3b82f6);
        color: #ffffff;
        padding: 13px 30px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.95rem;
        text-decoration: none;
        display: inline-block;
    }
    .btn-ghost {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.14);
        color: #e5e5ea;
        padding: 13px 26px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.95rem;
        text-decoration: none;
        display: inline-block;
    }

    /* ================= SECTION LABELS ================= */
    .section-eyebrow {
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: #fca86a;
        margin-bottom: 6px;
        text-align: center;
    }
    .section-heading {
        font-size: 2rem;
        font-weight: 700;
        color: #f5f5f7;
        text-align: center;
        margin-bottom: 8px;
    }
    .section-sub {
        text-align: center;
        color: #9c9ca6;
        font-size: 0.95rem;
        margin-bottom: 34px;
    }

    /* ================= FEATURE TILES ================= */
    .tile-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 20px;
        margin-bottom: 46px;
    }
    @media (max-width: 800px) { .tile-grid { grid-template-columns: 1fr; } }
    .tile {
        position: relative;
        height: 280px;
        border-radius: 24px;
        overflow: hidden;
        border: 1px solid rgba(255, 255, 255, 0.09);
        background: #0a0a10;
    }
    .tile-glow { position: absolute; border-radius: 50%; filter: blur(50px); opacity: 0.55; }
    .tile-glow-a { width: 220px; height: 220px; top: -40px; left: -30px; background: #fb923c; }
    .tile-glow-b { width: 220px; height: 220px; bottom: -50px; right: -30px; background: #3b82f6; }
    .tile svg { position: absolute; inset: 0; width: 100%; height: 100%; opacity: 0.4; }
    .tile-overlay {
        position: absolute; left: 0; right: 0; bottom: 0;
        padding: 20px 22px;
        background: linear-gradient(180deg, transparent 0%, rgba(5,5,8,0.75) 55%, rgba(5,5,8,0.97) 100%);
    }
    .tile-title { font-weight: 600; font-size: 1.05rem; color: #f5f5f7; margin-bottom: 6px; }
    .tile-text { font-size: 0.82rem; color: #b5b5c0; line-height: 1.5; }

    /* ================= GLASS CARD ================= */
    .glass-card {
        background: rgba(255, 255, 255, 0.035);
        border: 1px solid rgba(255, 255, 255, 0.09);
        border-radius: 20px;
        padding: 28px 30px;
        backdrop-filter: blur(20px);
        margin-bottom: 22px;
    }
    .card-eyebrow {
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: #93c5fd;
        margin-bottom: 6px;
    }
    .card-title {
        font-size: 1.35rem;
        font-weight: 600;
        color: #f5f5f7;
        margin-bottom: 4px;
    }

    /* ================= BUTTONS (streamlit) ================= */
    div.stButton > button {
        background: linear-gradient(90deg, #fb923c, #3b82f6);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.6rem 1.4rem;
        font-weight: 600;
        font-size: 0.92rem;
        transition: all 0.2s ease;
        width: 100%;
    }
    div.stButton > button:hover {
        box-shadow: 0 0 24px rgba(59, 130, 246, 0.45);
        transform: translateY(-1px);
    }

    /* ================= INPUTS ================= */
    div[data-baseweb="input"] > div, div[data-baseweb="select"] > div, div[data-baseweb="textarea"] > div {
        background: rgba(255, 255, 255, 0.04) !important;
        border: 1px solid rgba(255, 255, 255, 0.12) !important;
        border-radius: 10px !important;
        color: #f2f2f5 !important;
    }
    .stRadio > label, .stSelectbox > label, .stTextInput > label { color: #9ca3af !important; font-size: 0.85rem !important; }
    div[role="radiogroup"] label { color: #d1d1d6 !important; }

    /* ================= TABS ================= */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background: rgba(255, 255, 255, 0.03);
        padding: 6px;
        border-radius: 14px;
        border: 1px solid rgba(255, 255, 255, 0.07);
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px;
        color: #9ca3af;
        font-weight: 500;
        padding: 8px 16px;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(90deg, rgba(251,146,60,0.22), rgba(59,130,246,0.28));
        color: #f5f5f7 !important;
    }

    /* ================= CHAT ================= */
    div[data-testid="stChatMessage"] {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.07);
        border-radius: 14px;
    }

    /* ================= STEP TRACKER ================= */
    .step-row { display: flex; align-items: center; gap: 12px; padding: 9px 4px; font-size: 0.9rem; }
    .step-pending { color: #55555f; }
    .step-active { color: #93c5fd; font-weight: 600; }
    .step-done { color: #6ee7b7; }
    .step-dot { width: 8px; height: 8px; border-radius: 50%; display: inline-block; flex-shrink: 0; }
    .dot-pending { background: #33333d; }
    .dot-active { background: #60a5fa; box-shadow: 0 0 8px #60a5fa; }
    .dot-done { background: #34d399; }

    hr { border-color: rgba(255,255,255,0.08); }
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------------------
# SESSION STATE
# --------------------------------------------------------------------------------------
if "result" not in st.session_state:
    st.session_state.result = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --------------------------------------------------------------------------------------
# NAVBAR
# --------------------------------------------------------------------------------------
st.markdown(
    """
    <div class="navbar">
        <div class="logo">🎬 AIDEN</div>
        <div class="nav-links">
            <a href="#features">Features</a>
            <a href="#analyze">Analyze</a>
            <a href="#chat">Chat</a>
        </div>
        <a class="nav-cta" href="#analyze">Get Started</a>
    </div>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------------------
# HERO
# --------------------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero-shell">
        <div class="hero-glow-1"></div>
        <div class="hero-glow-2"></div>
        <div class="hero-rays"></div>
        <div class="hero-horizon"></div>
        <div class="hero-content">
            <div class="badge-pill">🚀 Now live — analyze any video in minutes</div>
            <div class="hero-title">Turn any video into<br><span class="grad">structured knowledge</span></div>
            <div class="hero-sub">Drop a YouTube link or upload a recording — get a transcript, summary,
                action items, key decisions, and a chat assistant that actually knows what was said.</div>
            <div class="hero-cta-row">
                <a class="btn-primary" href="#analyze">✨ Analyze a video</a>
                <a class="btn-ghost" href="#features">See how it works</a>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------------------
# FEATURE TILES
# --------------------------------------------------------------------------------------
st.markdown('<div id="features"></div>', unsafe_allow_html=True)
st.markdown('<div class="section-eyebrow">Capabilities</div>',
            unsafe_allow_html=True)
st.markdown('<div class="section-heading">What sets Aiden apart</div>',
            unsafe_allow_html=True)
st.markdown(
    '<div class="section-sub">Everything you need to go from raw footage to answers, in one pipeline.</div>',
    unsafe_allow_html=True,
)

waveform_svg = """
<svg viewBox="0 0 400 280" preserveAspectRatio="xMidYMid slice">
  <g stroke="#fdba74" stroke-width="6" stroke-linecap="round">
    <line x1="60" y1="140" x2="60" y2="110" /><line x1="80" y1="140" x2="80" y2="80" />
    <line x1="100" y1="140" x2="100" y2="160" /><line x1="120" y1="140" x2="120" y2="60" />
    <line x1="140" y1="140" x2="140" y2="190" /><line x1="160" y1="140" x2="160" y2="100" />
    <line x1="180" y1="140" x2="180" y2="170" /><line x1="200" y1="140" x2="200" y2="70" />
    <line x1="220" y1="140" x2="220" y2="150" /><line x1="240" y1="140" x2="240" y2="95" />
    <line x1="260" y1="140" x2="260" y2="180" /><line x1="280" y1="140" x2="280" y2="115" />
    <line x1="300" y1="140" x2="300" y2="90" /><line x1="320" y1="140" x2="320" y2="165" />
    <line x1="340" y1="140" x2="340" y2="105" />
  </g>
</svg>
"""

graph_svg = """
<svg viewBox="0 0 400 280" preserveAspectRatio="xMidYMid slice">
  <g stroke="#93c5fd" stroke-width="2.5" fill="none">
    <line x1="80" y1="60" x2="160" y2="120" /><line x1="160" y1="120" x2="270" y2="80" />
    <line x1="160" y1="120" x2="140" y2="200" /><line x1="140" y1="200" x2="250" y2="220" />
    <line x1="270" y1="80" x2="330" y2="150" /><line x1="140" y1="200" x2="70" y2="180" />
  </g>
  <g fill="#93c5fd">
    <circle cx="80" cy="60" r="7" /><circle cx="160" cy="120" r="9" />
    <circle cx="270" cy="80" r="6" /><circle cx="140" cy="200" r="8" />
    <circle cx="250" cy="220" r="6" /><circle cx="330" cy="150" r="6" />
    <circle cx="70" cy="180" r="5" />
  </g>
</svg>
"""

chat_svg = """
<svg viewBox="0 0 400 280" preserveAspectRatio="xMidYMid slice">
  <rect x="60" y="60" width="180" height="90" rx="20" fill="none" stroke="#fdba74" stroke-width="4" />
  <path d="M90 150 L90 180 L130 150 Z" fill="none" stroke="#fdba74" stroke-width="4" />
  <rect x="180" y="130" width="160" height="80" rx="20" fill="none" stroke="#93c5fd" stroke-width="4" />
  <path d="M340 210 L340 236 L302 210 Z" fill="none" stroke="#93c5fd" stroke-width="4" />
</svg>
"""

tiles = [
    (waveform_svg, "Transcription engine",
     "Chunked processing turns long recordings into clean, accurate text in minutes."),
    (graph_svg, "Smart extraction",
     "Automatically pulls action items, key decisions, and open questions from the conversation."),
    (chat_svg, "Chat with your meeting",
     "Ask follow-up questions and get answers grounded in the transcript, not guesses."),
]

tiles_html = '<div class="tile-grid">'
for svg, title, text in tiles:
    tiles_html += (
        '<div class="tile">'
        '<div class="tile-glow tile-glow-a"></div>'
        '<div class="tile-glow tile-glow-b"></div>'
        f"{svg}"
        '<div class="tile-overlay">'
        f'<div class="tile-title">{title}</div>'
        f'<div class="tile-text">{text}</div>'
        "</div></div>"
    )
tiles_html += "</div>"
st.markdown(tiles_html, unsafe_allow_html=True)

# --------------------------------------------------------------------------------------
# INPUT CARD
# --------------------------------------------------------------------------------------
st.markdown('<div id="analyze"></div>', unsafe_allow_html=True)
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.markdown('<div class="card-eyebrow">Get Started</div>',
            unsafe_allow_html=True)
st.markdown('<div class="card-title">Analyze a video</div>',
            unsafe_allow_html=True)
st.write("")

source_mode = st.radio("Source", [
                       "YouTube URL", "Upload file"], horizontal=True, label_visibility="collapsed")

source_value = None
tmp_path_to_cleanup = None
language = "english"

if source_mode == "YouTube URL":
    source_value = st.text_input(
        "YouTube URL", placeholder="https://www.youtube.com/watch?v=...", label_visibility="collapsed"
    )
else:
    uploaded = st.file_uploader(
        "Upload audio/video", type=["mp3", "wav", "m4a", "mp4", "mov", "webm"], label_visibility="collapsed"
    )
    if uploaded is not None:
        suffix = os.path.splitext(uploaded.name)[1]
        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        tmp_file.write(uploaded.read())
        tmp_file.close()
        source_value = tmp_file.name
        tmp_path_to_cleanup = tmp_file.name

analyze_clicked = st.button("✨ Analyze video", use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)

# --------------------------------------------------------------------------------------
# PIPELINE RUNNER WITH LIVE STEP TRACKER
# --------------------------------------------------------------------------------------
STEP_DEFS = [
    ("chunk", "Processing input & chunking audio"),
    ("transcribe", "Transcribing audio"),
    ("title", "Generating title"),
    ("summary", "Summarizing content"),
    ("actions", "Extracting action items"),
    ("decisions", "Extracting key decisions"),
    ("questions", "Extracting open questions"),
    ("rag", "Building chat knowledge base"),
]


def render_steps(tracker_slot, statuses):
    rows = []
    for key, label in STEP_DEFS:
        state = statuses.get(key, "pending")
        if state == "done":
            rows.append(
                f'<div class="step-row step-done"><span class="step-dot dot-done"></span>✓ {label}</div>')
        elif state == "active":
            rows.append(
                f'<div class="step-row step-active"><span class="step-dot dot-active"></span>{label}…</div>')
        else:
            rows.append(
                f'<div class="step-row step-pending"><span class="step-dot dot-pending"></span>{label}</div>')
    tracker_slot.markdown("".join(rows), unsafe_allow_html=True)


def run_pipeline_with_ui(source: str, language: str) -> dict:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="card-eyebrow">Working</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="card-title">Processing your video</div>',
                unsafe_allow_html=True)
    st.write("")
    tracker_slot = st.empty()
    statuses = {}
    render_steps(tracker_slot, statuses)

    def mark(key, state):
        statuses[key] = state
        render_steps(tracker_slot, statuses)

    result = {}

    mark("chunk", "active")
    chunks = process_input(source)
    mark("chunk", "done")

    mark("transcribe", "active")
    transcript = transcribe_all(chunks, language)
    mark("transcribe", "done")
    result["transcript"] = transcript

    mark("title", "active")
    result["title"] = generate_title(transcript)
    mark("title", "done")

    mark("summary", "active")
    result["summary"] = summarize(transcript)
    mark("summary", "done")

    mark("actions", "active")
    result["action_items"] = extract_action_items(transcript)
    mark("actions", "done")

    mark("decisions", "active")
    result["key_decisions"] = extract_key_decisions(transcript)
    mark("decisions", "done")

    mark("questions", "active")
    result["open_questions"] = extract_questions(transcript)
    mark("questions", "done")

    mark("rag", "active")
    result["rag_chain"] = build_rag_chain(transcript)
    mark("rag", "done")

    st.markdown("</div>", unsafe_allow_html=True)
    return result


if analyze_clicked:
    if not source_value:
        st.error("Add a YouTube URL or upload a file first.")
    else:
        try:
            st.session_state.result = run_pipeline_with_ui(
                source_value, language)
            st.session_state.chat_history = []
        except Exception as e:
            st.error(f"Something went wrong while processing: {e}")
        finally:
            if tmp_path_to_cleanup and os.path.exists(tmp_path_to_cleanup):
                try:
                    os.remove(tmp_path_to_cleanup)
                except OSError:
                    pass

# --------------------------------------------------------------------------------------
# RESULTS
# --------------------------------------------------------------------------------------
result = st.session_state.result

if result:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="card-eyebrow">Results</div>',
                unsafe_allow_html=True)
    st.markdown(
        f'<div class="card-title">📌 {result["title"]}</div>', unsafe_allow_html=True)
    st.write("")

    tabs = st.tabs(["📋 Summary", "✅ Action items",
                   "🔑 Key decisions", "❓ Open questions", "📝 Transcript"])

    with tabs[0]:
        st.write(result["summary"])
    with tabs[1]:
        st.write(result["action_items"])
    with tabs[2]:
        st.write(result["key_decisions"])
    with tabs[3]:
        st.write(result["open_questions"])
    with tabs[4]:
        st.text_area(
            "Full transcript", result["transcript"], height=300, label_visibility="collapsed")
        st.download_button(
            "Download transcript (.txt)",
            data=result["transcript"],
            file_name="transcript.txt",
            mime="text/plain",
        )

    st.markdown("</div>", unsafe_allow_html=True)

    # ---------------- Chat ----------------
    st.markdown('<div id="chat"></div>', unsafe_allow_html=True)
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="card-eyebrow">Phase 2</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="card-title">💬 Chat with your meeting</div>',
                unsafe_allow_html=True)
    st.write("")

    for role, msg in st.session_state.chat_history:
        with st.chat_message(role, avatar=("🧑" if role == "user" else "🤖")):
            st.write(msg)

    question = st.chat_input("Ask a question about this video...")
    if question:
        st.session_state.chat_history.append(("user", question))
        with st.chat_message("user", avatar="🧑"):
            st.write(question)
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("Thinking..."):
                try:
                    answer = ask_question(result["rag_chain"], question)
                except Exception as e:
                    answer = f"Sorry, I couldn't answer that: {e}"
            st.write(answer)
        st.session_state.chat_history.append(("assistant", answer))

    st.markdown("</div>", unsafe_allow_html=True)
else:
    st.markdown('<div id="chat"></div>', unsafe_allow_html=True)
    st.markdown(
        '<div style="text-align:center; color:#55555f; padding: 40px 0; font-size:0.9rem;">'
        "Results and chat will appear here once you analyze a video."
        "</div>",
        unsafe_allow_html=True,
    )
