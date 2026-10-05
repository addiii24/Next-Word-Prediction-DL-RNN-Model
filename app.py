import streamlit as st
import numpy as np
import pickle
import os
import json
import h5py
import tempfile
import shutil
from tensorflow.keras.models import load_model, model_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ─────────────────────────────────────────────
#  Page Configuration
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Next Word Predictor · LSTM",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
#  Custom CSS – Dark Glassmorphism Theme
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: linear-gradient(135deg, #0d0d1a 0%, #0f1729 40%, #0a1628 70%, #0d0d1a 100%);
    min-height: 100vh;
}

.stApp::before {
    content: '';
    position: fixed;
    top: -20%;
    left: -10%;
    width: 600px;
    height: 600px;
    background: radial-gradient(circle, rgba(99,102,241,0.12) 0%, transparent 70%);
    border-radius: 50%;
    pointer-events: none;
    z-index: 0;
    animation: drift 10s ease-in-out infinite alternate;
}

@keyframes drift {
    0%   { transform: translate(0, 0) scale(1); }
    100% { transform: translate(40px, 30px) scale(1.05); }
}

#MainMenu, footer, header { visibility: hidden; }

.main .block-container {
    padding: 2rem 2rem 4rem 2rem;
    max-width: 820px;
    margin: 0 auto;
}

.hero-header {
    text-align: center;
    padding: 2.5rem 0 1.5rem 0;
    animation: fadeInDown 0.8s ease both;
}

.hero-badge {
    display: inline-block;
    background: linear-gradient(90deg, rgba(99,102,241,0.2), rgba(139,92,246,0.2));
    border: 1px solid rgba(139,92,246,0.35);
    border-radius: 50px;
    padding: 0.3rem 1rem;
    font-size: 0.75rem;
    font-weight: 600;
    color: #a78bfa;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 1rem;
}

.hero-title {
    font-size: clamp(2rem, 5vw, 3rem);
    font-weight: 700;
    background: linear-gradient(135deg, #e0e7ff 0%, #a78bfa 50%, #818cf8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.2;
    margin-bottom: 0.6rem;
}

.hero-subtitle {
    color: rgba(167,139,250,0.7);
    font-size: 1rem;
    font-weight: 400;
    letter-spacing: 0.02em;
}

@keyframes fadeInDown {
    from { opacity: 0; transform: translateY(-24px); }
    to   { opacity: 1; transform: translateY(0); }
}

.glass-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(139,92,246,0.18);
    border-radius: 20px;
    padding: 1.8rem;
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    box-shadow: 0 8px 40px rgba(0,0,0,0.35), inset 0 1px 0 rgba(255,255,255,0.06);
    margin-bottom: 1.2rem;
    animation: fadeInUp 0.6s ease both;
}

@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(20px); }
    to   { opacity: 1; transform: translateY(0); }
}

.section-label {
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: rgba(139,92,246,0.8);
    margin-bottom: 0.5rem;
}

div[data-testid="stTextArea"] textarea {
    background: rgba(15,20,40,0.8) !important;
    border: 1.5px solid rgba(99,102,241,0.3) !important;
    border-radius: 14px !important;
    color: #e0e7ff !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 1rem !important;
    line-height: 1.7 !important;
    padding: 1rem 1.2rem !important;
    transition: border-color 0.3s, box-shadow 0.3s !important;
    resize: vertical !important;
}

div[data-testid="stTextArea"] textarea:focus {
    border-color: rgba(139,92,246,0.7) !important;
    box-shadow: 0 0 0 3px rgba(139,92,246,0.12) !important;
    outline: none !important;
}

div[data-testid="stTextArea"] label {
    color: rgba(199,210,254,0.8) !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
}

div[data-testid="stSlider"] label {
    color: rgba(199,210,254,0.8) !important;
    font-weight: 500 !important;
}

div[data-testid="stButton"] > button {
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.75rem 2rem !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
    font-family: 'Inter', sans-serif !important;
    letter-spacing: 0.03em !important;
    cursor: pointer !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 20px rgba(99,102,241,0.35) !important;
    width: 100% !important;
}

div[data-testid="stButton"] > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 30px rgba(99,102,241,0.55) !important;
    background: linear-gradient(135deg, #818cf8 0%, #a78bfa 100%) !important;
}

div[data-testid="stSelectbox"] label {
    color: rgba(199,210,254,0.8) !important;
    font-weight: 500 !important;
}

div[data-testid="stSelectbox"] > div > div {
    background: rgba(15,20,40,0.8) !important;
    border: 1.5px solid rgba(99,102,241,0.3) !important;
    border-radius: 10px !important;
    color: #e0e7ff !important;
}

.result-container {
    background: linear-gradient(135deg, rgba(99,102,241,0.08), rgba(139,92,246,0.12));
    border: 1.5px solid rgba(139,92,246,0.35);
    border-radius: 18px;
    padding: 1.6rem 2rem;
    margin-top: 1rem;
    animation: glowPulse 2.5s ease-in-out infinite alternate;
}

@keyframes glowPulse {
    from { box-shadow: 0 4px 30px rgba(99,102,241,0.15); }
    to   { box-shadow: 0 8px 50px rgba(139,92,246,0.30); }
}

.result-label {
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #a78bfa;
    margin-bottom: 0.8rem;
}

.result-text {
    font-size: 1.25rem;
    font-weight: 500;
    color: #e0e7ff;
    line-height: 1.75;
    font-family: 'Inter', sans-serif;
}

.predicted-word {
    display: inline-block;
    background: linear-gradient(90deg, #6366f1, #a78bfa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-weight: 700;
    font-size: 1.4rem;
    border-bottom: 2px solid rgba(167,139,250,0.4);
    padding-bottom: 1px;
}

.chips-container {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
    margin-top: 0.8rem;
}

.chip {
    background: rgba(99,102,241,0.15);
    border: 1px solid rgba(99,102,241,0.3);
    border-radius: 50px;
    padding: 0.25rem 0.85rem;
    font-size: 0.82rem;
    font-weight: 500;
    color: #c7d2fe;
}

.chip-rank {
    color: rgba(167,139,250,0.6);
    font-size: 0.7rem;
    margin-right: 2px;
}

.stat-row {
    display: flex;
    gap: 0.8rem;
    flex-wrap: wrap;
    margin-top: 1rem;
}

.stat-badge {
    flex: 1;
    min-width: 120px;
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(139,92,246,0.15);
    border-radius: 12px;
    padding: 0.8rem 1rem;
    text-align: center;
}

.stat-value {
    font-size: 1.4rem;
    font-weight: 700;
    color: #a78bfa;
    display: block;
}

.stat-key {
    font-size: 0.7rem;
    color: rgba(167,139,250,0.5);
    text-transform: uppercase;
    letter-spacing: 0.1em;
    font-weight: 600;
    margin-top: 0.2rem;
    display: block;
}

section[data-testid="stSidebar"] {
    background: rgba(10,15,35,0.95) !important;
    border-right: 1px solid rgba(99,102,241,0.12) !important;
}

hr {
    border-color: rgba(99,102,241,0.15) !important;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  Load Model & Artifacts (cached)
# ─────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource(show_spinner=False)
def load_artifacts():
    model = _load_model_safe(os.path.join(BASE_DIR, "lstm_model.h5"))
    with open(os.path.join(BASE_DIR, "tokenizer.pkl"), "rb") as f:
        tokenizer = pickle.load(f)
    with open(os.path.join(BASE_DIR, "max_len.pkl"), "rb") as f:
        max_len = pickle.load(f)
    return model, tokenizer, max_len


def _strip_unsupported_keys(config):
    """Recursively remove keys unknown to older Keras (e.g. quantization_config)."""
    UNSUPPORTED = {"quantization_config"}
    if isinstance(config, dict):
        return {
            k: _strip_unsupported_keys(v)
            for k, v in config.items()
            if k not in UNSUPPORTED
        }
    if isinstance(config, list):
        return [_strip_unsupported_keys(item) for item in config]
    return config


def _load_model_safe(h5_path):
    """Load an H5 model, patching quantization_config issues from Keras 3.x."""
    # First try a plain load (works when versions match)
    try:
        return load_model(h5_path, compile=False)
    except Exception:
        pass

    # Fallback: patch the model config stored inside the H5 file
    tmp_dir = tempfile.mkdtemp()
    try:
        tmp_path = os.path.join(tmp_dir, "model_patched.h5")
        shutil.copy2(h5_path, tmp_path)

        with h5py.File(tmp_path, "r+") as f:
            if "model_config" in f.attrs:
                raw = f.attrs["model_config"]
                # h5py may return bytes or str
                raw_str = raw.decode("utf-8") if isinstance(raw, bytes) else raw
                cfg = json.loads(raw_str)
                cfg = _strip_unsupported_keys(cfg)
                f.attrs["model_config"] = json.dumps(cfg)

        return load_model(tmp_path, compile=False)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ─────────────────────────────────────────────
#  Prediction Helpers
# ─────────────────────────────────────────────
def predict_next_word(model, tokenizer, max_len, text):
    token_list = tokenizer.texts_to_sequences([text])[0]
    if not token_list:
        return None
    token_list = pad_sequences([token_list], maxlen=max_len - 1, padding="pre")
    predicted = model.predict(token_list, verbose=0)
    predicted_idx = np.argmax(predicted, axis=-1)[0]
    for word, idx in tokenizer.word_index.items():
        if idx == predicted_idx:
            return word
    return None


def predict_top_k_words(model, tokenizer, max_len, text, k=5):
    token_list = tokenizer.texts_to_sequences([text])[0]
    if not token_list:
        return []
    token_list = pad_sequences([token_list], maxlen=max_len - 1, padding="pre")
    predictions = model.predict(token_list, verbose=0)[0]
    top_k_idx = np.argsort(predictions)[-k:][::-1]
    idx_to_word = {idx: word for word, idx in tokenizer.word_index.items()}
    return [(idx_to_word.get(i, "?"), float(predictions[i])) for i in top_k_idx]


def generate_sequence(model, tokenizer, max_len, seed, n_words):
    current = seed.strip()
    generated = []
    for _ in range(n_words):
        word = predict_next_word(model, tokenizer, max_len, current)
        if word is None:
            break
        generated.append(word)
        current = current + " " + word
    return seed.strip() + " " + " ".join(generated)


# ─────────────────────────────────────────────
#  Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 1rem 0 0.5rem 0;'>
        <div style='font-size:2.5rem;'>🧠</div>
        <div style='font-weight:700; font-size:1.1rem; color:#e0e7ff; margin-top:0.4rem;'>LSTM Predictor</div>
        <div style='font-size:0.75rem; color:rgba(167,139,250,0.6); margin-top:0.2rem;'>Neural Language Model</div>
    </div>
    <hr>
    """, unsafe_allow_html=True)

    st.markdown("#### ⚙️ Settings")

    mode = st.selectbox(
        "Prediction Mode",
        options=["Single Next Word", "Top-K Suggestions", "Auto-Generate Sequence"],
        help="Choose how the model predicts."
    )

    top_k = 5
    n_words = 5

    if mode == "Top-K Suggestions":
        top_k = st.slider("Number of suggestions (K)", min_value=2, max_value=15, value=5, step=1)

    if mode == "Auto-Generate Sequence":
        n_words = st.slider("Words to generate", min_value=1, max_value=30, value=10, step=1)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("""
    <div style='font-size:0.75rem; color:rgba(167,139,250,0.5); line-height:1.7;'>
    <b style='color:rgba(167,139,250,0.8)'>Model Architecture</b><br>
    Type &nbsp;&nbsp;&nbsp;→ LSTM<br>
    Task &nbsp;&nbsp;&nbsp;→ Next Word Prediction<br>
    Tokens → Keras Tokenizer<br>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  Hero Header
# ─────────────────────────────────────────────
st.markdown("""
<div class="hero-header">
    <div class="hero-badge">⚡ Deep Learning · LSTM</div>
    <div class="hero-title">Next Word Predictor</div>
    <div class="hero-subtitle">Powered by a Long Short-Term Memory neural network</div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  Load Model
# ─────────────────────────────────────────────
with st.spinner("🔄 Loading model & tokenizer…"):
    try:
        model, tokenizer, max_len = load_artifacts()
        model_loaded = True
    except Exception as e:
        model_loaded = False
        st.error(f"❌ Failed to load model artifacts: {e}")

if model_loaded:
    vocab_size = len(tokenizer.word_index) + 1

    # Stats row
    st.markdown("""
    <div class="stat-row">
        <div class="stat-badge">
            <span class="stat-value">{vocab}</span>
            <span class="stat-key">Vocabulary</span>
        </div>
        <div class="stat-badge">
            <span class="stat-value">{maxlen}</span>
            <span class="stat-key">Max Seq Len</span>
        </div>
        <div class="stat-badge">
            <span class="stat-value">LSTM</span>
            <span class="stat-key">Architecture</span>
        </div>
        <div class="stat-badge">
            <span class="stat-value">✅</span>
            <span class="stat-key">Model Status</span>
        </div>
    </div>
    """.format(vocab=f"{vocab_size:,}", maxlen=max_len), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Input card
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<p class="section-label">📝 Input Text</p>', unsafe_allow_html=True)

    seed_text = st.text_area(
        label="seed_text",
        placeholder="e.g.  The quick brown fox jumps over…",
        height=120,
        label_visibility="collapsed",
    )

    col1, col2 = st.columns([3, 1])
    with col1:
        predict_btn = st.button("🔮 Predict", use_container_width=True)
    with col2:
        clear_btn = st.button("🗑️ Clear", use_container_width=True)

    if clear_btn:
        st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)

    # Prediction output
    if predict_btn:
        if not seed_text.strip():
            st.warning("⚠️ Please enter some seed text before predicting.")
        else:
            with st.spinner("✨ Generating prediction…"):

                if mode == "Single Next Word":
                    word = predict_next_word(model, tokenizer, max_len, seed_text.strip())
                    if word:
                        st.markdown(f"""
                        <div class="result-container">
                            <div class="result-label">✦ Predicted Next Word</div>
                            <div class="result-text">
                                {seed_text.strip()} <span class="predicted-word">{word}</span>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.error("❌ Could not predict. Try a different input.")

                elif mode == "Top-K Suggestions":
                    top_words = predict_top_k_words(model, tokenizer, max_len, seed_text.strip(), k=top_k)
                    if top_words:
                        best_word, best_prob = top_words[0]
                        st.markdown(f"""
                        <div class="result-container">
                            <div class="result-label">✦ Top Prediction</div>
                            <div class="result-text">
                                {seed_text.strip()} <span class="predicted-word">{best_word}</span>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                        st.markdown("<br>", unsafe_allow_html=True)
                        st.markdown('<p class="section-label">🎯 All Suggestions (ranked by probability)</p>', unsafe_allow_html=True)

                        chips_html = '<div class="chips-container">'
                        for rank, (w, prob) in enumerate(top_words, 1):
                            chips_html += f'<div class="chip"><span class="chip-rank">#{rank}</span> {w} <span style="opacity:0.5;font-size:0.72rem;">({prob:.2%})</span></div>'
                        chips_html += '</div>'
                        st.markdown(chips_html, unsafe_allow_html=True)

                        st.markdown("<br>", unsafe_allow_html=True)
                        st.markdown('<p class="section-label">📊 Probability Distribution</p>', unsafe_allow_html=True)
                        import pandas as pd
                        df = pd.DataFrame(top_words, columns=["Word", "Probability"])
                        df = df.set_index("Word")
                        st.bar_chart(df, color="#6366f1")
                    else:
                        st.error("❌ Could not predict. Try a different input.")

                elif mode == "Auto-Generate Sequence":
                    generated = generate_sequence(model, tokenizer, max_len, seed_text.strip(), n_words)
                    seed_words = seed_text.strip().split()
                    seed_len = len(seed_words)
                    words = generated.split()
                    original = " ".join(words[:seed_len])
                    new_words = " ".join(words[seed_len:])

                    st.markdown(f"""
                    <div class="result-container">
                        <div class="result-label">✦ Generated Sequence ({n_words} new words)</div>
                        <div class="result-text">
                            {original} <span class="predicted-word">{new_words}</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

    # How to use expander
    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("💡 How to Use"):
        st.markdown("""
        | Mode | Description |
        |------|-------------|
        | **Single Next Word** | Predicts the single most likely word to follow your input |
        | **Top-K Suggestions** | Shows the top K candidate words with their probabilities |
        | **Auto-Generate Sequence** | Iteratively predicts to build a longer text sequence |

        **Tips:**
        - Provide at least 2–3 words for better predictions
        - The model learned patterns from its training corpus
        - Longer seed texts give more contextual predictions
        """)
