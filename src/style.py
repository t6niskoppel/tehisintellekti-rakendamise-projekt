MODERN_CSS = """
<style>
    /* ── Force readable colors everywhere ── */
    :root {
        --ut-blue: #005aa9;
        --ut-blue-dark: #003d73;
        --ut-blue-light: #e8f1fa;
        --ut-light: #f4f8fc;
        --ut-border: rgba(0, 90, 169, 0.15);
        --ut-text: #1a1a2e;
        --ut-text-muted: #5a6577;
    }

    /* ── Hide chrome ── */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* ── Layout ── */
    .main > div {
        max-width: 900px;
        margin: 0 auto;
    }

    /* ── Typography ── */
    h1 {
        color: var(--ut-blue-dark) !important;
        font-weight: 700 !important;
        letter-spacing: -0.02em;
    }
    h2, h3 {
        color: var(--ut-text) !important;
        letter-spacing: -0.01em;
    }
    p, li, span, label {
        color: var(--ut-text);
    }

    /* ── Chat messages ── */
    .stChatMessage {
        border-radius: 14px;
        padding: 14px 18px;
        margin-bottom: 10px;
        border: 1px solid var(--ut-border);
        background: #ffffff;
        color: var(--ut-text) !important;
    }
    .stChatMessage p,
    .stChatMessage li,
    .stChatMessage span,
    .stChatMessage a {
        color: var(--ut-text) !important;
    }
    .stChatMessage a {
        color: var(--ut-blue) !important;
        text-decoration: underline;
    }

    /* ── Chat input ── */
    .stChatInputContainer {
        border-radius: 12px;
        border: 1px solid var(--ut-border);
        box-shadow: 0 2px 8px rgba(0, 90, 169, 0.06);
        background-color: #ffffff;
    }
    .stChatInputContainer textarea,
    .stChatInputContainer input {
        color: var(--ut-text) !important;
        background-color: #ffffff !important;
    }

    /* ── Buttons ── */
    .stButton > button {
        border-radius: 10px;
        border: 1px solid var(--ut-blue);
        color: var(--ut-blue-dark);
        background: #ffffff;
        font-weight: 500;
        transition: all 0.15s ease;
    }
    .stButton > button:hover {
        color: #ffffff;
        background: var(--ut-blue);
        border-color: var(--ut-blue);
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background-color: var(--ut-light);
        border-right: 1px solid var(--ut-border);
    }
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] span {
        color: var(--ut-text) !important;
    }
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: var(--ut-blue-dark) !important;
    }

    /* ── Widgets (inputs, selects, sliders) ── */
    .stSelectbox label,
    .stMultiSelect label,
    .stSlider label,
    .stCheckbox label,
    .stNumberInput label {
        color: var(--ut-text) !important;
    }

    /* ── Metrics ── */
    [data-testid="stMetricValue"] {
        color: var(--ut-blue-dark) !important;
    }
    [data-testid="stMetricLabel"] {
        color: var(--ut-text-muted) !important;
    }

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        color: var(--ut-text) !important;
    }

    /* ── Dividers ── */
    hr {
        border-color: var(--ut-border) !important;
    }
</style>
"""

SIDEBAR_LOGO_HTML = """
<div style="text-align: center; padding: 8px 0 4px 0;">
    <div style="
        font-size: 1.3rem;
        font-weight: 700;
        color: #003d73;
        letter-spacing: -0.02em;
    ">TÜ Ainete Nõustaja</div>
    <div style="
        font-size: 0.78rem;
        color: #5a6577;
        margin-top: 2px;
    ">Tartu Ülikool</div>
</div>
"""
