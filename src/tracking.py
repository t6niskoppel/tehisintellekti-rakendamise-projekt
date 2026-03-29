import csv
import json
from datetime import datetime, timezone
from typing import Dict, List

import streamlit as st
import tiktoken

from .config import FEEDBACK_LOG_PATH, USAGE_DIR, USAGE_PATH

_enc = tiktoken.get_encoding("cl100k_base")


def count_text_tokens(text: str) -> int:
    return len(_enc.encode(text or ""))


def count_messages_tokens(messages: List[Dict[str, str]]) -> int:
    combined = "\n".join(
        f"{m.get('role', '')}: {m.get('content', '')}" for m in messages
    )
    return count_text_tokens(combined)


def log_feedback(
    timestamp, prompt, filters, context_ids, context_names,
    response, rating, error_category,
):
    file_exists = FEEDBACK_LOG_PATH.exists()
    with open(FEEDBACK_LOG_PATH, "a", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        if not file_exists:
            writer.writerow([
                "Aeg", "Kasutaja päring", "Filtrid", "Leitud ID-d",
                "Leitud ained", "LLM Vastus", "Hinnang", "Veatüüp",
            ])
        writer.writerow([
            timestamp, prompt, filters, str(context_ids),
            str(context_names), response, rating, error_category,
        ])


def _ensure_usage_store():
    USAGE_DIR.mkdir(parents=True, exist_ok=True)
    if not USAGE_PATH.exists():
        _save_usage({"totals": {}, "interactions": []})


def load_usage() -> Dict:
    _ensure_usage_store()
    try:
        with open(USAGE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {"totals": {}, "interactions": []}


def _save_usage(data: Dict):
    with open(USAGE_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _update_usage(
    usage: Dict, model: str, input_tokens: int, output_tokens: int, filters: Dict,
) -> Dict:
    if model not in usage["totals"]:
        usage["totals"][model] = {"input_tokens": 0, "output_tokens": 0}
    usage["totals"][model]["input_tokens"] += input_tokens
    usage["totals"][model]["output_tokens"] += output_tokens
    usage["interactions"].append({
        "ts": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "filters": filters,
    })
    return usage


def add_token_usage(
    model: str, input_tokens: int, output_tokens: int, filters: Dict, mode: str,
):
    input_tokens = int(input_tokens)
    output_tokens = int(output_tokens)

    if model not in st.session_state.session_tokens:
        st.session_state.session_tokens[model] = {"input": 0, "output": 0}
    st.session_state.session_tokens[model]["input"] += input_tokens
    st.session_state.session_tokens[model]["output"] += output_tokens

    if "mode_tokens" not in st.session_state:
        st.session_state.mode_tokens = {
            "chat": {"input": 0, "output": 0},
            "test": {"input": 0, "output": 0},
        }
    if mode not in st.session_state.mode_tokens:
        st.session_state.mode_tokens[mode] = {"input": 0, "output": 0}
    st.session_state.mode_tokens[mode]["input"] += input_tokens
    st.session_state.mode_tokens[mode]["output"] += output_tokens

    st.session_state.last_tokens = {"input": input_tokens, "output": output_tokens}

    usage = load_usage()
    usage = _update_usage(usage, model, input_tokens, output_tokens, filters)
    _save_usage(usage)
