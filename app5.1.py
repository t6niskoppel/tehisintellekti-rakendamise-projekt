import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import tiktoken
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


BASE_DIR = Path(__file__).resolve().parent
DATA_CSV_PATH = BASE_DIR / "andmed" / "puhtad_andmed.csv"
EMB_PKL_PATH = BASE_DIR / "andmed" / "puhtad_andmed_embeddings.pkl"
USAGE_DIR = BASE_DIR / ".usage"
USAGE_PATH = USAGE_DIR / "token_usage.json"

MODEL_OPTIONS = ["google/gemma-3-27b-it:free", "google/gemma-3-27b-it"]


enc = tiktoken.get_encoding("cl100k_base")


def count_text_tokens(text: str) -> int:
    return len(enc.encode(text or ""))


def count_messages_tokens(messages: List[Dict[str, str]]) -> int:
    serialized = "\n".join(f"{m.get('role', '')}: {m.get('content', '')}" for m in messages)
    return count_text_tokens(serialized)


def ensure_usage_store() -> None:
    USAGE_DIR.mkdir(parents=True, exist_ok=True)
    if not USAGE_PATH.exists():
        save_usage({"totals": {}, "interactions": []})


def load_usage() -> Dict:
    ensure_usage_store()
    with open(USAGE_PATH, "r", encoding="utf-8") as handle:
        data = json.load(handle)

    if not isinstance(data, dict):
        data = {"totals": {}, "interactions": []}

    totals = data.get("totals", {})
    interactions = data.get("interactions", [])

    if "input_tokens" in totals:
        totals = {}

    if not isinstance(totals, dict):
        totals = {}
    if not isinstance(interactions, list):
        interactions = []

    data["totals"] = totals
    data["interactions"] = interactions
    return data


def save_usage(data: Dict) -> None:
    USAGE_DIR.mkdir(parents=True, exist_ok=True)
    with open(USAGE_PATH, "w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)


def update_usage(
    usage: Dict,
    model: str,
    input_tokens: int,
    output_tokens: int,
    filters: Dict,
) -> Dict:
    if model not in usage["totals"]:
        usage["totals"][model] = {"input_tokens": 0, "output_tokens": 0}

    usage["totals"][model]["input_tokens"] += int(input_tokens)
    usage["totals"][model]["output_tokens"] += int(output_tokens)

    usage["interactions"].append(
        {
            "ts": datetime.now(timezone.utc).isoformat(),
            "model": model,
            "input_tokens": int(input_tokens),
            "output_tokens": int(output_tokens),
            "filters": filters,
        }
    )
    return usage


def print_usage_stats(
    model: str,
    query_input_tokens: int,
    query_output_tokens: int,
    session_input_tokens: int,
    session_output_tokens: int,
    total_input_tokens: int,
    total_output_tokens: int,
) -> None:
    print(
        f"[TOKENS][{model}] query_input={query_input_tokens} query_output={query_output_tokens}"
    )
    print(
        f"[TOKENS][{model}] session_input={session_input_tokens} session_output={session_output_tokens}"
    )
    print(
        f"[TOKENS][{model}] total_input={total_input_tokens} total_output={total_output_tokens}"
    )


@st.cache_data
def load_courses_df() -> pd.DataFrame:
    return pd.read_csv(DATA_CSV_PATH)


@st.cache_data
def load_embeddings_df() -> pd.DataFrame:
    return pd.read_pickle(EMB_PKL_PATH)


@st.cache_data
def get_merged_df(courses_df: pd.DataFrame, embeddings_df: pd.DataFrame) -> pd.DataFrame:
    return pd.merge(courses_df, embeddings_df, on="unique_ID", how="inner")


@st.cache_resource
def get_embedder() -> SentenceTransformer:
    return SentenceTransformer("BAAI/bge-m3")


def select_filter_options(df: pd.DataFrame) -> Tuple[List[str], List, List[str]]:
    semester_options = sorted(df["semester"].dropna().astype(str).unique().tolist())

    eap_counts = df["eap"].dropna().value_counts()
    eap_options = sorted(
        {
            int(value) if float(value).is_integer() else float(value)
            for value, count in eap_counts.items()
            if count >= 50
        }
        | {3, 6}
    )

    language_counts: Dict[str, int] = {}
    for raw_value in df["keel"].dropna().astype(str):
        for part in raw_value.split(","):
            language = part.strip().lower()
            if language:
                language_counts[language] = language_counts.get(language, 0) + 1

    keel_options = sorted(
        [language for language, count in language_counts.items() if count >= 20]
    )
    return semester_options, eap_options, keel_options


def apply_filters(
    merged_df: pd.DataFrame,
    selected_semesters: List[str],
    selected_eap: List[int],
    selected_keel: List[str],
) -> pd.DataFrame:
    mask = np.ones(len(merged_df), dtype=bool)

    if selected_semesters:
        mask &= merged_df["semester"].astype(str).isin(selected_semesters)

    # EAP filter: match any selected EAP
    if selected_eap:
        mask &= merged_df["eap"].isin(selected_eap)

    # Language filter: match if any selected language is in course's language string
    if selected_keel:
        lang_mask = np.zeros(len(merged_df), dtype=bool)
        for lang in selected_keel:
            lang_mask |= merged_df["keel"].astype(str).str.lower().str.contains(lang.lower(), regex=False)
        mask &= lang_mask

    return merged_df[mask].copy()


def retrieve_top_courses(
    embedder: SentenceTransformer,
    filtered_df: pd.DataFrame,
    query: str,
    top_n: int,
) -> pd.DataFrame:
    if filtered_df.empty:
        return filtered_df

    query_vec = embedder.encode([query])[0]
    similarities = cosine_similarity([query_vec], np.stack(filtered_df["embedding"].to_numpy()))[0]

    scored_df = filtered_df.copy()
    scored_df["score"] = similarities
    top_df = scored_df.sort_values("score", ascending=False).head(int(top_n))
    return top_df


def format_context_for_llm(results_df: pd.DataFrame) -> str:
    if results_df.empty:
        return "Sobivaid kursusi ei leitud valitud filtrite ja päringu põhjal."

    context_rows = results_df.drop(columns=["embedding", "score"], errors="ignore")
    context_text = context_rows.to_string(index=False)
    # Limit context to max tokens (e.g. 10000)
    MAX_CONTEXT_TOKENS = 10000
    tokens = count_text_tokens(context_text)
    if tokens > MAX_CONTEXT_TOKENS:
        # Truncate context to fit token limit
        # Find approximate cutoff by character
        cutoff = int(len(context_text) * MAX_CONTEXT_TOKENS / tokens)
        context_text = context_text[:cutoff] + "\n... (truncated)"
    return context_text


def build_system_prompt(context_text: str) -> Dict[str, str]:
    prompt = (
        "Sa oled Tartu Ülikooli kursuste nõustaja. "
        "Sinu ülesanne on soovitada 3-5 sobivat kursust ainult antud konteksti põhjal. "
        "Põhjenda iga soovituse sobivust kasutaja eesmärgiga. "
        "Kui kontekst ei kata kasutaja soovi, ütle see selgelt ja küsi täpsustav küsimus. "
        "Ära mõtle kursusi välja ega kasuta väliseid allikaid. "
        "Kui kursuse info on mitmes keeles, sobib see mõlema keele filtriga. "
        "Kui kontekst on liiga pikk, võib osa sellest olla kärbitud.\n\n"
        f"Kursuste kontekst:\n{context_text}"
    )
    return {"role": "system", "content": prompt}


def stream_completion(client: OpenAI, model: str, messages: List[Dict[str, str]]) -> str:
    stream = client.chat.completions.create(model=model, messages=messages, stream=True)

    def chunk_generator() -> Iterable[str]:
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta

    return st.write_stream(chunk_generator())


st.set_page_config(page_title="AI Kursuse Nõustaja 5.1", layout="centered")
st.title("🎓 AI Kursuse Nõustaja 5.1")
# st.caption("RAG + metaandmete filtrid + mudelipõhine tokenistatistika.")

# Load only the light-weight course table up-front so UI appears fast.
courses_df = load_courses_df()
semester_options, eap_options, keel_options = select_filter_options(courses_df)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_tokens" not in st.session_state:
    st.session_state.session_tokens = {}

usage = load_usage()

with st.sidebar:
    api_key = st.text_input("OpenRouter API Key", type="password")
    model = st.selectbox("Model", MODEL_OPTIONS)
    top_n = st.number_input("Top N", min_value=1, max_value=10, value=5)

    st.subheader("Filtrid")
    semester = st.multiselect("Semester", semester_options, default=[])
    eap = st.multiselect("EAP", eap_options, default=[])
    keel = st.multiselect("Keel", keel_options, default=[])

    st.caption("Tokenistatistika prinditakse terminali iga päringu järel.")

    st.subheader("Kogusummad mudelipõhiselt")
    for model_name, token_data in usage.get("totals", {}).items():
        if isinstance(token_data, dict):
            st.write(
                f"{model_name}: input={token_data.get('input_tokens', 0)}, "
                f"output={token_data.get('output_tokens', 0)}"
            )

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


if prompt := st.chat_input("Kirjelda, mida soovid õppida..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        if not api_key:
            error_msg = "Palun sisesta API võti!"
            st.error(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})
        else:
            with st.spinner("Valmistan andmed ette ja otsin sobivaid kursusi..."):
                embeddings_df = load_embeddings_df()
                merged_df = get_merged_df(courses_df, embeddings_df)
                embedder = get_embedder()

                filtered_df = apply_filters(merged_df, semester, eap, keel)
                results_df = retrieve_top_courses(embedder, filtered_df, prompt, int(top_n))
                context_text = format_context_for_llm(results_df)

            system_prompt = build_system_prompt(context_text)
            messages_to_send = [system_prompt] + st.session_state.messages
            input_tokens = count_messages_tokens(messages_to_send)

            client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)

            try:
                response = stream_completion(client, model, messages_to_send)
                st.session_state.messages.append({"role": "assistant", "content": response})

                output_tokens = count_text_tokens(response)

                if model not in st.session_state.session_tokens:
                    st.session_state.session_tokens[model] = {"input": 0, "output": 0}

                st.session_state.session_tokens[model]["input"] += input_tokens
                st.session_state.session_tokens[model]["output"] += output_tokens

                selected_filters = {
                    "semester": semester or None,
                    "eap": eap or None,
                    "keel": keel or None,
                }

                usage = load_usage()
                usage = update_usage(
                    usage=usage,
                    model=model,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    filters=selected_filters,
                )
                save_usage(usage)

                model_totals = usage["totals"].get(model, {"input_tokens": 0, "output_tokens": 0})

                print_usage_stats(
                    model=model,
                    query_input_tokens=input_tokens,
                    query_output_tokens=output_tokens,
                    session_input_tokens=st.session_state.session_tokens[model]["input"],
                    session_output_tokens=st.session_state.session_tokens[model]["output"],
                    total_input_tokens=model_totals.get("input_tokens", 0),
                    total_output_tokens=model_totals.get("output_tokens", 0),
                )

            except Exception as exc:
                st.error(f"Viga: {exc}")