import numpy as np
import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from .config import COURSES_PATH, EMBEDDINGS_PATH


@st.cache_data
def load_courses() -> pd.DataFrame:
    return pd.read_csv(COURSES_PATH)


@st.cache_data
def load_embeddings() -> pd.DataFrame:
    return pd.read_pickle(EMBEDDINGS_PATH)


@st.cache_resource
def get_embedder() -> SentenceTransformer:
    return SentenceTransformer("BAAI/bge-m3")


def merge_courses_and_embeddings(
    courses_df: pd.DataFrame, embeddings_df: pd.DataFrame
) -> pd.DataFrame:
    return pd.merge(courses_df, embeddings_df, on="unique_ID", how="inner")


def get_filter_options(df: pd.DataFrame):
    semester_options = sorted(df["semester"].dropna().astype(str).unique().tolist())
    max_eap = float(df["eap"].max()) if "eap" in df.columns else 30.0

    language_counts: dict[str, int] = {}
    for raw in df["keel"].dropna().astype(str):
        for part in raw.split(","):
            lang = part.strip().lower()
            if lang:
                language_counts[lang] = language_counts.get(lang, 0) + 1
    keel_options = sorted(
        lang for lang, count in language_counts.items() if count >= 20
    )

    veeb_options = (
        sorted(df["veebiope"].dropna().astype(str).unique().tolist())
        if "veebiope" in df.columns
        else []
    )
    return semester_options, max_eap, keel_options, veeb_options


def apply_filters(df: pd.DataFrame, **filters) -> pd.DataFrame:
    filtered = df.copy()

    if filters.get("selected_semesters"):
        filtered = filtered[filtered["semester"].isin(filters["selected_semesters"])]

    if filters.get("selected_keel"):
        targets = filters["selected_keel"]

        def _has_language(val):
            if not val:
                return False
            langs = [x.strip().lower() for x in str(val).split(",")]
            return any(t in langs for t in targets)

        filtered = filtered[filtered["keel"].apply(_has_language)]

    if filters.get("eap_range"):
        low, high = filters["eap_range"]
        filtered = filtered[filtered["eap"].between(low, high)]

    if filters.get("selected_hindamis"):
        filtered = filtered[filtered["hindamisviis"].isin(filters["selected_hindamis"])]

    if filters.get("selected_linn"):
        filtered = filtered[filtered["linn"].isin(filters["selected_linn"])]

    if filters.get("selected_aste"):
        pattern = "|".join(filters["selected_aste"])
        filtered = filtered[
            filtered["oppeaste"].str.contains(pattern, case=False, na=False)
        ]

    if filters.get("selected_veeb"):
        filtered = filtered[filtered["veebiope"].isin(filters["selected_veeb"])]

    if filters.get("no_prereqs") and "eelduse_olemasolu" in filtered.columns:
        filtered = filtered[filtered["eelduse_olemasolu"] == 0]

    return filtered


def retrieve_top_courses(
    embedder: SentenceTransformer,
    df: pd.DataFrame,
    query: str,
    k: int = 5,
) -> pd.DataFrame:
    if df.empty:
        return df
    query_embedding = embedder.encode([query], normalize_embeddings=True)
    course_embeddings = np.stack(df["embedding"].values)
    similarities = cosine_similarity(query_embedding, course_embeddings)[0]
    result = df.copy()
    result["score"] = similarities
    return result.sort_values(by="score", ascending=False).head(k)
