import re
from typing import Dict, List

import pandas as pd
from openai import OpenAI

from .data import (
    apply_filters,
    get_embedder,
    load_courses,
    load_embeddings,
    merge_courses_and_embeddings,
    retrieve_top_courses,
)
from .tracking import count_messages_tokens, count_text_tokens

OIS2_BASE_URL = "https://ois2.ut.ee/#/courses"


def make_ois2_url(course_code: str) -> str:
    return f"{OIS2_BASE_URL}/{course_code}/"


def format_context(results_df: pd.DataFrame) -> str:
    if results_df.empty:
        return "Sobivaid kursusi ei leitud."
    lines = []
    for _, row in results_df.iterrows():
        course_id = row.get("aine_kood") or row.get("unique_ID")
        ois_url = row.get("ois2_url") or make_ois2_url(str(course_id))
        lines.append(
            f"ID: {course_id} | Nimi: {row.get('nimi_et')} | OIS2_URL: {ois_url} | "
            f"EAP: {row.get('eap')} | Hindamine: {row.get('hindamisviis')} | "
            f"Linn: {row.get('linn')} | Oppeaste: {row.get('oppeaste')} | "
            f"Kirjeldus: {str(row.get('kirjeldus'))[:400]}..."
        )
    return "\n\n".join(lines)


def build_system_prompt(context_text: str, filters_str: str = "") -> Dict[str, str]:
    filter_note = (
        f"(Kasutaja on valinud filtrid: {filters_str})\n" if filters_str else ""
    )
    rules = (
        "Oled Tartu Ülikooli kursuste nõustaja. Sinu ülesanne on soovitada kursusi "
        "vastavalt antud kontekstile. Vasta eesti keeles ja kasuta Markdowni. "
        "Iga soovitatud kursuse juures lisa ÕIS2 link Markdown lingina kujul "
        "[Kursuse nimi](OIS2_URL). Ära väljasta URL-i ilma lingivormita."
    )
    body = f"\n\n{filter_note}Kontekst:\n{context_text}"
    return {"role": "system", "content": rules + body}


def check_query_relevance(client: OpenAI, model: str, query: str):
    system_msg = (
        "Sa oled Tartu Ülikooli kursuste nõustaja assistent. "
        "Vasta ainult 'RELEVANT' või 'NOT_RELEVANT'."
    )
    messages = [
        {"role": "system", "content": system_msg},
        {
            "role": "user",
            "content": (
                f"Kas see päring on seotud ülikooli kursuste, õppimise "
                f"või haridusega: '{query}'?"
            ),
        },
    ]
    response = client.chat.completions.create(
        model=model, messages=messages, temperature=0
    )
    answer = response.choices[0].message.content.strip().upper()
    is_relevant = answer == "RELEVANT"
    return is_relevant, count_messages_tokens(messages), count_text_tokens(answer)


def run_rag_pipeline(
    client: OpenAI,
    model: str,
    query: str,
    filters: Dict,
    top_n: int = 5,
    stream: bool = True,
) -> Dict:
    is_relevant, rel_in, rel_out = check_query_relevance(client, model, query)
    if not is_relevant:
        return {"success": False, "reason": "off_topic", "tokens": (rel_in, rel_out)}

    courses_df = load_courses()
    embeddings_df = load_embeddings()
    merged_df = merge_courses_and_embeddings(courses_df, embeddings_df)
    filtered_df = apply_filters(merged_df, **filters)

    if filtered_df.empty:
        return {"success": False, "reason": "no_results", "tokens": (rel_in, rel_out)}

    embedder = get_embedder()
    results_df = retrieve_top_courses(embedder, filtered_df, query, top_n)
    id_col = "aine_kood" if "aine_kood" in results_df.columns else "unique_ID"
    results_df["ois2_url"] = results_df[id_col].apply(make_ois2_url)

    context_text = format_context(results_df)
    system_prompt = build_system_prompt(context_text, filters_str=str(filters))
    messages = [system_prompt, {"role": "user", "content": query}]

    input_tokens = count_messages_tokens(messages) + rel_in
    result = {
        "success": True,
        "results_df": results_df,
        "context_text": context_text,
        "system_prompt": system_prompt,
        "messages": messages,
        "tokens": (input_tokens, rel_out),
        "filtered_count": len(filtered_df),
    }
    if stream:
        result["stream"] = client.chat.completions.create(
            model=model, messages=messages, stream=True
        )
    return result


def ensure_ois_links(
    response_text: str, results_df: pd.DataFrame, max_links: int = 3
) -> str:
    if "https://ois2.ut.ee/#/courses/" in (response_text or ""):
        return response_text
    if results_df is None or results_df.empty:
        return response_text

    id_col = "aine_kood" if "aine_kood" in results_df.columns else "unique_ID"
    links: List[str] = []
    for _, row in results_df.head(max_links).iterrows():
        course_id = str(row.get(id_col, "")).strip()
        course_name = str(row.get("nimi_et", course_id)).strip() or course_id
        if not course_id:
            continue
        url = str(row.get("ois2_url", "")).strip() or make_ois2_url(course_id)
        links.append(f"- [{course_name}]({url}) ({course_id})")

    if not links:
        return response_text
    return (response_text or "").rstrip() + "\n\n### ÕIS lingid\n" + "\n".join(links)


def safe_extract_delta(chunk) -> str:
    choices = getattr(chunk, "choices", None)
    if not choices:
        return ""
    delta = getattr(choices[0], "delta", None)
    if not delta:
        return ""
    return getattr(delta, "content", "") or ""


def parse_expected_ids(raw: str) -> List[str]:
    value = (raw or "").strip()
    if not value or value == "-":
        return []
    return [part.strip() for part in re.split(r"[;,]", value) if part.strip()]


def contains_course_code(text: str) -> bool:
    return bool(re.search(r"\b[A-ZÕÄÖÜ]{2,6}\.\d{2}\.\d{3}\b", text or ""))
