import pandas as pd
import streamlit as st
from openai import OpenAI

from src.config import (
    API_KEY,
    APP_ENV,
    DEFAULT_MODEL,
    LAST_TEST_RESULTS_PATH,
    MODEL_OPTIONS,
    TEST_CASES_PATH,
    USAGE_DIR,
)
from src.data import apply_filters, get_filter_options, load_courses
from src.pipeline import (
    contains_course_code,
    ensure_ois_links,
    parse_expected_ids,
    run_rag_pipeline,
    safe_extract_delta,
)
from src.safety import is_prompt_injection
from src.style import MODERN_CSS, SIDEBAR_LOGO_HTML
from src.tracking import add_token_usage, count_text_tokens, load_usage

WELCOME_MSG = (
    "Tere! Olen Tartu Ülikooli ainete nõustaja. "
    "Kirjelda, mida soovid õppida, ja soovitan sulle sobivaid aineid.\n\n"
    "Näiteks: *\"Tahan õppida masinõpet\"* või *\"Soovita mulle spordiga seotud aineid\"*"
)


# ---------------------------------------------------------------------------
# Session helpers
# ---------------------------------------------------------------------------

def _init_session():
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": WELCOME_MSG}
        ]
    if "session_tokens" not in st.session_state:
        st.session_state.session_tokens = {}
    if "last_tokens" not in st.session_state:
        st.session_state.last_tokens = {"input": 0, "output": 0}
    if "mode_tokens" not in st.session_state:
        st.session_state.mode_tokens = {
            "chat": {"input": 0, "output": 0},
            "test": {"input": 0, "output": 0},
        }


# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="TÜ Ainete Nõustaja",
    page_icon="https://ut.ee/favicon.ico",
    layout="wide" if APP_ENV == "dev" else "centered",
    initial_sidebar_state="expanded",
)
st.markdown(MODERN_CSS, unsafe_allow_html=True)

_init_session()
courses_df = load_courses()
sem_opts, max_eap, keel_opts, veeb_opts = get_filter_options(courses_df)
usage = load_usage()


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown(SIDEBAR_LOGO_HTML, unsafe_allow_html=True)
    st.divider()

    if APP_ENV == "dev":
        model = st.selectbox(
            "Mudel",
            MODEL_OPTIONS,
            index=(
                MODEL_OPTIONS.index(DEFAULT_MODEL)
                if DEFAULT_MODEL in MODEL_OPTIONS
                else 0
            ),
        )
        top_n = st.number_input("Tulemuste arv", 1, 10, 5)
        st.divider()
    else:
        model = DEFAULT_MODEL
        top_n = 5

    st.markdown(
        '<p style="font-weight:600; font-size:0.95rem; margin-bottom:4px;">Filtrid</p>',
        unsafe_allow_html=True,
    )

    eap_range = st.slider("EAP vahemik", 0.0, max_eap, (0.0, max_eap), 1.0)
    semester = st.multiselect("Semester", sem_opts)
    keel = st.multiselect("Keel", keel_opts)
    hindamis = st.multiselect("Hindamisviis", ["Eristav", "Eristamata"])
    linn = st.multiselect(
        "Linn", ["Tartu", "Tallinn", "Narva", "Pärnu", "Viljandi"]
    )
    aste = st.multiselect("Õppeaste", ["bakalaureuse", "magistri", "doktori"])
    veeb = st.multiselect(
        "Õppevorm", veeb_opts or ["põimõpe", "lähiõpe", "veebiõpe"]
    )
    no_prereqs = st.checkbox("Ainult ilma eeldusaineteta")

    current_filters = {
        "selected_semesters": semester,
        "selected_keel": keel,
        "eap_range": eap_range,
        "selected_hindamis": hindamis,
        "selected_linn": linn,
        "selected_aste": aste,
        "selected_veeb": veeb,
        "no_prereqs": no_prereqs,
    }

    if APP_ENV == "dev":
        st.divider()
        st.markdown(
            '<p style="font-weight:600; font-size:0.95rem;">Token kasutus</p>',
            unsafe_allow_html=True,
        )
        m_sess = st.session_state.session_tokens.get(
            model, {"input": 0, "output": 0}
        )
        m_tot = usage.get("totals", {}).get(
            model, {"input_tokens": 0, "output_tokens": 0}
        )
        c1, c2 = st.columns(2)
        c1.metric("Sessioon in", m_sess["input"])
        c2.metric("Sessioon out", m_sess["output"])
        c1.metric("Kokku in", m_tot["input_tokens"])
        c2.metric("Kokku out", m_tot["output_tokens"])

    st.divider()
    if st.button("Alusta uut vestlust", use_container_width=True):
        st.session_state.messages = [
            {"role": "assistant", "content": WELCOME_MSG}
        ]
        st.rerun()

    st.caption(
        "Kasutab [ÕIS2](https://ois2.ut.ee) avalikke andmeid "
        "ja Gemma 3 keelemudelit."
    )


# ---------------------------------------------------------------------------
# Chat UI
# ---------------------------------------------------------------------------

if APP_ENV == "dev":
    tab_chat, tab_test = st.tabs(["Vestlus", "Testimine"])
else:
    tab_chat = st.container()

with tab_chat:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if (
                msg["role"] == "assistant"
                and "debug" in msg
                and APP_ENV == "dev"
            ):
                with st.expander("Debug info"):
                    st.write(
                        f"Filtrite järel alles: {msg['debug'].get('filtered_count')} ainet"
                    )
                    st.dataframe(
                        msg["debug"]["results_df"].drop(columns=["embedding"]),
                        use_container_width=True,
                    )
                    st.text_area(
                        "System prompt",
                        msg["debug"]["system_prompt"]["content"],
                        height=100,
                    )

    if prompt := st.chat_input("Kirjelda, mida soovid õppida..."):
        if not API_KEY or API_KEY == "your_key_here":
            st.error(
                "API võti on seadistamata. "
                "Lisa see `.env` faili (vt `.env.example`)."
            )
            st.stop()

        if len(prompt) > 1000:
            st.warning("Päring on liiga pikk (max 1000 tähemärki).")
            st.stop()

        if is_prompt_injection(prompt):
            st.error("Tuvastati keelatud sisend. Palun esita tavaline päring.")
            st.stop()

        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            client = OpenAI(
                base_url="https://openrouter.ai/api/v1", api_key=API_KEY
            )
            res = run_rag_pipeline(client, model, prompt, current_filters, top_n)

            if not res["success"]:
                if res["reason"] == "off_topic":
                    ans = (
                        "Saan aidata ainult Tartu Ülikooli ainetega seotud "
                        "küsimustes. Palun kirjelda, mida soovid õppida."
                    )
                else:
                    ans = (
                        "Valitud filtritega ei leitud ühtegi ainet. "
                        "Proovi filtreid leevendada."
                    )
                st.warning(ans)
                st.session_state.messages.append(
                    {"role": "assistant", "content": ans}
                )
            else:
                full_resp = ""
                with st.empty():
                    for chunk in res["stream"]:
                        delta = safe_extract_delta(chunk)
                        if delta:
                            full_resp += delta
                            st.markdown(full_resp + "▌")
                    full_resp = ensure_ois_links(full_resp, res["results_df"])
                    st.markdown(full_resp)

                in_t, precheck_out_t = res["tokens"]
                out_t = count_text_tokens(full_resp) + precheck_out_t
                add_token_usage(
                    model, in_t, out_t,
                    {**current_filters, "mode": "chat"},
                    mode="chat",
                )

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": full_resp,
                    "debug": {
                        "results_df": res["results_df"],
                        "system_prompt": res["system_prompt"],
                        "filtered_count": res["filtered_count"],
                    },
                })
                st.rerun()


# ---------------------------------------------------------------------------
# Test runner (dev only)
# ---------------------------------------------------------------------------

if APP_ENV == "dev":
    with tab_test:
        st.subheader("RAG Test Runner")

        if not TEST_CASES_PATH.exists():
            st.error(f"Testfaili ei leitud: {TEST_CASES_PATH}")
        else:
            tests_df = pd.read_csv(TEST_CASES_PATH)
            if len(tests_df.columns) < 2:
                st.error(
                    "test_cases.csv peab sisaldama vähemalt 2 tulpa: "
                    "päring ja oodatud ainekoodid."
                )
                st.stop()

            query_col = tests_df.columns[0]
            expected_col = tests_df.columns[1]
            total_tests = len(tests_df)

            num_to_run = st.slider(
                "Testide arv",
                min_value=1,
                max_value=total_tests,
                value=min(10, total_tests),
            )

            if LAST_TEST_RESULTS_PATH.exists():
                if st.toggle("Eelmised tulemused", value=False):
                    prev_df = pd.read_csv(LAST_TEST_RESULTS_PATH)
                    p1, p2 = st.columns(2)
                    p1.metric(
                        "Input tokenid",
                        int(prev_df.get(
                            "Sisend tokenid", pd.Series(dtype=int)
                        ).fillna(0).sum()),
                    )
                    p2.metric(
                        "Output tokenid",
                        int(prev_df.get(
                            "Väljund tokenid", pd.Series(dtype=int)
                        ).fillna(0).sum()),
                    )
                    st.dataframe(
                        prev_df, use_container_width=True, hide_index=True
                    )

            if st.button("Käivita testid", use_container_width=True):
                if not API_KEY or API_KEY == "your_key_here":
                    st.error("API võti on seadistamata.")
                    st.stop()

                client = OpenAI(
                    base_url="https://openrouter.ai/api/v1", api_key=API_KEY
                )
                results = []
                progress_bar = st.progress(0)
                status_box = st.empty()
                metrics_box = st.empty()
                table_box = st.empty()

                for idx, (_, row) in enumerate(
                    tests_df.head(num_to_run).iterrows(), start=1
                ):
                    query = str(row[query_col])
                    raw_expected = str(row[expected_col])
                    expected_ids = parse_expected_ids(raw_expected)

                    res = run_rag_pipeline(
                        client, model, query, current_filters,
                        top_n, stream=False,
                    )

                    rag_pass = False
                    llm_pass = False
                    llm_text = ""
                    found_ids = []
                    reason = ""
                    test_in = 0
                    test_out = 0

                    if res["success"]:
                        id_col = (
                            "aine_kood"
                            if "aine_kood" in res["results_df"].columns
                            else "unique_ID"
                        )
                        found_ids = (
                            res["results_df"][id_col].astype(str).tolist()
                        )
                        rag_pass = (
                            all(eid in found_ids for eid in expected_ids)
                            if expected_ids
                            else len(found_ids) == 0
                        )

                        llm_res = client.chat.completions.create(
                            model=model, messages=res["messages"]
                        )
                        llm_text = llm_res.choices[0].message.content or ""

                        base_in, precheck_out = res["tokens"]
                        test_in = base_in
                        test_out = (
                            count_text_tokens(llm_text) + precheck_out
                        )
                        add_token_usage(
                            model, test_in, test_out,
                            {**current_filters, "mode": "test"},
                            mode="test",
                        )

                        if expected_ids:
                            llm_pass = all(
                                eid in llm_text for eid in expected_ids
                            )
                        else:
                            llm_pass = not contains_course_code(llm_text)
                    else:
                        reason = res.get("reason", "unknown")
                        base_in, precheck_out = res["tokens"]
                        test_in = base_in
                        test_out = precheck_out
                        add_token_usage(
                            model, test_in, test_out,
                            {**current_filters, "mode": "test"},
                            mode="test",
                        )
                        if not expected_ids and reason in {
                            "off_topic", "no_results"
                        }:
                            rag_pass = True
                            llm_pass = True

                    if not reason:
                        if rag_pass and llm_pass:
                            reason = "OK"
                        elif not rag_pass and not llm_pass:
                            reason = "RAG + LLM failed"
                        elif not rag_pass:
                            reason = "RAG miss"
                        else:
                            reason = "LLM hallucination"

                    results.append({
                        "#": idx,
                        "Päring": query,
                        "Oodatud": raw_expected,
                        "RAG leidis": ", ".join(found_ids[:8]),
                        "RAG": "pass" if rag_pass else "FAIL",
                        "LLM": "pass" if llm_pass else "FAIL",
                        "In": test_in,
                        "Out": test_out,
                        "Staatus": reason,
                        "Vastus": (
                            (llm_text[:120] + "...")
                            if len(llm_text) > 120
                            else llm_text
                        ),
                    })

                    progress_bar.progress(idx / num_to_run)
                    status_box.info(f"Test {idx}/{num_to_run}: {query[:80]}")

                    live_df = pd.DataFrame(results)
                    rag_acc = (live_df["RAG"] == "pass").mean() * 100
                    llm_acc = (live_df["LLM"] == "pass").mean() * 100
                    both_acc = (
                        (live_df["RAG"] == "pass")
                        & (live_df["LLM"] == "pass")
                    ).mean() * 100

                    with metrics_box.container():
                        mc1, mc2, mc3 = st.columns(3)
                        mc1.metric("RAG", f"{rag_acc:.0f}%")
                        mc2.metric("LLM", f"{llm_acc:.0f}%")
                        mc3.metric("End-to-end", f"{both_acc:.0f}%")

                    table_box.dataframe(
                        live_df, use_container_width=True, hide_index=True
                    )

                status_box.empty()
                res_df = pd.DataFrame(results)
                USAGE_DIR.mkdir(parents=True, exist_ok=True)
                res_df.to_csv(LAST_TEST_RESULTS_PATH, index=False)
                st.success(
                    f"Testijooks lõpetatud: "
                    f"{(res_df['RAG'] == 'pass').mean() * 100:.0f}% RAG, "
                    f"{(res_df['LLM'] == 'pass').mean() * 100:.0f}% LLM"
                )
