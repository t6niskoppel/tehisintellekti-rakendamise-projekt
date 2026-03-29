# UT Course Advisor

AI-powered course recommendation chatbot for **University of Tartu** students. Uses Retrieval-Augmented Generation (RAG) to help students discover courses through natural language queries across 2,700+ courses from the [ÕIS2](https://ois2.ut.ee) course catalog.

> The application interface is in Estonian, as it is built for University of Tartu students.

## Features

- **Semantic search** — finds courses by meaning, not just keyword matching (BAAI/bge-m3 embeddings)
- **Metadata filtering** — semester, credits (EAP), language, grading, location, study level, format
- **LLM recommendations** — natural language course suggestions with direct ÕIS2 links
- **Prompt injection detection** — input validation against adversarial prompts
- **Query relevance checking** — rejects off-topic queries before running the pipeline
- **Built-in evaluation** — automated test runner with RAG and LLM accuracy metrics (dev mode)
- **Token usage tracking** — monitors API consumption per session and cumulatively

## Architecture

```
User query
    │
    ▼
┌─────────────────┐
│  Safety checks   │  Prompt injection detection + relevance check
└────────┬────────┘
         ▼
┌─────────────────┐
│ Metadata filters │  Semester, EAP, language, location, etc.
└────────┬────────┘
         ▼
┌─────────────────┐
│  Vector search   │  bge-m3 embeddings + cosine similarity → top-k courses
└────────┬────────┘
         ▼
┌─────────────────┐
│  LLM generation  │  Gemma 3 27B generates recommendations from retrieved context
└────────┬────────┘
         ▼
    Response with ÕIS2 links
```

## Quick Start

### Prerequisites

- Python 3.10+
- [OpenRouter](https://openrouter.ai/) API key (free tier available with Gemma 3 27B)

### Installation

```bash
git clone https://github.com/your-username/ut-course-advisor.git
cd ut-course-advisor

# Option A: pip
pip install -r requirements.txt

# Option B: conda
conda env create -f environment.yml
conda activate ut_course_advisor
```

### Configuration

```bash
cp .env.example .env
# Edit .env and add your OpenRouter API key
```

### Run

```bash
streamlit run app.py
```

Set `APP_ENV=dev` in `.env` to enable the debug panel and test runner.

## Project Structure

```
├── app.py                  # Streamlit application entry point
├── src/
│   ├── config.py           # Configuration and paths
│   ├── data.py             # Data loading, filtering, retrieval
│   ├── pipeline.py         # RAG pipeline and LLM interaction
│   ├── safety.py           # Input validation
│   ├── style.py            # UI styling
│   └── tracking.py         # Token usage and feedback logging
├── data/
│   ├── courses.csv         # Cleaned course catalog (2,768 courses)
│   ├── embeddings.pkl      # Pre-computed bge-m3 embeddings
│   └── test_cases.csv      # Evaluation test queries
├── notebooks/
│   ├── data_collection.ipynb    # ÕIS2 API data fetching
│   └── data_preparation.ipynb   # Data cleaning pipeline
└── docs/
    └── error_analysis.md   # Evaluation results and failure analysis
```

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Frontend | Streamlit |
| Embeddings | [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3) via sentence-transformers |
| LLM | Google Gemma 3 27B via [OpenRouter](https://openrouter.ai/) |
| Vector search | scikit-learn cosine similarity |
| Data | pandas, numpy |

## Data Pipeline

1. **Collection** — fetch course data from ÕIS2 public API (3,031 courses, 223 columns)
2. **Cleaning** — normalize, deduplicate, extract key fields (→ 2,768 courses, 16 columns)
3. **Embedding** — generate vector representations with bge-m3
4. **Storage** — CSV for metadata, pickle for embeddings (fast load)

See [notebooks/](notebooks/) for the full data pipeline implementation.

## Evaluation

The built-in test runner (`APP_ENV=dev`) evaluates the pipeline against predefined test cases:

| Component | Accuracy |
|-----------|----------|
| Metadata filtering | 100% |
| RAG vector search | 100% |
| End-to-end (incl. LLM) | 82.4% |

The 17.6% failure rate is caused by LLM hallucination on edge cases where no courses match the filters. See [docs/error_analysis.md](docs/error_analysis.md) for details.

## License

This project uses publicly available course data from the University of Tartu ÕIS2 system.
