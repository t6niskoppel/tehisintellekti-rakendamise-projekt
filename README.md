# tehisintellekti-rakendamise-projekt

# Tõnis Kristian Koppel

## Käivitamine (Streamlit)

- Keskkond: `conda env create -f environment.yml` ja `conda activate oisi_projekt`
- Käivita: `streamlit run app5.2.py`

## API võti (turvaliselt)

Soovituslikult seadista API võti ilma seda repo failidesse salvestamata:

- Keskkonnamuutuja kaudu: `export OPENROUTER_API_KEY="..."`
- Või Streamlit secrets kaudu: loo `.streamlit/secrets.toml` sisuga:

	`OPENROUTER_API_KEY = "..."`

Rakendus loeb võtme eelisjärjekorras `st.secrets["OPENROUTER_API_KEY"]` → `OPENROUTER_API_KEY` → `OPENAI_API_KEY`.