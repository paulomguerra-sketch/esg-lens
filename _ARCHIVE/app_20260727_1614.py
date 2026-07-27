"""
ESG Lens — Streamlit prototype
Generated: 2026-07-27 16:15 (Europe/Lisbon)

Two-tier prototype:
  1. Precomputed core dataset (Galp, EDP, Amorim x 2024/2025) -- Company view
     + Comparison view, reading data/processed/*.csv (built by score_reports.py).
  2. "Analyze a new report" -- live PDF upload, classified on demand with the
     same trained model. English-only, marked provisional, never mixed into
     the Comparison view (per project rules).

Framing guardrail (binding, shown in every view): this app measures language
specificity and transparency. It does not adjudicate greenwashing.
"""

import os
import re
import sys

import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATA_DIR = os.path.join("data", "processed")
SCORED_SENTENCES_PATH = os.path.join(DATA_DIR, "scored_sentences.csv")
COMPANY_SUMMARY_PATH = os.path.join(DATA_DIR, "company_summary.csv")

LABELS = ["commitment", "aspiration", "reported_result"]
LABEL_DISPLAY = {
    "commitment": "Commitment (specific, checkable)",
    "aspiration": "Aspiration (vague intent)",
    "reported_result": "Reported result (already happened)",
}

FRAMING_NOTE = (
    "ESG Lens measures language **specificity and transparency** — "
    "it does not adjudicate greenwashing."
)

MAX_LIVE_PAGES = 150  # soft cap so a huge PDF doesn't stall a live demo

# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------

@st.cache_data
def load_summary():
    df = pd.read_csv(COMPANY_SUMMARY_PATH)
    df["company_year"] = df["company"].str.capitalize() + " " + df["year"].astype(str)
    return df


@st.cache_data
def load_scored_sentences():
    return pd.read_csv(SCORED_SENTENCES_PATH)


@st.cache_resource
def load_classifier_model():
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from predict_commitment import load_model
    return load_model()


@st.cache_resource
def load_spacy_model():
    import spacy
    return spacy.load("en_core_web_md", disable=["ner"])


# ---------------------------------------------------------------------------
# Live-upload text processing (simplified: whole PDF, no chapter-location)
# ---------------------------------------------------------------------------

TAXONOMY_CODE_PATTERN = re.compile(
    r"\b(CCM|CE|CCA|DNSH|N/EL|Capex|Opex)\b|\bA\.\d\.|\bB\.\d?\b", re.IGNORECASE
)
PAGE_NUMBER_PATTERN = re.compile(r"^\s*\d{1,4}\s*$")

EN_MARKERS = {"the", "and", "for", "with", "our", "this", "will", "have", "are"}
PT_MARKERS = {"que", "para", "com", "uma", "não", "são", "está", "nossa", "foi"}


def guess_is_english(text: str) -> bool:
    """Very lightweight heuristic -- not a real language detector, just enough
    to warn the user if a non-English PDF was uploaded (model is English-only,
    see project scope decision)."""
    words = re.findall(r"[a-zà-ÿ']+", text.lower())
    sample = words[:2000]
    if not sample:
        return True
    en_hits = sum(1 for w in sample if w in EN_MARKERS)
    pt_hits = sum(1 for w in sample if w in PT_MARKERS)
    return en_hits >= pt_hits


def is_table_like(sent, min_real_word_ratio: float = 0.45) -> bool:
    """Same heuristic as Catarina's C3 notebook -- filters out EU Taxonomy
    annex tables and other non-prose fragments that spaCy still segments as
    'sentences' but which would be meaningless classifier input."""
    tokens = [t for t in sent if not t.is_space and not t.is_punct]
    if not tokens:
        return True
    real_words = [t for t in tokens if t.is_alpha and len(t.text) >= 3]
    if len(real_words) / len(tokens) < min_real_word_ratio:
        return True
    if len(TAXONOMY_CODE_PATTERN.findall(sent.text)) >= 2:
        return True
    if not any(t.pos_ in ("VERB", "AUX") for t in sent):
        return True
    return False


def extract_pdf_text_per_page(uploaded_file):
    import pdfplumber
    pages_text = []
    with pdfplumber.open(uploaded_file) as pdf:
        n_pages = len(pdf.pages)
        limit = min(n_pages, MAX_LIVE_PAGES)
        for page in pdf.pages[:limit]:
            text = page.extract_text() or ""
            text = "\n".join(
                line for line in text.splitlines()
                if not PAGE_NUMBER_PATTERN.match(line.strip())
            )
            pages_text.append(text)
    return pages_text, n_pages, limit


def process_uploaded_pdf(uploaded_file, nlp):
    pages_text, n_pages, limit = extract_pdf_text_per_page(uploaded_file)
    full_text = "\n".join(pages_text)

    rows = []
    for page_text in pages_text:
        if not page_text.strip():
            continue
        doc = nlp(page_text)
        for sent in doc.sents:
            raw = sent.text.strip()
            if len(raw) < 5:
                continue
            if is_table_like(sent):
                continue
            tokens = [t for t in sent if not t.is_space]
            lemma = " ".join(t.lemma_ for t in tokens if not t.is_punct)
            if not lemma.strip():
                continue
            rows.append({"raw_sentence": raw, "lemmatized_sentence": lemma})

    return rows, full_text, n_pages, limit


# ---------------------------------------------------------------------------
# Views
# ---------------------------------------------------------------------------

def render_company_view():
    st.header("Company view")
    st.caption(FRAMING_NOTE)

    summary_df = load_summary()
    scored_df = load_scored_sentences()

    companies = sorted(summary_df["company"].str.capitalize().unique())
    company = st.selectbox("Company", companies)
    company_key = company.lower()

    years = sorted(summary_df.loc[summary_df["company"] == company_key, "year"].unique())
    year = st.selectbox("Year", years)

    row = summary_df[(summary_df["company"] == company_key) & (summary_df["year"] == year)]
    if row.empty:
        st.warning("No precomputed data for this company/year.")
        return
    row = row.iloc[0]

    st.metric("Commitment Specificity Ratio", f"{row['commitment_specificity_ratio']:.1%}")

    counts = pd.DataFrame({
        "label": ["Commitment", "Aspiration", "Reported result"],
        "count": [row["n_commitment"], row["n_aspiration"], row["n_reported_result"]],
    }).set_index("label")
    st.bar_chart(counts)

    st.subheader("Example sentences")
    subset = scored_df[(scored_df["company"] == company_key) & (scored_df["year"] == year)]
    for label in LABELS:
        with st.expander(LABEL_DISPLAY[label]):
            examples = subset[subset["predicted_label"] == label]["raw_sentence"].head(5)
            if examples.empty:
                st.write("No examples found.")
            for s in examples:
                st.write(f"- {s}")


def render_comparison_view():
    st.header("Comparison view")
    st.caption(FRAMING_NOTE)

    summary_df = load_summary().sort_values(["company", "year"])
    st.bar_chart(summary_df.set_index("company_year")["commitment_specificity_ratio"])

    st.subheader("Full summary table")
    display_cols = ["company", "year", "n_commitment", "n_aspiration",
                     "n_reported_result", "n_total_classified", "commitment_specificity_ratio"]
    st.dataframe(summary_df[display_cols], use_container_width=True)


def render_live_upload_view():
    st.header("Analyze a new report")
    st.caption(FRAMING_NOTE)
    st.info(
        "**Provisional / live path.** English-language sustainability reports only. "
        "Results here are never mixed into the Comparison view above — this proves "
        "the pipeline runs on unseen input, it isn't a vetted addition to the core dataset."
    )

    uploaded_file = st.file_uploader("Upload a sustainability report (PDF)", type=["pdf"])
    if uploaded_file is None:
        return

    with st.spinner("Extracting and classifying — this can take a minute on a large PDF..."):
        nlp = load_spacy_model()
        rows, full_text, n_pages, limit = process_uploaded_pdf(uploaded_file, nlp)

        if not guess_is_english(full_text):
            st.warning(
                "This document doesn't look like English text. The classifier is "
                "trained on English-language reports only — results below may be unreliable."
            )

        if limit < n_pages:
            st.caption(f"Note: processed the first {limit} of {n_pages} pages "
                       f"(demo performance cap).")

        if not rows:
            st.warning("No usable sentences found in this PDF.")
            return

        vectorizer, classifier = load_classifier_model()
        from predict_commitment import predict_commitment
        lemmas = [r["lemmatized_sentence"] for r in rows]
        labels = predict_commitment(lemmas, vectorizer, classifier)
        for r, label in zip(rows, labels):
            r["predicted_label"] = label

    result_df = pd.DataFrame(rows)
    counts = result_df["predicted_label"].value_counts()
    n_commitment = int(counts.get("commitment", 0))
    n_aspiration = int(counts.get("aspiration", 0))
    n_result = int(counts.get("reported_result", 0))
    n_total = n_commitment + n_aspiration + n_result

    st.success(f"Classified {n_total} sentences from {limit} page(s).")

    ratio = (n_commitment / n_total) if n_total else 0
    st.metric("Commitment Specificity Ratio", f"{ratio:.1%}")

    chart_df = pd.DataFrame({
        "label": ["Commitment", "Aspiration", "Reported result"],
        "count": [n_commitment, n_aspiration, n_result],
    }).set_index("label")
    st.bar_chart(chart_df)

    st.subheader("Example sentences")
    for label in LABELS:
        with st.expander(LABEL_DISPLAY[label]):
            examples = result_df[result_df["predicted_label"] == label]["raw_sentence"].head(5)
            if examples.empty:
                st.write("No examples found.")
            for s in examples:
                st.write(f"- {s}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    st.set_page_config(page_title="ESG Lens", layout="wide")
    st.title("ESG Lens")
    st.caption("Commitment Specificity Ratio across sustainability reports")

    page = st.sidebar.radio("View", ["Company", "Comparison", "Analyze a new report"])

    if page == "Company":
        render_company_view()
    elif page == "Comparison":
        render_comparison_view()
    else:
        render_live_upload_view()


if __name__ == "__main__":
    main()
