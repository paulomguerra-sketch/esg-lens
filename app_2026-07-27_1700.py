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

LABEL_SHORT = {
    "commitment": "COMMITMENT",
    "aspiration": "ASPIRATION",
    "reported_result": "REPORTED RESULT",
}

LABEL_COLORS = {
    "commitment": "#C9922B",       # amber -- checkable, "stamped"
    "aspiration": "#7C8798",       # slate -- vague, unfalsifiable
    "reported_result": "#3C6E71",  # teal -- already settled
}

LABEL_EXPLANATION = {
    "commitment": "Has a number **and** a deadline/year, plus a forward action verb — "
                  "checkable at a specific future date.",
    "aspiration": "Forward-looking language with no verifiable number+deadline pair — "
                  "direction without a checkable commitment.",
    "reported_result": "Describes something that already happened — past tense, "
                        "not a forward pledge.",
}

FRAMING_NOTE = (
    "ESG Lens measures language **specificity and transparency** — "
    "it does not adjudicate greenwashing."
)

MAX_LIVE_PAGES = 150  # soft cap so a huge PDF doesn't stall a live demo

CUE_PATTERN = re.compile(r"\b(19|20)\d{2}\b|\b\d+(\.\d+)?\s?%")

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
# UI helpers -- design, cue highlighting, confidence
# ---------------------------------------------------------------------------

def inject_custom_css():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:wght@600;700&family=IBM+Plex+Sans:wght@400;500;600&display=swap');

        [data-testid="stMarkdownContainer"] {
            font-family: 'IBM Plex Sans', sans-serif;
        }
        [data-testid="stMarkdownContainer"] h1,
        [data-testid="stMarkdownContainer"] h2,
        [data-testid="stMarkdownContainer"] h3 {
            font-family: 'Source Serif 4', serif;
            color: #1B2430;
        }
        [data-testid="stMetricValue"] {
            font-family: 'Source Serif 4', serif;
            color: #1F4E4C;
        }
        [data-testid="stMetric"] {
            background-color: #F5F7F8;
            border: 1px solid #D8DEE2;
            border-radius: 8px;
            padding: 12px 16px;
        }
        .cue-badge {
            display: inline-block;
            font-family: 'IBM Plex Sans', sans-serif;
            font-size: 0.72rem;
            font-weight: 600;
            letter-spacing: 0.03em;
            padding: 2px 8px;
            border-radius: 4px;
            color: white;
            margin-right: 8px;
        }
        .confidence-tag {
            font-size: 0.78rem;
            color: #5B6572;
            font-style: italic;
        }
        .example-row {
            padding: 6px 0;
            border-bottom: 1px solid #EAEDEF;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def highlight_cues(text: str) -> str:
    """Bold every year/percentage in a sentence so the number+deadline signal
    the classifier actually looks for is visually obvious."""
    return CUE_PATTERN.sub(lambda m: f"**{m.group(0)}**", text)


def label_badge_html(label: str) -> str:
    color = LABEL_COLORS.get(label, "#999999")
    short = LABEL_SHORT.get(label, label.upper())
    return f'<span class="cue-badge" style="background-color:{color};">{short}</span>'


def render_example_row(raw_sentence: str, label: str, confidence: float | None = None):
    conf_html = (
        f'<span class="confidence-tag">{confidence:.0%} confidence</span>'
        if confidence is not None else ""
    )
    st.markdown(
        f'<div class="example-row">{label_badge_html(label)}{conf_html}<br>'
        f'{highlight_cues(raw_sentence)}</div>',
        unsafe_allow_html=True,
    )


def get_confidences(lemmas, vectorizer, classifier, labels):
    """Returns the model's probability for each row's OWN predicted label
    (not the max across all classes -- we already know the predicted label,
    this just reports how confident the model was in that specific call)."""
    if not lemmas:
        return []
    X = vectorizer.transform(lemmas)
    probas = classifier.predict_proba(X)
    classes = list(classifier.classes_)
    confidences = []
    for row_proba, label in zip(probas, labels):
        idx = classes.index(label)
        confidences.append(row_proba[idx])
    return confidences


def render_how_to_read():
    with st.expander("How to read this"):
        st.markdown(
            "**Commitment Specificity Ratio** = `commitment ÷ (commitment + aspiration + "
            "reported result)`. A higher ratio means a greater share of the report's forward "
            "and result language is specific and checkable, not that the company is more "
            "or less truthful."
        )
        for label in LABELS:
            st.markdown(f"{label_badge_html(label)} {LABEL_EXPLANATION[label]}", unsafe_allow_html=True)
        st.caption(FRAMING_NOTE)


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
    vectorizer, classifier = load_classifier_model()
    for label in LABELS:
        with st.expander(LABEL_DISPLAY[label]):
            examples = subset[subset["predicted_label"] == label].head(5)
            if examples.empty:
                st.write("No examples found.")
                continue
            confidences = get_confidences(
                examples["lemmatized_sentence"].tolist(), vectorizer, classifier,
                examples["predicted_label"].tolist(),
            )
            for (_, ex_row), conf in zip(examples.iterrows(), confidences):
                render_example_row(ex_row["raw_sentence"], label, conf)


def render_comparison_view():
    st.header("Comparison view")
    st.caption(FRAMING_NOTE)

    summary_df = load_summary().sort_values(["company", "year"]).reset_index(drop=True)

    st.caption(
        "The ratio is naturally small — most sustainability-report language is "
        "aspirational or describes past results, not specific forward pledges. "
        "A few percentage points is typical; the comparison is about relative "
        "differences between reports, not an absolute pass/fail threshold."
    )

    chart_df = summary_df.copy()
    chart_df["Ratio (%)"] = chart_df["commitment_specificity_ratio"] * 100
    st.bar_chart(chart_df.set_index("company_year")["Ratio (%)"])

    best = summary_df.loc[summary_df["commitment_specificity_ratio"].idxmax()]
    worst = summary_df.loc[summary_df["commitment_specificity_ratio"].idxmin()]
    st.markdown(
        f"**Highest specificity:** {best['company'].capitalize()} {int(best['year'])} "
        f"({best['commitment_specificity_ratio']:.1%}) — **Lowest:** "
        f"{worst['company'].capitalize()} {int(worst['year'])} "
        f"({worst['commitment_specificity_ratio']:.1%})"
    )

    st.subheader("Full summary table")
    display_df = summary_df[["company", "year", "n_commitment", "n_aspiration",
                              "n_reported_result", "n_total_classified",
                              "commitment_specificity_ratio"]].copy()
    display_df["commitment_specificity_ratio"] = display_df["commitment_specificity_ratio"].map(
        lambda x: f"{x:.1%}"
    )
    display_df.columns = ["Company", "Year", "# Commitment", "# Aspiration",
                           "# Reported result", "Total classified",
                           "Commitment Specificity Ratio"]
    display_df["Company"] = display_df["Company"].str.capitalize()
    st.dataframe(display_df, use_container_width=True, hide_index=True)


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
        from predict_commitment import predict_commitment_proba
        lemmas = [r["lemmatized_sentence"] for r in rows]
        predictions = predict_commitment_proba(lemmas, vectorizer, classifier)
        for r, pred in zip(rows, predictions):
            r["predicted_label"] = pred["label"]
            r["confidence"] = pred["probabilities"][pred["label"]]

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
            examples = result_df[result_df["predicted_label"] == label].head(5)
            if examples.empty:
                st.write("No examples found.")
                continue
            for _, ex_row in examples.iterrows():
                render_example_row(ex_row["raw_sentence"], label, ex_row["confidence"])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    st.set_page_config(page_title="ESG Lens", layout="wide")
    inject_custom_css()
    st.title("ESG Lens")
    st.caption("Commitment Specificity Ratio across sustainability reports")
    render_how_to_read()

    page = st.sidebar.radio("View", ["Company", "Comparison", "Analyze a new report"])

    if page == "Company":
        render_company_view()
    elif page == "Comparison":
        render_comparison_view()
    else:
        render_live_upload_view()


if __name__ == "__main__":
    main()
