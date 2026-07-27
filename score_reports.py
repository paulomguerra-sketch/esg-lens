"""
ESG Lens — Offline scoring: classifier -> precomputed app data
Generated: 2026-07-27 16:00 (Europe/Lisbon)

Run this AFTER:
  - Catarina's C3 notebook has produced `all_companies_sentences.csv`
    (columns: company, year, sentence_id, raw_sentence, lemmatized_sentence, n_tokens)
  - T3 training has produced `vectorizer.pkl` + `classifier.pkl`
  - `predict_commitment.py` is in the same folder as this script

Scope (confirmed in Catarina's C1/C2/C3 notebooks): Galp, EDP, Amorim only,
2024 + 2025 -> 6 report-years, English-language reports. Sonae, Mota-Engil,
and Millennium BCP are out of scope -- this script does not filter for them
specially; it simply scores whatever is in the input CSV, which should
already be limited to the 3 in-scope companies.

What this produces (both written to data/processed/):
  1. scored_sentences.csv   -- one row per sentence, for the Company deep-dive view
  2. company_summary.csv    -- one row per company-year, for the Comparison view,
                               including the Commitment Specificity Ratio

Framing guardrail: the Commitment Specificity Ratio measures language
specificity and transparency. It is not a greenwashing verdict -- keep this
framing in any UI text or slide that reports it.
"""

import os
import sys
import csv
from datetime import datetime, timezone

import pandas as pd

# ---------------------------------------------------------------------------
# Config -- adjust these paths if your files live somewhere else
# ---------------------------------------------------------------------------

# The script looks in these locations, in order, for the C3 output.
# Simplest fix if neither matches: copy/rename your file to
# "all_companies_sentences.csv" in the same folder as this script.
CANDIDATE_INPUT_PATHS = [
    "all_companies_sentences.csv",
    os.path.join("clean_sentences", "all_companies_sentences.csv"),
]

OUTPUT_DIR = os.path.join("data", "processed")
SCORED_SENTENCES_OUT = os.path.join(OUTPUT_DIR, "scored_sentences.csv")
COMPANY_SUMMARY_OUT = os.path.join(OUTPUT_DIR, "company_summary.csv")

SCORED_AT_UTC = datetime.now(timezone.utc).isoformat(timespec="seconds")

# ---------------------------------------------------------------------------
# Step 1 -- locate and load the C3 sentence dataset
# ---------------------------------------------------------------------------

def find_input_csv():
    for path in CANDIDATE_INPUT_PATHS:
        if os.path.exists(path):
            return path
    tried = "\n  ".join(CANDIDATE_INPUT_PATHS)
    raise FileNotFoundError(
        f"Could not find all_companies_sentences.csv. Tried:\n  {tried}\n"
        f"Copy your C3 output CSV into this folder (or edit CANDIDATE_INPUT_PATHS)."
    )


def load_sentences():
    input_path = find_input_csv()
    print(f"Loading sentences from: {input_path}")
    df = pd.read_csv(input_path)

    required_cols = {"company", "year", "sentence_id", "raw_sentence", "lemmatized_sentence"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Input CSV is missing expected column(s): {missing}")

    print(f"Loaded {len(df)} sentences across {df['company'].nunique()} companies, "
          f"{df['year'].nunique()} years")
    print(df.groupby(["company", "year"]).size())

    # Drop rows with no usable lemmatized text -- can't classify these
    before = len(df)
    df = df[df["lemmatized_sentence"].notna()]
    df = df[df["lemmatized_sentence"].astype(str).str.strip() != ""]
    dropped = before - len(df)
    if dropped:
        print(f"Dropped {dropped} row(s) with empty/missing lemmatized_sentence")

    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Step 2 -- load the trained model and classify every sentence
# ---------------------------------------------------------------------------

def classify_all(df):
    # predict_commitment.py, vectorizer.pkl, classifier.pkl must be in the
    # same folder as this script (see accompanying setup instructions)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from predict_commitment import load_model, predict_commitment

    vectorizer, classifier = load_model()
    print(f"Model loaded. Classes: {list(classifier.classes_)}")

    labels = predict_commitment(df["lemmatized_sentence"].tolist(), vectorizer, classifier)
    df = df.copy()
    df["predicted_label"] = labels
    df["scored_at_utc"] = SCORED_AT_UTC
    return df


# ---------------------------------------------------------------------------
# Step 3 -- build the per-company-year summary + Commitment Specificity Ratio
# ---------------------------------------------------------------------------

def build_summary(scored_df):
    rows = []
    for (company, year), group in scored_df.groupby(["company", "year"]):
        counts = group["predicted_label"].value_counts()
        n_commitment = int(counts.get("commitment", 0))
        n_aspiration = int(counts.get("aspiration", 0))
        n_result = int(counts.get("reported_result", 0))
        n_total = n_commitment + n_aspiration + n_result

        ratio = (n_commitment / n_total) if n_total > 0 else None

        rows.append({
            "company": company,
            "year": year,
            "n_commitment": n_commitment,
            "n_aspiration": n_aspiration,
            "n_reported_result": n_result,
            "n_total_classified": n_total,
            "commitment_specificity_ratio": round(ratio, 4) if ratio is not None else None,
            "scored_at_utc": SCORED_AT_UTC,
        })

    summary_df = pd.DataFrame(rows).sort_values(["company", "year"]).reset_index(drop=True)
    return summary_df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = load_sentences()
    scored_df = classify_all(df)

    # Row-level output, for the Company deep-dive view
    out_cols = ["company", "year", "sentence_id", "raw_sentence",
                "lemmatized_sentence", "predicted_label", "scored_at_utc"]
    scored_df[out_cols].to_csv(SCORED_SENTENCES_OUT, index=False, encoding="utf-8-sig")
    print(f"\nWrote {len(scored_df)} scored sentences -> {SCORED_SENTENCES_OUT}")

    # Company-year summary, for the Comparison view
    summary_df = build_summary(scored_df)
    summary_df.to_csv(COMPANY_SUMMARY_OUT, index=False, encoding="utf-8-sig")
    print(f"Wrote {len(summary_df)} company-year summary rows -> {COMPANY_SUMMARY_OUT}")

    print("\nCommitment Specificity Ratio by company-year:")
    print(summary_df[["company", "year", "commitment_specificity_ratio"]].to_string(index=False))

    print("\nReminder: this ratio measures language specificity/transparency, "
          "not greenwashing -- keep that framing in the app and slides.")


if __name__ == "__main__":
    main()
