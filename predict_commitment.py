"""
T3 — Commitment classifier: importable functions for the live pipeline.

Usage (Paulo's app):
    from predict_commitment import load_model, predict_commitment

    vectorizer, classifier = load_model()
    labels = predict_commitment(sentences, vectorizer, classifier)

`sentences` should be lemmatized text (spaCy en_core_web, same pipeline as
Catarina's C3), matching what the model was trained on.
"""
import pickle
import os

_DIR = os.path.dirname(os.path.abspath(__file__))


def load_model(vectorizer_path=None, classifier_path=None):
    vectorizer_path = vectorizer_path or os.path.join(_DIR, "vectorizer.pkl")
    classifier_path = classifier_path or os.path.join(_DIR, "classifier.pkl")
    with open(vectorizer_path, "rb") as f:
        vectorizer = pickle.load(f)
    with open(classifier_path, "rb") as f:
        classifier = pickle.load(f)
    return vectorizer, classifier


def predict_commitment(lemmatized_sentences, vectorizer, classifier):
    """
    lemmatized_sentences: list[str] of lemmatized English sentences.
    Returns: list[str] of predicted labels — 'commitment' | 'aspiration' | 'reported_result'
    (the classifier is only trained on these 3 classes; 'skip' sentences should
    be filtered out upstream before calling this, same as in training).
    """
    X = vectorizer.transform(lemmatized_sentences)
    return classifier.predict(X).tolist()


def predict_commitment_proba(lemmatized_sentences, vectorizer, classifier):
    """Same as predict_commitment, but returns per-class probabilities too."""
    X = vectorizer.transform(lemmatized_sentences)
    preds = classifier.predict(X).tolist()
    probas = classifier.predict_proba(X).tolist()
    classes = classifier.classes_.tolist()
    return [
        {"label": p, "probabilities": dict(zip(classes, pr))}
        for p, pr in zip(preds, probas)
    ]
