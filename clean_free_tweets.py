from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import List

import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

MENTION_RE = re.compile(r"(?i)@\w+")
URL_RE = re.compile(r"https?://\S+")
WHITESPACE_RE = re.compile(r"\s+")

ON_TOPIC_KEYWORDS = [
    "free",
    "freebox",
    "free mobile",
    "free pro",
    "free fibre",
    "free delta",
    "free pop",
    "free 5g",
    "free 4g",
    "réseau free",
    "reseau free",
    "freewifi",
    "free assistance",
    "freebox pop",
    "freebox delta",
]

THEME_KEYWORDS = {
    "reseau": [
        "panne",
        "coupure",
        "réseau",
        "reseau",
        "connexion",
        "internet",
        "debit",
        "upload",
        "download",
        "ping",
        "fibre",
        "4g",
        "5g",
        "latence",
    ],
    "facturation": [
        "facture",
        "prelevement",
        "prélèvement",
        "paiement",
        "remboursement",
        "surfacturation",
        "montant",
        "tarif",
        "prix",
    ],
    "abonnement": [
        "abonnement",
        "resiliation",
        "résiliation",
        "inscription",
        "offre",
        "contrat",
        "portabilite",
        "portabilité",
    ],
    "equipement": [
        "box",
        "modem",
        "routeur",
        "player",
        "décodeur",
        "decodeur",
        "tv",
        "serveur",
    ],
    "support": [
        "service client",
        "hotline",
        "assistance",
        "sav",
        "help",
        "support",
        "ticket",
    ],
}

URGENCY_PATTERNS = [
    r"\burgent[e]?\b",
    r"\bimpossible\b",
    r"\bdepuis\s+\d+\s*(?:jours?|heures?)",
    r"\bdepuis\s+(?:hier|ce matin)\b",
    r"\bhelp\b",
    r"\bsvp\b",
    r"\basap\b",
    r"\bperdu\b",
    r"\baucun service\b",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clean raw Free tweets export and enrich with annotations."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/free tweet export.csv"),
        help="Path to the raw CSV export.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/free_tweet_export_clean.csv"),
        help="Destination path for the cleaned dataset.",
    )
    return parser.parse_args()


def load_dataset(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, keep_default_na=False, na_values=["null"])
    # Standardize string columns
    if "full_text" not in df.columns:
        raise ValueError("Expected 'full_text' column not found in dataset.")
    df["full_text"] = df["full_text"].astype(str)
    return df


def is_retweet(row: pd.Series) -> bool:
    retweeted_status = row.get("retweeted_status")
    if isinstance(retweeted_status, float) and pd.isna(retweeted_status):
        retweeted_status = None
    return (
        (isinstance(retweeted_status, str) and retweeted_status.strip() != "")
        or str(row.get("full_text", "")).strip().lower().startswith("rt @")
    )


def remove_retweets(df: pd.DataFrame) -> pd.DataFrame:
    mask = df.apply(lambda row: not is_retweet(row), axis=1)
    return df[mask].copy()


def drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop_duplicates(subset=["full_text"]).copy()


def is_on_topic(text: str) -> bool:
    lower_text = text.lower()
    return any(keyword in lower_text for keyword in ON_TOPIC_KEYWORDS)


def filter_off_topic(df: pd.DataFrame) -> pd.DataFrame:
    mask = df["full_text"].fillna("").apply(is_on_topic)
    return df[mask].copy()


def normalize_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text_no_urls = URL_RE.sub(" ", text)
    text_no_mentions = MENTION_RE.sub(" ", text_no_urls)
    ascii_text = (
        text_no_mentions.encode("ascii", "ignore").decode("ascii", errors="ignore")
    )
    normalized = WHITESPACE_RE.sub(" ", ascii_text)
    return normalized.strip()


def detect_theme(text: str) -> str:
    if not text:
        return "autre"
    lower_text = text.lower()
    detected: List[str] = []
    for theme, keywords in THEME_KEYWORDS.items():
        if any(keyword in lower_text for keyword in keywords):
            detected.append(theme)
    return ";".join(sorted(set(detected))) if detected else "autre"


def detect_urgency(text: str) -> bool:
    if not text:
        return False
    for pattern in URGENCY_PATTERNS:
        if re.search(pattern, text.lower()):
            return True
    return False


def annotate_sentiment(
    analyzer: SentimentIntensityAnalyzer, text: str
) -> tuple[str, float]:
    if not text:
        return "neutre", 0.0
    score = analyzer.polarity_scores(text)["compound"]
    if score >= 0.05:
        label = "positif"
    elif score <= -0.05:
        label = "negatif"
    else:
        label = "neutre"
    return label, float(score)


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=["full_text"])
    df = remove_retweets(df)
    df = drop_duplicates(df)
    df = filter_off_topic(df)

    analyzer = SentimentIntensityAnalyzer()
    clean_texts = df["full_text"].apply(normalize_text)
    df["clean_text"] = clean_texts

    sentiments = clean_texts.apply(lambda text: annotate_sentiment(analyzer, text))
    df["sentiment_label"] = sentiments.apply(lambda tup: tup[0])
    df["sentiment_score"] = sentiments.apply(lambda tup: tup[1])

    df["theme"] = clean_texts.apply(detect_theme)
    df["urgent"] = clean_texts.apply(detect_urgency)

    df["has_media"] = df.get("media", "").apply(
        lambda media: bool(media) and str(media) != "[]"
    )
    df["text_length"] = clean_texts.str.len()

    return df.reset_index(drop=True)


def main() -> None:
    args = parse_args()
    input_path = args.input
    output_path = args.output

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df_raw = load_dataset(input_path)
    df_clean = clean_dataset(df_raw)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_csv(output_path, index=False)
    print(
        f"✅ Clean dataset saved to {output_path} ({len(df_clean)} tweets after filtering)"
    )


if __name__ == "__main__":
    main()

