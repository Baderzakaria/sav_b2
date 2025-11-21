
"""Tweet cleaning and enrichment pipeline.

Usage:
    python clean_free_tweets.py \
        --input "/home/bader/Desktop/sav_b2/data/free tweet export.csv"

This script removes noisy tweets (retweets, duplicates, off-topic),
normalizes text, and exports a clean dataset ready for the orchestrator
(which handles detection, sentiment, and categorisation). Outputs are
saved as timestamped and "latest" CSV/JSON artifacts under `data/cleaned`,
with matching metadata stored in `data/results`.
"""

from __future__ import annotations

import argparse
import html
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict

import pandas as pd
from pandas.api import types as pd_types


RE_EMOJI = re.compile(
    "["
    "\U0001F600-\U0001F64F"
    "\U0001F300-\U0001F5FF"
    "\U0001F680-\U0001F6FF"
    "\U0001F1E0-\U0001F1FF"
    "\U00002702-\U000027B0"
    "\U000024C2-\U0001F251"
    "\U00010000-\U0010FFFF"
    "]+",
    flags=re.UNICODE,
)
RE_URL = re.compile(r"https?://\S+|www\.\S+", flags=re.IGNORECASE)
RE_MENTION = re.compile(r"@\w+")
RE_HASHTAG = re.compile(r"#")
RE_NON_ALNUM = re.compile(r"[^\w\s]", flags=re.UNICODE)

RT_PREFIX = re.compile(r"^RT\s+@.+", flags=re.IGNORECASE)

OFFICIAL_ACCOUNTS = {"Freebox", "Free", "Free_1337", "FreeboxNews"}

AD_SIGNATURES = [
    "free recrute",
    "rejoignez free",
    "free proxi",
    "freeproxi",
    "jobs.free",
    "offre d'emploi free",
    "free x 2024",
    "abonnez vous à free",
    "jeu concours free",
    "gagne ta freebox",
    "freebox pop dès",
    "free mobile 2€",
]

AUTO_REPLY_SNIPPETS = [
    "la messagerie privée x n’est plus disponible",
    "dm pour échanger vos informations",
    "merci de nous contacter en dm",
]

MASS_PROMO_THRESHOLD = 3


@dataclass
class PipelineArtifacts:
    csv_latest: Path
    csv_timestamped: Path
    json_latest: Path
    json_timestamped: Path
    metadata_latest: Path
    metadata_timestamped: Path


def normalize_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = html.unescape(text)
    text = RE_EMOJI.sub(" ", text)
    text = RE_URL.sub(" ", text)
    text = RE_MENTION.sub(" ", text)
    text = RE_HASHTAG.sub("", text)
    text = RE_NON_ALNUM.sub(" ", text)
    text = re.sub(r"\s+", " ", text, flags=re.UNICODE).strip().lower()
    return text


def remove_retweets(df: pd.DataFrame) -> pd.DataFrame:
    mask = ~df["full_text"].astype(str).str.match(RT_PREFIX)
    return df[mask].copy()


def drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    if "id" in df.columns:
        df = df.drop_duplicates(subset=["id"])
    return df.drop_duplicates(subset=["full_text"]).copy()


def remove_official_accounts(df: pd.DataFrame) -> pd.DataFrame:
    if "screen_name" not in df.columns:
        return df
    mask = ~df["screen_name"].isin(OFFICIAL_ACCOUNTS)
    return df[mask].copy()


def filter_off_topic(df: pd.DataFrame) -> pd.DataFrame:
    def is_ad_or_auto_reply(text: str) -> bool:
        t = str(text).lower()
        if not t.strip():
            return True
        if any(signature in t for signature in AD_SIGNATURES):
            return True
        if any(snippet in t for snippet in AUTO_REPLY_SNIPPETS):
            return True
        return False

    mask = ~df["full_text"].apply(is_ad_or_auto_reply)
    filtered = df[mask].copy()
    return filtered


def remove_mass_promotions(df: pd.DataFrame) -> pd.DataFrame:
    if "clean_text" not in df.columns:
        return df
    counts = df["clean_text"].value_counts()
    promo_texts = [
        text for text, count in counts.items()
        if count >= MASS_PROMO_THRESHOLD and ("free" in text or "freebox" in text)
    ]
    if not promo_texts:
        return df
    mask = ~df["clean_text"].isin(promo_texts)
    return df[mask].copy()


def load_dataset(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".xls", ".xlsx"}:
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)
    if "full_text" not in df.columns:
        raise ValueError("Expected column 'full_text' in dataset.")
    return df


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.dropna(subset=["full_text"])
    df = remove_retweets(df)
    df = remove_official_accounts(df)
    df = drop_duplicates(df)
    df = filter_off_topic(df)

    if "created_at" in df.columns:
        df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce", utc=True)
        df["date_iso"] = df["created_at"].dt.tz_convert(None)

    df["clean_text"] = df["full_text"].apply(normalize_text)
    df = df.drop_duplicates(subset=["clean_text"])
    df = remove_mass_promotions(df)

    useful_cols = [
        "id",
        "date_iso",
        "screen_name",
        "full_text",
        "clean_text",
        "favorite_count",
        "reply_count",
    ]
    available_cols = [col for col in useful_cols if col in df.columns]
    return df[available_cols].reset_index(drop=True)


def prepare_artifacts(base_name: str, output_dir: Path, results_dir: Path, timestamp: str) -> PipelineArtifacts:
    csv_latest = output_dir / f"{base_name}-latest.csv"
    csv_timestamped = output_dir / f"{base_name}-{timestamp}.csv"
    json_latest = output_dir / f"{base_name}-latest.json"
    json_timestamped = output_dir / f"{base_name}-{timestamp}.json"
    metadata_latest = results_dir / f"{base_name}-latest.json"
    metadata_timestamped = results_dir / f"{base_name}-{timestamp}.json"
    return PipelineArtifacts(
        csv_latest=csv_latest,
        csv_timestamped=csv_timestamped,
        json_latest=json_latest,
        json_timestamped=json_timestamped,
        metadata_latest=metadata_latest,
        metadata_timestamped=metadata_timestamped,
    )


def serialize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    serializable = df.copy()
    for column in serializable.columns:
        series = serializable[column]
        if pd_types.is_datetime64_any_dtype(series):
            serializable[column] = series.dt.strftime("%Y-%m-%dT%H:%M:%S")
    return serializable.where(pd.notnull(serializable), None)


def write_dataframe_outputs(df: pd.DataFrame, artifacts: PipelineArtifacts) -> None:
    df.to_csv(artifacts.csv_latest, index=False)
    df.to_csv(artifacts.csv_timestamped, index=False)

    records = serialize_dataframe(df).to_dict(orient="records")

    for path in (artifacts.json_latest, artifacts.json_timestamped):
        with path.open("w", encoding="utf-8") as handle:
            json.dump(records, handle, ensure_ascii=False, indent=2)


def build_metadata(
    df_raw: pd.DataFrame,
    df_clean: pd.DataFrame,
    artifacts: PipelineArtifacts,
    input_path: Path,
    timestamp: str,
) -> Dict[str, object]:
    clean_lengths = df_clean["clean_text"].str.len() if "clean_text" in df_clean.columns else pd.Series(dtype=int)
    stats = {
        "avg_clean_length": float(clean_lengths.mean()) if not clean_lengths.empty else 0.0,
        "median_clean_length": float(clean_lengths.median()) if not clean_lengths.empty else 0.0,
        "unique_screen_names": int(df_clean["screen_name"].nunique()) if "screen_name" in df_clean.columns else 0,
    }

    return {
        "input_file": str(input_path),
        "records_in": int(len(df_raw)),
        "records_out": int(len(df_clean)),
        "removed": int(len(df_raw) - len(df_clean)),
        "timestamp_utc": timestamp,
        "outputs": {
            "csv_latest": str(artifacts.csv_latest),
            "csv_timestamped": str(artifacts.csv_timestamped),
            "json_latest": str(artifacts.json_latest),
            "json_timestamped": str(artifacts.json_timestamped),
        },
        "stats": stats,
    }


def write_metadata(metadata: Dict[str, object], artifacts: PipelineArtifacts) -> None:
    for path in (artifacts.metadata_latest, artifacts.metadata_timestamped):
        with path.open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, ensure_ascii=False, indent=2)


def sanitize_base_name(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", name).strip("_").lower()
    return cleaned or "tweets_clean"


def parse_args() -> argparse.Namespace:
    default_input = Path(__file__).resolve().parent / "data" / "free tweet export.csv"
    parser = argparse.ArgumentParser(description="Nettoyage et enrichissement des tweets Free.")
    parser.add_argument("--input", type=Path, default=default_input, help="Chemin du fichier CSV/XLSX à nettoyer.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "data" / "cleaned",
        help="Répertoire où stocker les CSV/JSON nettoyés.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "data" / "results",
        help="Répertoire pour les métadonnées de run.",
    )
    parser.add_argument("--name", type=str, default=None, help="Nom de base des fichiers de sortie.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = args.input.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    results_dir = args.results_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    base_name = sanitize_base_name(args.name or input_path.stem)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    artifacts = prepare_artifacts(base_name, output_dir, results_dir, timestamp)

    print(f"📥 Chargement du dataset depuis {input_path}")
    df_raw = load_dataset(input_path)
    print(f"   -> {len(df_raw)} lignes brutes")

    df_clean = clean_dataset(df_raw)
    print(f"✅ Dataset nettoyé : {len(df_clean)} lignes conservées")

    write_dataframe_outputs(df_clean, artifacts)
    metadata = build_metadata(df_raw, df_clean, artifacts, input_path, timestamp)
    write_metadata(metadata, artifacts)

    print("📦 Fichiers générés :")
    print(f"   CSV latest : {artifacts.csv_latest}")
    print(f"   CSV horodaté : {artifacts.csv_timestamped}")
    print(f"   JSON latest : {artifacts.json_latest}")
    print(f"   JSON horodaté : {artifacts.json_timestamped}")
    print(f"   Résumé : {artifacts.metadata_timestamped}")


if __name__ == "__main__":
    main()

