from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
import streamlit as st


@dataclass(frozen=True)
class DataQualityReport:
    rows: int
    missing_sale_price_pct: float
    missing_listing_price_pct: float
    missing_images_pct: float
    missing_description_pct: float


EXPECTED_COLUMNS: tuple[str, ...] = (
    "product_name",
    "product_id",
    "listing_price",
    "sale_price",
    "discount",
    "brand",
    "description",
    "rating",
    "reviews",
    "images",
)


def _strip_object_columns(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    for col in columns:
        if col in df.columns:
            df[col] = df[col].astype("string").fillna("").str.strip()
    return df


def _coerce_numeric(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _infer_gender(text: str) -> str:
    t = (text or "").lower()
    if re.search(r"\b(women|woman|women's|womens)\b", t):
        return "Women"
    if re.search(r"\b(men|man|men's|mens)\b", t):
        return "Men"
    if re.search(r"\b(kids|kid|youth|boys|girls)\b", t):
        return "Kids"
    return "Unisex/Unknown"


FAMILY_PATTERNS: list[tuple[str, str]] = [
    ("Air Max", r"\bair\s+max\b"),
    ("Air Force", r"\bair\s+force\b"),
    ("Jordan", r"\bjordan\b|\bair\s+jordan\b"),
    ("Mercurial", r"\bmercurial\b"),
    ("Pegasus", r"\bpegasus\b"),
    ("VaporMax", r"\bvapormax\b|\bvapor\s*max\b"),
    ("Metcon", r"\bmetcon\b"),
    ("React", r"\breact\b"),
    ("Joyride", r"\bjoyride\b"),
    ("Free", r"\bfree\b"),
    ("Court", r"\bcourt\b"),
    ("SB", r"\bnike\s+sb\b|\bsb\b"),
]


def _infer_product_family(product_name: str) -> str:
    t = (product_name or "").lower()
    for family, pattern in FAMILY_PATTERNS:
        if re.search(pattern, t):
            return family
    if not t.strip():
        return "Unknown"
    first = re.split(r"[-,()]+|\s{2,}", product_name.strip(), maxsplit=1)[0]
    return first[:32] if first else "Other"


def _safe_qcut(series: pd.Series, labels: list[str]) -> pd.Series:
    clean = series.dropna()
    if clean.nunique() < 4:
        return pd.Series(["All"] * len(series), index=series.index, dtype="string")
    try:
        return pd.qcut(series, q=len(labels), labels=labels, duplicates="drop").astype("string")
    except Exception:
        return pd.Series(["All"] * len(series), index=series.index, dtype="string")


def preprocess_nike_df(raw_df: pd.DataFrame) -> tuple[pd.DataFrame, DataQualityReport]:
    df = raw_df.copy()

    missing_cols = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Dataset is missing required columns: {missing_cols}")

    df = _strip_object_columns(df, ["product_name", "product_id", "brand", "description", "images"])
    df = _coerce_numeric(df, ["listing_price", "sale_price", "discount", "rating", "reviews"])

    # Unit normalization (you chose divide by 100).
    for col in ("listing_price", "sale_price"):
        df[col] = df[col] / 100.0

    df["rating"] = df["rating"].fillna(0.0).clip(lower=0.0, upper=5.0)
    df["reviews"] = df["reviews"].fillna(0.0).clip(lower=0.0)
    df["description"] = df["description"].fillna("").astype("string")

    # Drop rows only if both prices are missing.
    df = df[~(df["listing_price"].isna() & df["sale_price"].isna())].copy()

    # Prefer a computed discount_pct when listing_price is present.
    computed_discount = pd.Series(pd.NA, index=df.index, dtype="Float64")
    valid_listing = df["listing_price"].fillna(0) > 0
    computed_discount.loc[valid_listing] = (
        (df.loc[valid_listing, "listing_price"] - df.loc[valid_listing, "sale_price"])
        / df.loc[valid_listing, "listing_price"]
    )
    computed_discount = computed_discount.clip(lower=0.0, upper=1.0)

    # Some datasets store discount as 0/1; some as percent (0-100); some as already fraction.
    discount_raw = df["discount"].astype("Float64")
    discount_norm = discount_raw.copy()
    discount_norm = discount_norm.where(discount_norm.isna(), discount_norm)
    discount_norm = discount_norm.where(discount_norm.isna() | (discount_norm <= 1.0), discount_norm / 100.0)
    discount_norm = discount_norm.clip(lower=0.0, upper=1.0)

    df["discount_pct"] = computed_discount.fillna(discount_norm).fillna(0.0).astype("float")
    df["is_discounted"] = df["discount_pct"] >= 0.01

    df["product_family"] = df["product_name"].apply(_infer_product_family).astype("string")
    df["gender_inferred"] = (
        (df["product_name"].fillna("") + " " + df["description"].fillna(""))
        .apply(_infer_gender)
        .astype("string")
    )
    df["price_band"] = _safe_qcut(
        df["sale_price"].astype("Float64"),
        labels=["Low", "Mid", "High", "Premium"],
    )

    # Data quality stats for the header.
    dq = DataQualityReport(
        rows=int(len(df)),
        missing_sale_price_pct=float(df["sale_price"].isna().mean()),
        missing_listing_price_pct=float(df["listing_price"].isna().mean()),
        missing_images_pct=float((df["images"].astype("string").fillna("").str.len() == 0).mean()),
        missing_description_pct=float((df["description"].astype("string").fillna("").str.len() == 0).mean()),
    )

    return df, dq


@st.cache_data(show_spinner="Loading and preparing Nike dataset…")
def load_data(csv_path: str) -> tuple[pd.DataFrame, DataQualityReport]:
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"CSV not found at: {path}")

    df = pd.read_csv(path)
    return preprocess_nike_df(df)

