from __future__ import annotations

import math

import pandas as pd


def _share(part: float, whole: float) -> float:
    if whole <= 0:
        return 0.0
    return float(part) / float(whole)


def insight_top_family(df: pd.DataFrame) -> str:
    if df.empty:
        return "No data available after filtering."

    agg = (
        df.groupby("product_family", as_index=False)
        .agg(catalog_value=("sale_price", "sum"), products=("product_id", "nunique"))
        .sort_values("catalog_value", ascending=False)
    )
    top = agg.iloc[0]
    total = float(agg["catalog_value"].sum())
    top_share = _share(float(top["catalog_value"]), total)

    second_val = float(agg.iloc[1]["catalog_value"]) if len(agg) > 1 else 0.0
    gap = float(top["catalog_value"]) - second_val

    return (
        f"**{top['product_family']}** leads the catalog with **{top_share:.1%}** of total catalog value "
        f"and **{int(top['products']):,}** unique products. "
        f"The gap vs #2 is **{gap:,.2f}** in catalog value."
    )


def insight_price_distribution(df: pd.DataFrame) -> str:
    if df.empty:
        return "No data available after filtering."

    s = pd.to_numeric(df["sale_price"], errors="coerce").dropna()
    if s.empty:
        return "Sale price is missing for all rows in the current selection."

    median = float(s.median())
    q1 = float(s.quantile(0.25))
    q3 = float(s.quantile(0.75))
    most_common = (
        df["price_band"].astype("string").value_counts(dropna=True).index[0]
        if "price_band" in df.columns and not df["price_band"].dropna().empty
        else "N/A"
    )
    return (
        f"Median sale price is **{median:,.2f}** with an interquartile range of **{q1:,.2f}–{q3:,.2f}**. "
        f"Most products fall into the **{most_common}** band."
    )


def insight_discount_price_relationship(df: pd.DataFrame) -> str:
    if df.empty:
        return "No data available after filtering."

    x = pd.to_numeric(df["discount_pct"], errors="coerce")
    y = pd.to_numeric(df["sale_price"], errors="coerce")
    valid = x.notna() & y.notna()
    if valid.sum() < 10:
        return "Not enough data points in the current selection to estimate a stable relationship."

    corr = float(x[valid].corr(y[valid], method="spearman"))
    corr_abs = abs(corr)
    if corr_abs < 0.1:
        strength = "weak"
    elif corr_abs < 0.3:
        strength = "moderate"
    else:
        strength = "strong"

    direction = "higher discounts tend to align with lower prices" if corr < 0 else "higher discounts tend to align with higher prices"

    fam = (
        df.groupby("product_family", as_index=False)
        .agg(median_discount=("discount_pct", "median"))
        .sort_values("median_discount", ascending=False)
        .head(1)
    )
    fam_name = fam.iloc[0]["product_family"]
    fam_disc = float(fam.iloc[0]["median_discount"])

    return (
        f"The discount–price relationship is **{strength}** (Spearman \(\\rho={corr:.2f}\\)), "
        f"meaning **{direction}** in this selection. "
        f"**{fam_name}** shows the highest median discount at **{fam_disc:.0%}**."
    )


def insight_sentiment_engagement(df: pd.DataFrame) -> str:
    if df.empty:
        return "No data available after filtering."

    agg = (
        df.groupby("product_family", as_index=False)
        .agg(products=("product_id", "nunique"), avg_rating=("rating", "mean"), avg_reviews=("reviews", "mean"))
    )
    agg = agg[agg["products"] >= 5].copy()
    if agg.empty:
        return "Not enough products per family (need ≥5) to compare sentiment reliably."

    agg["score"] = agg["avg_rating"].fillna(0) * (agg["avg_reviews"].fillna(0) + 1).apply(math.log1p)
    best = agg.sort_values("score", ascending=False).iloc[0]

    return (
        f"**{best['product_family']}** stands out as the best balance of sentiment and engagement "
        f"with an average rating of **{float(best['avg_rating']):.2f}** "
        f"and **{float(best['avg_reviews']):,.1f}** average reviews (min 5 products per family)."
    )


def insight_gender_pricing(df: pd.DataFrame) -> str:
    if df.empty:
        return "No data available after filtering."

    agg = (
        df.groupby("gender_inferred", as_index=False)
        .agg(median_price=("sale_price", "median"), avg_discount=("discount_pct", "mean"), products=("product_id", "nunique"))
        .sort_values("median_price", ascending=False)
    )
    top = agg.iloc[0]
    disc_top = agg.sort_values("avg_discount", ascending=False).iloc[0]

    return (
        f"**{top['gender_inferred']}** is the priciest segment by median sale price (**{float(top['median_price']):,.2f}**). "
        f"Deepest average discount appears in **{disc_top['gender_inferred']}** at **{float(disc_top['avg_discount']):.0%}** "
        f"(based on **{int(disc_top['products']):,}** products in the current selection)."
    )

