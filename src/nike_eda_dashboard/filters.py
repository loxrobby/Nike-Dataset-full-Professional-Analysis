from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import streamlit as st


@dataclass(frozen=True)
class FilterState:
    brands: list[str]
    families: list[str]
    genders: list[str]
    price_range: tuple[float, float]
    rating_range: tuple[float, float]
    reviews_range: tuple[int, int]
    discount_range: tuple[float, float]
    search: str
    discounted_only: bool


def _series_min_max(series: pd.Series, default_min: float, default_max: float) -> tuple[float, float]:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return default_min, default_max
    return float(clean.min()), float(clean.max())


def build_sidebar_filters(df: pd.DataFrame) -> FilterState:
    st.sidebar.header("Controls")

    top_controls = st.sidebar.columns([1, 1])
    if top_controls[0].button("Reset", use_container_width=True):
        st.session_state.clear()
        st.rerun()
    top_controls[1].caption("Collapse below ↓")

    brands = sorted([b for b in df["brand"].dropna().unique().tolist() if str(b).strip()])
    families = sorted([f for f in df["product_family"].dropna().unique().tolist() if str(f).strip()])
    genders = sorted([g for g in df["gender_inferred"].dropna().unique().tolist() if str(g).strip()])

    # Start collapsed so filters aren't always shown.
    with st.sidebar.expander("Filters", expanded=False):
        selected_brands = st.multiselect("Brand", options=brands, default=brands)
        selected_families = st.multiselect("Product family", options=families, default=families)
        selected_genders = st.multiselect("Inferred gender segment", options=genders, default=genders)

        p_min, p_max = _series_min_max(df["sale_price"], 0.0, 1.0)
        price_range = st.slider(
            "Sale price range",
            min_value=float(p_min),
            max_value=float(p_max),
            value=(float(p_min), float(p_max)),
            step=max((p_max - p_min) / 200.0, 0.01),
        )

        r_min, r_max = _series_min_max(df["rating"], 0.0, 5.0)
        rating_range = st.slider(
            "Rating range",
            min_value=0.0,
            max_value=5.0,
            value=(float(max(0.0, r_min)), float(min(5.0, r_max))),
            step=0.1,
        )

        rev_min, rev_max = _series_min_max(df["reviews"], 0.0, 1.0)
        reviews_range = st.slider(
            "Reviews range",
            min_value=int(max(0, rev_min)),
            max_value=int(max(1, rev_max)),
            value=(int(max(0, rev_min)), int(max(1, rev_max))),
            step=max(int((rev_max - rev_min) / 200) or 1, 1),
        )

        d_min, d_max = _series_min_max(df["discount_pct"], 0.0, 1.0)
        discount_range = st.slider(
            "Discount range",
            min_value=0.0,
            max_value=1.0,
            value=(float(max(0.0, d_min)), float(min(1.0, d_max))),
            step=0.01,
            format="%.0f%%",
        )

        discounted_only = st.toggle("Discounted only", value=False)

        search = st.text_input("Search (name/description)", value="").strip()

    with st.sidebar.expander("About the metrics", expanded=False):
        st.caption(
            "This dataset does not include units sold. "
            "**Total Catalog Value** is the sum of listed sale prices for the current selection."
        )

    return FilterState(
        brands=selected_brands,
        families=selected_families,
        genders=selected_genders,
        price_range=price_range,
        rating_range=rating_range,
        reviews_range=reviews_range,
        discount_range=discount_range,
        search=search,
        discounted_only=discounted_only,
    )


def apply_filters(df: pd.DataFrame, state: FilterState) -> pd.DataFrame:
    out = df.copy()

    out = out[out["brand"].isin(state.brands)]
    out = out[out["product_family"].isin(state.families)]
    out = out[out["gender_inferred"].isin(state.genders)]

    out = out[out["sale_price"].between(state.price_range[0], state.price_range[1], inclusive="both")]
    out = out[out["rating"].between(state.rating_range[0], state.rating_range[1], inclusive="both")]
    out = out[out["reviews"].between(state.reviews_range[0], state.reviews_range[1], inclusive="both")]
    out = out[out["discount_pct"].between(state.discount_range[0], state.discount_range[1], inclusive="both")]

    if state.discounted_only:
        out = out[out["is_discounted"]]

    if state.search:
        q = state.search.lower()
        hay = (out["product_name"].fillna("") + " " + out["description"].fillna("")).str.lower()
        out = out[hay.str.contains(q, na=False)]

    return out

