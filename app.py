from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from src.nike_eda_dashboard.charts import (
    discount_by_gender_bar,
    discount_vs_price_scatter,
    family_sentiment_bubble,
    mini_donut,
    price_by_band_box,
    price_discount_by_gender,
    price_distribution_hist,
    top_families_bar,
)
from src.nike_eda_dashboard.data import load_data
from src.nike_eda_dashboard.filters import apply_filters, build_sidebar_filters
from src.nike_eda_dashboard.insights import (
    insight_discount_price_relationship,
    insight_gender_pricing,
    insight_price_distribution,
    insight_sentiment_engagement,
    insight_top_family,
)
from src.nike_eda_dashboard.style import format_currency


DATASET_PATH = str(Path(__file__).parent / "Dataset" / "nike_shoes_sales.csv")
ASSETS_DIR = Path(__file__).parent / "assets"
GIF_SHOE = str(ASSETS_DIR / "nike_shoe.gif")
GIF_RUNNER = str(ASSETS_DIR / "nike_runner.gif")

def inject_ui_css() -> None:
    st.markdown(
        """
        <style>
          /* Metric polish */
          div[data-testid="stMetric"] {
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid rgba(255, 255, 255, 0.08);
            padding: 16px 16px;
            border-radius: 14px;
          }
          div[data-testid="stMetricLabel"] > div {
            font-size: 0.9rem;
            opacity: 0.85;
          }
          div[data-testid="stMetricValue"] > div {
            font-weight: 700;
            letter-spacing: 0.2px;
          }
          /* Section headers spacing */
          h2, h3 {
            letter-spacing: 0.2px;
          }
          /* Footer */
          .nike-footer {
            margin-top: 40px;
            padding-top: 18px;
            border-top: 1px solid rgba(255, 255, 255, 0.08);
            opacity: 0.75;
            text-align: center;
            font-size: 0.9rem;
          }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header(rows_total: int, rows_filtered: int, dq) -> None:
    top = st.columns([3.2, 1], vertical_alignment="center")
    with top[0]:
        st.title("Nike Product Pricing & Engagement — EDA Dashboard")
        st.caption(
            "A professional EDA view of Nike product catalog pricing, discounting, ratings, and reviews. "
            "No units-sold are provided, so \"Total Catalog Value\" is the sum of sale prices."
        )
    with top[1]:
        if Path(GIF_SHOE).exists():
            st.image(GIF_SHOE, width=180)

    st.subheader("Data quality & coverage")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows in view", f"{rows_filtered:,}", delta=f"{rows_filtered - rows_total:+,} vs full data")
    c2.metric("Missing sale price", f"{dq.missing_sale_price_pct:.1%}")
    c3.metric("Missing images", f"{dq.missing_images_pct:.1%}")
    c4.metric("Missing descriptions", f"{dq.missing_description_pct:.1%}")


def render_kpis(df: pd.DataFrame, df_full: pd.DataFrame) -> None:
    catalog_value = float(df["sale_price"].sum())
    total_products = int(df["product_id"].nunique())
    avg_price = float(df["sale_price"].mean()) if not df["sale_price"].dropna().empty else 0.0

    fam = (
        df.groupby("product_family", as_index=False)
        .agg(catalog_value=("sale_price", "sum"))
        .sort_values("catalog_value", ascending=False)
    )
    top_family = fam.iloc[0]["product_family"] if not fam.empty else "N/A"

    discounted_share = float(df["is_discounted"].mean()) if len(df) else 0.0
    overall_discounted_share = float(df_full["is_discounted"].mean()) if len(df_full) else 0.0
    delta = discounted_share - overall_discounted_share

    st.subheader("Key metrics")
    k1, k2, k3, k4, k5 = st.columns(5)

    k1.metric("Total Catalog Value", format_currency(catalog_value))
    k2.metric("Total Products", f"{total_products:,}")
    k3.metric("Avg Sale Price", format_currency(avg_price))
    k4.metric("Top Product Family", str(top_family))

    # PowerBI-style metric: number + circular donut indicator
    with k5.container():
        top_row = st.columns([2.2, 1], vertical_alignment="center")
        with top_row[0]:
            st.metric("Discounted Share", f"{discounted_share:.0%}", delta=f"{delta:+.0%}")
        with top_row[1]:
            st.plotly_chart(
                mini_donut(discounted_share),
                use_container_width=False,
                config={"displayModeBar": False},
            )


def render_top_products_table(df: pd.DataFrame) -> None:
    st.subheader("Top products (quick scan)")
    cols = [
        "product_name",
        "product_id",
        "brand",
        "product_family",
        "gender_inferred",
        "sale_price",
        "listing_price",
        "discount_pct",
        "rating",
        "reviews",
    ]
    view = df[cols].copy()
    view = view.sort_values(["rating", "reviews", "sale_price"], ascending=[False, False, False]).head(50)
    st.dataframe(
        view,
        use_container_width=True,
        hide_index=True,
        column_config={
            "sale_price": st.column_config.NumberColumn(format="%.2f"),
            "listing_price": st.column_config.NumberColumn(format="%.2f"),
            "discount_pct": st.column_config.ProgressColumn(
                "Discount",
                format="%.0f%%",
                min_value=0.0,
                max_value=1.0,
            ),
        },
    )


def render_download(df: pd.DataFrame) -> None:
    st.download_button(
        "Download filtered data (CSV)",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="nike_filtered.csv",
        mime="text/csv",
        use_container_width=True,
    )


def main() -> None:
    st.set_page_config(page_title="Nike EDA Dashboard", page_icon="👟", layout="wide")
    inject_ui_css()

    try:
        df_full, dq = load_data(DATASET_PATH)
    except Exception as exc:
        st.error(f"Failed to load dataset. Details: {exc}")
        st.stop()

    state = build_sidebar_filters(df_full)
    df = apply_filters(df_full, state)

    render_header(rows_total=len(df_full), rows_filtered=len(df), dq=dq)

    if df.empty:
        st.error("No rows match your filters. Try widening the ranges or resetting filters.")
        st.stop()

    st.divider()
    render_kpis(df, df_full=df_full)

    st.divider()
    st.subheader("Executive summary (auto-generated)")
    st.info(
        "This selection is dominated by "
        f"**{df['product_family'].value_counts().idxmax()}** by product count, "
        f"with a discounted share of **{df['is_discounted'].mean():.0%}**. "
        "Use the Business Questions section below for driver-level insights."
    )

    st.divider()
    render_top_products_table(df)
    render_download(df)

    st.divider()
    st.header("Business Questions & Insights")

    qcols = st.columns([1, 1, 1])
    metric_choice = qcols[0].radio(
        "Family ranking metric",
        options=[("Catalog value", "catalog_value"), ("Product count", "count")],
        index=0,
        horizontal=True,
    )[1]
    top_n = qcols[1].slider("Top N families", min_value=5, max_value=25, value=10, step=1)
    min_products = qcols[2].slider("Min products per family (sentiment)", 1, 20, 5, 1)

    st.subheader("Q1: Which product families drive the most catalog value and products?")
    st.plotly_chart(top_families_bar(df, metric=metric_choice, top_n=top_n), use_container_width=True)
    st.success(insight_top_family(df))

    st.subheader("Q2: How are prices distributed, and where do most products sit?")
    c1, c2 = st.columns(2)
    c1.plotly_chart(price_distribution_hist(df), use_container_width=True)
    c2.plotly_chart(price_by_band_box(df), use_container_width=True)
    st.info(insight_price_distribution(df))

    st.subheader("Q3: Do discounts meaningfully shift price positioning?")
    st.plotly_chart(discount_vs_price_scatter(df), use_container_width=True)
    st.info(insight_discount_price_relationship(df))

    st.subheader("Q4: Which families show the strongest sentiment and engagement?")
    st.plotly_chart(family_sentiment_bubble(df, min_products=min_products), use_container_width=True)
    st.success(insight_sentiment_engagement(df))

    st.subheader("Q5: How do inferred segments compare on price and discount?")
    c3, c4 = st.columns(2)
    c3.plotly_chart(price_discount_by_gender(df), use_container_width=True)
    c4.plotly_chart(discount_by_gender_bar(df), use_container_width=True)
    st.info(insight_gender_pricing(df))

    st.divider()
    end = st.columns([1, 3], vertical_alignment="center")
    with end[0]:
        if Path(GIF_RUNNER).exists():
            st.image(GIF_RUNNER, width=180)
    with end[1]:
        st.markdown('<div class="nike-footer">All rights reserved to TheDevTeam</div>', unsafe_allow_html=True)



if __name__ == "__main__":
    main()

