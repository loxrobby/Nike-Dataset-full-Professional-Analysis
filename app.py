from __future__ import annotations

import base64
from pathlib import Path

import pandas as pd
import plotly.express as px
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


def render_gif(path: str, *, width_px: int) -> None:
    p = Path(path)
    if not p.exists():
        return

    data = base64.b64encode(p.read_bytes()).decode("ascii")
    st.markdown(
        f"""
        <img
          src="data:image/gif;base64,{data}"
          style="width:{width_px}px; height:auto; display:block; margin-left:auto;"
          alt=""
        />
        """,
        unsafe_allow_html=True,
    )


def inject_ui_css() -> None:
    st.markdown(
        """
        <style>
          /* Reduce Streamlit's default top padding */
          [data-testid="stAppViewBlockContainer"]{
            padding-top: 2rem;
            padding-bottom: 2rem;
          }

          /* Metric polish */
          div[data-testid="stMetric"] {
            background: rgba(18, 18, 18, 0.85);
            border: 1px solid #333;
            padding: 16px 16px;
            border-radius: 8px;
            transition: 0.3s;
          }
          div[data-testid="stMetric"]:hover {
            transform: translateY(-3px);
            box-shadow: 0 4px 12px rgba(255, 165, 0, 0.25);
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
          .nike-footer-wrap{
            width: 100%;
            margin-top: 34px;
            padding-top: 18px;
            border-top: 1px solid rgba(255, 255, 255, 0.10);
            display: flex;
            justify-content: center;
          }
          .nike-footer{
            width: min(1100px, 92vw);
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 10px;
            color: rgba(255, 255, 255, 0.72);
            font-size: 0.9rem;
            letter-spacing: 0.2px;
            padding: 10px 12px;
            background: rgba(18, 18, 18, 0.55);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 10px;
          }
          .nike-footer strong{
            color: rgba(255, 255, 255, 0.88);
            font-weight: 600;
          }

          /* Executive summary (Power BI narrative card) */
          .pbi-card{
            background: rgba(18, 18, 18, 0.75);
            border: 1px solid rgba(255, 255, 255, 0.10);
            border-radius: 10px;
            padding: 14px 16px;
          }
          .pbi-card .pbi-title{
            display:flex;
            align-items:center;
            justify-content:space-between;
            gap: 12px;
            margin-bottom: 10px;
          }
          .pbi-card .pbi-title .label{
            font-weight: 700;
            letter-spacing: 0.2px;
            color: rgba(255,255,255,0.92);
          }
          .pbi-card .pbi-title .sub{
            font-size: 0.85rem;
            color: rgba(255,255,255,0.60);
            white-space: nowrap;
          }
          .pbi-card .pbi-body{
            color: rgba(255,255,255,0.80);
            font-size: 0.95rem;
            line-height: 1.45;
          }
          .pbi-card strong{
            color: rgba(255,255,255,0.95);
            font-weight: 700;
          }
          .pbi-card ul{
            margin: 0.35rem 0 0 1.1rem;
          }
          .pbi-card li{
            margin: 0.2rem 0;
          }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header(rows_total: int, rows_filtered: int, dq) -> None:
    top = st.columns([3.2, 0.9, 0.9], vertical_alignment="center")
    with top[0]:
        st.title("Nike Product Pricing & Engagement — EDA Dashboard")
        st.caption(
            "A professional EDA view of Nike product catalog pricing, discounting, ratings, and reviews. "
            "No units-sold are provided, so \"Total Catalog Value\" is the sum of sale prices."
        )
    with top[1]:
        render_gif(GIF_SHOE, width_px=180)
    with top[2]:
        render_gif(GIF_RUNNER, width_px=180)


def build_powerbi_sidebar(df_full: pd.DataFrame) -> dict:
    with st.sidebar:
        st.markdown("## ⚙️ Dashboard Controls")

        families = sorted([f for f in df_full["product_family"].dropna().unique().tolist() if str(f).strip()])
        min_price = float(df_full["sale_price"].min()) if not df_full["sale_price"].dropna().empty else 0.0
        max_price = float(df_full["sale_price"].max()) if not df_full["sale_price"].dropna().empty else 0.0
        step = max(0.01, (max_price - min_price) / 200) if max_price > min_price else 0.01

        with st.expander("📦 Product scope", expanded=True):
            st.caption("Tip: leave empty to include all families.")
            selected_families = st.multiselect(
                "Product Family",
                options=families,
                default=[],
                placeholder="All product families",
            )

        with st.expander("💲 Pricing", expanded=True):
            price_min, price_max = st.slider(
                "Sale Price Range",
                min_value=min_price,
                max_value=max_price,
                value=(min_price, max_price),
                step=step,
            )

        with st.expander("🏷️ Discounting", expanded=False):
            discounted_only = st.toggle("Show Discounted Items Only", value=False)

    return {
        "families": selected_families,
        "price_min": float(price_min),
        "price_max": float(price_max),
        "discounted_only": bool(discounted_only),
    }


def apply_powerbi_filters(df_full: pd.DataFrame, state: dict) -> pd.DataFrame:
    df = df_full.copy()

    families = state.get("families") or []
    # Empty selection means "All families"
    if len(families) > 0:
        df = df[df["product_family"].isin(families)]

    pmin = float(state.get("price_min", float("-inf")))
    pmax = float(state.get("price_max", float("inf")))
    df = df[df["sale_price"].between(pmin, pmax, inclusive="both")]

    if state.get("discounted_only"):
        df = df[df["is_discounted"] == True]  # noqa: E712

    return df


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

    k1.metric(
        "Total Catalog Value",
        format_currency(catalog_value),
        help="Sum of all sale prices in the current filtered view.",
    )
    k2.metric(
        "Total Products",
        f"{total_products:,}",
        help="Count of unique products (by product_id) in the current filtered view.",
    )
    k3.metric(
        "Avg Sale Price",
        format_currency(avg_price),
        help="Average sale price across products in the current filtered view.",
    )
    k4.metric(
        "Top Product Family",
        str(top_family),
        help="Product family with the highest total catalog value (sum of sale prices) in the current filtered view.",
    )

    # PowerBI-style metric: number + circular donut indicator
    with k5.container():
        top_row = st.columns([2.2, 1], vertical_alignment="center")
        with top_row[0]:
            st.metric(
                "Discounted Share",
                f"{discounted_share:.0%}",
                delta=f"{delta:+.0%}",
                help="Share of products marked discounted in the filtered view; delta compares to the full catalog.",
            )
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

    state = build_powerbi_sidebar(df_full)
    df = apply_powerbi_filters(df_full, state)

    render_header(rows_total=len(df_full), rows_filtered=len(df), dq=dq)

    if df.empty:
        st.error("No rows match your filters. Try widening the ranges or resetting filters.")
        st.stop()

    st.divider()
    render_kpis(df, df_full=df_full)

    st.divider()
    st.subheader("Executive summary")

    top_family = str(df["product_family"].value_counts().idxmax()) if len(df) else "N/A"
    overall_avg = float(df_full["sale_price"].mean()) if not df_full["sale_price"].dropna().empty else 0.0
    overall_median = float(df_full["sale_price"].median()) if not df_full["sale_price"].dropna().empty else 0.0

    view_value = float(df["sale_price"].sum()) if len(df) else 0.0
    view_avg = float(df["sale_price"].mean()) if not df["sale_price"].dropna().empty else 0.0
    view_median = float(df["sale_price"].median()) if not df["sale_price"].dropna().empty else 0.0

    top_family_avg = (
        float(df.loc[df["product_family"] == top_family, "sale_price"].mean())
        if top_family != "N/A" and not df.loc[df["product_family"] == top_family, "sale_price"].dropna().empty
        else 0.0
    )
    price_gap = top_family_avg - overall_avg
    gap_dir = "above" if price_gap >= 0 else "below"

    discounted_pct = float(df["is_discounted"].mean()) if len(df) else 0.0
    avg_discount = (
        float(df.loc[df["is_discounted"] == True, "discount_pct"].mean())  # noqa: E712
        if len(df) and not df.loc[df["is_discounted"] == True, "discount_pct"].dropna().empty  # noqa: E712
        else 0.0
    )

    top3 = (
        df.groupby("product_family", as_index=False)
        .agg(catalog_value=("sale_price", "sum"))
        .sort_values("catalog_value", ascending=False)
        .head(3)
    )
    top3_str = ", ".join([f"{r['product_family']} ({format_currency(float(r['catalog_value']))})" for _, r in top3.iterrows()]) or "N/A"

    summary = (
        f"""
        <div class="pbi-card">
          <div class="pbi-title">
            <div class="label">Executive summary</div>
            <div class="sub">Current filters</div>
          </div>
          <div class="pbi-body">
            <ul>
              <li>
                Catalog value <strong>{format_currency(view_value)}</strong> across
                <strong>{df['product_id'].nunique():,}</strong> products; median price
                <strong>{format_currency(view_median)}</strong> (avg <strong>{format_currency(view_avg)}</strong>).
              </li>
              <li>
                Top family <strong>{top_family}</strong> at <strong>{format_currency(top_family_avg)}</strong> avg price
                (<strong>{format_currency(abs(price_gap))}</strong> {gap_dir} overall avg <strong>{format_currency(overall_avg)}</strong>;
                overall median <strong>{format_currency(overall_median)}</strong>).
              </li>
              <li>
                Discounting: <strong>{discounted_pct:.1%}</strong> of items discounted; average discount
                <strong>{avg_discount:.1%}</strong>. Top 3 families by value: <strong>{top3_str}</strong>.
              </li>
            </ul>
          </div>
        </div>
        """
    )
    st.markdown(summary, unsafe_allow_html=True)

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

    st.subheader("Deep dives (Power BI-style)")
    d1, d2 = st.columns(2)

    top5_families = (
        df["product_family"].value_counts().head(5).index.tolist()
        if "product_family" in df.columns and not df.empty
        else []
    )
    box_df = df[df["product_family"].isin(top5_families)].copy() if top5_families else df.copy()

    with d1:
        fig_box = px.box(
            box_df,
            x="product_family",
            y="sale_price",
            color="product_family",
            template="plotly_dark",
            title="Sale Price distribution (Top 5 families)",
            labels={"product_family": "Product Family", "sale_price": "Sale Price ($)"},
        )
        fig_box.update_xaxes(showgrid=False)
        fig_box.update_yaxes(showgrid=False)
        fig_box.update_traces(
            hovertemplate="Family=%{x}<br>Sale price=$%{y:,.2f}<extra></extra>"
        )
        st.plotly_chart(fig_box, use_container_width=True)

    with d2:
        fig_scatter = px.scatter(
            df,
            x="sale_price",
            y="discount_pct",
            color="product_family",
            template="plotly_dark",
            title="Sale Price vs Discount %",
            labels={"sale_price": "Sale Price ($)", "discount_pct": "Discount %", "product_family": "Product Family"},
        )
        fig_scatter.update_xaxes(showgrid=False)
        fig_scatter.update_yaxes(showgrid=False, tickformat=".0%")
        fig_scatter.update_traces(
            marker=dict(size=7, opacity=0.75),
            hovertemplate="Family=%{legendgroup}<br>Sale price=$%{x:,.2f}<br>Discount=%{y:.0%}<extra></extra>",
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

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
        pass
    with end[1]:
        st.markdown(
            """
            <div class="nike-footer-wrap">
              <div class="nike-footer">
                <span>©</span>
                <span><strong>TheDevTeam</strong></span>
                <span>— All rights reserved</span>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )



if __name__ == "__main__":
    main()

