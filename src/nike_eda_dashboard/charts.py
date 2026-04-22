from __future__ import annotations

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

from .style import PALETTE


def apply_nike_layout(fig: go.Figure, title: str | None = None) -> go.Figure:
    fig.update_layout(
        title=title,
        template="plotly_white",
        paper_bgcolor=PALETTE.white,
        plot_bgcolor=PALETTE.white,
        font=dict(color=PALETTE.black),
        legend=dict(title=None, font=dict(color=PALETTE.black)),
        hoverlabel=dict(font=dict(color=PALETTE.black)),
        margin=dict(l=30, r=20, t=60, b=40),
    )
    fig.update_xaxes(
        showgrid=True,
        gridcolor="rgba(0,0,0,0.06)",
        zeroline=False,
        tickfont=dict(color=PALETTE.black),
        title_font=dict(color=PALETTE.black),
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor="rgba(0,0,0,0.06)",
        zeroline=False,
        tickfont=dict(color=PALETTE.black),
        title_font=dict(color=PALETTE.black),
    )
    fig.update_layout(
        coloraxis_colorbar=dict(
            tickfont=dict(color=PALETTE.black),
            title=dict(font=dict(color=PALETTE.black)),
        )
    )
    return fig


def mini_donut(percent: float, color: str = PALETTE.accent) -> go.Figure:
    p = max(0.0, min(1.0, float(percent)))
    fig = go.Figure(
        data=[
            go.Pie(
                values=[p, 1 - p],
                hole=0.72,
                sort=False,
                direction="clockwise",
                marker=dict(colors=[color, "rgba(255,255,255,0.08)"], line=dict(color="rgba(0,0,0,0)", width=0)),
                textinfo="none",
                hoverinfo="skip",
                showlegend=False,
            )
        ]
    )
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        height=86,
    )
    fig.add_annotation(
        text=f"{p:.0%}",
        x=0.5,
        y=0.5,
        showarrow=False,
        font=dict(size=16, color=PALETTE.black),
    )
    return fig


def top_families_bar(df: pd.DataFrame, metric: str, top_n: int = 10) -> go.Figure:
    if metric == "catalog_value":
        agg = (
            df.groupby("product_family", as_index=False)
            .agg(catalog_value=("sale_price", "sum"), products=("product_id", "nunique"))
            .sort_values("catalog_value", ascending=False)
            .head(top_n)
        )
        fig = px.bar(
            agg,
            x="catalog_value",
            y="product_family",
            orientation="h",
            color_discrete_sequence=[PALETTE.accent],
            labels={"catalog_value": "Total catalog value", "product_family": "Product family"},
        )
        fig.update_traces(hovertemplate="Family=%{y}<br>Catalog value=%{x:,.2f}<extra></extra>")
        return apply_nike_layout(fig, title=f"Top {top_n} product families by catalog value")

    agg = (
        df.groupby("product_family", as_index=False)
        .agg(products=("product_id", "nunique"))
        .sort_values("products", ascending=False)
        .head(top_n)
    )
    fig = px.bar(
        agg,
        x="products",
        y="product_family",
        orientation="h",
        color_discrete_sequence=[PALETTE.accent],
        labels={"products": "Unique products", "product_family": "Product family"},
    )
    fig.update_traces(hovertemplate="Family=%{y}<br>Products=%{x:,}<extra></extra>")
    return apply_nike_layout(fig, title=f"Top {top_n} product families by product count")


def price_distribution_hist(df: pd.DataFrame, nbins: int = 40) -> go.Figure:
    fig = px.histogram(
        df,
        x="sale_price",
        nbins=nbins,
        color_discrete_sequence=[PALETTE.black],
        labels={"sale_price": "Sale price"},
    )
    fig.update_traces(hovertemplate="Sale price=%{x:,.2f}<br>Count=%{y:,}<extra></extra>")
    return apply_nike_layout(fig, title="Sale price distribution")


def price_by_band_box(df: pd.DataFrame) -> go.Figure:
    order = ["Low", "Mid", "High", "Premium", "All"]
    fig = px.box(
        df,
        x="price_band",
        y="sale_price",
        category_orders={"price_band": [o for o in order if o in df["price_band"].unique().tolist()]},
        color="price_band",
        color_discrete_sequence=[PALETTE.light_gray, PALETTE.gray, PALETTE.accent, PALETTE.black, PALETTE.accent_2],
        labels={"sale_price": "Sale price", "price_band": "Price band"},
    )
    fig.update_traces(hovertemplate="Band=%{x}<br>Sale price=%{y:,.2f}<extra></extra>")
    return apply_nike_layout(fig, title="Sale price by price band")


def discount_vs_price_scatter(df: pd.DataFrame, max_points: int = 4000) -> go.Figure:
    plot_df = df
    if len(plot_df) > max_points:
        plot_df = plot_df.sample(max_points, random_state=7)

    fig = px.scatter(
        plot_df,
        x="discount_pct",
        y="sale_price",
        color="product_family",
        hover_data={"product_name": True, "rating": True, "reviews": True},
        labels={"discount_pct": "Discount", "sale_price": "Sale price", "product_family": "Product family"},
    )
    fig.update_traces(
        marker=dict(size=7, opacity=0.7),
        hovertemplate=(
            "Discount=%{x:.0%}<br>Sale price=%{y:,.2f}<br>"
            "Rating=%{customdata[0]:.1f}<br>Reviews=%{customdata[1]:,}<extra></extra>"
        ),
    )
    fig.update_xaxes(tickformat=".0%")
    return apply_nike_layout(fig, title="Discount vs sale price (sampled if large)")


def family_sentiment_bubble(df: pd.DataFrame, min_products: int = 5) -> go.Figure:
    agg = (
        df.groupby("product_family", as_index=False)
        .agg(
            products=("product_id", "nunique"),
            avg_rating=("rating", "mean"),
            avg_reviews=("reviews", "mean"),
            catalog_value=("sale_price", "sum"),
        )
    )
    agg = agg[agg["products"] >= min_products].copy()
    fig = px.scatter(
        agg,
        x="avg_rating",
        y="avg_reviews",
        size="products",
        color="catalog_value",
        color_continuous_scale=["#f2f2f2", PALETTE.accent],
        hover_name="product_family",
        labels={
            "avg_rating": "Avg rating",
            "avg_reviews": "Avg reviews",
            "products": "Products (bubble size)",
            "catalog_value": "Catalog value (color)",
        },
    )
    fig.update_traces(
        hovertemplate=(
            "%{hovertext}<br>Avg rating=%{x:.2f}<br>Avg reviews=%{y:,.1f}"
            "<br>Products=%{marker.size:,}<extra></extra>"
        )
    )
    return apply_nike_layout(fig, title=f"Family sentiment & engagement (min {min_products} products)")


def price_discount_by_gender(df: pd.DataFrame) -> go.Figure:
    fig = px.box(
        df,
        x="gender_inferred",
        y="sale_price",
        color="gender_inferred",
        color_discrete_sequence=[PALETTE.black, PALETTE.accent, PALETTE.gray, PALETTE.accent_2],
        labels={"gender_inferred": "Inferred segment", "sale_price": "Sale price"},
    )
    fig.update_traces(hovertemplate="Segment=%{x}<br>Sale price=%{y:,.2f}<extra></extra>")
    return apply_nike_layout(fig, title="Sale price by inferred gender segment")


def discount_by_gender_bar(df: pd.DataFrame) -> go.Figure:
    agg = df.groupby("gender_inferred", as_index=False).agg(avg_discount=("discount_pct", "mean"))
    fig = px.bar(
        agg,
        x="gender_inferred",
        y="avg_discount",
        color_discrete_sequence=[PALETTE.accent],
        labels={"gender_inferred": "Inferred segment", "avg_discount": "Avg discount"},
    )
    fig.update_yaxes(tickformat=".0%")
    fig.update_traces(hovertemplate="Segment=%{x}<br>Avg discount=%{y:.1%}<extra></extra>")
    return apply_nike_layout(fig, title="Average discount by inferred segment")

