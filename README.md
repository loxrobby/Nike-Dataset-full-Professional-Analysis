# Nike Streamlit EDA Dashboard

An interview-ready, production-style **Exploratory Data Analysis (EDA) dashboard** built with **Streamlit** + **Plotly**.

This project analyzes Nike product catalog data (pricing, discounting, ratings, reviews) and turns it into a clean, interactive analytics experience with business questions and data-driven narrative insights.

## What’s in the dashboard
- **Robust data pipeline**: cached loading, type coercion, missing-value handling, feature engineering
- **Professional filtering**: sidebar controls that update the entire app
- **KPI header**: catalog value, products, pricing, top family, discount share deltas
- **Business Questions & Insights (Q&A)**: interactive charts + dynamic, computed takeaways
- **Exports**: download the filtered dataset as CSV

> Note: the dataset does not include units sold. The app uses **Total Catalog Value** as `sum(sale_price)` for the current selection.

## Project structure
```
Nike/
  app.py
  Dataset/
    nike_shoes_sales.csv
  src/
    nike_eda_dashboard/
      __init__.py
      data.py
      filters.py
      charts.py
      insights.py
      style.py
  requirements.txt
  .gitignore
```

## Run locally
From the project root:

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Customize
- **Dataset path**: update `DATASET_PATH` in `app.py` if you move/rename the CSV.
- **Theme/colors**: tweak `src/nike_eda_dashboard/style.py`.
- **Business questions**: add/edit questions in `app.py` and corresponding insight logic in `insights.py`.

