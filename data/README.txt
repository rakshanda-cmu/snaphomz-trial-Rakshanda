README
Project: Real Estate Dataset Analysis
Setup
Requirements

Python 3.8+
Install dependencies:
pip install pandas numpy matplotlib seaborn scikit-learn rank_bm25 sentence-transformers
Data file: data/listings_sample.csv (must be present).
Notebook Usage

Open notebooks/trial.ipynb in VS Code or Jupyter.
Run cells top-to-bottom for data loading, EDA, feature engineering, modeling, and LLM features.
Key Decisions
Data Cleaning

Numeric columns: impute missing values with median, add missing indicators.
Winsorize numeric features (clip outliers at 1st/99th percentiles).
Drop non-essential columns (address, media_urls, remarks) after feature engineering.
Categorical Encoding

One-hot encode columns with <20 unique values.
Label encode high-cardinality categoricals.
Modeling

Baseline regression: LinearRegression and RandomForestRegressor (target: price_per_sqft).
Feature importance printed for RandomForest.
EDA

Visualizations: countplot by city/state, boxplot by city/state, heatmap of price_per_sqft by beds/baths, scatterplot sqft vs price.
LLM Features

RAG Q&A: BM25 keyword retriever and optional embedding retriever over remarks column.
Auto-summary: deterministic template for listing summary; optional local LLM prompt (llama.cpp).
How to Run
Data Analysis

Run all notebook cells for full workflow.
Visualizations and summary tables are generated inline.
Model Training

Baseline regression runs automatically; metrics printed in output.
RAG Q&A

BM25 retriever: build index and query with natural language (see notebook cell for usage).
Embedding retriever: install sentence-transformers for semantic search.
Auto-summary

Use summarize_listing_row(row) for deterministic summaries.
For LLM-based summaries, generate prompt and run with local LLM binary (see notebook for example).
Limits & Assumptions
Data

Assumes clean CSV with expected columns; missing values handled as described.
Outlier handling is basic (winsorization).
Modeling

No hyperparameter tuning; models are for baseline only.
Only numeric and encoded categorical features used.
LLM Features

BM25 is keyword-based; embedding retriever needs extra install and RAM.
Auto-summary is rule-based unless local LLM is set up.
General

No web UI; all interaction via notebook.
No production-grade error handling or persistence.
For local LLM, user must install and configure model binaries separately.
Contact: For questions, open an issue or email the project maintainer. - Data file: data/listings_sample.csv (must be present).

Notebook Usage
Open notebooks/trial.ipynb in VS Code or Jupyter.
Run cells top-to-bottom for data loading, EDA, feature engineering, modeling, and LLM features.
Key Decisions
Data Cleaning

Numeric columns: impute missing values with median, add missing indicators.
Winsorize numeric features (clip outliers at 1st/99th percentiles).
Drop non-essential columns (address, media_urls, remarks) after feature engineering.
Categorical Encoding

One-hot encode columns with <20 unique values.
Label encode high-cardinality categoricals.
Modeling

Baseline regression: LinearRegression and RandomForestRegressor (target: price_per_sqft).
Feature importance printed for RandomForest.
EDA

Visualizations: countplot by city/state, boxplot by city/state, heatmap of price_per_sqft by beds/baths, scatterplot sqft vs price.
LLM Features

RAG Q&A: BM25 keyword retriever and optional embedding retriever over remarks column.
Auto-summary: deterministic template for listing summary; optional local LLM prompt (llama.cpp).
How to Run
Data Analysis

Run all notebook cells for full workflow.
Visualizations and summary tables are generated inline.
Model Training

Baseline regression runs automatically; metrics printed in output.
RAG Q&A

BM25 retriever: build index and query with natural language (see notebook cell for usage).
Embedding retriever: install sentence-transformers for semantic search.
Auto-summary

Use summarize_listing_row(row) for deterministic summaries.
For LLM-based summaries, generate prompt and run with local LLM binary (see notebook for example).
Limits & Assumptions
Data

Assumes clean CSV with expected columns; missing values handled as described.
Outlier handling is basic (winsorization).
Modeling

No hyperparameter tuning; models are for baseline only.
Only numeric and encoded categorical features used.
LLM Features

BM25 is keyword-based; embedding retriever needs extra install and RAM.
Auto-summary is rule-based unless local LLM is set up.
General

No web UI; all interaction via notebook.
No production-grade error handling or persistence.
For local LLM, user must install and configure model binaries separately.
Contact: For questions, open an issue or email the project maintainer.

GPT-4.1 • 1x
