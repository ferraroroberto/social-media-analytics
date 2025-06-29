# Content Performance Predictor

## Overview

A portfolio-ready, end-to-end data science project to analyze and predict social media content performance. Built for content creators with 2.5+ years of data, this project leverages open-source tools and best practices to deliver actionable insights, predictive analytics, and interactive dashboards.

- **Data Source:** Notion database (synced to Supabase/PostgreSQL)
- **Platforms Tracked:** LinkedIn, Instagram, Twitter, Threads
- **Key Use Cases:** Performance prediction, repost analysis, cross-platform comparison, caption effectiveness, trend analysis

---

## Project Structure

```
content-performance-predictor/
├── .github/workflows/         # CI/CD pipelines
├── data/                      # Raw, processed, and external data
├── notebooks/                 # EDA, feature engineering, modeling
├── src/                       # Source code (data, features, models, API)
├── mlflow/                    # MLflow artifacts
├── dash_app/                  # Dashboard app (Dash)
├── tests/                     # Unit tests
├── config/                    # Config files
├── requirements.txt
├── setup.py
├── Dockerfile
├── docker-compose.yml
├── .env.example
├── README.md
└── Makefile
```

---

## Tech Stack

- **Data Storage:** Supabase (PostgreSQL), DuckDB, Parquet
- **Python Libraries:** pandas, polars, scikit-learn, xgboost, lightgbm, catboost, torch, transformers, spacy, textblob, sentence-transformers, feature-engine, prophet, statsmodels
- **Experiment Tracking:** MLflow, Optuna
- **Visualization:** plotly, seaborn, matplotlib
- **Dashboard:** Dash, dash-bootstrap-components
- **API:** FastAPI, Uvicorn, Pydantic
- **DevOps:** GitHub Actions, Docker, pre-commit, pytest, black
- **Documentation:** MkDocs, mkdocs-material

---

## Workflow

1. **Data Pipeline & EDA**
   - Connect to Supabase, extract and clean data
   - Explore data in Jupyter notebooks

2. **Feature Engineering**
   - Temporal, content, engagement, cross-platform, visual, and network features

3. **Model Development**
   - Baseline (moving average, linear regression)
   - ML models (Random Forest, XGBoost, LightGBM, CatBoost)
   - Advanced (deep learning, time series, ensemble, multi-output)

4. **Evaluation & Experiment Tracking**
   - MLflow for tracking, Optuna for hyperparameter tuning

5. **Dashboard & API**
   - Dash app for interactive analytics
   - FastAPI for prediction and insights endpoints

6. **Deployment**
   - Dockerized app, CI/CD via GitHub Actions
   - Optional: Streamlit Cloud, Vercel, Hugging Face Spaces

---

## Key Features

- **Performance Prediction:** Forecast post engagement before publishing
- **Repost Analysis:** Predict repost performance based on trends
- **Cross-Platform Comparison:** Analyze content effectiveness across platforms
- **Caption Analyzer:** NLP-driven insights on caption style and engagement
- **Trend Analysis:** Identify and visualize performance patterns

---

## Deliverables

- **Interactive Dashboard:** Post predictions, optimal posting times, caption analyzer, cross-platform comparison
- **REST API:** Predict engagement, analyze captions, get best times, platform trends
- **Jupyter Notebooks:** EDA, feature engineering, model evaluation, business insights
- **Documentation:** Technical docs, model cards, API docs (Swagger/OpenAPI)

---

## Cost & Hosting

- **Free Tiers:** Supabase, GitHub, Streamlit Cloud, MLflow (self-hosted), Hugging Face
- **Optional Paid:** Weights & Biases, higher Supabase tier, Vercel Pro

---

## Success Metrics

- **Model:** MAE/RMSE, R² > 0.7, cross-platform accuracy
- **Business:** Increased engagement, time saved, improved consistency
- **Portfolio:** Clean, documented codebase, live demo, clear business value

---

## Getting Started

1. Clone the repo and install dependencies:
   ```
   git clone https://github.com/yourusername/content-performance-predictor.git
   cd content-performance-predictor
   pip install -r requirements.txt
   ```
2. Configure `.env` with your Supabase credentials.
3. Run notebooks for EDA and modeling.
4. Launch the dashboard:
   ```
   cd dash_app
   python app.py
   ```
5. Start the API:
   ```
   uvicorn src.api.prediction_api:app --reload
   ```

---

## 📝 License and contact

This project is free software for personal use from Roberto Ferraro 😇

https://www.linkedin.com/in/ferraroroberto/

Built with ❤️ for automated social media analytics and reporting