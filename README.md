# Content Performance Predictor

A portfolio-ready, end-to-end data science project to analyze and predict social media content performance. Built for content creators with 2.5+ years of data, this project leverages open-source tools and best practices to deliver actionable insights, predictive analytics, and interactive dashboards.

## 🎯 Overview

This project provides:
- **Performance Prediction:** Forecast post engagement before publishing
- **Repost Analysis:** Predict repost performance based on trends
- **Cross-Platform Comparison:** Analyze content effectiveness across platforms
- **Caption Analyzer:** NLP-driven insights on caption style and engagement
- **Trend Analysis:** Identify and visualize performance patterns

## 🏗️ Project Structure

```
content-performance-predictor/
├── .github/workflows/         # CI/CD pipelines
├── data/                      # Raw, processed, and external data
│   ├── raw/                   # Raw data files
│   ├── processed/             # Processed data files
│   └── external/              # External data sources
├── notebooks/                 # EDA, feature engineering, modeling
├── src/                       # Source code
│   ├── data/                  # Data loading and processing
│   │   ├── database.py        # Supabase client
│   │   └── data_loader.py     # Data loading utilities
│   ├── features/              # Feature engineering
│   │   └── feature_engineering.py
│   ├── models/                # ML models
│   │   ├── base_model.py      # Base model class
│   │   └── prediction_models.py # Specific models
│   ├── api/                   # FastAPI application
│   │   └── prediction_api.py  # REST API
│   ├── utils/                 # Utilities
│   │   └── config.py          # Configuration management
│   └── cli.py                 # Command-line interface
├── dash_app/                  # Dashboard app (Dash)
│   └── app.py                 # Dash application
├── tests/                     # Unit tests
├── config/                    # Configuration files
│   └── config.yaml            # Main configuration
├── mlflow/                    # MLflow artifacts
├── models/                    # Trained models
├── logs/                      # Application logs
├── requirements.txt           # Python dependencies
├── setup.py                   # Package setup
├── Dockerfile                 # Docker configuration
├── docker-compose.yml         # Docker Compose setup
├── Makefile                   # Development tasks
├── .env.example               # Environment variables template
└── README.md                  # This file
```

## 🛠️ Tech Stack

### Data & Storage
- **Database:** Supabase (PostgreSQL)
- **Data Processing:** pandas, polars, numpy
- **File Storage:** DuckDB, Parquet

### Machine Learning
- **Models:** scikit-learn, XGBoost, LightGBM, CatBoost
- **Deep Learning:** PyTorch, Transformers
- **NLP:** spaCy, TextBlob, sentence-transformers
- **Feature Engineering:** feature-engine
- **Time Series:** Prophet, statsmodels

### Experiment Tracking & Optimization
- **Experiment Tracking:** MLflow
- **Hyperparameter Tuning:** Optuna

### Visualization & Dashboard
- **Visualization:** plotly, seaborn, matplotlib
- **Dashboard:** Dash, dash-bootstrap-components

### API & Web Framework
- **API:** FastAPI, Uvicorn
- **Validation:** Pydantic
- **HTTP Client:** requests

### DevOps & Development
- **Containerization:** Docker, Docker Compose
- **CI/CD:** GitHub Actions
- **Code Quality:** black, flake8, pre-commit
- **Testing:** pytest
- **Documentation:** MkDocs, mkdocs-material

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Docker (optional)
- Supabase account and project

### 1. Clone and Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/content-performance-predictor.git
cd content-performance-predictor

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -e .
```

### 2. Environment Configuration

```bash
# Copy environment template
cp .env.example .env

# Edit .env with your Supabase credentials
nano .env
```

Required environment variables:
```bash
# Supabase Configuration
SUPABASE_URL=your_supabase_url_here
SUPABASE_KEY=your_supabase_anon_key_here
SUPABASE_SERVICE_ROLE_KEY=your_supabase_service_role_key_here

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_RELOAD=true

# Dashboard Configuration
DASHBOARD_HOST=0.0.0.0
DASHBOARD_PORT=8050
DASHBOARD_DEBUG=true
```

### 3. Run the Application

#### Option A: Using Make (Recommended)

```bash
# Setup development environment
make dev-setup

# Run API server
make run-api

# Run dashboard (in another terminal)
make run-dashboard

# Run MLflow tracking server (in another terminal)
make run-mlflow
```

#### Option B: Using Docker

```bash
# Build and run with Docker Compose
docker-compose up --build
```

#### Option C: Manual Execution

```bash
# Run API server
uvicorn src.api.prediction_api:app --host 0.0.0.0 --port 8000 --reload

# Run dashboard
python dash_app/app.py

# Run MLflow server
mlflow server --host 0.0.0.0 --port 5000
```

## 📊 Usage

### API Endpoints

The API provides the following endpoints:

- `GET /` - API information
- `GET /health` - Health check
- `POST /predict` - Predict content engagement
- `POST /analyze-caption` - Analyze caption text
- `POST /best-times` - Get best posting times
- `POST /platform-trends` - Get platform trends
- `GET /models` - List available models

#### Example API Usage

```bash
# Predict engagement
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "platform": "linkedin",
    "content_type": "no_video",
    "date": "2023-12-01"
  }'

# Analyze caption
curl -X POST "http://localhost:8000/analyze-caption" \
  -H "Content-Type: application/json" \
  -d '{
    "caption": "Check out this amazing content! #socialmedia #content"
  }'
```

### Dashboard

Access the dashboard at `http://localhost:8050` to:
- Predict content performance
- Analyze captions
- View best posting times
- Explore platform trends
- Compare cross-platform performance

### CLI Usage

```bash
# Train a model
python -m src.cli train --platform linkedin --content-type no_video --model-type xgboost

# Make predictions
python -m src.cli predict --platform linkedin --content-type no_video --date 2023-12-01

# Analyze data
python -m src.cli analyze --start-date 2023-01-01 --end-date 2023-12-31
```

## 🔧 Development

### Project Setup

```bash
# Install development dependencies
pip install -r requirements.txt
pip install -e .[dev]

# Setup pre-commit hooks
pre-commit install

# Run tests
make test

# Run linting
make lint

# Format code
make format
```

### Adding New Models

1. Create a new model class in `src/models/prediction_models.py`
2. Inherit from `BaseModel`
3. Implement the `_create_model` method
4. Add to the `create_model` factory function

### Adding New Features

1. Add feature engineering logic in `src/features/feature_engineering.py`
2. Update the `FeatureEngineer` class
3. Add tests in `tests/test_feature_engineering.py`

### Database Schema

The project expects the following Supabase tables:

#### `posts` table
- `date` (date) - Primary key
- `post_id_linkedin_no_video` (text)
- `num_likes_linkedin_no_video` (integer)
- `num_comments_linkedin_no_video` (integer)
- `num_reshares_linkedin_no_video` (integer)
- ... (similar columns for other platforms and content types)

#### `profile` table
- `date` (date) - Primary key
- `num_followers_linkedin` (integer)
- `num_followers_instagram` (integer)
- `num_followers_twitter` (integer)
- `num_followers_substack` (integer)
- `num_followers_threads` (integer)

## 📈 Model Performance

The project includes multiple model types:

- **Random Forest:** Good baseline performance, interpretable
- **XGBoost:** High performance, handles non-linear relationships
- **LightGBM:** Fast training, good for large datasets
- **CatBoost:** Handles categorical features well
- **Linear Regression:** Simple baseline, interpretable
- **Ensemble:** Combines multiple models for better performance

## 🚀 Deployment

### Docker Deployment

```bash
# Build image
docker build -t content-performance-predictor .

# Run container
docker run -p 8000:8000 -p 8050:8050 content-performance-predictor
```

### Production Deployment

1. Set up environment variables for production
2. Use a production WSGI server (e.g., Gunicorn)
3. Set up reverse proxy (e.g., Nginx)
4. Configure SSL certificates
5. Set up monitoring and logging

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Roberto Ferraro**
- LinkedIn: [https://www.linkedin.com/in/ferraroroberto/](https://www.linkedin.com/in/ferraroroberto/)
- Email: roberto.ferraro@example.com

## 🙏 Acknowledgments

- Built with ❤️ for automated social media analytics and reporting
- Inspired by the need for data-driven content strategy
- Thanks to the open-source community for the amazing tools

## 📞 Support

For support and questions:
- Open an issue on GitHub
- Contact: roberto.ferraro@example.com
- Documentation: [docs/](docs/)

---

**Built with ❤️ for automated social media analytics and reporting**