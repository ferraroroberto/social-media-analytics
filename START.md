# Content Performance Predictor - LLM Agent Recreation Guide

This document provides all the necessary information for an LLM agent to recreate the Content Performance Predictor project from scratch.

## 🎯 Project Overview

**Content Performance Predictor** is a portfolio-ready, end-to-end data science project to analyze and predict social media content performance. Built for content creators with 2.5+ years of data, this project leverages open-source tools and best practices to deliver actionable insights, predictive analytics, and interactive dashboards.

### Key Features
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
│   │   ├── __init__.py
│   │   ├── database.py        # Supabase client
│   │   └── data_loader.py     # Data loading utilities
│   ├── features/              # Feature engineering
│   │   ├── __init__.py
│   │   └── feature_engineering.py
│   ├── models/                # ML models
│   │   ├── __init__.py
│   │   ├── base_model.py      # Base model class
│   │   └── prediction_models.py # Specific models
│   ├── api/                   # FastAPI application
│   │   ├── __init__.py
│   │   └── prediction_api.py  # REST API
│   ├── utils/                 # Utilities
│   │   ├── __init__.py
│   │   └── config.py          # Configuration management
│   ├── __init__.py
│   └── cli.py                 # Command-line interface
├── dash_app/                  # Dashboard app (Dash)
│   └── app.py                 # Dash application
├── tests/                     # Unit tests
│   ├── __init__.py
│   └── test_data_loader.py
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
├── .gitignore                 # Git ignore file
├── README.md                  # Project documentation
└── START.md                   # This file
```

## 🛠️ Tech Stack Requirements

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

## 📊 Database Schema

### Supabase Tables Structure

#### `posts` table (Consolidated)
- **Primary Key:** `date` (date)
- **Columns:**
  - `post_id_linkedin_no_video` (text)
  - `posted_at_linkedin_no_video` (date)
  - `num_likes_linkedin_no_video` (integer)
  - `num_comments_linkedin_no_video` (integer)
  - `num_reshares_linkedin_no_video` (integer)
  - `post_id_linkedin_video` (text)
  - `posted_at_linkedin_video` (date)
  - `num_likes_linkedin_video` (integer)
  - `num_comments_linkedin_video` (integer)
  - `num_reshares_linkedin_video` (integer)
  - Similar columns for: `instagram`, `twitter`, `substack`, `threads`

#### `profile` table (Consolidated)
- **Primary Key:** `date` (date)
- **Columns:**
  - `num_followers_linkedin` (integer)
  - `num_followers_instagram` (integer)
  - `num_followers_twitter` (integer)
  - `num_followers_substack` (integer)
  - `num_followers_threads` (integer)

#### Platform-specific tables (Raw data)
- `linkedin_posts`, `instagram_posts`, `twitter_posts`, `substack_posts`, `threads_posts`
- `linkedin_profile`, `instagram_profile`, `twitter_profile`, `substack_profile`, `threads_profile`

#### Notion-synced tables
- `notion_posts`, `notion_editorial`, `notion_articles`, `notion_books`, etc.

## 🔧 Implementation Steps

### Step 1: Project Setup

1. **Create project structure**
2. **Initialize git repository**
3. **Create virtual environment**
4. **Set up basic files**

### Step 2: Dependencies

Create `requirements.txt` with all necessary packages:
- Data processing: pandas, polars, numpy, scipy
- ML: scikit-learn, xgboost, lightgbm, catboost, torch, transformers
- NLP: spacy, textblob, nltk, sentence-transformers
- Visualization: plotly, seaborn, matplotlib, dash, dash-bootstrap-components
- API: fastapi, uvicorn, pydantic, requests
- Database: supabase, psycopg2-binary, duckdb, pyarrow
- Experiment tracking: mlflow, optuna
- Development: pytest, black, flake8, pre-commit
- Documentation: mkdocs, mkdocs-material
- Utilities: python-dotenv, pyyaml, tqdm, joblib

### Step 3: Core Modules

#### 3.1 Data Layer (`src/data/`)

**database.py** - Supabase client:
- `SupabaseClient` class
- Methods: `get_posts_data()`, `get_profile_data()`, `get_notion_data()`, `get_platform_data()`
- Error handling and logging

**data_loader.py** - Data processing:
- `DataLoader` class
- Methods: `load_consolidated_data()`, `preprocess_posts_data()`, `preprocess_profile_data()`
- Platform-specific dataset creation
- Cross-platform dataset creation

#### 3.2 Feature Engineering (`src/features/`)

**feature_engineering.py** - Feature creation:
- `FeatureEngineer` class
- Temporal features: day_of_week, month, quarter, year, cyclical features
- Lag features: rolling statistics, trend analysis
- Engagement features: rates, ratios, interactions
- Cross-platform features: comparisons, aggregations
- Content features: text analysis, sentiment, complexity
- Scaling and normalization

#### 3.3 Models (`src/models/`)

**base_model.py** - Abstract base class:
- `BaseModel` abstract class
- Methods: `fit()`, `predict()`, `evaluate()`, `save_model()`, `load_model()`
- MLflow integration
- Feature importance extraction

**prediction_models.py** - Specific models:
- `RandomForestModel`, `XGBoostModel`, `LightGBMModel`, `CatBoostModel`
- `LinearRegressionModel`, `RidgeModel`, `SVRModel`, `MLPModel`
- `EnsembleModel` for combining multiple models
- Factory function `create_model()`

#### 3.4 API (`src/api/`)

**prediction_api.py** - FastAPI application:
- FastAPI app with CORS middleware
- Endpoints:
  - `GET /` - API information
  - `GET /health` - Health check
  - `POST /predict` - Predict content engagement
  - `POST /analyze-caption` - Analyze caption text
  - `POST /best-times` - Get best posting times
  - `POST /platform-trends` - Get platform trends
  - `GET /models` - List available models
- Pydantic models for request/response validation
- Error handling and logging

#### 3.5 Dashboard (`dash_app/`)

**app.py** - Dash application:
- Multi-tab dashboard with Bootstrap styling
- Tabs: Performance Prediction, Caption Analyzer, Best Posting Times, Platform Trends, Cross-Platform Analysis
- Interactive components: dropdowns, date pickers, buttons
- Real-time API integration
- Visualization with Plotly

#### 3.6 Utilities (`src/utils/`)

**config.py** - Configuration management:
- `Config` class for YAML and environment variable management
- Methods for accessing different config sections
- Environment variable override functionality

#### 3.7 CLI (`src/cli.py`)

**cli.py** - Command-line interface:
- Argument parsing with subcommands
- Commands: `train`, `predict`, `analyze`
- Integration with all core modules
- Logging and error handling

### Step 4: Configuration

**config.yaml** - Main configuration:
- Data configuration (platforms, content types)
- Model configuration (types, default parameters)
- Feature engineering configuration
- API and dashboard configuration
- MLflow configuration
- Paths configuration

**.env.example** - Environment variables template:
- Supabase credentials
- API configuration
- Dashboard configuration
- Model paths
- External API keys

### Step 5: Testing

**tests/test_data_loader.py** - Unit tests:
- Test data loading functionality
- Test preprocessing methods
- Test dataset creation
- Mock Supabase client

### Step 6: Deployment

**Dockerfile** - Container configuration:
- Python 3.11 slim base image
- System dependencies installation
- Python dependencies installation
- Non-root user creation
- Health check configuration

**docker-compose.yml** - Multi-service setup:
- API service
- Dashboard service
- MLflow service
- PostgreSQL service (optional)

**Makefile** - Development tasks:
- Install dependencies
- Run tests
- Lint and format code
- Run services
- Docker operations

### Step 7: Documentation

**README.md** - Comprehensive documentation:
- Project overview
- Installation instructions
- Usage examples
- API documentation
- Development guide
- Deployment instructions

## 🎯 Key Implementation Details

### Data Processing Pipeline

1. **Data Loading**: Connect to Supabase, extract posts and profile data
2. **Preprocessing**: Clean data, handle missing values, create derived features
3. **Feature Engineering**: Create temporal, engagement, cross-platform features
4. **Model Training**: Train multiple model types, evaluate performance
5. **Prediction**: Make predictions for new content
6. **Analysis**: Provide insights and recommendations

### Model Architecture

- **Base Model**: Abstract class with common functionality
- **Specific Models**: Implementation of different ML algorithms
- **Ensemble Model**: Combines multiple models for better performance
- **Model Persistence**: Save/load models with joblib
- **MLflow Integration**: Track experiments and model versions

### API Design

- **RESTful Design**: Standard HTTP methods and status codes
- **Request/Response Validation**: Pydantic models for type safety
- **Error Handling**: Comprehensive error responses
- **Documentation**: Auto-generated OpenAPI/Swagger docs
- **CORS Support**: Cross-origin resource sharing

### Dashboard Features

- **Interactive UI**: Modern, responsive design with Bootstrap
- **Real-time Data**: Live API integration
- **Visualizations**: Rich charts and graphs with Plotly
- **Multi-tab Interface**: Organized analysis sections
- **User-friendly**: Intuitive navigation and controls

## 🚀 Deployment Strategy

### Development Environment

1. **Local Setup**: Virtual environment, local dependencies
2. **Docker Development**: Docker Compose for local development
3. **Testing**: Unit tests, integration tests
4. **Code Quality**: Linting, formatting, pre-commit hooks

### Production Environment

1. **Containerization**: Docker images for all services
2. **Orchestration**: Docker Compose or Kubernetes
3. **Environment Variables**: Secure configuration management
4. **Monitoring**: Health checks, logging, metrics
5. **Scaling**: Load balancing, horizontal scaling

## 📊 Success Metrics

### Technical Metrics
- **Model Performance**: MAE/RMSE < 0.3, R² > 0.7
- **API Response Time**: < 500ms for predictions
- **Dashboard Load Time**: < 3 seconds
- **Test Coverage**: > 80%

### Business Metrics
- **Prediction Accuracy**: > 70% for engagement predictions
- **User Adoption**: Dashboard usage and API calls
- **Time Savings**: Reduced manual analysis time
- **Content Performance**: Improved engagement rates

## 🔄 Development Workflow

1. **Feature Development**: Create feature branch, implement, test
2. **Code Review**: Pull request, code review, approval
3. **Testing**: Automated tests, manual testing
4. **Deployment**: Staging environment, production deployment
5. **Monitoring**: Performance monitoring, error tracking

## 📝 Notes for LLM Agent

### Critical Implementation Points

1. **Database Schema**: Follow the exact Supabase schema provided
2. **Error Handling**: Implement comprehensive error handling throughout
3. **Logging**: Use structured logging for debugging and monitoring
4. **Configuration**: Make all settings configurable via environment variables
5. **Testing**: Write tests for all core functionality
6. **Documentation**: Maintain comprehensive documentation
7. **Security**: Follow security best practices (no hardcoded secrets)
8. **Performance**: Optimize for speed and scalability
9. **User Experience**: Focus on intuitive and responsive UI
10. **Maintainability**: Write clean, modular, and well-documented code

### Common Pitfalls to Avoid

1. **Hardcoded Values**: Use configuration files and environment variables
2. **Poor Error Handling**: Implement proper exception handling
3. **Missing Tests**: Write comprehensive test coverage
4. **Inconsistent Code Style**: Use linting and formatting tools
5. **Poor Documentation**: Maintain up-to-date documentation
6. **Security Issues**: Follow security best practices
7. **Performance Issues**: Optimize database queries and API responses
8. **User Experience**: Design intuitive and responsive interfaces

### Quality Assurance

1. **Code Review**: All code must be reviewed before merging
2. **Testing**: Automated tests must pass before deployment
3. **Documentation**: All features must be documented
4. **Performance**: Performance benchmarks must be met
5. **Security**: Security review for all new features

This document provides all the necessary information for an LLM agent to recreate the Content Performance Predictor project from scratch. Follow the implementation steps carefully and ensure all requirements are met.