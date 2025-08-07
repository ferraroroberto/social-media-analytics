"""FastAPI application for content performance prediction."""

import os
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, date
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn
from dotenv import load_dotenv

# Import project modules
from ..data.database import SupabaseClient
from ..data.data_loader import DataLoader
from ..features.feature_engineering import FeatureEngineer
from ..models.prediction_models import create_model, BaseModel as MLBaseModel

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Content Performance Predictor API",
    description="API for predicting social media content performance",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for API requests/responses
class PredictionRequest(BaseModel):
    """Request model for content performance prediction."""
    platform: str = Field(..., description="Platform (linkedin, instagram, twitter, substack, threads)")
    content_type: str = Field(..., description="Content type (no_video, video)")
    date: date = Field(..., description="Date for prediction")
    features: Optional[Dict[str, Any]] = Field(default=None, description="Additional features")
    
class PredictionResponse(BaseModel):
    """Response model for content performance prediction."""
    platform: str
    content_type: str
    date: date
    predicted_engagement: float
    confidence: Optional[float] = None
    model_used: str
    features_used: List[str]
    
class CaptionAnalysisRequest(BaseModel):
    """Request model for caption analysis."""
    caption: str = Field(..., description="Caption text to analyze")
    platform: Optional[str] = Field(default=None, description="Platform for analysis")
    
class CaptionAnalysisResponse(BaseModel):
    """Response model for caption analysis."""
    caption: str
    sentiment_score: float
    word_count: int
    char_count: int
    hashtag_count: int
    mention_count: int
    url_count: int
    complexity_score: float
    recommendations: List[str]
    
class BestTimeRequest(BaseModel):
    """Request model for best posting time analysis."""
    platform: str = Field(..., description="Platform to analyze")
    content_type: Optional[str] = Field(default=None, description="Content type")
    days_ahead: int = Field(default=7, description="Number of days to analyze")
    
class BestTimeResponse(BaseModel):
    """Response model for best posting time analysis."""
    platform: str
    content_type: Optional[str]
    best_times: List[Dict[str, Any]]
    analysis_period: str
    
class PlatformTrendRequest(BaseModel):
    """Request model for platform trend analysis."""
    platform: str = Field(..., description="Platform to analyze")
    days_back: int = Field(default=30, description="Number of days to look back")
    
class PlatformTrendResponse(BaseModel):
    """Response model for platform trend analysis."""
    platform: str
    trend_data: List[Dict[str, Any]]
    summary_stats: Dict[str, float]
    recommendations: List[str]

# Global variables for model and data
supabase_client: Optional[SupabaseClient] = None
data_loader: Optional[DataLoader] = None
feature_engineer: Optional[FeatureEngineer] = None
models: Dict[str, MLBaseModel] = {}

def get_supabase_client() -> SupabaseClient:
    """Get Supabase client instance."""
    global supabase_client
    if supabase_client is None:
        supabase_client = SupabaseClient()
    return supabase_client

def get_data_loader() -> DataLoader:
    """Get data loader instance."""
    global data_loader
    if data_loader is None:
        supabase_client = get_supabase_client()
        data_loader = DataLoader(supabase_client)
    return data_loader

def get_feature_engineer() -> FeatureEngineer:
    """Get feature engineer instance."""
    global feature_engineer
    if feature_engineer is None:
        feature_engineer = FeatureEngineer()
    return feature_engineer

def load_models() -> Dict[str, MLBaseModel]:
    """Load trained models."""
    global models
    if not models:
        model_path = os.getenv("MODEL_PATH", "./models")
        if os.path.exists(model_path):
            for model_file in os.listdir(model_path):
                if model_file.endswith(".joblib"):
                    try:
                        model_name = model_file.replace(".joblib", "")
                        model_path_full = os.path.join(model_path, model_file)
                        model = MLBaseModel.load_model(model_path_full)
                        models[model_name] = model
                        logger.info(f"Loaded model: {model_name}")
                    except Exception as e:
                        logger.error(f"Error loading model {model_file}: {e}")
    return models

@app.on_event("startup")
async def startup_event():
    """Initialize app on startup."""
    logger.info("Starting Content Performance Predictor API")
    load_models()

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Content Performance Predictor API",
        "version": "0.1.0",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "models_loaded": len(load_models())
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_engagement(request: PredictionRequest):
    """
    Predict content engagement for a specific platform and content type.
    """
    try:
        # Load models
        models = load_models()
        if not models:
            raise HTTPException(status_code=500, detail="No trained models available")
        
        # Get data loader and feature engineer
        data_loader = get_data_loader()
        feature_engineer = get_feature_engineer()
        
        # Load recent data for feature engineering
        end_date = request.date
        start_date = (end_date - pd.Timedelta(days=60)).strftime('%Y-%m-%d')
        
        posts_df, profile_df = data_loader.load_consolidated_data(start_date, end_date.strftime('%Y-%m-%d'))
        
        if posts_df.empty:
            raise HTTPException(status_code=400, detail="No data available for the specified date range")
        
        # Preprocess data
        posts_df = data_loader.preprocess_posts_data(posts_df)
        profile_df = data_loader.preprocess_profile_data(profile_df)
        
        # Create features
        posts_df = feature_engineer.create_temporal_features(posts_df)
        posts_df = feature_engineer.create_engagement_features(posts_df)
        posts_df = feature_engineer.create_cross_platform_features(posts_df)
        
        # Create prediction input
        prediction_date = pd.to_datetime(request.date)
        prediction_data = posts_df[posts_df['date'] == prediction_date].copy()
        
        if prediction_data.empty:
            # Create synthetic data for prediction
            prediction_data = pd.DataFrame([{
                'date': prediction_date,
                'day_of_week': prediction_date.dayofweek,
                'month': prediction_date.month,
                'quarter': prediction_date.quarter,
                'year': prediction_date.year,
                'is_weekend': 1 if prediction_date.dayofweek in [5, 6] else 0
            }])
        
        # Add additional features if provided
        if request.features:
            for key, value in request.features.items():
                prediction_data[key] = value
        
        # Select appropriate model
        model_key = f"{request.platform}_{request.content_type}"
        if model_key not in models:
            # Use a default model
            model_key = list(models.keys())[0]
        
        model = models[model_key]
        
        # Make prediction
        prediction = model.predict(prediction_data)
        predicted_engagement = float(prediction[0]) if len(prediction) > 0 else 0.0
        
        return PredictionResponse(
            platform=request.platform,
            content_type=request.content_type,
            date=request.date,
            predicted_engagement=predicted_engagement,
            model_used=model_key,
            features_used=model.feature_columns or []
        )
        
    except Exception as e:
        logger.error(f"Error in prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze-caption", response_model=CaptionAnalysisResponse)
async def analyze_caption(request: CaptionAnalysisRequest):
    """
    Analyze caption text and provide insights.
    """
    try:
        feature_engineer = get_feature_engineer()
        
        # Create a dummy DataFrame for analysis
        df = pd.DataFrame([{'caption': request.caption}])
        
        # Apply content features
        df = feature_engineer.create_content_features(df, 'caption')
        
        # Extract features
        caption_length = df['caption_length'].iloc[0] if 'caption_length' in df.columns else len(request.caption)
        word_count = df['caption_word_count'].iloc[0] if 'caption_word_count' in df.columns else len(request.caption.split())
        char_count = df['caption_char_count'].iloc[0] if 'caption_char_count' in df.columns else len(request.caption.replace(' ', ''))
        sentiment_score = df['caption_sentiment'].iloc[0] if 'caption_sentiment' in df.columns else 0.0
        hashtag_count = df['caption_hashtag_count'].iloc[0] if 'caption_hashtag_count' in df.columns else request.caption.count('#')
        mention_count = df['caption_mention_count'].iloc[0] if 'caption_mention_count' in df.columns else request.caption.count('@')
        url_count = df['caption_url_count'].iloc[0] if 'caption_url_count' in df.columns else request.caption.count('http')
        
        # Calculate complexity score
        complexity_score = word_count / max(caption_length, 1) if caption_length > 0 else 0
        
        # Generate recommendations
        recommendations = []
        if word_count < 10:
            recommendations.append("Consider adding more detail to your caption")
        elif word_count > 100:
            recommendations.append("Consider shortening your caption for better engagement")
        
        if hashtag_count < 3:
            recommendations.append("Consider adding relevant hashtags")
        elif hashtag_count > 10:
            recommendations.append("Consider reducing hashtag count for better readability")
        
        if sentiment_score < -0.1:
            recommendations.append("Consider using more positive language")
        elif sentiment_score > 0.5:
            recommendations.append("Great positive sentiment! Keep it up")
        
        if mention_count == 0:
            recommendations.append("Consider mentioning relevant accounts or people")
        
        return CaptionAnalysisResponse(
            caption=request.caption,
            sentiment_score=sentiment_score,
            word_count=word_count,
            char_count=char_count,
            hashtag_count=hashtag_count,
            mention_count=mention_count,
            url_count=url_count,
            complexity_score=complexity_score,
            recommendations=recommendations
        )
        
    except Exception as e:
        logger.error(f"Error in caption analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/best-times", response_model=BestTimeResponse)
async def get_best_posting_times(request: BestTimeRequest):
    """
    Get best posting times for a platform.
    """
    try:
        data_loader = get_data_loader()
        feature_engineer = get_feature_engineer()
        
        # Load recent data
        end_date = datetime.now().date()
        start_date = (end_date - pd.Timedelta(days=request.days_ahead + 30)).strftime('%Y-%m-%d')
        
        posts_df, profile_df = data_loader.load_consolidated_data(start_date, end_date.strftime('%Y-%m-%d'))
        
        if posts_df.empty:
            raise HTTPException(status_code=400, detail="No data available for analysis")
        
        # Preprocess data
        posts_df = data_loader.preprocess_posts_data(posts_df)
        posts_df = feature_engineer.create_temporal_features(posts_df)
        
        # Analyze engagement by day of week
        engagement_col = f'engagement_{request.platform}_{request.content_type}' if request.content_type else f'engagement_{request.platform}_no_video'
        
        if engagement_col not in posts_df.columns:
            # Use any available engagement column
            engagement_cols = [col for col in posts_df.columns if 'engagement' in col and request.platform in col]
            if not engagement_cols:
                raise HTTPException(status_code=400, detail=f"No engagement data available for {request.platform}")
            engagement_col = engagement_cols[0]
        
        # Group by day of week and calculate average engagement
        day_engagement = posts_df.groupby('day_of_week')[engagement_col].mean().reset_index()
        day_engagement = day_engagement.sort_values(engagement_col, ascending=False)
        
        # Map day numbers to names
        day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        day_engagement['day_name'] = day_engagement['day_of_week'].apply(lambda x: day_names[x])
        
        best_times = []
        for _, row in day_engagement.head(3).iterrows():
            best_times.append({
                'day': row['day_name'],
                'day_of_week': int(row['day_of_week']),
                'avg_engagement': float(row[engagement_col]),
                'recommendation': f"Best day: {row['day_name']}"
            })
        
        return BestTimeResponse(
            platform=request.platform,
            content_type=request.content_type,
            best_times=best_times,
            analysis_period=f"Last {request.days_ahead + 30} days"
        )
        
    except Exception as e:
        logger.error(f"Error in best times analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/platform-trends", response_model=PlatformTrendResponse)
async def get_platform_trends(request: PlatformTrendRequest):
    """
    Get platform performance trends.
    """
    try:
        data_loader = get_data_loader()
        feature_engineer = get_feature_engineer()
        
        # Load recent data
        end_date = datetime.now().date()
        start_date = (end_date - pd.Timedelta(days=request.days_back)).strftime('%Y-%m-%d')
        
        posts_df, profile_df = data_loader.load_consolidated_data(start_date, end_date.strftime('%Y-%m-%d'))
        
        if posts_df.empty:
            raise HTTPException(status_code=400, detail="No data available for analysis")
        
        # Preprocess data
        posts_df = data_loader.preprocess_posts_data(posts_df)
        posts_df = feature_engineer.create_temporal_features(posts_df)
        posts_df = feature_engineer.create_engagement_features(posts_df)
        
        # Get engagement columns for the platform
        engagement_cols = [col for col in posts_df.columns if 'engagement' in col and request.platform in col]
        
        if not engagement_cols:
            raise HTTPException(status_code=400, detail=f"No engagement data available for {request.platform}")
        
        # Calculate trends
        trend_data = []
        for col in engagement_cols:
            col_data = posts_df[['date', col]].dropna()
            if not col_data.empty:
                trend_data.append({
                    'metric': col,
                    'current_avg': float(col_data[col].mean()),
                    'trend': float(col_data[col].pct_change().mean()),
                    'volatility': float(col_data[col].std())
                })
        
        # Calculate summary stats
        all_engagement = posts_df[engagement_cols].mean().mean()
        summary_stats = {
            'avg_engagement': float(all_engagement),
            'total_posts': len(posts_df),
            'days_analyzed': request.days_back
        }
        
        # Generate recommendations
        recommendations = []
        if all_engagement > 100:
            recommendations.append("Great engagement! Keep up the quality content")
        elif all_engagement < 50:
            recommendations.append("Consider experimenting with different content types")
        
        return PlatformTrendResponse(
            platform=request.platform,
            trend_data=trend_data,
            summary_stats=summary_stats,
            recommendations=recommendations
        )
        
    except Exception as e:
        logger.error(f"Error in platform trends analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/models")
async def list_models():
    """List available trained models."""
    models = load_models()
    return {
        "models": list(models.keys()),
        "total_models": len(models)
    }

@app.get("/models/{model_name}")
async def get_model_info(model_name: str):
    """Get information about a specific model."""
    models = load_models()
    if model_name not in models:
        raise HTTPException(status_code=404, detail="Model not found")
    
    model = models[model_name]
    return model.get_model_summary()

if __name__ == "__main__":
    uvicorn.run(
        "src.api.prediction_api:app",
        host=os.getenv("API_HOST", "0.0.0.0"),
        port=int(os.getenv("API_PORT", 8000)),
        reload=os.getenv("API_RELOAD", "true").lower() == "true"
    )