"""Prediction models for content performance."""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
import logging
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import lightgbm as lgb
from catboost import CatBoostRegressor
from .base_model import BaseModel

logger = logging.getLogger(__name__)

# Suppress LightGBM logging completely
class SilentLogger:
    def info(self, msg):
        pass
    def warning(self, msg):
        pass

lgb.register_logger(SilentLogger())

class RandomForestModel(BaseModel):
    """Random Forest model for content performance prediction."""
    
    def __init__(self, target_column: str, feature_columns: List[str] = None, **kwargs):
        """
        Initialize Random Forest model.
        
        Args:
            target_column: Target column for prediction
            feature_columns: List of feature columns
            **kwargs: Additional parameters for RandomForestRegressor
        """
        super().__init__("random_forest", target_column, feature_columns)
        self.model_params = kwargs
    
    def _create_model(self) -> RandomForestRegressor:
        """Create Random Forest model."""
        default_params = {
            'n_estimators': 100,
            'max_depth': 10,
            'random_state': 42,
            'n_jobs': -1
        }
        default_params.update(self.model_params)
        return RandomForestRegressor(**default_params)


class XGBoostModel(BaseModel):
    """XGBoost model for content performance prediction."""
    
    def __init__(self, target_column: str, feature_columns: List[str] = None, **kwargs):
        """
        Initialize XGBoost model.
        
        Args:
            target_column: Target column for prediction
            feature_columns: List of feature columns
            **kwargs: Additional parameters for XGBRegressor
        """
        super().__init__("xgboost", target_column, feature_columns)
        self.model_params = kwargs
    
    def _create_model(self) -> XGBRegressor:
        """Create XGBoost model."""
        default_params = {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'random_state': 42,
            'n_jobs': -1
        }
        default_params.update(self.model_params)
        return XGBRegressor(**default_params)


class LightGBMModel(BaseModel):
    """LightGBM model for content performance prediction."""
    
    def __init__(self, target_column: str, feature_columns: List[str] = None, **kwargs):
        """
        Initialize LightGBM model.
        
        Args:
            target_column: Target column for prediction
            feature_columns: List of feature columns
            **kwargs: Additional parameters for LGBMRegressor
        """
        super().__init__("lightgbm", target_column, feature_columns)
        self.model_params = kwargs
    
    def _create_model(self) -> LGBMRegressor:
        """Create LightGBM model."""
        default_params = {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'random_state': 42,
            'n_jobs': -1
        }
        default_params.update(self.model_params)
        return LGBMRegressor(**default_params)


class CatBoostModel(BaseModel):
    """CatBoost model for content performance prediction."""
    
    def __init__(self, target_column: str, feature_columns: List[str] = None, **kwargs):
        """
        Initialize CatBoost model.
        
        Args:
            target_column: Target column for prediction
            feature_columns: List of feature columns
            **kwargs: Additional parameters for CatBoostRegressor
        """
        super().__init__("catboost", target_column, feature_columns)
        self.model_params = kwargs
    
    def _create_model(self) -> CatBoostRegressor:
        """Create CatBoost model."""
        default_params = {
            'iterations': 100,
            'depth': 6,
            'learning_rate': 0.1,
            'random_state': 42,
            'verbose': False
        }
        default_params.update(self.model_params)
        return CatBoostRegressor(**default_params)


class LinearRegressionModel(BaseModel):
    """Linear Regression model for content performance prediction."""
    
    def __init__(self, target_column: str, feature_columns: List[str] = None, **kwargs):
        """
        Initialize Linear Regression model.
        
        Args:
            target_column: Target column for prediction
            feature_columns: List of feature columns
            **kwargs: Additional parameters for LinearRegression
        """
        super().__init__("linear_regression", target_column, feature_columns)
        self.model_params = kwargs
    
    def _create_model(self) -> LinearRegression:
        """Create Linear Regression model."""
        default_params = {
            'n_jobs': -1
        }
        default_params.update(self.model_params)
        return LinearRegression(**default_params)


class RidgeModel(BaseModel):
    """Ridge Regression model for content performance prediction."""
    
    def __init__(self, target_column: str, feature_columns: List[str] = None, **kwargs):
        """
        Initialize Ridge Regression model.
        
        Args:
            target_column: Target column for prediction
            feature_columns: List of feature columns
            **kwargs: Additional parameters for Ridge
        """
        super().__init__("ridge", target_column, feature_columns)
        self.model_params = kwargs
    
    def _create_model(self) -> Ridge:
        """Create Ridge Regression model."""
        default_params = {
            'alpha': 1.0,
            'random_state': 42
        }
        default_params.update(self.model_params)
        return Ridge(**default_params)


class SVRModel(BaseModel):
    """Support Vector Regression model for content performance prediction."""
    
    def __init__(self, target_column: str, feature_columns: List[str] = None, **kwargs):
        """
        Initialize SVR model.
        
        Args:
            target_column: Target column for prediction
            feature_columns: List of feature columns
            **kwargs: Additional parameters for SVR
        """
        super().__init__("svr", target_column, feature_columns)
        self.model_params = kwargs
    
    def _create_model(self) -> SVR:
        """Create SVR model."""
        default_params = {
            'kernel': 'rbf',
            'C': 1.0,
            'epsilon': 0.1
        }
        default_params.update(self.model_params)
        return SVR(**default_params)


class MLPModel(BaseModel):
    """Multi-layer Perceptron model for content performance prediction."""
    
    def __init__(self, target_column: str, feature_columns: List[str] = None, **kwargs):
        """
        Initialize MLP model.
        
        Args:
            target_column: Target column for prediction
            feature_columns: List of feature columns
            **kwargs: Additional parameters for MLPRegressor
        """
        super().__init__("mlp", target_column, feature_columns)
        self.model_params = kwargs
    
    def _create_model(self) -> MLPRegressor:
        """Create MLP model."""
        default_params = {
            'hidden_layer_sizes': (100, 50),
            'activation': 'relu',
            'solver': 'adam',
            'alpha': 0.0001,
            'learning_rate': 'adaptive',
            'max_iter': 1000,
            'random_state': 42
        }
        default_params.update(self.model_params)
        return MLPRegressor(**default_params)


class EnsembleModel(BaseModel):
    """Ensemble model combining multiple base models."""
    
    def __init__(self, target_column: str, feature_columns: List[str] = None, 
                 base_models: List[BaseModel] = None, weights: List[float] = None):
        """
        Initialize Ensemble model.
        
        Args:
            target_column: Target column for prediction
            feature_columns: List of feature columns
            base_models: List of base models to ensemble
            weights: Weights for each model (optional)
        """
        super().__init__("ensemble", target_column, feature_columns)
        self.base_models = base_models or []
        self.weights = weights or []
        self.is_fitted = False
    
    def _create_model(self) -> None:
        """
        Create the underlying model instance.
        For ensemble models, this is not needed as we use base models.
        """
        return None
    
    def add_model(self, model: BaseModel, weight: float = 1.0) -> None:
        """
        Add a model to the ensemble.
        
        Args:
            model: Base model to add
            weight: Weight for the model
        """
        self.base_models.append(model)
        self.weights.append(weight)
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'EnsembleModel':
        """
        Fit all base models.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            
        Returns:
            Self
        """
        if not self.base_models:
            raise ValueError("No base models added to ensemble")
        
        # Fit each base model
        for model in self.base_models:
            model.fit(X, y)
        
        self.is_fitted = True
        logger.info(f"Fitted ensemble with {len(self.base_models)} models")
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make ensemble predictions.
        
        Args:
            X: Feature DataFrame
            
        Returns:
            Weighted average of predictions
        """
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before making predictions")
        
        predictions = []
        for model in self.base_models:
            pred = model.predict(X)
            predictions.append(pred)
        
        # Weighted average
        if self.weights and len(self.weights) == len(predictions):
            weights = np.array(self.weights)
            weights = weights / weights.sum()  # Normalize weights
            ensemble_pred = np.average(predictions, axis=0, weights=weights)
        else:
            ensemble_pred = np.mean(predictions, axis=0)
        
        return ensemble_pred
    
    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """
        Evaluate the ensemble model.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Evaluate individual models
        individual_metrics = {}
        for i, model in enumerate(self.base_models):
            metrics = model.evaluate(X, y)
            individual_metrics[f"model_{i}_{model.model_name}"] = metrics
        
        # Evaluate ensemble
        ensemble_metrics = super().evaluate(X, y)
        
        # Combine metrics
        all_metrics = {**individual_metrics, **ensemble_metrics}
        self.metrics = all_metrics
        
        return all_metrics


def create_model(model_type: str, target_column: str, feature_columns: List[str] = None, **kwargs) -> BaseModel:
    """
    Factory function to create models.
    
    Args:
        model_type: Type of model to create
        target_column: Target column for prediction
        feature_columns: List of feature columns
        **kwargs: Additional model parameters
        
    Returns:
        Model instance
    """
    model_classes = {
        'random_forest': RandomForestModel,
        'xgboost': XGBoostModel,
        'lightgbm': LightGBMModel,
        'catboost': CatBoostModel,
        'linear_regression': LinearRegressionModel,
        'ridge': RidgeModel,
        'svr': SVRModel,
        'mlp': MLPModel,
        'ensemble': EnsembleModel
    }
    
    if model_type not in model_classes:
        raise ValueError(f"Unknown model type: {model_type}. Available types: {list(model_classes.keys())}")
    
    return model_classes[model_type](target_column, feature_columns, **kwargs)