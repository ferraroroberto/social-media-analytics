"""Base model class for content performance prediction."""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
import joblib
import os
from abc import ABC, abstractmethod
from sklearn.base import BaseEstimator
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import mlflow
import mlflow.sklearn

logger = logging.getLogger(__name__)


class BaseModel(ABC, BaseEstimator):
    """Base class for all content performance prediction models."""
    
    def __init__(self, model_name: str, target_column: str, feature_columns: List[str] = None):
        """
        Initialize base model.
        
        Args:
            model_name: Name of the model
            target_column: Target column for prediction
            feature_columns: List of feature columns
        """
        self.model_name = model_name
        self.target_column = target_column
        self.feature_columns = feature_columns or []
        self.model = None
        self.is_fitted = False
        self.metrics = {}
        self.feature_importance = {}
        
    @abstractmethod
    def _create_model(self) -> Any:
        """Create the underlying model instance."""
        pass
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'BaseModel':
        """
        Fit the model to the data.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            
        Returns:
            Self
        """
        if self.model is None:
            self.model = self._create_model()
        
        # Ensure we have the right features
        if self.feature_columns:
            X = X[self.feature_columns]
        
        # Handle different column types
        X_clean = X.copy()
        for col in X_clean.columns:
            if X_clean[col].dtype in ['int64', 'float64']:
                # For numeric columns, fill missing values with median
                median_val = X_clean[col].median()
                if pd.isna(median_val):
                    median_val = 0
                X_clean[col] = X_clean[col].fillna(median_val)
            else:
                # For categorical/string columns, encode all values as categories
                X_clean[col] = X_clean[col].fillna('missing')
                X_clean[col] = pd.Categorical(X_clean[col]).codes
        
        # Remove any rows with missing target values
        mask = ~y.isnull()
        X_clean = X_clean[mask]
        y_clean = y[mask]
        
        if len(X_clean) == 0:
            raise ValueError("No valid data after removing missing values")
        
        logger.info(f"Fitting {self.model_name} with {len(X_clean)} samples and {len(X_clean.columns)} features")
        
        try:
            self.model.fit(X_clean, y_clean)
            self.is_fitted = True
            logger.info(f"Successfully fitted {self.model_name}")
        except Exception as e:
            logger.error(f"Error fitting {self.model_name}: {e}")
            raise
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Feature DataFrame
            
        Returns:
            Predictions array
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        if self.feature_columns:
            X = X[self.feature_columns]
        
        # Handle different column types
        X_clean = X.copy()
        for col in X_clean.columns:
            if X_clean[col].dtype in ['int64', 'float64']:
                # For numeric columns, fill missing values with median
                median_val = X_clean[col].median()
                if pd.isna(median_val):
                    median_val = 0
                X_clean[col] = X_clean[col].fillna(median_val)
            else:
                # For categorical/string columns, encode all values as categories
                X_clean[col] = X_clean[col].fillna('missing')
                X_clean[col] = pd.Categorical(X_clean[col]).codes
        
        try:
            predictions = self.model.predict(X_clean)
            logger.info(f"Made predictions for {len(X_clean)} samples")
            return predictions
        except Exception as e:
            logger.error(f"Error making predictions: {e}")
            raise
    
    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """
        Evaluate the model.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            
        Returns:
            Dictionary of evaluation metrics
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before evaluation")
        
        predictions = self.predict(X)
        
        # Calculate metrics
        mae = mean_absolute_error(y, predictions)
        mse = mean_squared_error(y, predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(y, predictions)
        
        self.metrics = {
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'r2': r2
        }
        
        logger.info(f"Model evaluation - MAE: {mae:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}")
        
        return self.metrics
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance if available.
        
        Returns:
            Dictionary of feature importance scores
        """
        if not self.is_fitted:
            return {}
        
        if hasattr(self.model, 'feature_importances_'):
            importance = self.model.feature_importances_
            features = self.feature_columns if self.feature_columns else [f'feature_{i}' for i in range(len(importance))]
            self.feature_importance = dict(zip(features, importance))
        elif hasattr(self.model, 'coef_'):
            coef = self.model.coef_
            features = self.feature_columns if self.feature_columns else [f'feature_{i}' for i in range(len(coef))]
            self.feature_importance = dict(zip(features, np.abs(coef)))
        else:
            self.feature_importance = {}
        
        return self.feature_importance
    
    def save_model(self, filepath: str) -> None:
        """
        Save the model to disk.
        
        Args:
            filepath: Path to save the model
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving")
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        model_data = {
            'model_name': self.model_name,
            'target_column': self.target_column,
            'feature_columns': self.feature_columns,
            'model': self.model,
            'is_fitted': self.is_fitted,
            'metrics': self.metrics,
            'feature_importance': self.feature_importance
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")
    
    @classmethod
    def load_model(cls, filepath: str) -> 'BaseModel':
        """
        Load a model from disk.
        
        Args:
            filepath: Path to the saved model
            
        Returns:
            Loaded model instance
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        model_data = joblib.load(filepath)
        
        # Create new instance
        instance = cls(
            model_name=model_data['model_name'],
            target_column=model_data['target_column'],
            feature_columns=model_data['feature_columns']
        )
        
        # Restore state
        instance.model = model_data['model']
        instance.is_fitted = model_data['is_fitted']
        instance.metrics = model_data['metrics']
        instance.feature_importance = model_data['feature_importance']
        
        logger.info(f"Model loaded from {filepath}")
        return instance
    
    def log_to_mlflow(self, experiment_name: str = None) -> None:
        """
        Log model and metrics to MLflow.
        
        Args:
            experiment_name: MLflow experiment name
        """
        if not self.is_fitted:
            logger.warning("Model not fitted, skipping MLflow logging")
            return
        
        try:
            if experiment_name:
                mlflow.set_experiment(experiment_name)
            
            with mlflow.start_run():
                # Log model
                mlflow.sklearn.log_model(self.model, self.model_name)
                
                # Log parameters
                if hasattr(self.model, 'get_params'):
                    mlflow.log_params(self.model.get_params())
                
                # Log metrics
                if self.metrics:
                    mlflow.log_metrics(self.metrics)
                
                # Log feature importance
                if self.feature_importance:
                    importance_df = pd.DataFrame(
                        list(self.feature_importance.items()),
                        columns=['feature', 'importance']
                    ).sort_values('importance', ascending=False)
                    
                    mlflow.log_artifact(importance_df.to_csv(index=False), "feature_importance.csv")
                
                logger.info(f"Model logged to MLflow experiment: {experiment_name}")
                
        except Exception as e:
            logger.error(f"Error logging to MLflow: {e}")
    
    def get_model_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the model.
        
        Returns:
            Dictionary with model summary
        """
        summary = {
            'model_name': self.model_name,
            'target_column': self.target_column,
            'feature_columns': self.feature_columns,
            'is_fitted': self.is_fitted,
            'metrics': self.metrics,
            'feature_importance': self.feature_importance
        }
        
        if hasattr(self.model, 'get_params'):
            summary['model_params'] = self.model.get_params()
        
        return summary