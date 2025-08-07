"""Command-line interface for content performance predictor."""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

from .data.database import SupabaseClient
from .data.data_loader import DataLoader
from .features.feature_engineering import FeatureEngineer
from .models.prediction_models import create_model
from .utils.config import get_config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = False):
    """Set up logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def train_model(args):
    """Train a model."""
    try:
        # Load configuration
        config = get_config()
        
        # Initialize components
        supabase_client = SupabaseClient()
        data_loader = DataLoader(supabase_client)
        feature_engineer = FeatureEngineer()
        
        # Load data
        logger.info("Loading data...")
        posts_df, profile_df = data_loader.load_consolidated_data(
            start_date=args.start_date,
            end_date=args.end_date
        )
        
        if posts_df.empty:
            logger.error("No data found for the specified date range")
            return 1
        
        # Preprocess data
        logger.info("Preprocessing data...")
        posts_df = data_loader.preprocess_posts_data(posts_df)
        profile_df = data_loader.preprocess_profile_data(profile_df)
        
        # Create features
        logger.info("Creating features...")
        posts_df = feature_engineer.create_temporal_features(posts_df)
        posts_df = feature_engineer.create_engagement_features(posts_df)
        posts_df = feature_engineer.create_cross_platform_features(posts_df)
        
        # Create dataset
        if args.platform:
            dataset = data_loader.create_platform_dataset(args.platform, posts_df, profile_df)
            target_col = f'engagement_{args.platform}_{args.content_type}'
        else:
            dataset = data_loader.create_cross_platform_dataset(posts_df, profile_df)
            target_col = f'total_engagement_{args.content_type}'
        
        # Prepare features and target
        feature_cols = data_loader.get_feature_columns(args.platform)
        available_features = [col for col in feature_cols if col in dataset.columns]
        
        if not available_features:
            logger.error("No features available for training")
            return 1
        
        X = dataset[available_features].fillna(0)
        y = dataset[target_col].fillna(0)
        
        # Remove rows with missing target
        mask = ~y.isnull()
        X = X[mask]
        y = y[mask]
        
        if len(X) == 0:
            logger.error("No valid data for training")
            return 1
        
        # Create and train model
        logger.info(f"Training {args.model_type} model...")
        model = create_model(
            model_type=args.model_type,
            target_column=target_col,
            feature_columns=available_features
        )
        
        model.fit(X, y)
        
        # Evaluate model
        logger.info("Evaluating model...")
        metrics = model.evaluate(X, y)
        
        for metric, value in metrics.items():
            logger.info(f"{metric}: {value:.4f}")
        
        # Save model
        model_path = Path(config.get_paths()['models'])
        model_path.mkdir(exist_ok=True)
        
        model_name = f"{args.platform or 'cross_platform'}_{args.content_type}_{args.model_type}"
        model_file = model_path / f"{model_name}.joblib"
        
        model.save_model(str(model_file))
        logger.info(f"Model saved to {model_file}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Error training model: {e}")
        return 1


def predict(args):
    """Make predictions."""
    try:
        # Load configuration
        config = get_config()
        
        # Load model
        model_path = Path(config.get_paths()['models'])
        model_name = f"{args.platform or 'cross_platform'}_{args.content_type}_{args.model_type}"
        model_file = model_path / f"{model_name}.joblib"
        
        if not model_file.exists():
            logger.error(f"Model not found: {model_file}")
            return 1
        
        # Load model (this would need to be implemented in the base model)
        logger.info(f"Loading model from {model_file}")
        # model = BaseModel.load_model(str(model_file))
        
        # For now, just print a message
        logger.info(f"Prediction for {args.platform} {args.content_type} on {args.date}")
        logger.info("Prediction functionality to be implemented")
        
        return 0
        
    except Exception as e:
        logger.error(f"Error making prediction: {e}")
        return 1


def analyze_data(args):
    """Analyze data."""
    try:
        # Initialize components
        supabase_client = SupabaseClient()
        data_loader = DataLoader(supabase_client)
        
        # Load data
        logger.info("Loading data...")
        posts_df, profile_df = data_loader.load_consolidated_data(
            start_date=args.start_date,
            end_date=args.end_date
        )
        
        if posts_df.empty:
            logger.error("No data found for the specified date range")
            return 1
        
        # Basic analysis
        logger.info(f"Data summary:")
        logger.info(f"  Posts: {len(posts_df)} records")
        logger.info(f"  Profiles: {len(profile_df)} records")
        logger.info(f"  Date range: {posts_df['date'].min()} to {posts_df['date'].max()}")
        
        # Platform analysis
        platforms = ['linkedin', 'instagram', 'twitter', 'substack', 'threads']
        for platform in platforms:
            engagement_cols = [col for col in posts_df.columns if f'engagement_{platform}' in col]
            if engagement_cols:
                total_engagement = posts_df[engagement_cols].sum().sum()
                logger.info(f"  {platform.title()}: {total_engagement:.0f} total engagement")
        
        return 0
        
    except Exception as e:
        logger.error(f"Error analyzing data: {e}")
        return 1


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Content Performance Predictor CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s train --platform linkedin --content-type no_video --model-type xgboost
  %(prog)s predict --platform linkedin --content-type no_video --date 2023-12-01
  %(prog)s analyze --start-date 2023-01-01 --end-date 2023-12-31
        """
    )
    
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train a model')
    train_parser.add_argument('--platform', choices=['linkedin', 'instagram', 'twitter', 'substack', 'threads'],
                             help='Platform to train for')
    train_parser.add_argument('--content-type', choices=['no_video', 'video'], default='no_video',
                             help='Content type')
    train_parser.add_argument('--model-type', 
                             choices=['random_forest', 'xgboost', 'lightgbm', 'catboost', 'linear_regression'],
                             default='xgboost', help='Model type')
    train_parser.add_argument('--start-date', help='Start date (YYYY-MM-DD)')
    train_parser.add_argument('--end-date', help='End date (YYYY-MM-DD)')
    
    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Make predictions')
    predict_parser.add_argument('--platform', choices=['linkedin', 'instagram', 'twitter', 'substack', 'threads'],
                               help='Platform to predict for')
    predict_parser.add_argument('--content-type', choices=['no_video', 'video'], default='no_video',
                               help='Content type')
    predict_parser.add_argument('--model-type', 
                               choices=['random_forest', 'xgboost', 'lightgbm', 'catboost', 'linear_regression'],
                               default='xgboost', help='Model type')
    predict_parser.add_argument('--date', required=True, help='Date to predict for (YYYY-MM-DD)')
    
    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze data')
    analyze_parser.add_argument('--start-date', help='Start date (YYYY-MM-DD)')
    analyze_parser.add_argument('--end-date', help='End date (YYYY-MM-DD)')
    
    args = parser.parse_args()
    
    # Set up logging
    setup_logging(args.verbose)
    
    # Execute command
    if args.command == 'train':
        return train_model(args)
    elif args.command == 'predict':
        return predict(args)
    elif args.command == 'analyze':
        return analyze_data(args)
    else:
        parser.print_help()
        return 1


if __name__ == '__main__':
    sys.exit(main())