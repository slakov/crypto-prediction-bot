#!/usr/bin/env python3
"""
Improved Crypto Prediction Model
Advanced ML with proper feature engineering and validation
"""

import numpy as np
import pandas as pd
import requests
import time
import json
import os
from typing import Dict, List, Optional, Tuple, Any
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import TimeSeriesSplit, KFold, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import warnings
warnings.filterwarnings('ignore')

class AdvancedCryptoPredictionModel:
    """
    Advanced crypto prediction model with proper ML pipeline
    """
    
    def __init__(self):
        self.models = {}
        self.scaler = RobustScaler()
        self.feature_names = []
        self.model_weights = {'rf': 0.3, 'gb': 0.3, 'ridge': 0.2, 'elastic': 0.2}
        self.is_trained = False
        
        # Model parameters optimized for crypto prediction
        self.rf_params = {
            'n_estimators': 100,
            'max_depth': 8,
            'min_samples_split': 10,
            'min_samples_leaf': 5,
            'random_state': 42,
            'n_jobs': -1
        }
        
        self.gb_params = {
            'n_estimators': 80,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'random_state': 42
        }
        
        self.ridge_params = {
            'alpha': 1.0,
            'random_state': 42
        }
        
        self.elastic_params = {
            'alpha': 0.5,
            'l1_ratio': 0.5,
            'random_state': 42
        }
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Advanced feature engineering for crypto prediction
        """
        if df.empty:
            return df
            
        features_df = df.copy()
        
        # Basic price features
        features_df['price_log'] = np.log(features_df['current_price'].clip(lower=1e-8))
        features_df['market_cap_log'] = np.log(features_df['market_cap'].clip(lower=1))
        features_df['volume_log'] = np.log(features_df['total_volume'].clip(lower=1))
        
        # Volume ratios and velocity
        features_df['volume_to_mcap'] = features_df['total_volume'] / features_df['market_cap'].clip(lower=1)
        features_df['price_volume_trend'] = features_df['price_change_percentage_24h_in_currency'] * features_df['volume_to_mcap']
        
        # Momentum features (properly handle missing values)
        for period in ['1h', '24h', '7d']:
            col = f'price_change_percentage_{period}_in_currency'
            if col in features_df.columns:
                features_df[f'momentum_{period}'] = features_df[col].fillna(0).clip(-50, 50)
                features_df[f'momentum_{period}_abs'] = features_df[f'momentum_{period}'].abs()
                features_df[f'momentum_{period}_sign'] = np.sign(features_df[f'momentum_{period}'])
                # Percentile ranks for monotonic transformations
                features_df[f'momentum_{period}_rank'] = features_df[f'momentum_{period}'].rank(pct=True)
        
        # Cross-period momentum relationships
        if all(col in features_df.columns for col in ['momentum_1h', 'momentum_24h']):
            features_df['momentum_acceleration'] = features_df['momentum_1h'] - (features_df['momentum_24h'] / 24)
            features_df['momentum_consistency'] = (features_df['momentum_1h'] * features_df['momentum_24h'] > 0).astype(int)
        
        # Market cap tiers (one-hot encoded)
        mcap_tiers = pd.cut(features_df['market_cap'], 
                           bins=[0, 1e8, 1e9, 1e10, 1e11, np.inf],
                           labels=['micro', 'small', 'mid', 'large', 'mega'])
        features_df = pd.concat([features_df, pd.get_dummies(mcap_tiers, prefix='mcap_tier')], axis=1)
        
        # Volume anomaly detection
        if len(features_df) > 10:
            volume_median = features_df['volume_to_mcap'].median()
            volume_mad = (features_df['volume_to_mcap'] - volume_median).abs().median()
            features_df['volume_anomaly'] = (features_df['volume_to_mcap'] - volume_median) / (volume_mad + 1e-8)
            features_df['high_volume_flag'] = (features_df['volume_anomaly'] > 3).astype(int)
        else:
            features_df['volume_anomaly'] = 0
            features_df['high_volume_flag'] = 0

        # Rank transform for volume ratio
        features_df['volume_to_mcap_rank'] = features_df['volume_to_mcap'].rank(pct=True)
        
        # Relative performance vs market leaders
        btc_perf = features_df[features_df['symbol'].str.upper() == 'BTC']['momentum_24h'].iloc[0] if any(features_df['symbol'].str.upper() == 'BTC') else 0
        eth_perf = features_df[features_df['symbol'].str.upper() == 'ETH']['momentum_24h'].iloc[0] if any(features_df['symbol'].str.upper() == 'ETH') else 0
        
        features_df['outperform_btc'] = features_df['momentum_24h'] - btc_perf
        features_df['outperform_eth'] = features_df['momentum_24h'] - eth_perf
        
        # Market cap rank features
        if 'market_cap_rank' in features_df.columns:
            features_df['rank_log'] = np.log(features_df['market_cap_rank'].clip(lower=1))
            features_df['rank_percentile'] = features_df['market_cap_rank'].rank(pct=True, ascending=False)
        
        # Technical strength indicators
        features_df['price_strength'] = (
            0.4 * features_df['momentum_24h'].fillna(0) +
            0.3 * features_df['momentum_1h'].fillna(0) + 
            0.2 * features_df['volume_anomaly'] +
            0.1 * features_df['outperform_btc']
        )
        
        # Clean up infinite and NaN values
        features_df = features_df.replace([np.inf, -np.inf], 0)
        features_df = features_df.fillna(0)
        
        return features_df
    
    def select_features(self, df: pd.DataFrame) -> List[str]:
        """
        Select the most predictive features for the model
        """
        # Core momentum and volume features
        base_features = [
            'momentum_1h', 'momentum_24h', 'momentum_7d',
            'momentum_1h_abs', 'momentum_24h_abs', 'momentum_7d_abs',
            'volume_to_mcap', 'volume_anomaly', 'high_volume_flag',
            'momentum_acceleration', 'momentum_consistency',
            'outperform_btc', 'outperform_eth', 'price_strength'
        ]
        
        # Market cap features
        mcap_features = [col for col in df.columns if col.startswith('mcap_tier_')]
        
        # Log-transformed features
        log_features = ['price_log', 'market_cap_log', 'volume_log']
        
        # Rank features
        rank_features = ['volume_to_mcap_rank']
        if 'rank_log' in df.columns:
            rank_features.extend(['rank_log', 'rank_percentile'])
        # Momentum rank features if present
        for period in ['1h', '24h', '7d']:
            col = f'momentum_{period}_rank'
            if col in df.columns:
                rank_features.append(col)
        
        # Combine all features
        all_features = base_features + mcap_features + log_features + rank_features
        
        # Filter to only include features that exist in the dataframe
        available_features = [f for f in all_features if f in df.columns]
        
        self.feature_names = available_features
        return available_features
    
    def train_models(self, df: pd.DataFrame, target_col: str = 'target_24h_change') -> Dict[str, float]:
        """
        Train ensemble of ML models on the data
        """
        if len(df) < 20:
            print(f"⚠️  Insufficient data for training: {len(df)} samples")
            return {}
        
        # Engineer features
        df_features = self.engineer_features(df)
        
        # Select features
        feature_cols = self.select_features(df_features)
        
        if not feature_cols:
            print("❌ No valid features found for training")
            return {}
        
        X = df_features[feature_cols].copy()
        y = df_features[target_col].copy()
        
        # Remove samples with invalid targets
        valid_mask = pd.notna(y) & np.isfinite(y)
        X = X[valid_mask]
        y = y[valid_mask]
        
        if len(X) < 10:
            print(f"⚠️  Insufficient valid samples after cleaning: {len(X)}")
            return {}
        
        print(f"📊 Training on {len(X)} samples with {len(feature_cols)} features")
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        X_scaled_df = pd.DataFrame(X_scaled, columns=feature_cols, index=X.index)
        
        # Initialize base models
        rf_model = RandomForestRegressor(**self.rf_params)
        gb_model = GradientBoostingRegressor(**self.gb_params)
        ridge_model = Ridge(**self.ridge_params)
        elastic_model = ElasticNet(**self.elastic_params)

        self.models = {
            'rf': rf_model,
            'gb': gb_model,
            'ridge': ridge_model,
            'elastic': elastic_model
        }
        
        # Train models and evaluate
        scores = {}
        
        # Use TimeSeriesSplit for time-aware validation
        tscv = TimeSeriesSplit(n_splits=min(5, len(X) // 10))
        
        for name, model in self.models.items():
            try:
                # Fit model
                model.fit(X_scaled_df, y)
                
                # Cross-validation score
                cv_scores = cross_val_score(model, X_scaled_df, y, cv=tscv, scoring='neg_mean_absolute_error')
                scores[name] = -cv_scores.mean()
                
                print(f"✅ {name.upper()}: MAE = {scores[name]:.3f}")
                
            except Exception as e:
                print(f"❌ Failed to train {name}: {e}")
                scores[name] = float('inf')

        # Stacking meta-learner using out-of-fold predictions
        try:
            base_estimators = [
                ('rf', rf_model),
                ('gb', gb_model),
                ('ridge', ridge_model),
                ('elastic', elastic_model)
            ]
            meta_learner = Ridge(alpha=0.5, random_state=42)
            stacking_reg = StackingRegressor(
                estimators=base_estimators,
                final_estimator=meta_learner,
                cv=KFold(n_splits=min(5, max(2, len(X) // 50)), shuffle=False),
                passthrough=False,
                n_jobs=None
            )
            stacking_reg.fit(X_scaled_df, y)
            self.models['stack'] = stacking_reg
            # Cross-validated MAE for stack
            cv_scores_stack = cross_val_score(stacking_reg, X_scaled_df, y, cv=tscv, scoring='neg_mean_absolute_error')
            scores['stack'] = -cv_scores_stack.mean()
            print(f"✅ STACK: MAE = {scores['stack']:.3f}")
        except Exception as e:
            print(f"⚠️  Failed to train stacking meta-learner: {e}")
        
        # Update model weights based on performance
        if scores:
            # Inverse weighting - better models get higher weights
            inv_scores = {k: 1 / (v + 0.1) for k, v in scores.items() if v < float('inf')}
            total_inv = sum(inv_scores.values())
            
            if total_inv > 0:
                self.model_weights = {k: v / total_inv for k, v in inv_scores.items()}
                print(f"📈 Updated model weights: {self.model_weights}")
        
        self.is_trained = True
        return scores
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Generate predictions using the trained ensemble
        """
        if not self.is_trained or not self.models:
            print("⚠️  Model not trained yet")
            return np.zeros(len(df))
        
        # Engineer features
        df_features = self.engineer_features(df)
        
        # Select same features as training
        if not self.feature_names:
            print("❌ No feature names stored from training")
            return np.zeros(len(df))
        
        X = df_features[self.feature_names].copy()
        
        # Scale features using fitted scaler
        try:
            X_scaled = self.scaler.transform(X)
            X_scaled_df = pd.DataFrame(X_scaled, columns=self.feature_names, index=X.index)
        except Exception as e:
            print(f"❌ Error scaling features: {e}")
            return np.zeros(len(df))
        
        # Generate ensemble predictions
        predictions = np.zeros(len(df))
        total_weight = 0
        
        for name, model in self.models.items():
            if name in self.model_weights:
                try:
                    pred = model.predict(X_scaled_df)
                    weight = self.model_weights[name]
                    predictions += weight * pred
                    total_weight += weight
                except Exception as e:
                    print(f"⚠️  Error predicting with {name}: {e}")
                    continue
        
        # Normalize by total weight
        if total_weight > 0:
            predictions /= total_weight
        
        # Apply realistic bounds for crypto predictions
        predictions = np.clip(predictions, -25, 25)
        
        return predictions
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance from trained models
        """
        if not self.is_trained:
            return {}
        
        importance_dict = {}
        
        # Random Forest feature importance
        if 'rf' in self.models and hasattr(self.models['rf'], 'feature_importances_'):
            rf_importance = dict(zip(self.feature_names, self.models['rf'].feature_importances_))
            for feature, importance in rf_importance.items():
                importance_dict[f'rf_{feature}'] = importance
        
        # Gradient Boosting feature importance  
        if 'gb' in self.models and hasattr(self.models['gb'], 'feature_importances_'):
            gb_importance = dict(zip(self.feature_names, self.models['gb'].feature_importances_))
            for feature, importance in gb_importance.items():
                importance_dict[f'gb_{feature}'] = importance
        
        return importance_dict
    
    def save_model(self, filepath: str):
        """
        Save the trained model to disk
        """
        import pickle
        
        model_data = {
            'models': self.models,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'model_weights': self.model_weights,
            'is_trained': self.is_trained
        }
        
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
            print(f"✅ Model saved to {filepath}")
        except Exception as e:
            print(f"❌ Error saving model: {e}")
    
    def load_model(self, filepath: str):
        """
        Load a trained model from disk
        """
        import pickle
        
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            self.models = model_data['models']
            self.scaler = model_data['scaler']
            self.feature_names = model_data['feature_names']
            self.model_weights = model_data['model_weights']
            self.is_trained = model_data['is_trained']
            
            print(f"✅ Model loaded from {filepath}")
        except Exception as e:
            print(f"❌ Error loading model: {e}")


def fetch_coingecko_data(limit: int = 100) -> Optional[pd.DataFrame]:
    """
    Fetch current market data from CoinGecko
    """
    try:
        url = "https://api.coingecko.com/api/v3/coins/markets"
        params = {
            'vs_currency': 'usd',
            'order': 'market_cap_desc',
            'per_page': limit,
            'page': 1,
            'sparkline': False,
            'price_change_percentage': '1h,24h,7d'
        }
        
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        
        data = response.json()
        df = pd.DataFrame(data)
        
        if df.empty:
            return None
        
        # Standardize column names
        rename_map = {
            'price_change_percentage_1h_in_currency': 'price_change_percentage_1h_in_currency',
            'price_change_percentage_24h_in_currency': 'price_change_percentage_24h_in_currency', 
            'price_change_percentage_7d_in_currency': 'price_change_percentage_7d_in_currency'
        }
        df = df.rename(columns=rename_map)
        
        return df
        
    except Exception as e:
        print(f"❌ Error fetching CoinGecko data: {e}")
        return None


def create_synthetic_training_data(df: pd.DataFrame, num_samples: int = 1000) -> pd.DataFrame:
    """
    Create synthetic training data for model development
    """
    if df.empty:
        return df
    
    training_data = []
    
    for _ in range(num_samples):
        # Sample a random row as base
        base_row = df.sample(1).iloc[0].copy()
        
        # Add realistic noise and variations
        synthetic_row = base_row.copy()
        
        # Vary momentum with realistic crypto volatility
        momentum_noise = np.random.normal(0, 2.5)  # 2.5% standard noise
        synthetic_row['price_change_percentage_1h_in_currency'] = base_row.get('price_change_percentage_1h_in_currency', 0) + momentum_noise * 0.3
        synthetic_row['price_change_percentage_24h_in_currency'] = base_row.get('price_change_percentage_24h_in_currency', 0) + momentum_noise
        
        # Vary volume with market cap correlation
        volume_factor = np.random.lognormal(0, 0.3)  # 30% log-normal variation
        synthetic_row['total_volume'] = base_row.get('total_volume', 0) * volume_factor
        
        # Create realistic target (future 24h change)
        # Based on current momentum + mean reversion + random walk
        current_momentum = synthetic_row.get('price_change_percentage_24h_in_currency', 0)
        volume_factor = synthetic_row.get('total_volume', 0) / max(synthetic_row.get('market_cap', 1), 1)
        
        # Mean reversion component (negative correlation with current performance)
        mean_reversion = -0.15 * current_momentum
        
        # Volume impact (higher volume -> more follow-through)
        volume_impact = 0.1 * np.log(volume_factor + 1e-8) if volume_factor > 0 else 0
        
        # Random walk component
        random_component = np.random.normal(0, 3.0)  # 3% daily volatility
        
        # Market regime factor (simulate bull/bear conditions)
        regime_factor = np.random.choice([0.8, 1.0, 1.2], p=[0.2, 0.6, 0.2])
        
        target_change = (mean_reversion + volume_impact + random_component) * regime_factor
        target_change = np.clip(target_change, -30, 30)  # Realistic bounds
        
        synthetic_row['target_24h_change'] = target_change
        
        training_data.append(synthetic_row)
    
    return pd.DataFrame(training_data)


def evaluate_model_performance(model: AdvancedCryptoPredictionModel, test_df: pd.DataFrame) -> Dict[str, float]:
    """
    Evaluate model performance on test data
    """
    if test_df.empty or 'target_24h_change' not in test_df.columns:
        return {}
    
    # Generate predictions
    predictions = model.predict(test_df)
    actual = test_df['target_24h_change'].values
    
    # Remove invalid values
    valid_mask = pd.notna(actual) & pd.notna(predictions) & np.isfinite(actual) & np.isfinite(predictions)
    actual_clean = actual[valid_mask]
    pred_clean = predictions[valid_mask]
    
    if len(actual_clean) < 2:
        return {}
    
    # Calculate metrics
    mae = mean_absolute_error(actual_clean, pred_clean)
    rmse = np.sqrt(mean_squared_error(actual_clean, pred_clean))
    
    # Correlation
    correlation = np.corrcoef(actual_clean, pred_clean)[0, 1] if len(actual_clean) > 1 else 0
    
    # Bias (directional accuracy)
    bias = np.mean(pred_clean - actual_clean)
    
    # R-squared
    r2 = r2_score(actual_clean, pred_clean)
    
    # Directional accuracy
    actual_direction = np.sign(actual_clean)
    pred_direction = np.sign(pred_clean)
    directional_accuracy = np.mean(actual_direction == pred_direction)
    
    return {
        'mae': mae,
        'rmse': rmse,
        'correlation': correlation,
        'bias': bias,
        'r2': r2,
        'directional_accuracy': directional_accuracy,
        'n_samples': len(actual_clean)
    }


def main():
    """
    Main function to test the improved model
    """
    print("🚀 Testing Improved Crypto Prediction Model")
    print("=" * 50)
    
    # Fetch current market data
    print("📊 Fetching market data...")
    market_df = fetch_coingecko_data(100)
    
    if market_df is None or market_df.empty:
        print("❌ Failed to fetch market data")
        return
    
    print(f"✅ Loaded {len(market_df)} coins")
    
    # Create synthetic training data
    print("🧪 Creating synthetic training data...")
    training_df = create_synthetic_training_data(market_df, num_samples=2000)
    print(f"✅ Created {len(training_df)} training samples")
    
    # Initialize and train model
    model = AdvancedCryptoPredictionModel()
    
    print("🎯 Training ensemble models...")
    train_scores = model.train_models(training_df)
    
    # Create test data
    test_df = create_synthetic_training_data(market_df, num_samples=500)
    
    # Evaluate performance
    print("\n📈 Evaluating model performance...")
    performance = evaluate_model_performance(model, test_df)
    
    print("\n🎯 Model Performance Metrics:")
    for metric, value in performance.items():
        if metric == 'n_samples':
            print(f"📊 {metric}: {value}")
        else:
            print(f"📈 {metric}: {value:.4f}")
    
    # Generate current predictions
    print("\n🔮 Current Market Predictions:")
    current_predictions = model.predict(market_df)
    
    # Add predictions to dataframe
    market_df['predicted_24h_change'] = current_predictions
    
    # Show top predictions
    top_predictions = market_df.nlargest(10, 'predicted_24h_change')[
        ['symbol', 'name', 'current_price', 'price_change_percentage_24h_in_currency', 'predicted_24h_change']
    ]
    
    print("\n🚀 Top 10 Predictions:")
    for _, row in top_predictions.iterrows():
        symbol = row['symbol'].upper()
        name = row['name']
        price = row['current_price']
        current_change = row.get('price_change_percentage_24h_in_currency', 0) or 0
        predicted = row['predicted_24h_change']
        
        print(f"💎 {symbol} ({name})")
        print(f"   💰 Price: ${price:.4f}")
        print(f"   📊 24h: {current_change:+.2f}% → Predicted: {predicted:+.2f}%")
        print()
    
    # Show feature importance
    importance = model.get_feature_importance()
    if importance:
        print("🔍 Top Feature Importances:")
        sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]
        for feature, score in sorted_importance:
            print(f"   {feature}: {score:.4f}")
    
    # Save model
    model_path = os.path.join(os.path.dirname(__file__), "improved_crypto_model.pkl")
    model.save_model(model_path)
    
    print(f"\n✅ Model training completed and saved to {model_path}")


if __name__ == "__main__":
    main()
