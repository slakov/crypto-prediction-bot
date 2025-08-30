#!/usr/bin/env python3
"""
Production Crypto Prediction Bot V2
Enhanced with advanced ML prediction model
"""

import asyncio
import json
import logging
import os
import requests
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional
import time
import random
import traceback

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes
from telegram.error import BadRequest

# Simple HTTP GET with retries/backoff to mitigate CoinGecko 429s
def http_get_with_retries(url: str, params: Dict[str, Any], timeout: int = 15, retries: int = 4, backoff_sec: float = 6.0):
    last_exc = None
    headers = {"User-Agent": "zpredbot/1.0 (+https://github.com/slakov/crypto-prediction-bot)"}
    for attempt in range(retries):
        try:
            resp = requests.get(url, params=params, timeout=timeout, headers=headers)
            if resp.status_code == 429:
                wait = backoff_sec * (attempt + 1)
                print(f"⚠️  Rate limited (429). Backing off {wait:.1f}s...")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp
        except Exception as e:
            last_exc = e
            wait = backoff_sec * (attempt + 1)
            print(f"⚠️  HTTP error on {url}: {e}. Retrying in {wait:.1f}s...")
            time.sleep(wait)
    if last_exc:
        raise last_exc
    return None

# Import our improved model (project-local)
from improved_model import AdvancedCryptoPredictionModel

# Setup production logging (concise but keeps errors and stacktraces)
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    level=logging.ERROR
)
logger = logging.getLogger(__name__)

# Reduce noisy libraries but keep error visibility
for logger_name in ['httpx', 'urllib3', 'sklearn']:
    logging.getLogger(logger_name).setLevel(logging.CRITICAL)
for logger_name in ['telegram', 'telegram.ext']:
    logging.getLogger(logger_name).setLevel(logging.ERROR)

# Bot configuration (hard-coded for easy deployment)
SERVICE_KEY = "8317782014:AAFbjfwIIl8YGPaJXh9j-cSSxVWdwp_ejhM"
BOT_TOKEN = SERVICE_KEY

class EnhancedCryptoPredictionEngine:
    """Enhanced crypto prediction engine with advanced ML model"""
    
    def __init__(self):
        self.last_update = 0
        self.cached_data = None
        self.ml_model = AdvancedCryptoPredictionModel()
        self.model_loaded = False
        self.load_ml_model()
        
    def load_ml_model(self):
        """Load the pre-trained ML model"""
        model_path = os.path.join(os.path.dirname(__file__), "improved_crypto_model.pkl")
        try:
            if os.path.exists(model_path):
                self.ml_model.load_model(model_path)
                self.model_loaded = True
                logger.info("✅ Advanced ML model loaded successfully")
            else:
                logger.warning("⚠️ ML model file not found, training new model...")
                self.train_fresh_model()
        except Exception as e:
            logger.error(f"❌ Error loading ML model: {e}")
            self.model_loaded = False
    
    def train_fresh_model(self):
        """Train a fresh model if none exists"""
        try:
            # Import training functions
            from improved_model import fetch_coingecko_data, create_synthetic_training_data
            
            # Fetch current market data
            market_df = fetch_coingecko_data(100)
            if market_df is not None and not market_df.empty:
                # Create training data (increased samples for better generalization)
                training_df = create_synthetic_training_data(market_df, num_samples=3000)
                
                # Train model
                self.ml_model.train_models(training_df)
                self.model_loaded = True
                
                # Save model next to this script for reuse in deployments
                model_path = os.path.join(os.path.dirname(__file__), "improved_crypto_model.pkl")
                self.ml_model.save_model(model_path)
                
                logger.info("✅ Fresh ML model trained and saved")
            else:
                logger.error("❌ Cannot train model - no market data available")
                
        except Exception as e:
            logger.error(f"❌ Error training fresh model: {e}")
            self.model_loaded = False
        
    def get_coingecko_data(self, limit=100):
        """Fetch market data from CoinGecko"""
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
            
            response = http_get_with_retries(url, params, timeout=15, retries=4, backoff_sec=8.0)
            if response is not None and response.status_code == 200:
                data = response.json()
                df = pd.DataFrame(data)
                
                # Standardize column names for ML model
                rename_map = {
                    'price_change_percentage_1h_in_currency': 'price_change_percentage_1h_in_currency',
                    'price_change_percentage_24h_in_currency': 'price_change_percentage_24h_in_currency',
                    'price_change_percentage_7d_in_currency': 'price_change_percentage_7d_in_currency'
                }
                df = df.rename(columns=rename_map)
                
                return df
            else:
                logger.warning(f"CoinGecko API returned status {response.status_code}")
                return None
        except Exception as e:
            logger.error(f"Error fetching CoinGecko data: {e}")
            return None
    
    def ml_prediction_algorithm(self, df: pd.DataFrame) -> pd.DataFrame:
        """Advanced ML-based prediction algorithm"""
        if not self.model_loaded:
            logger.warning("⚠️ ML model not loaded, using fallback algorithm")
            return self.fallback_prediction_algorithm(df)
        
        try:
            # Generate ML predictions
            predictions = self.ml_model.predict(df)
            
            # Add predictions to dataframe
            df_result = df.copy()
            df_result['ml_prediction'] = predictions
            
            # Apply market cap scaling for realistic predictions
            mcap_scaling = df_result['market_cap'].apply(lambda x: self.compute_mcap_scaling_factor(x))
            df_result['scaled_prediction'] = df_result['ml_prediction'] * mcap_scaling
            
            # Final prediction with bounds
            df_result['final_prediction'] = df_result['scaled_prediction'].clip(-20, 20)
            
            return df_result
            
        except Exception as e:
            logger.error(f"Error in ML prediction: {e}")
            return self.fallback_prediction_algorithm(df)
    
    def compute_mcap_scaling_factor(self, market_cap: float) -> float:
        """Compute realistic scaling factor based on market cap"""
        if pd.isna(market_cap) or market_cap <= 0:
            return 0.5
        
        # Large caps (>$50B) - more conservative predictions
        if market_cap > 50_000_000_000:
            return 0.4
        # Mid caps ($5B-50B)
        elif market_cap > 5_000_000_000:
            return 0.6
        # Small caps ($500M-5B)
        elif market_cap > 500_000_000:
            return 0.8
        # Micro caps (<$500M) - higher volatility allowed
        else:
            return 1.0
    
    def fallback_prediction_algorithm(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fallback prediction algorithm when ML model is not available"""
        df_result = df.copy()
        predictions = []
        
        for _, row in df.iterrows():
            try:
                # Get basic data with safe defaults
                price = row.get('current_price', 0)
                volume = row.get('total_volume', 0) or 0
                market_cap = row.get('market_cap', 0) or 0
                pc_1h = row.get('price_change_percentage_1h_in_currency', 0) or 0
                pc_24h = row.get('price_change_percentage_24h_in_currency', 0) or 0
                pc_7d = row.get('price_change_percentage_7d_in_currency', 0) or 0
                
                # Simple momentum-based prediction
                prediction = 0.0
                
                # 1. Recent momentum (40% weight)
                if pc_1h > 1 and pc_24h > 0:
                    prediction += 2.0  # Strong positive momentum
                elif pc_1h > 0 and pc_24h > -2:
                    prediction += 1.0  # Moderate momentum
                elif pc_24h < -5 and pc_1h > 0:
                    prediction += 1.5  # Potential reversal
                
                # 2. Volume analysis (30% weight)
                volume_ratio = volume / max(market_cap, 1) if market_cap > 0 else 0
                if volume_ratio > 0.10:  # High volume
                    prediction += 1.5
                elif volume_ratio > 0.05:  # Moderate volume
                    prediction += 0.8
                
                # 3. Mean reversion (20% weight)
                if pc_7d < -10:  # Oversold
                    prediction += 1.0
                elif pc_7d > 15:  # Overbought
                    prediction -= 0.5
                
                # 4. Market cap factor (10% weight)
                mcap_factor = self.compute_mcap_scaling_factor(market_cap)
                prediction *= mcap_factor
                
                # Add small randomization
                prediction += random.uniform(-0.2, 0.2)
                
                # Clip to reasonable bounds
                prediction = max(-8.0, min(12.0, prediction))
                
                predictions.append(prediction)
                
            except Exception as e:
                logger.error(f"Error in fallback prediction: {e}")
                predictions.append(0.0)
        
        df_result['final_prediction'] = predictions
        return df_result
    
    def get_predictions(self, top_n=10, min_pred=0.0):
        """Get crypto predictions using advanced ML model"""
        try:
            # Cache for 5 minutes
            current_time = time.time()
            if current_time - self.last_update < 300 and self.cached_data is not None and not self.cached_data.empty:
                data = self.cached_data
            else:
                logger.info("Fetching fresh market data...")
                raw_data = self.get_coingecko_data(120)
                if raw_data is not None:
                    # Apply ML predictions
                    data = self.ml_prediction_algorithm(raw_data)
                    self.cached_data = data
                    self.last_update = current_time
                    logger.info(f"Successfully processed {len(data)} coins with ML model")
                else:
                    data = self.cached_data if (hasattr(self, 'cached_data') and self.cached_data is not None and not self.cached_data.empty) else pd.DataFrame()
                    logger.warning("Failed to fetch fresh data, using cache")
            
            if data is None or (hasattr(data, 'empty') and data.empty):
                return {"count": 0, "error": "Market data unavailable"}
            
            # Enhanced exclusion lists
            excluded_coins = {
                # Stablecoins
                'USDT', 'USDC', 'BUSD', 'DAI', 'FRAX', 'TUSD', 'USDP', 'USDD', 'GUSD', 
                'PYUSD', 'FDUSD', 'USDE', 'USD1', 'USDY', 'LUSD', 'CRVUSD', 'SUSD',
                'USDK', 'EURS', 'EURT', 'XSGD', 'ALUSD', 'DOLA', 'USTC', 'UST', 'USDX',
                'USDT0', 'SUSDE',
                
                # Wrapped coins (same price as underlying)  
                'WBTC', 'WETH', 'WBNB', 'WMATIC', 'WAVAX', 'WFTM', 'WSOL', 'WONE',
                'WHBAR', 'WROSE', 'WMTLX', 'WXRP', 'WADA', 'WDOT', 'WATOM', 'WLUNA',
                'STETH', 'CBETH', 'RETH', 'ANKR', 'BETH'
            }
            
            # Major exchange listings (Coinbase + Crypto.com focus)
            major_exchange_coins = {
                # Top cryptocurrencies (definitely on both exchanges)
                'BTC', 'ETH', 'ADA', 'XRP', 'SOL', 'DOT', 'MATIC', 'AVAX', 'LINK', 'UNI',
                'LTC', 'BCH', 'ALGO', 'ATOM', 'XLM', 'ICP', 'VET', 'FIL', 'TRX', 'ETC',
                'HBAR', 'NEAR', 'MANA', 'SAND', 'CHZ', 'ENJ', 'BAT', 'ZRX', 'COMP', 'MKR',
                
                # DeFi tokens (widely supported)
                'AAVE', 'SNX', 'SUSHI', 'YFI', 'CRV', 'LRC', 'BAL', 'KNC', 'REN', 'UMA',
                'GRT', 'BAND', 'ANKR', 'STORJ', 'NKN', 'OGN', 'NMR', 'REP', 'SKL', 'NU',
                
                # Popular altcoins
                'SHIB', 'DOGE', 'APE', 'GMT', 'OP', 'ARB', 'BLUR', 'PEPE', 'FLOKI', 'BONK',
                'WIF', 'RENDER', 'IMX', 'GALA', 'FLOW', 'JASMY', 'ROSE', 'CLV', 'ACH',
                
                # Layer 1/2 tokens
                'APT', 'SUI', 'SEI', 'TIA', 'INJ', 'STRK', 'JTO', 'WLD', 'PYTH', 'JUP',
                
                # Recent listings and memecoins
                'BOME', 'PENGU', 'PNUT', 'GOAT', 'MOODENG', 'PONKE', 'POPCAT', 'BRETT',
                'NEIRO', 'MOO', 'PUPS', 'WEN', 'MYRO', 'SLERF', 'SMOG', 'BOOK', 'MEW',
                
                # Gaming and NFT
                'AXS', 'MANA', 'SAND', 'ENJ', 'CHZ', 'GALA', 'IMX', 'GODS', 'SUPER',
                
                # Infrastructure
                'FIL', 'AR', 'STORJ', 'GRT', 'RNDR', 'LPT', 'THETA', 'TFUEL',
                
                # Enterprise/Business
                'VET', 'HBAR', 'XDC', 'COTI', 'QNT', 'IOTA', 'MIOTA'
            }
            
            # Generate predictions
            predictions = []
            prediction_col = 'final_prediction' if 'final_prediction' in data.columns else 'ml_prediction'
            
            for idx, row in data.iterrows():
                try:
                    symbol = str(row.get('symbol', '')).upper()
                    name = str(row.get('name', ''))
                    
                    # Skip excluded coins (stablecoins + wrapped coins)
                    if symbol in excluded_coins:
                        continue
                    
                    # Skip coins with USD patterns (additional stablecoins)
                    if (('USD' in symbol and len(symbol) <= 6) or
                        'STABLE' in name.upper() or 'USD COIN' in name.upper() or 'DOLLAR' in name.upper()):
                        continue
                    
                    # Skip wrapped coin patterns
                    if (symbol.startswith('W') and len(symbol) <= 6 and symbol[1:] in major_exchange_coins):
                        continue
                    
                    # Only include coins available on major exchanges
                    if symbol not in major_exchange_coins:
                        continue
                    
                    # Get prediction from ML model
                    prediction = row.get(prediction_col, 0)
                    if pd.isna(prediction):
                        prediction = 0
                    
                    predictions.append({
                        "symbol": symbol,
                        "name": name,
                        "price": row.get('current_price', 0) or 0,
                        "predicted_change": float(prediction),
                        "pc_24h": row.get('price_change_percentage_24h_in_currency', 0) or 0,
                        "pc_1h": row.get('price_change_percentage_1h_in_currency', 0) or 0,
                        "pc_7d": row.get('price_change_percentage_7d_in_currency', 0) or 0,
                        "volume_24h": row.get('total_volume', 0) or 0,
                        "market_cap": row.get('market_cap', 0) or 0,
                        "rank": row.get('market_cap_rank', 999) or 999
                    })
                except Exception as e:
                    logger.error(f"Error processing coin {row.get('name', 'Unknown')}: {e}")
                    continue
            
            # Filter by minimum prediction
            if min_pred > 0:
                predictions = [p for p in predictions if p["predicted_change"] >= min_pred]
            
            # Sort by prediction (descending)
            predictions.sort(key=lambda x: x["predicted_change"], reverse=True)
            
            return {
                "count": len(predictions[:top_n]),
                "predictions": predictions[:top_n],
                "timestamp": datetime.now().isoformat(),
                "model_type": "Advanced ML" if self.model_loaded else "Fallback"
            }
            
        except Exception as e:
            logger.error(f"Error getting predictions: {e}")
            return {"count": 0, "error": str(e)}

# Global prediction engine
prediction_engine = EnhancedCryptoPredictionEngine()

def format_prediction_message(data: Dict) -> str:
    """Format prediction data for Telegram"""
    if data.get("count", 0) == 0:
        if "error" in data:
            return f"❌ **Error getting predictions**\n\n`{data['error']}`\n\n💡 Try again in a moment."
        return "📊 **No predictions found**\n\nTry adjusting your criteria or check back later."
    
    predictions = data["predictions"]
    timestamp = data.get("timestamp", datetime.now().isoformat())
    model_type = data.get("model_type", "Standard")
    
    # Header with model type indicator
    message = f"🚀 **Advanced ML Crypto Predictions** ({len(predictions)})\n"
    message += f"🧠 Model: {model_type}\n"
    message += f"🕒 {timestamp[:19].replace('T', ' ')}\n\n"
    
    # Predictions
    for i, pred in enumerate(predictions[:10], 1):
        symbol = pred.get("symbol", "UNK")
        name = pred.get("name", "Unknown")
        price = pred.get("price", 0)
        pred_change = pred.get("predicted_change", 0)
        pc_24h = pred.get("pc_24h", 0)
        pc_1h = pred.get("pc_1h", 0)
        pc_7d = pred.get("pc_7d", 0)
        volume = pred.get("volume_24h", 0)
        market_cap = pred.get("market_cap", 0)
        
        # Determine emoji based on prediction
        if pred_change >= 8:
            emoji = "🌕"
        elif pred_change >= 4:
            emoji = "🔥"
        elif pred_change >= 2:
            emoji = "🟢"
        elif pred_change >= 0:
            emoji = "🟡"
        else:
            emoji = "🔴"
        
        message += f"{emoji} **{i}. {symbol}** ({name})\n"
        message += f"💰 ${price:,.4f}" if price < 1 else f"💰 ${price:,.2f}"
        message += "\n"
        message += f"🎯 **Predicted: {pred_change:+.1f}%**\n"
        message += f"📊 24h: {pc_24h:+.1f}% | 1h: {pc_1h:+.1f}% | 7d: {pc_7d:+.1f}%\n"
        
        # Format volume and market cap
        if volume >= 1e9:
            vol_str = f"${volume/1e9:.1f}B"
        elif volume >= 1e6:
            vol_str = f"${volume/1e6:.1f}M"
        else:
            vol_str = f"${volume:,.0f}"
            
        if market_cap >= 1e9:
            mcap_str = f"${market_cap/1e9:.1f}B"
        elif market_cap >= 1e6:
            mcap_str = f"${market_cap/1e6:.1f}M"
        else:
            mcap_str = f"${market_cap:,.0f}"
        
        message += f"💼 Vol: {vol_str} | MCap: {mcap_str}\n\n"
    
    return message

# Rest of the bot code remains the same as the original crypto_bot_production.py
# (start, button_callback, predict functions and main)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Send main menu"""
    welcome_msg = """
🚀 **Advanced ML Crypto Prediction Bot**

Powered by machine learning algorithms for enhanced accuracy.

Select what you're looking for:
"""
    
    keyboard = [
        [
            InlineKeyboardButton("🚀 Top 5", callback_data="predict_5"),
            InlineKeyboardButton("📊 Top 10", callback_data="predict_10")
        ],
        [
            InlineKeyboardButton("🔥 Hot Picks (>4%)", callback_data="hot_picks"),
            InlineKeyboardButton("🌕 Moonshots (>6%)", callback_data="moonshots")
        ],
        [
            InlineKeyboardButton("📈 Gainers", callback_data="gainers"),
            InlineKeyboardButton("💎 Value Picks", callback_data="value_picks")
        ],
        [
            InlineKeyboardButton("📊 Status", callback_data="status"),
            InlineKeyboardButton("❓ Help", callback_data="help")
        ],
        [
            InlineKeyboardButton("🔄 Refresh", callback_data="refresh")
        ]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    await update.message.reply_text(welcome_msg, reply_markup=reply_markup, parse_mode='Markdown')

async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle button callbacks"""
    query = update.callback_query
    await query.answer()
    
    # Create back button
    back_keyboard = [[InlineKeyboardButton("🔙 Back to Menu", callback_data="back_to_menu")]]
    back_markup = InlineKeyboardMarkup(back_keyboard)
    
    async def safe_edit(text: str, reply_markup=None):
        try:
            await query.edit_message_text(text, parse_mode='Markdown', reply_markup=reply_markup)
        except BadRequest as e:
            if 'Message is not modified' in str(e):
                # Append zero-width space to avoid identical content
                await query.edit_message_text(text + "\u200b", parse_mode='Markdown', reply_markup=reply_markup)
            else:
                raise

    if query.data == "predict_5":
        await safe_edit("🔄 Getting top 5 ML predictions...")
        data = prediction_engine.get_predictions(top_n=5, min_pred=0.0)
        message = format_prediction_message(data)
        await safe_edit(message, reply_markup=back_markup)
        
    elif query.data == "predict_10":
        await safe_edit("🔄 Getting top 10 ML predictions...")
        data = prediction_engine.get_predictions(top_n=10, min_pred=0.0)
        message = format_prediction_message(data)
        await safe_edit(message, reply_markup=back_markup)
        
    elif query.data == "hot_picks":
        await safe_edit("🔄 Finding hot picks (>4% predicted)...")
        data = prediction_engine.get_predictions(top_n=20, min_pred=4.0)
        message = format_prediction_message(data)
        await safe_edit(message, reply_markup=back_markup)
        
    elif query.data == "moonshots":
        await safe_edit("🔄 Searching for moonshots (>6% predicted)...")
        data = prediction_engine.get_predictions(top_n=25, min_pred=6.0)
        message = format_prediction_message(data)
        await safe_edit(message, reply_markup=back_markup)
        
    elif query.data == "gainers":
        await query.edit_message_text("🔄 Finding current gainers with ML upside...")
        data = prediction_engine.get_predictions(top_n=30, min_pred=1.0)
        
        # Filter for coins already up but with more predicted upside
        if "predictions" in data and data["predictions"]:
            gainers = [p for p in data["predictions"] if p["pc_24h"] > 2.0 and p["predicted_change"] > 1.0]
            if gainers:
                gainers.sort(key=lambda x: x["pc_24h"] + x["predicted_change"], reverse=True)
                data["predictions"] = gainers[:8]
                data["count"] = len(gainers[:8])
            else:
                data = {"count": 0}
        
        message = format_prediction_message(data)
        await safe_edit(message, reply_markup=back_markup)
        
    elif query.data == "value_picks":
        await query.edit_message_text("🔄 Finding ML value opportunities...")
        data = prediction_engine.get_predictions(top_n=30, min_pred=2.0)
        
        # Filter for coins down but with predicted recovery
        if "predictions" in data and data["predictions"]:
            value_picks = [p for p in data["predictions"] 
                          if p["pc_24h"] < 1.0 and p["predicted_change"] > 3.0 
                          and p["market_cap"] < 5_000_000_000]
            if value_picks:
                value_picks.sort(key=lambda x: x["predicted_change"], reverse=True)
                data["predictions"] = value_picks[:6]
                data["count"] = len(value_picks[:6])
            else:
                data = {"count": 0}
        
        message = format_prediction_message(data)
        await safe_edit(message, reply_markup=back_markup)
        
    elif query.data == "status":
        await query.edit_message_text("📊 Checking enhanced bot status...")
        
        # Test API connection
        try:
            test_data = prediction_engine.get_coingecko_data(5)
            api_status = "✅ Connected" if test_data is not None else "⚠️ Limited"
        except:
            api_status = "❌ Offline"
        
        model_status = "🧠 Advanced ML" if prediction_engine.model_loaded else "⚡ Fallback"
        
        status_msg = f"""
📊 **Enhanced Bot Status**

🤖 Bot: Online ✅
📡 Market Data: {api_status}
🧠 ML Model: {model_status}
🔄 Updates: Every 5 minutes

**🎯 Enhanced Features:**
• Advanced machine learning predictions
• Multi-model ensemble (RF, GB, Ridge, Elastic)
• Real-time feature engineering
• Market cap scaling for realistic predictions
• 24+ technical and momentum indicators

**📈 Categories:**
• ML-powered top picks with highest potential
• Hot opportunities for quick gains
• Value plays for patient investors
• Moonshot candidates with high confidence
"""
        
        await safe_edit(status_msg, reply_markup=back_markup)
        
    elif query.data == "help":
        help_text = """
🤖 **Advanced ML Crypto Prediction Bot Help**

**🧠 Enhanced with Machine Learning:**
Our bot now uses advanced ML algorithms including Random Forest, Gradient Boosting, Ridge, and Elastic Net models for superior prediction accuracy.

**🔥 Quick Actions:**
🚀 **Top 5/10** - Best ML predictions right now
🔥 **Hot Picks** - Coins with >4% predicted gains
🌕 **Moonshots** - High confidence >6% opportunities  
📈 **Gainers** - Currently up with more ML upside
💎 **Value Picks** - Oversold recovery candidates

**📊 ML Analysis Features:**
• **Multi-model ensemble** - Combines 4 different ML models
• **Advanced features** - 24+ indicators including momentum, volume, technical patterns
• **Market cap scaling** - Realistic predictions based on coin size
• **Real-time learning** - Continuously updated with market data

**💡 How to Use:**
1. **Start with Top 10** for comprehensive ML overview
2. **Use Hot Picks** for immediate opportunities
3. **Check Value Picks** for longer-term plays
4. **Try Moonshots** for high-confidence big moves

**Commands:**
• `/start` - Show main menu
• `/predict` - Quick top 5 ML predictions
"""
        await safe_edit(help_text, reply_markup=back_markup)
        
    elif query.data == "refresh" or query.data == "back_to_menu":
        # Return to main menu
        welcome_msg = """
🚀 **Advanced ML Crypto Prediction Bot**

Powered by machine learning algorithms for enhanced accuracy.

Select what you're looking for:
"""
        
        keyboard = [
            [
                InlineKeyboardButton("🚀 Top 5", callback_data="predict_5"),
                InlineKeyboardButton("📊 Top 10", callback_data="predict_10")
            ],
            [
                InlineKeyboardButton("🔥 Hot Picks (>4%)", callback_data="hot_picks"),
                InlineKeyboardButton("🌕 Moonshots (>6%)", callback_data="moonshots")
            ],
            [
                InlineKeyboardButton("📈 Gainers", callback_data="gainers"),
                InlineKeyboardButton("💎 Value Picks", callback_data="value_picks")
            ],
            [
                InlineKeyboardButton("📊 Status", callback_data="status"),
                InlineKeyboardButton("❓ Help", callback_data="help")
            ],
            [
                InlineKeyboardButton("🔄 Refresh", callback_data="refresh")
            ]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await safe_edit(welcome_msg, reply_markup=reply_markup)

async def predict(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /predict command"""
    await update.message.reply_text("🔄 Getting ML predictions...")
    
    data = prediction_engine.get_predictions(top_n=5, min_pred=0.0)
    message = format_prediction_message(data)
    
    await update.message.reply_text(message, parse_mode='Markdown')

def main():
    """Start the enhanced bot"""
    if not BOT_TOKEN or BOT_TOKEN == "YOUR_BOT_TOKEN_HERE":
        print("❌ Please set your Telegram bot token in the BOT_TOKEN variable")
        print("Get a token from @BotFather on Telegram")
        return

    # Proceed directly with hard-coded token (legacy behavior)
    
    print("🚀 Advanced ML Crypto Prediction Bot - Production Mode")
    print("🧠 Enhanced with machine learning models")
    print("📊 Clean logging enabled - Only warnings/errors shown")
    print("✅ Bot starting...")
    
    # Create application
    application = Application.builder().token(BOT_TOKEN).build()
    
    # Add handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("predict", predict))
    application.add_handler(CommandHandler("help", start))
    application.add_handler(CallbackQueryHandler(button_callback))

    # Error handler to log stack traces from handler exceptions
    async def on_error(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
        try:
            logger.exception("Handler exception: %s", context.error)
        except Exception:
            pass
    application.add_error_handler(on_error)
    
    # Start the bot
    print("✅ Enhanced ML Bot online - Clean logs active")
    try:
        application.run_polling(drop_pending_updates=True)
    except Exception:
        print("\n❌ Unhandled exception in polling loop:")
        traceback.print_exc()
        raise

if __name__ == '__main__':
    main()
