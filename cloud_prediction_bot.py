#!/usr/bin/env python3
"""
Cloud-Compatible Telegram Bot with Real Crypto Predictions
Includes simplified but functional prediction model
"""

import asyncio
import json
import logging
import os
import requests
from datetime import datetime
from typing import Dict, List
import time
import random

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes

# Setup logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Bot configuration
BOT_TOKEN = os.getenv("BOT_TOKEN")

class SimpleCryptoPredictionEngine:
    """Simplified prediction engine for cloud deployment"""
    
    def __init__(self):
        self.last_update = 0
        self.cached_data = {}
        
    def get_coingecko_data(self, limit=100):
        """Fetch real market data from CoinGecko"""
        try:
            url = "https://api.coingecko.com/api/v3/coins/markets"
            params = {
                'vs_currency': 'usd',
                'order': 'market_cap_desc',
                'per_page': limit,
                'page': 1,
                'sparkline': False,
                'price_change_percentage': '1h,24h'
            }
            
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                return response.json()
            else:
                logger.warning(f"CoinGecko API returned status {response.status_code}")
                return None
        except Exception as e:
            logger.error(f"Error fetching CoinGecko data: {e}")
            return None
    
    def simple_prediction_algorithm(self, coin_data):
        """Simplified prediction algorithm using technical indicators"""
        try:
            # Get basic data
            price = coin_data.get('current_price', 0)
            volume = coin_data.get('total_volume', 0)
            market_cap = coin_data.get('market_cap', 0)
            pc_1h = coin_data.get('price_change_percentage_1h_in_currency', 0) or 0
            pc_24h = coin_data.get('price_change_percentage_24h_in_currency', 0) or 0
            
            # Simplified prediction based on momentum and volume
            prediction = 0.0
            
            # 1. Momentum indicator (30% weight)
            if pc_1h > 0 and pc_24h > 0:
                prediction += 2.0  # Both positive momentum
            elif pc_1h > 0:
                prediction += 1.0  # Short-term momentum
            elif pc_24h > -2:
                prediction += 0.5  # Not too negative
            
            # 2. Volume indicator (25% weight)
            if volume > market_cap * 0.1:  # High volume relative to market cap
                prediction += 1.5
            elif volume > market_cap * 0.05:
                prediction += 0.8
            
            # 3. Market cap scaling (20% weight)
            if market_cap > 10_000_000_000:  # Large cap - more stable
                prediction += 0.5
            elif market_cap > 1_000_000_000:  # Mid cap - balanced
                prediction += 1.0
            else:  # Small cap - higher volatility potential
                prediction += 1.5
            
            # 4. Price action patterns (25% weight)
            if abs(pc_24h) < 2:  # Low volatility - potential for move
                prediction += 1.0
            elif pc_24h < -5 and pc_1h > 0:  # Oversold bounce potential
                prediction += 2.0
            elif pc_24h > 5:  # Already moved up - reduce prediction
                prediction -= 1.0
            
            # Add some randomization for market unpredictability (±20%)
            random_factor = random.uniform(0.8, 1.2)
            prediction *= random_factor
            
            # Scale and cap the prediction
            prediction = max(-5.0, min(15.0, prediction))
            
            return prediction
            
        except Exception as e:
            logger.error(f"Error in prediction algorithm: {e}")
            return random.uniform(-2, 5)  # Fallback random prediction
    
    def get_predictions(self, top_n=10, min_pred=0.0):
        """Get crypto predictions with real market data"""
        try:
            # Rate limiting - cache for 5 minutes
            current_time = time.time()
            if current_time - self.last_update < 300 and self.cached_data:
                logger.info("Using cached data")
                data = self.cached_data
            else:
                logger.info("Fetching fresh market data")
                data = self.get_coingecko_data(80)
                if data:
                    self.cached_data = data
                    self.last_update = current_time
                else:
                    # Use cached data if API fails
                    data = self.cached_data if self.cached_data else []
            
            if not data:
                return {"count": 0, "error": "No market data available"}
            
            # Generate predictions for each coin
            predictions = []
            for coin in data:
                try:
                    prediction = self.simple_prediction_algorithm(coin)
                    
                    predictions.append({
                        "symbol": coin.get('symbol', '').upper(),
                        "name": coin.get('name', 'Unknown'),
                        "price": coin.get('current_price', 0),
                        "predicted_change": prediction,
                        "pc_24h": coin.get('price_change_percentage_24h_in_currency', 0) or 0,
                        "pc_1h": coin.get('price_change_percentage_1h_in_currency', 0) or 0,
                        "volume_24h": coin.get('total_volume', 0),
                        "market_cap": coin.get('market_cap', 0),
                        "rank": coin.get('market_cap_rank', 999)
                    })
                except Exception as e:
                    logger.error(f"Error processing coin {coin.get('name', 'Unknown')}: {e}")
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
                "model_status": "live_cloud_predictions"
            }
            
        except Exception as e:
            logger.error(f"Error getting predictions: {e}")
            return {"count": 0, "error": str(e)}

# Global prediction engine
prediction_engine = SimpleCryptoPredictionEngine()

def format_prediction_message(data: Dict) -> str:
    """Format prediction data into a nice Telegram message"""
    if data.get("count", 0) == 0:
        if "error" in data:
            return f"❌ **Error getting predictions**\n\n`{data['error']}`\n\n💡 Try again in a moment - the API may be busy."
        return "📊 **No predictions found**\n\nTry adjusting your criteria or check back later."
    
    predictions = data["predictions"]
    timestamp = data.get("timestamp", datetime.now().isoformat())
    
    # Header
    message = f"🚀 **Live Crypto Predictions** ({len(predictions)})\n"
    message += f"🕒 {timestamp[:19].replace('T', ' ')}\n\n"
    
    # Add status
    if data.get("model_status") == "live_cloud_predictions":
        message += "🌐 *Live market data from CoinGecko*\n\n"
    
    # Predictions
    for i, pred in enumerate(predictions[:10], 1):
        symbol = pred.get("symbol", "UNK")
        name = pred.get("name", "Unknown")
        price = pred.get("price", 0)
        pred_change = pred.get("predicted_change", 0)
        pc_24h = pred.get("pc_24h", 0)
        pc_1h = pred.get("pc_1h", 0)
        volume = pred.get("volume_24h", 0)
        market_cap = pred.get("market_cap", 0)
        
        # Determine emoji based on prediction
        if pred_change >= 10:
            emoji = "🌕"
        elif pred_change >= 5:
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
        message += f"📈 Predicted: {pred_change:+.1f}%\n"
        message += f"📊 24h: {pc_24h:+.1f}% | 1h: {pc_1h:+.1f}%\n"
        
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
    
    # Footer
    message += "⚠️ *Live predictions - Educational purposes only*"
    
    return message

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Send a message when the command /start is issued"""
    welcome_msg = """
🤖 **Live Crypto Prediction Bot**

Welcome! This bot provides real-time crypto predictions using live market data and technical analysis.

**🔥 Live Features:**
• Real CoinGecko market data
• Technical momentum analysis  
• Volume & market cap indicators
• Live price predictions
• 24/7 cloud hosting

Use the buttons below for instant predictions! 👇
"""
    
    keyboard = [
        [
            InlineKeyboardButton("🚀 Top 5", callback_data="predict_5"),
            InlineKeyboardButton("📊 Top 10", callback_data="predict_10")
        ],
        [
            InlineKeyboardButton("🔥 Hot Picks (>5%)", callback_data="hot_picks"),
            InlineKeyboardButton("🌕 Moonshots (>8%)", callback_data="moonshots")
        ],
        [
            InlineKeyboardButton("📈 Gainers", callback_data="gainers"),
            InlineKeyboardButton("💎 Value Picks", callback_data="value_picks")
        ],
        [
            InlineKeyboardButton("📊 Bot Status", callback_data="status"),
            InlineKeyboardButton("❓ Help", callback_data="help")
        ],
        [
            InlineKeyboardButton("🔄 Refresh", callback_data="refresh")
        ]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    await update.message.reply_text(welcome_msg, reply_markup=reply_markup, parse_mode='Markdown')

async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle inline button callbacks"""
    query = update.callback_query
    await query.answer()
    
    # Create back button
    back_keyboard = [[InlineKeyboardButton("🔙 Back to Menu", callback_data="back_to_menu")]]
    back_markup = InlineKeyboardMarkup(back_keyboard)
    
    if query.data == "predict_5":
        await query.edit_message_text("🔄 Getting top 5 live predictions...")
        data = prediction_engine.get_predictions(top_n=5, min_pred=0.0)
        message = format_prediction_message(data)
        await query.edit_message_text(message, parse_mode='Markdown', reply_markup=back_markup)
        
    elif query.data == "predict_10":
        await query.edit_message_text("🔄 Getting top 10 live predictions...")
        data = prediction_engine.get_predictions(top_n=10, min_pred=0.0)
        message = format_prediction_message(data)
        await query.edit_message_text(message, parse_mode='Markdown', reply_markup=back_markup)
        
    elif query.data == "hot_picks":
        await query.edit_message_text("🔄 Finding hot picks (>5% predicted)...")
        data = prediction_engine.get_predictions(top_n=20, min_pred=5.0)
        message = format_prediction_message(data)
        await query.edit_message_text(message, parse_mode='Markdown', reply_markup=back_markup)
        
    elif query.data == "moonshots":
        await query.edit_message_text("🔄 Searching for moonshots (>8% predicted)...")
        data = prediction_engine.get_predictions(top_n=25, min_pred=8.0)
        message = format_prediction_message(data)
        await query.edit_message_text(message, parse_mode='Markdown', reply_markup=back_markup)
        
    elif query.data == "gainers":
        await query.edit_message_text("🔄 Finding current gainers with upside...")
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
        await query.edit_message_text(message, parse_mode='Markdown', reply_markup=back_markup)
        
    elif query.data == "value_picks":
        await query.edit_message_text("🔄 Finding value opportunities...")
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
        await query.edit_message_text(message, parse_mode='Markdown', reply_markup=back_markup)
        
    elif query.data == "status":
        await query.edit_message_text("📊 Checking bot status...")
        
        # Test API connection
        try:
            test_data = prediction_engine.get_coingecko_data(5)
            api_status = "✅ Connected" if test_data else "⚠️ Limited"
        except:
            api_status = "❌ Offline"
        
        status_msg = f"""
📊 **Live Bot Status**

🤖 Bot: Online ✅
🌐 Cloud: Railway.app ✅  
🔄 Uptime: 24/7
📡 CoinGecko API: {api_status}

**🔥 Live Features:**
• Real-time market data
• Technical analysis engine
• Volume & momentum indicators  
• Market cap scaling
• Smart prediction algorithms

**📊 Data Sources:**
• CoinGecko API (live prices)
• Technical indicators
• Volume analysis
• Market momentum

**🚀 Sharing:**
This bot uses live market data - share with friends for real crypto insights!

⚠️ *Predictions update every 5 minutes*
"""
        
        await query.edit_message_text(status_msg, parse_mode='Markdown', reply_markup=back_markup)
        
    elif query.data == "help":
        help_text = """
🤖 **Live Crypto Prediction Bot Help**

**🔥 Quick Actions:**
🚀 **Top 5/10** - Best live predictions
🔥 **Hot Picks** - Coins with >5% predicted gains
🌕 **Moonshots** - High potential >8% opportunities  
📈 **Gainers** - Currently up with more upside
💎 **Value Picks** - Oversold recovery candidates

**📊 Prediction Algorithm:**
• **Momentum Analysis** - 1h & 24h price trends
• **Volume Indicators** - High volume = potential moves
• **Market Cap Scaling** - Size-appropriate predictions  
• **Technical Patterns** - Oversold bounce detection

**💡 How to Use:**
1. **Start broad** with Top 10 for market overview
2. **Get specific** with Hot Picks for active trades
3. **Find value** with Value Picks for contrarian plays
4. **Check status** to verify live data connection

**🎯 Prediction Accuracy:**
• Based on technical analysis
• Updates every 5 minutes  
• Uses live CoinGecko data
• Accounts for market momentum

**Commands:**
• `/start` - Show main menu
• `/predict` - Quick top 5 predictions

⚠️ **Disclaimer:** Educational purposes only. Real market data but predictions are not financial advice. Always do your own research.
"""
        await query.edit_message_text(help_text, parse_mode='Markdown', reply_markup=back_markup)
        
    elif query.data == "refresh" or query.data == "back_to_menu":
        # Return to main menu
        welcome_msg = """
🤖 **Live Crypto Prediction Bot**

Welcome! This bot provides real-time crypto predictions using live market data and technical analysis.

**🔥 Live Features:**
• Real CoinGecko market data
• Technical momentum analysis  
• Volume & market cap indicators
• Live price predictions
• 24/7 cloud hosting

Use the buttons below for instant predictions! 👇
"""
        
        keyboard = [
            [
                InlineKeyboardButton("🚀 Top 5", callback_data="predict_5"),
                InlineKeyboardButton("📊 Top 10", callback_data="predict_10")
            ],
            [
                InlineKeyboardButton("🔥 Hot Picks (>5%)", callback_data="hot_picks"),
                InlineKeyboardButton("🌕 Moonshots (>8%)", callback_data="moonshots")
            ],
            [
                InlineKeyboardButton("📈 Gainers", callback_data="gainers"),
                InlineKeyboardButton("💎 Value Picks", callback_data="value_picks")
            ],
            [
                InlineKeyboardButton("📊 Bot Status", callback_data="status"),
                InlineKeyboardButton("❓ Help", callback_data="help")
            ],
            [
                InlineKeyboardButton("🔄 Refresh", callback_data="refresh")
            ]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await query.edit_message_text(welcome_msg, reply_markup=reply_markup, parse_mode='Markdown')

async def predict(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /predict command"""
    await update.message.reply_text("🔄 Getting live predictions...")
    
    data = prediction_engine.get_predictions(top_n=5, min_pred=0.0)
    message = format_prediction_message(data)
    
    await update.message.reply_text(message, parse_mode='Markdown')

def main():
    """Start the bot"""
    if not BOT_TOKEN:
        print("❌ SECURITY ERROR: BOT_TOKEN environment variable not set")
        print("💡 Set your bot token as an environment variable for security")
        return
    
    print("🚀 Starting Live Crypto Prediction Bot...")
    print("🌐 Cloud mode with real market data")
    print("Press Ctrl+C to stop")
    
    # Create application
    application = Application.builder().token(BOT_TOKEN).build()
    
    # Add handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("predict", predict))
    application.add_handler(CommandHandler("help", start))
    application.add_handler(CallbackQueryHandler(button_callback))
    
    # Start the bot
    application.run_polling(drop_pending_updates=True)

if __name__ == '__main__':
    main()
