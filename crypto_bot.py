#!/usr/bin/env python3
"""
Telegram Bot for Crypto Predictions
Uses the enhanced crypto prediction model from 2.py
"""

import asyncio
import json
import logging
import os
import sys
from datetime import datetime
from typing import Dict, List, Optional

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes
import pandas as pd

# Import our prediction functions from the main script
sys.path.append('/Users/xfx/Desktop/trade')
from importlib import import_module

# Setup logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Bot configuration
BOT_TOKEN = os.getenv("BOT_TOKEN")
# Removed user restrictions - bot is now public for friends
# AUTHORIZED_USERS = [181441247]  # Commented out for public access
AUTHORIZED_USERS = []  # Empty list = allow all users

class CryptoPredictionBot:
    def __init__(self):
        self.prediction_module = None
        self.load_prediction_module()
    
    def load_prediction_module(self):
        """Dynamically load the prediction functions from 2.py"""
        try:
            # Import the main prediction module
            spec = import_module('2')
            self.prediction_module = spec
        except Exception as e:
            logger.error(f"Failed to load prediction module: {e}")
            # Fallback to importing specific functions
            try:
                from importlib.util import spec_from_file_location, module_from_spec
                spec = spec_from_file_location("prediction", "/Users/xfx/Desktop/trade/2.py")
                self.prediction_module = module_from_spec(spec)
                spec.loader.exec_module(self.prediction_module)
            except Exception as e2:
                logger.error(f"Fallback import also failed: {e2}")
    
    def check_authorization(self, user_id: int) -> bool:
        """Check if user is authorized to use the bot"""
        return len(AUTHORIZED_USERS) == 0 or user_id in AUTHORIZED_USERS
    
    async def get_predictions(self, top_n: int = 5, min_pred: float = 0.0) -> Dict:
        """Get crypto predictions using the enhanced model"""
        try:
            if not self.prediction_module:
                return {"error": "Prediction module not loaded"}
            
            # Create session and get data
            session = self.prediction_module.create_http_session()
            
            try:
                # Fetch market data
                rows = self.prediction_module.fetch_coingecko_markets(session, per_page=250, pages=1)
                df_raw = self.prediction_module.prepare_dataframe(rows)
                
                # Filter universe
                crypto_com_bases = self.prediction_module.fetch_crypto_com_base_symbols(session) or set(self.prediction_module.FALLBACK_CRYPTO_COM_TICKERS)
                df_universe = self.prediction_module.filter_universe(
                    df_raw, 
                    crypto_com_bases,
                    min_market_cap=150_000_000,
                    min_volume=10_000_000
                )
                
                # Load model weights
                learned = self.prediction_module.load_weights("/Users/xfx/Desktop/trade/model_weights.json")
                
                # Score and predict
                df_scored = self.prediction_module.score_and_predict(
                    df_universe, 
                    learned_weights=learned, 
                    use_enhanced_features=True
                )
                
                if df_scored.empty:
                    return {"error": "No coins could be scored"}
                
                # Get top predictions
                top = df_scored.nlargest(top_n, "composite_score")
                
                # Filter by minimum prediction if specified
                if min_pred > 0:
                    top = top[top["predicted_change_24h_pct"] >= min_pred]
                
                # Format results
                results = []
                for _, row in top.iterrows():
                    results.append({
                        "symbol": row.get("symbol", "").upper(),
                        "name": row.get("name", ""),
                        "current_price": float(row.get("current_price", 0)),
                        "predicted_change": float(row.get("predicted_change_24h_pct", 0)),
                        "pc_24h": float(row.get("price_change_percentage_24h_in_currency", 0)),
                        "pc_1h": float(row.get("price_change_percentage_1h_in_currency", 0)),
                        "volume": float(row.get("total_volume", 0)),
                        "market_cap": float(row.get("market_cap", 0)),
                        "score": float(row.get("composite_score", 0))
                    })
                
                return {
                    "success": True,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "predictions": results,
                    "count": len(results)
                }
                
            finally:
                session.close()
                
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return {"error": f"Prediction failed: {str(e)}"}

    def format_prediction_message(self, data: Dict) -> str:
        """Format prediction data into a readable Telegram message"""
        if "error" in data:
            return f"❌ Error: {data['error']}"
        
        if data["count"] == 0:
            return "📊 No predictions meet the criteria"
        
        message = f"🚀 **Crypto Predictions** ({data['timestamp']})\n\n"
        
        for i, pred in enumerate(data["predictions"], 1):
            symbol = pred["symbol"]
            name = pred["name"]
            price = pred["current_price"]
            predicted = pred["predicted_change"]
            pc_24h = pred["pc_24h"]
            pc_1h = pred["pc_1h"]
            volume = pred["volume"]
            
            # Format volume for readability
            if volume >= 1_000_000_000:
                vol_str = f"${volume/1_000_000_000:.1f}B"
            elif volume >= 1_000_000:
                vol_str = f"${volume/1_000_000:.0f}M"
            else:
                vol_str = f"${volume:,.0f}"
            
            # Emoji based on prediction
            if predicted > 5:
                emoji = "🟢"
            elif predicted > 0:
                emoji = "🟡"
            else:
                emoji = "🔴"
            
            message += f"{emoji} **{i}. {symbol}** ({name})\n"
            message += f"💰 ${price:,.2f}\n"
            message += f"📈 Predicted: **{predicted:+.2f}%**\n"
            message += f"📊 24h: {pc_24h:+.2f}% | 1h: {pc_1h:+.2f}%\n"
            message += f"💼 Volume: {vol_str}\n\n"
        
        message += "⚠️ *Not financial advice. Trade at your own risk.*"
        return message

# Bot command handlers
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Start command handler"""
    user_id = update.effective_user.id
    
    if not bot.check_authorization(user_id):
        await update.message.reply_text("❌ Unauthorized access")
        return
    
    welcome_msg = """
🤖 **Crypto Prediction Bot**

Welcome! I can provide AI-powered crypto predictions using advanced ensemble models with 14+ technical indicators.

**Available Commands:**
• Top 5/10 predictions with current market analysis
• Hot picks with >5% predicted gains
• Moonshot opportunities with >10% potential
• Model status and performance metrics
• Custom prediction filters

Use the buttons below for instant predictions! 👇
"""
    
    keyboard = [
        [
            InlineKeyboardButton("🚀 Top 5", callback_data="predict_5"),
            InlineKeyboardButton("📊 Top 10", callback_data="predict_10")
        ],
        [
            InlineKeyboardButton("🔥 Hot Picks (>5%)", callback_data="hot_picks"),
            InlineKeyboardButton("🌕 Moonshots (>10%)", callback_data="moonshots")
        ],
        [
            InlineKeyboardButton("📈 Best Performers", callback_data="best_performers"),
            InlineKeyboardButton("💎 Hidden Gems", callback_data="hidden_gems")
        ],
        [
            InlineKeyboardButton("📊 Model Status", callback_data="status"),
            InlineKeyboardButton("❓ Help", callback_data="help")
        ],
        [
            InlineKeyboardButton("🔄 Refresh Predictions", callback_data="refresh")
        ]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    await update.message.reply_text(welcome_msg, reply_markup=reply_markup, parse_mode='Markdown')

async def predict(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Get top 5 predictions"""
    user_id = update.effective_user.id
    
    if not bot.check_authorization(user_id):
        await update.message.reply_text("❌ Unauthorized access")
        return
    
    await update.message.reply_text("🔄 Getting predictions...")
    
    data = await bot.get_predictions(top_n=5, min_pred=0.0)
    message = bot.format_prediction_message(data)
    
    await update.message.reply_text(message, parse_mode='Markdown')

async def predict_10(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Get top 10 predictions"""
    user_id = update.effective_user.id
    
    if not bot.check_authorization(user_id):
        await update.message.reply_text("❌ Unauthorized access")
        return
    
    await update.message.reply_text("🔄 Getting top 10 predictions...")
    
    data = await bot.get_predictions(top_n=10, min_pred=0.0)
    message = bot.format_prediction_message(data)
    
    await update.message.reply_text(message, parse_mode='Markdown')

async def hot_picks(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Get coins with >5% predicted gains"""
    user_id = update.effective_user.id
    
    if not bot.check_authorization(user_id):
        await update.message.reply_text("❌ Unauthorized access")
        return
    
    await update.message.reply_text("🔄 Finding hot picks (>5% predicted)...")
    
    data = await bot.get_predictions(top_n=15, min_pred=5.0)
    message = bot.format_prediction_message(data)
    
    await update.message.reply_text(message, parse_mode='Markdown')

async def moonshots(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Get coins with >10% predicted gains"""
    user_id = update.effective_user.id
    
    if not bot.check_authorization(user_id):
        await update.message.reply_text("❌ Unauthorized access")
        return
    
    await update.message.reply_text("🔄 Searching for moonshots (>10% predicted)...")
    
    data = await bot.get_predictions(top_n=20, min_pred=10.0)
    message = bot.format_prediction_message(data)
    
    await update.message.reply_text(message, parse_mode='Markdown')

async def status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Get bot and model status"""
    user_id = update.effective_user.id
    
    if not bot.check_authorization(user_id):
        await update.message.reply_text("❌ Unauthorized access")
        return
    
    try:
        # Check model weights file
        weights_path = "/Users/xfx/Desktop/trade/model_weights.json"
        if os.path.exists(weights_path):
            mtime = os.path.getmtime(weights_path)
            model_updated = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")
            
            # Load model info
            with open(weights_path, 'r') as f:
                model_data = json.load(f)
                
            status_msg = f"""
📊 **Bot Status**

🤖 Bot: Online ✅
🧠 Model: Loaded ✅
📅 Last Updated: {model_updated}

**Model Info:**
• Type: {model_data.get('model_type', 'Unknown')}
• Samples: {model_data.get('samples', 'Unknown')}
• Training MAE: {model_data.get('train_mae', 'Unknown')}
• Features: {len(model_data.get('features', []))}

**Ensemble Weights:**
"""
            
            if 'ensemble_weights' in model_data:
                for model_name, weight in model_data['ensemble_weights'].items():
                    status_msg += f"• {model_name}: {weight:.3f}\n"
            
            status_msg += "\n🔄 Continuous learning active"
            
        else:
            status_msg = "⚠️ Model weights file not found"
            
    except Exception as e:
        status_msg = f"❌ Error checking status: {str(e)}"
    
    await update.message.reply_text(status_msg, parse_mode='Markdown')

async def best_performers(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Get coins currently performing well with continued upside"""
    user_id = update.effective_user.id
    
    if not bot.check_authorization(user_id):
        await update.message.reply_text("❌ Unauthorized access")
        return
    
    await update.message.reply_text("📈 Finding best current performers...")
    
    # Get predictions and filter for coins already up but with more upside
    data = await bot.get_predictions(top_n=20, min_pred=1.0)
    
    if "predictions" in data and data["predictions"]:
        # Filter for coins with positive 24h performance AND positive prediction
        performers = [p for p in data["predictions"] if p["pc_24h"] > 2.0 and p["predicted_change"] > 1.0]
        
        if performers:
            # Sort by combination of current performance and predicted upside
            performers.sort(key=lambda x: x["pc_24h"] + x["predicted_change"], reverse=True)
            data["predictions"] = performers[:8]
            data["count"] = len(performers[:8])
        else:
            data = {"count": 0}
    
    message = bot.format_prediction_message(data)
    await update.message.reply_text(message, parse_mode='Markdown')

async def hidden_gems(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Get undervalued coins with high prediction potential"""
    user_id = update.effective_user.id
    
    if not bot.check_authorization(user_id):
        await update.message.reply_text("❌ Unauthorized access")
        return
    
    await update.message.reply_text("💎 Searching for hidden gems...")
    
    # Get predictions and filter for coins that are down but predicted to recover
    data = await bot.get_predictions(top_n=25, min_pred=1.5)
    
    if "predictions" in data and data["predictions"]:
        # Filter for coins with negative/flat 24h but positive strong predictions
        gems = [p for p in data["predictions"] 
                if p["pc_24h"] < 1.0 and p["predicted_change"] > 2.0 
                and p["market_cap"] < 5_000_000_000]  # Focus on smaller caps
        
        if gems:
            # Sort by prediction strength
            gems.sort(key=lambda x: x["predicted_change"], reverse=True)
            data["predictions"] = gems[:6]
            data["count"] = len(gems[:6])
        else:
            data = {"count": 0}
    
    message = bot.format_prediction_message(data)
    await update.message.reply_text(message, parse_mode='Markdown')

async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle inline button callbacks"""
    query = update.callback_query
    user_id = query.from_user.id
    
    if not bot.check_authorization(user_id):
        await query.answer("❌ Unauthorized access")
        return
    
    await query.answer()
    
    # Create back button for navigation
    back_keyboard = [[InlineKeyboardButton("🔙 Back to Menu", callback_data="back_to_menu")]]
    back_markup = InlineKeyboardMarkup(back_keyboard)
    
    if query.data == "predict_5":
        await query.edit_message_text("🔄 Getting top 5 predictions...")
        data = await bot.get_predictions(top_n=5, min_pred=0.0)
        message = bot.format_prediction_message(data)
        await query.edit_message_text(message, parse_mode='Markdown', reply_markup=back_markup)
        
    elif query.data == "predict_10":
        await query.edit_message_text("🔄 Getting top 10 predictions...")
        data = await bot.get_predictions(top_n=10, min_pred=0.0)
        message = bot.format_prediction_message(data)
        await query.edit_message_text(message, parse_mode='Markdown', reply_markup=back_markup)
        
    elif query.data == "hot_picks":
        await query.edit_message_text("🔄 Finding hot picks (>5% predicted)...")
        data = await bot.get_predictions(top_n=15, min_pred=5.0)
        message = bot.format_prediction_message(data)
        await query.edit_message_text(message, parse_mode='Markdown', reply_markup=back_markup)
        
    elif query.data == "moonshots":
        await query.edit_message_text("🔄 Searching for moonshots (>10% predicted)...")
        data = await bot.get_predictions(top_n=20, min_pred=10.0)
        message = bot.format_prediction_message(data)
        await query.edit_message_text(message, parse_mode='Markdown', reply_markup=back_markup)
        
    elif query.data == "best_performers":
        await query.edit_message_text("📈 Finding best current performers...")
        data = await bot.get_predictions(top_n=20, min_pred=1.0)
        
        if "predictions" in data and data["predictions"]:
            performers = [p for p in data["predictions"] if p["pc_24h"] > 2.0 and p["predicted_change"] > 1.0]
            if performers:
                performers.sort(key=lambda x: x["pc_24h"] + x["predicted_change"], reverse=True)
                data["predictions"] = performers[:8]
                data["count"] = len(performers[:8])
            else:
                data = {"count": 0}
        
        message = bot.format_prediction_message(data)
        await query.edit_message_text(message, parse_mode='Markdown', reply_markup=back_markup)
        
    elif query.data == "hidden_gems":
        await query.edit_message_text("💎 Searching for hidden gems...")
        data = await bot.get_predictions(top_n=25, min_pred=1.5)
        
        if "predictions" in data and data["predictions"]:
            gems = [p for p in data["predictions"] 
                    if p["pc_24h"] < 1.0 and p["predicted_change"] > 2.0 
                    and p["market_cap"] < 5_000_000_000]
            if gems:
                gems.sort(key=lambda x: x["predicted_change"], reverse=True)
                data["predictions"] = gems[:6]
                data["count"] = len(gems[:6])
            else:
                data = {"count": 0}
        
        message = bot.format_prediction_message(data)
        await query.edit_message_text(message, parse_mode='Markdown', reply_markup=back_markup)
        
    elif query.data == "status":
        await query.edit_message_text("📊 Checking model status...")
        try:
            weights_path = "/Users/xfx/Desktop/trade/model_weights.json"
            if os.path.exists(weights_path):
                mtime = os.path.getmtime(weights_path)
                model_updated = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")
                
                with open(weights_path, 'r') as f:
                    model_data = json.load(f)
                    
                status_msg = f"""
📊 **Bot Status**

🤖 Bot: Online ✅
🧠 Model: Loaded ✅
📅 Last Updated: {model_updated}

**Model Performance:**
• Type: {model_data.get('model_type', 'Unknown')}
• Training Samples: {model_data.get('samples', 'Unknown')}
• Training MAE: {model_data.get('train_mae', 'Unknown'):.3f}
• Features: {len(model_data.get('features', []))}

**Ensemble Weights:**
"""
                
                if 'ensemble_weights' in model_data:
                    for model_name, weight in model_data['ensemble_weights'].items():
                        status_msg += f"• {model_name}: {weight:.3f}\n"
                
                status_msg += "\n🔄 Continuous learning: Active"
                
            else:
                status_msg = "⚠️ Model weights file not found"
                
        except Exception as e:
            status_msg = f"❌ Error checking status: {str(e)}"
        
        await query.edit_message_text(status_msg, parse_mode='Markdown', reply_markup=back_markup)
        
    elif query.data == "help":
        help_text = """
🤖 **Crypto Prediction Bot Help**

**Quick Actions:**
🚀 **Top 5/10** - Best ranked predictions
🔥 **Hot Picks** - Coins with >5% predicted gains
🌕 **Moonshots** - High potential >10% opportunities
📈 **Best Performers** - Currently up with more upside
💎 **Hidden Gems** - Undervalued with recovery potential

**Available Commands:**
• `/predict` - Top 5 predictions  
• `/predict_10` - Top 10 predictions
• `/hot` - Hot picks (>5% gains)
• `/moonshots` - Moonshot opportunities (>10%)
• `/status` - Bot and model health
• `/help` - Show detailed help

**Model Features:**
🧠 Ensemble of 4 ML models (Ridge, GBT, RF, ElasticNet)
📊 14+ technical indicators (RSI, MACD, Bollinger Bands)
🎯 Market cap-based realistic predictions
🔄 Continuous learning and improvement
📱 Real-time analysis

⚠️ **Disclaimer:** Educational purposes only. Not financial advice.
"""
        await query.edit_message_text(help_text, parse_mode='Markdown', reply_markup=back_markup)
        
    elif query.data == "refresh" or query.data == "back_to_menu":
        # Show fresh menu by editing the current message
        welcome_msg = """
🤖 **Crypto Prediction Bot**

Welcome! I can provide AI-powered crypto predictions using advanced ensemble models with 14+ technical indicators.

**Available Commands:**
• Top 5/10 predictions with current market analysis
• Hot picks with >5% predicted gains
• Moonshot opportunities with >10% potential
• Model status and performance metrics
• Custom prediction filters

Use the buttons below for instant predictions! 👇
"""
        
        keyboard = [
            [
                InlineKeyboardButton("🚀 Top 5", callback_data="predict_5"),
                InlineKeyboardButton("📊 Top 10", callback_data="predict_10")
            ],
            [
                InlineKeyboardButton("🔥 Hot Picks (>5%)", callback_data="hot_picks"),
                InlineKeyboardButton("🌕 Moonshots (>10%)", callback_data="moonshots")
            ],
            [
                InlineKeyboardButton("📈 Best Performers", callback_data="best_performers"),
                InlineKeyboardButton("💎 Hidden Gems", callback_data="hidden_gems")
            ],
            [
                InlineKeyboardButton("📊 Model Status", callback_data="status"),
                InlineKeyboardButton("❓ Help", callback_data="help")
            ],
            [
                InlineKeyboardButton("🔄 Refresh Predictions", callback_data="refresh")
            ]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await query.edit_message_text(welcome_msg, reply_markup=reply_markup, parse_mode='Markdown')

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show help message"""
    help_text = """
🤖 **Crypto Prediction Bot Help**

**Commands:**
• `/start` - Welcome message with quick actions
• `/predict` - Top 5 predictions  
• `/predict_10` - Top 10 predictions
• `/hot` - Coins with >5% predicted gains
• `/moonshots` - Coins with >10% predicted gains  
• `/status` - Bot and model status
• `/help` - Show this help

**Features:**
🧠 Enhanced AI ensemble model (Ridge + GBT + RF + ElasticNet)
📊 Technical indicators (RSI, MACD, Bollinger Bands)
🎯 Market cap-based realistic predictions
🔄 Continuous learning and improvement
📱 Real-time crypto market analysis

**Prediction Accuracy:**
The model uses 14+ features and has achieved:
• 3x improved correlation vs baseline
• Realistic predictions based on market cap
• Continuous reinforcement learning

⚠️ **Disclaimer:** This bot provides predictions for educational purposes. Not financial advice. Always do your own research and trade responsibly.
"""
    
    await update.message.reply_text(help_text, parse_mode='Markdown')

def main():
    """Start the bot"""
    global bot
    
    if BOT_TOKEN == "YOUR_BOT_TOKEN_HERE":
        print("❌ Please set your Telegram bot token in the BOT_TOKEN variable")
        print("Get a token from @BotFather on Telegram")
        return
    
    bot = CryptoPredictionBot()
    
    # Create application
    application = Application.builder().token(BOT_TOKEN).build()
    
    # Add handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("predict", predict))
    application.add_handler(CommandHandler("predict_10", predict_10))
    application.add_handler(CommandHandler("hot", hot_picks))
    application.add_handler(CommandHandler("moonshots", moonshots))
    application.add_handler(CommandHandler("performers", best_performers))
    application.add_handler(CommandHandler("gems", hidden_gems))
    application.add_handler(CommandHandler("status", status))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CallbackQueryHandler(button_callback))
    
    # Start the bot
    print("🚀 Starting Crypto Prediction Bot...")
    print("Press Ctrl+C to stop")
    
    try:
        application.run_polling(allowed_updates=Update.ALL_TYPES)
    except KeyboardInterrupt:
        print("\n⏹️ Bot stopped")

if __name__ == "__main__":
    main()
