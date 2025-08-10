#!/usr/bin/env python3
"""
Cloud-Optimized Telegram Bot for Crypto Predictions
Lightweight version optimized for free hosting platforms
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
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from importlib import import_module

# Setup logging for cloud
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Bot configuration - get from environment variables for security
BOT_TOKEN = os.getenv("BOT_TOKEN", "8317782014:AAGnV4eXAqc03xtRFg_LuCM3mWJq1uBtPuE")

class CryptoPredictionBot:
    def __init__(self):
        self.prediction_module = None
        self.load_prediction_module()
    
    def load_prediction_module(self):
        """Dynamically load the prediction functions from 2.py"""
        try:
            # Try to import the main prediction module
            if os.path.exists('2.py'):
                spec = import_module('2')
                self.prediction_module = spec
                logger.info("Successfully loaded prediction module")
            else:
                logger.warning("2.py not found, creating fallback predictions")
                self.prediction_module = None
        except Exception as e:
            logger.error(f"Failed to load prediction module: {e}")
            self.prediction_module = None
    
    def check_authorization(self, user_id: int) -> bool:
        """Check if user is authorized to use the bot"""
        return True  # Allow all users - public access enabled
    
    async def get_predictions(self, top_n: int = 5, min_pred: float = 0.0) -> Dict:
        """Get crypto predictions using the enhanced model"""
        try:
            if self.prediction_module and hasattr(self.prediction_module, 'main'):
                # Run the prediction with arguments
                result = self.prediction_module.main([
                    '--top', str(top_n),
                    '--min-prediction', str(min_pred)
                ])
                return result if result else {"count": 0}
            else:
                # Fallback demo predictions for testing
                return self.get_demo_predictions(top_n)
                
        except Exception as e:
            logger.error(f"Error getting predictions: {e}")
            return {"count": 0, "error": str(e)}
    
    def get_demo_predictions(self, top_n: int = 5) -> Dict:
        """Fallback demo predictions when main model isn't available"""
        demo_predictions = [
            {
                "symbol": "BTC",
                "name": "Bitcoin",
                "price": 45000.0,
                "predicted_change": 3.2,
                "pc_24h": 1.8,
                "pc_1h": 0.5,
                "volume_24h": 25000000000,
                "market_cap": 850000000000
            },
            {
                "symbol": "ETH", 
                "name": "Ethereum",
                "price": 2800.0,
                "predicted_change": 5.1,
                "pc_24h": 2.3,
                "pc_1h": 0.8,
                "volume_24h": 15000000000,
                "market_cap": 350000000000
            },
            {
                "symbol": "ADA",
                "name": "Cardano", 
                "price": 0.45,
                "predicted_change": 8.7,
                "pc_24h": -1.2,
                "pc_1h": 0.3,
                "volume_24h": 800000000,
                "market_cap": 15000000000
            }
        ]
        
        return {
            "count": len(demo_predictions[:top_n]),
            "predictions": demo_predictions[:top_n],
            "timestamp": datetime.now().isoformat(),
            "model_status": "demo_mode"
        }
    
    def format_prediction_message(self, data: Dict) -> str:
        """Format prediction data into a nice Telegram message"""
        if data.get("count", 0) == 0:
            if "error" in data:
                return f"❌ **Error getting predictions**\n\n`{data['error']}`\n\n💡 The bot may be starting up. Try again in a moment."
            return "📊 **No predictions found**\n\nTry adjusting your criteria or check back later."
        
        predictions = data["predictions"]
        timestamp = data.get("timestamp", datetime.now().isoformat())
        
        # Header
        message = f"🚀 **Crypto Predictions** ({len(predictions)})\n"
        message += f"🕒 {timestamp[:19].replace('T', ' ')}\n\n"
        
        # Add model status if in demo mode
        if data.get("model_status") == "demo_mode":
            message += "⚠️ *Demo mode - connect to full model for live predictions*\n\n"
        
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
            elif pred_change >= 1:
                emoji = "🟢"
            else:
                emoji = "🟡"
            
            message += f"{emoji} **{i}. {symbol}** ({name})\n"
            message += f"💰 ${price:,.2f}\n"
            message += f"📈 Predicted: +{pred_change:.1f}%\n"
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
        message += "⚠️ *Educational purposes only. Not financial advice.*"
        
        return message

# Global bot instance
bot = CryptoPredictionBot()

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Send a message when the command /start is issued"""
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
        
    elif query.data in ["best_performers", "hidden_gems"]:
        await query.edit_message_text("🔄 Analyzing market data...")
        # Simplified versions for cloud deployment
        data = await bot.get_predictions(top_n=8, min_pred=1.0)
        message = bot.format_prediction_message(data)
        await query.edit_message_text(message, parse_mode='Markdown', reply_markup=back_markup)
        
    elif query.data == "status":
        await query.edit_message_text("📊 Checking model status...")
        
        status_msg = """
📊 **Bot Status**

🤖 Bot: Online ✅
🌐 Cloud: Deployed ✅  
🔄 Uptime: 24/7

**Model Info:**
• Platform: Cloud-hosted
• Access: Public (friends welcome!)
• Features: Real-time predictions
• Indicators: 14+ technical signals

🚀 **Sharing:**
Send friends this bot username to get started!

💡 **Usage Tips:**
• Use buttons for best experience
• Check multiple timeframes  
• Combine with your own research

🔄 Continuous updates from market data
"""
        
        await query.edit_message_text(status_msg, parse_mode='Markdown', reply_markup=back_markup)
        
    elif query.data == "help":
        help_text = """
🤖 **Crypto Prediction Bot Help**

**Quick Actions:**
🚀 **Top 5/10** - Best ranked predictions
🔥 **Hot Picks** - Coins with >5% predicted gains
🌕 **Moonshots** - High potential >10% opportunities  
📈 **Best Performers** - Currently trending up
💎 **Hidden Gems** - Undervalued opportunities

**Commands:**
• `/start` - Show main menu
• `/predict` - Top 5 predictions
• `/help` - This help message

**Features:**
🧠 AI ensemble models (4 algorithms)
📊 14+ technical indicators  
🎯 Market cap-based scaling
🔄 Real-time analysis
📱 Mobile-optimized interface

**Sharing:**
✅ **Public bot** - share with friends!
✅ **No setup required** for new users
✅ **24/7 availability** via cloud hosting

⚠️ **Disclaimer:** Educational purposes only. Not financial advice. Always do your own research.
"""
        await query.edit_message_text(help_text, parse_mode='Markdown', reply_markup=back_markup)
        
    elif query.data == "refresh" or query.data == "back_to_menu":
        # Return to main menu
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

async def predict(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /predict command"""
    user_id = update.effective_user.id
    
    if not bot.check_authorization(user_id):
        await update.message.reply_text("❌ Unauthorized access")
        return
    
    await update.message.reply_text("🔄 Getting predictions...")
    
    data = await bot.get_predictions(top_n=5, min_pred=0.0)
    message = bot.format_prediction_message(data)
    
    await update.message.reply_text(message, parse_mode='Markdown')

def main():
    """Start the bot"""
    if not BOT_TOKEN or BOT_TOKEN == "YOUR_BOT_TOKEN_HERE":
        print("❌ Please set your BOT_TOKEN environment variable")
        print("Get a token from @BotFather on Telegram")
        return
    
    print("🚀 Starting Crypto Prediction Bot...")
    print(f"🌐 Cloud deployment mode")
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
