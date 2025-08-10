#!/usr/bin/env python3
"""
Minimal Cloud-Compatible Telegram Bot for Crypto Predictions
Simplified version that avoids build issues
"""

import asyncio
import json
import logging
import os
from datetime import datetime
from typing import Dict, List

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes

# Setup logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Bot configuration
BOT_TOKEN = os.getenv("BOT_TOKEN", "8317782014:AAGnV4eXAqc03xtRFg_LuCM3mWJq1uBtPuE")

def get_demo_predictions(top_n: int = 5) -> Dict:
    """Generate demo predictions for testing"""
    import random
    
    # Demo crypto data
    cryptos = [
        {"symbol": "BTC", "name": "Bitcoin", "price": 45000 + random.randint(-2000, 2000)},
        {"symbol": "ETH", "name": "Ethereum", "price": 2800 + random.randint(-200, 200)},
        {"symbol": "ADA", "name": "Cardano", "price": 0.45 + random.uniform(-0.05, 0.05)},
        {"symbol": "DOT", "name": "Polkadot", "price": 7.5 + random.uniform(-1, 1)},
        {"symbol": "LINK", "name": "Chainlink", "price": 15 + random.uniform(-2, 2)},
        {"symbol": "SOL", "name": "Solana", "price": 25 + random.uniform(-3, 3)},
        {"symbol": "MATIC", "name": "Polygon", "price": 0.85 + random.uniform(-0.1, 0.1)},
        {"symbol": "AVAX", "name": "Avalanche", "price": 18 + random.uniform(-2, 2)},
    ]
    
    predictions = []
    for crypto in cryptos[:top_n]:
        predictions.append({
            "symbol": crypto["symbol"],
            "name": crypto["name"],
            "price": crypto["price"],
            "predicted_change": random.uniform(-5, 15),  # -5% to +15%
            "pc_24h": random.uniform(-8, 12),
            "pc_1h": random.uniform(-2, 3),
            "volume_24h": random.randint(100_000_000, 50_000_000_000),
            "market_cap": random.randint(1_000_000_000, 1_000_000_000_000)
        })
    
    # Sort by predicted change
    predictions.sort(key=lambda x: x["predicted_change"], reverse=True)
    
    return {
        "count": len(predictions),
        "predictions": predictions,
        "timestamp": datetime.now().isoformat(),
        "model_status": "demo_cloud_mode"
    }

def format_prediction_message(data: Dict) -> str:
    """Format prediction data into a nice Telegram message"""
    if data.get("count", 0) == 0:
        return "📊 **No predictions found**\n\nTry adjusting your criteria or check back later."
    
    predictions = data["predictions"]
    timestamp = data.get("timestamp", datetime.now().isoformat())
    
    # Header
    message = f"🚀 **Crypto Predictions** ({len(predictions)})\n"
    message += f"🕒 {timestamp[:19].replace('T', ' ')}\n\n"
    
    # Add status
    message += "🌐 *Cloud-hosted demo mode*\n\n"
    
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
    message += "⚠️ *Demo mode - Educational purposes only*"
    
    return message

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Send a message when the command /start is issued"""
    welcome_msg = """
🤖 **Crypto Prediction Bot**

Welcome! This is a demo version running in the cloud.

**Available Features:**
• Top 5/10 AI predictions
• Hot picks with >5% potential
• Moonshot opportunities >10%
• Interactive button interface
• 24/7 cloud availability

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
        await query.edit_message_text("🔄 Getting top 5 predictions...")
        data = get_demo_predictions(top_n=5)
        message = format_prediction_message(data)
        await query.edit_message_text(message, parse_mode='Markdown', reply_markup=back_markup)
        
    elif query.data == "predict_10":
        await query.edit_message_text("🔄 Getting top 10 predictions...")
        data = get_demo_predictions(top_n=8)
        message = format_prediction_message(data)
        await query.edit_message_text(message, parse_mode='Markdown', reply_markup=back_markup)
        
    elif query.data == "hot_picks":
        await query.edit_message_text("🔄 Finding hot picks (>5% predicted)...")
        data = get_demo_predictions(top_n=8)
        # Filter for >5% predictions
        hot_picks = [p for p in data["predictions"] if p["predicted_change"] > 5.0]
        if hot_picks:
            data["predictions"] = hot_picks[:6]
            data["count"] = len(hot_picks[:6])
        else:
            data = {"count": 0}
        message = format_prediction_message(data)
        await query.edit_message_text(message, parse_mode='Markdown', reply_markup=back_markup)
        
    elif query.data == "moonshots":
        await query.edit_message_text("🔄 Searching for moonshots (>10% predicted)...")
        data = get_demo_predictions(top_n=8)
        # Filter for >10% predictions
        moonshots = [p for p in data["predictions"] if p["predicted_change"] > 10.0]
        if moonshots:
            data["predictions"] = moonshots[:5]
            data["count"] = len(moonshots[:5])
        else:
            data = {"count": 0}
        message = format_prediction_message(data)
        await query.edit_message_text(message, parse_mode='Markdown', reply_markup=back_markup)
        
    elif query.data == "status":
        await query.edit_message_text("📊 Checking bot status...")
        
        status_msg = """
📊 **Bot Status**

🤖 Bot: Online ✅
🌐 Cloud: Railway.app ✅  
🔄 Uptime: 24/7
🧠 Mode: Demo predictions

**Features:**
• Interactive button interface
• Multiple prediction categories
• Mobile-optimized design
• Public access for friends

**Demo Mode:**
This version generates sample predictions to demonstrate the interface. The full model with real-time market data is available locally.

🚀 **Sharing:**
Send friends this bot username to try it out!
"""
        
        await query.edit_message_text(status_msg, parse_mode='Markdown', reply_markup=back_markup)
        
    elif query.data == "help":
        help_text = """
🤖 **Crypto Prediction Bot Help**

**Quick Actions:**
🚀 **Top 5/10** - Best prediction demos
🔥 **Hot Picks** - Samples with >5% predicted gains
🌕 **Moonshots** - High potential >10% demos

**Commands:**
• `/start` - Show main menu
• `/predict` - Quick top 5 predictions

**About This Demo:**
This cloud-hosted version demonstrates the bot interface using sample data. It shows how the real prediction bot would work with live market data.

**Features:**
✅ 24/7 cloud availability
✅ Interactive button interface  
✅ Mobile-optimized design
✅ Public access for sharing

⚠️ **Note:** This is a demo version. Real predictions require the full model with live market data feeds.
"""
        await query.edit_message_text(help_text, parse_mode='Markdown', reply_markup=back_markup)
        
    elif query.data == "refresh" or query.data == "back_to_menu":
        # Return to main menu
        welcome_msg = """
🤖 **Crypto Prediction Bot**

Welcome! This is a demo version running in the cloud.

**Available Features:**
• Top 5/10 AI predictions
• Hot picks with >5% potential
• Moonshot opportunities >10%
• Interactive button interface
• 24/7 cloud availability

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
    await update.message.reply_text("🔄 Getting predictions...")
    
    data = get_demo_predictions(top_n=5)
    message = format_prediction_message(data)
    
    await update.message.reply_text(message, parse_mode='Markdown')

def main():
    """Start the bot"""
    if not BOT_TOKEN or BOT_TOKEN == "YOUR_BOT_TOKEN_HERE":
        print("❌ Please set your BOT_TOKEN environment variable")
        return
    
    print("🚀 Starting Simple Crypto Prediction Bot...")
    print("🌐 Cloud demo mode")
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
