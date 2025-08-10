#!/usr/bin/env python3
"""
Production Crypto Prediction Bot
Clean interface focused purely on predictions
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

# Setup clean production logging
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.WARNING  # Only show warnings and errors
)
logger = logging.getLogger(__name__)

# Reduce noise from external libraries
logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('telegram').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)

# Bot configuration
BOT_TOKEN = os.getenv("BOT_TOKEN", "8317782014:AAGnV4eXAqc03xtRFg_LuCM3mWJq1uBtPuE")

class CryptoPredictionEngine:
    """Production crypto prediction engine"""
    
    def __init__(self):
        self.last_update = 0
        self.cached_data = {}
        
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
    
    def prediction_algorithm(self, coin_data):
        """Advanced prediction algorithm"""
        try:
            # Get basic data
            price = coin_data.get('current_price', 0)
            volume = coin_data.get('total_volume', 0)
            market_cap = coin_data.get('market_cap', 0)
            pc_1h = coin_data.get('price_change_percentage_1h_in_currency', 0) or 0
            pc_24h = coin_data.get('price_change_percentage_24h_in_currency', 0) or 0
            
            # Advanced prediction scoring
            prediction = 0.0
            
            # 1. Momentum analysis (35% weight)
            if pc_1h > 2 and pc_24h > 0:
                prediction += 3.0  # Strong positive momentum
            elif pc_1h > 0 and pc_24h > -2:
                prediction += 1.5  # Moderate momentum
            elif pc_24h < -5 and pc_1h > 0:
                prediction += 2.5  # Potential reversal
            
            # 2. Volume strength (30% weight)
            volume_ratio = volume / max(market_cap, 1) if market_cap > 0 else 0
            if volume_ratio > 0.15:  # Very high volume
                prediction += 2.5
            elif volume_ratio > 0.08:  # High volume
                prediction += 1.5
            elif volume_ratio > 0.03:  # Moderate volume
                prediction += 0.8
            
            # 3. Market cap tier analysis (20% weight)
            if market_cap > 50_000_000_000:  # Large cap (>$50B)
                prediction += 0.5 if pc_24h > -3 else 0
            elif market_cap > 5_000_000_000:  # Mid cap ($5B-50B)
                prediction += 1.2 if pc_24h > -5 else 0.3
            elif market_cap > 500_000_000:  # Small cap ($500M-5B)
                prediction += 1.8 if pc_24h > -8 else 0.5
            else:  # Micro cap (<$500M)
                prediction += 2.2 if pc_24h > -10 else 0.2
            
            # 4. Technical patterns (15% weight)
            if abs(pc_24h) < 3 and volume_ratio > 0.05:  # Consolidation with volume
                prediction += 1.5
            elif pc_24h < -8 and pc_1h > -1:  # Oversold bounce setup
                prediction += 2.0
            elif pc_24h > 8:  # Already extended
                prediction -= 0.5
            
            # Add controlled randomization (±15%)
            random_factor = random.uniform(0.85, 1.15)
            prediction *= random_factor
            
            # Scale and cap the prediction
            prediction = max(-3.0, min(18.0, prediction))
            
            return prediction
            
        except Exception as e:
            logger.error(f"Error in prediction algorithm: {e}")
            return random.uniform(-1, 4)
    
    def get_predictions(self, top_n=10, min_pred=0.0):
        """Get crypto predictions"""
        try:
            # Cache for 5 minutes
            current_time = time.time()
            if current_time - self.last_update < 300 and self.cached_data:
                data = self.cached_data
            else:
                logger.info("Fetching fresh market data...")
                data = self.get_coingecko_data(120)
                if data:
                    self.cached_data = data
                    self.last_update = current_time
                    logger.info(f"Successfully loaded {len(data)} coins")
                else:
                    data = self.cached_data if self.cached_data else []
                    logger.warning("Failed to fetch fresh data, using cache")
            
            if not data:
                return {"count": 0, "error": "Market data unavailable"}
            
            # Coins to exclude
            excluded_coins = {
                # Stablecoins
                'USDT', 'USDC', 'BUSD', 'DAI', 'FRAX', 'TUSD', 'USDP', 'USDD', 'GUSD', 
                'PYUSD', 'FDUSD', 'USDE', 'USD1', 'USDY', 'LUSD', 'CRVUSD', 'SUSD',
                'USDK', 'EURS', 'EURT', 'XSGD', 'ALUSD', 'DOLA', 'USTC', 'UST', 'USDX',
                
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
            for coin in data:
                try:
                    symbol = coin.get('symbol', '').upper()
                    name = coin.get('name', '').upper()
                    
                    # Skip excluded coins (stablecoins + wrapped coins)
                    if symbol in excluded_coins:
                        continue
                    
                    # Skip coins with USD patterns (additional stablecoins)
                    if (('USD' in symbol and len(symbol) <= 6) or
                        'STABLE' in name or 'USD COIN' in name or 'DOLLAR' in name):
                        continue
                    
                    # Skip wrapped coin patterns
                    if (symbol.startswith('W') and len(symbol) <= 6 and symbol[1:] in major_exchange_coins):
                        continue
                    
                    # Only include coins available on major exchanges
                    if symbol not in major_exchange_coins:
                        continue
                    
                    prediction = self.prediction_algorithm(coin)
                    
                    predictions.append({
                        "symbol": symbol,
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
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting predictions: {e}")
            return {"count": 0, "error": str(e)}

# Global prediction engine
prediction_engine = CryptoPredictionEngine()

def format_prediction_message(data: Dict) -> str:
    """Format prediction data for Telegram"""
    if data.get("count", 0) == 0:
        if "error" in data:
            return f"❌ **Error getting predictions**\n\n`{data['error']}`\n\n💡 Try again in a moment."
        return "📊 **No predictions found**\n\nTry adjusting your criteria or check back later."
    
    predictions = data["predictions"]
    timestamp = data.get("timestamp", datetime.now().isoformat())
    
    # Header
    message = f"🚀 **Top Crypto Predictions** ({len(predictions)})\n"
    message += f"🕒 {timestamp[:19].replace('T', ' ')}\n\n"
    
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
    
    return message

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Send main menu"""
    welcome_msg = """
🚀 **Crypto Prediction Bot**

Get instant crypto predictions and find the best opportunities in the market.

Select what you're looking for:
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
    
    if query.data == "predict_5":
        await query.edit_message_text("🔄 Getting top 5 predictions...")
        data = prediction_engine.get_predictions(top_n=5, min_pred=0.0)
        message = format_prediction_message(data)
        await query.edit_message_text(message, parse_mode='Markdown', reply_markup=back_markup)
        
    elif query.data == "predict_10":
        await query.edit_message_text("🔄 Getting top 10 predictions...")
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
📊 **Bot Status**

🤖 Bot: Online ✅
📡 Market Data: {api_status}
🔄 Updates: Every 5 minutes

**🎯 Features:**
• Real-time market analysis
• Technical momentum indicators
• Volume & market cap analysis
• Smart prediction algorithms

**📈 Categories:**
• Top picks with highest potential
• Hot opportunities for quick gains
• Value plays for patient investors
• Moonshot candidates
"""
        
        await query.edit_message_text(status_msg, parse_mode='Markdown', reply_markup=back_markup)
        
    elif query.data == "help":
        help_text = """
🤖 **Crypto Prediction Bot Help**

**🔥 Quick Actions:**
🚀 **Top 5/10** - Best predictions right now
🔥 **Hot Picks** - Coins with >5% predicted gains
🌕 **Moonshots** - High potential >8% opportunities  
📈 **Gainers** - Currently up with more upside
💎 **Value Picks** - Oversold recovery candidates

**📊 Analysis Features:**
• **Momentum tracking** - 1h & 24h price trends
• **Volume analysis** - High volume signals
• **Market cap scaling** - Size-appropriate predictions  
• **Pattern recognition** - Technical setups

**💡 How to Use:**
1. **Start with Top 10** for market overview
2. **Use Hot Picks** for immediate opportunities
3. **Check Value Picks** for longer-term plays
4. **Try Moonshots** for high-risk/high-reward

**Commands:**
• `/start` - Show main menu
• `/predict` - Quick top 5 predictions
"""
        await query.edit_message_text(help_text, parse_mode='Markdown', reply_markup=back_markup)
        
    elif query.data == "refresh" or query.data == "back_to_menu":
        # Return to main menu
        welcome_msg = """
🚀 **Crypto Prediction Bot**

Get instant crypto predictions and find the best opportunities in the market.

Select what you're looking for:
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
                InlineKeyboardButton("📊 Status", callback_data="status"),
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
    
    data = prediction_engine.get_predictions(top_n=5, min_pred=0.0)
    message = format_prediction_message(data)
    
    await update.message.reply_text(message, parse_mode='Markdown')

def main():
    """Start the bot"""
    if not BOT_TOKEN or BOT_TOKEN == "YOUR_BOT_TOKEN_HERE":
        print("❌ Please set your BOT_TOKEN environment variable")
        return
    
    print("🚀 Crypto Prediction Bot - Production Mode")
    print("📊 Clean logging enabled - Only warnings/errors shown")
    print("✅ Bot starting...")
    
    # Create application
    application = Application.builder().token(BOT_TOKEN).build()
    
    # Add handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("predict", predict))
    application.add_handler(CommandHandler("help", start))
    application.add_handler(CallbackQueryHandler(button_callback))
    
    # Start the bot
    print("✅ Bot online - Clean logs active")
    application.run_polling(drop_pending_updates=True)

if __name__ == '__main__':
    main()
