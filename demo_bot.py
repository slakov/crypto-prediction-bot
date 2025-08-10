#!/usr/bin/env python3
"""
Demo script showing the Telegram bot functionality without Telegram
Run this to see what the bot would return
"""

import sys
import asyncio
sys.path.append('/Users/xfx/Desktop/trade')

async def demo_predictions():
    """Demo the prediction functionality"""
    print("🤖 Crypto Prediction Bot Demo\n")
    
    # Import the bot class
    from crypto_bot import CryptoPredictionBot
    
    # Create bot instance
    bot = CryptoPredictionBot()
    
    print("🔄 Getting top 5 predictions...")
    data = await bot.get_predictions(top_n=5, min_pred=0.0)
    message = bot.format_prediction_message(data)
    print("\n📱 Telegram message would be:")
    print("=" * 50)
    print(message)
    
    print("\n" + "=" * 50)
    print("\n🔥 Getting hot picks (>5% predicted)...")
    data = await bot.get_predictions(top_n=10, min_pred=5.0)
    message = bot.format_prediction_message(data)
    print("\n📱 Telegram message would be:")
    print("=" * 50)
    print(message)

if __name__ == "__main__":
    asyncio.run(demo_predictions())
