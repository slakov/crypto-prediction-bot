#!/usr/bin/env python3
"""
Interactive setup script for the Telegram bot
"""

import os
import re

def setup_bot_token():
    """Interactive setup for bot token"""
    print("🤖 Telegram Bot Setup\n")
    
    print("Step 1: Create a Telegram Bot")
    print("1. Open Telegram and message @BotFather")
    print("2. Send /newbot and follow the instructions")
    print("3. Choose a name like 'My Crypto Predictor Bot'")
    print("4. Choose a username ending in 'bot' like 'my_crypto_predictor_bot'")
    print("5. Copy the bot token (looks like 123456789:ABCdef...)\n")
    
    while True:
        token = input("Enter your bot token: ").strip()
        
        # Basic validation of token format
        if re.match(r'^\d{8,10}:[A-Za-z0-9_-]{35}$', token):
            break
        else:
            print("❌ Invalid token format. Should look like: 123456789:ABCdef...")
            continue
    
    return token

def setup_user_id():
    """Interactive setup for user ID"""
    print("\nStep 2: Get Your User ID")
    print("1. Open Telegram and message @userinfobot")
    print("2. Copy your user ID (a number like 123456789)\n")
    
    while True:
        try:
            user_id = input("Enter your Telegram user ID: ").strip()
            user_id = int(user_id)
            if user_id > 0:
                break
            else:
                print("❌ User ID should be a positive number")
        except ValueError:
            print("❌ Please enter a valid number")
            continue
    
    return user_id

def update_bot_file(token, user_id):
    """Update the bot file with the new configuration"""
    bot_file = "/Users/xfx/Desktop/trade/crypto_bot.py"
    
    # Read current content
    with open(bot_file, 'r') as f:
        content = f.read()
    
    # Replace token
    content = re.sub(
        r'BOT_TOKEN = "YOUR_BOT_TOKEN_HERE"',
        f'BOT_TOKEN = "{token}"',
        content
    )
    
    # Replace user ID
    content = re.sub(
        r'AUTHORIZED_USERS = \[\s*# Add your Telegram user ID here.*?\]',
        f'AUTHORIZED_USERS = [\n    {user_id}  # Your Telegram user ID\n]',
        content,
        flags=re.DOTALL
    )
    
    # Write back
    with open(bot_file, 'w') as f:
        f.write(content)
    
    print(f"✅ Updated {bot_file}")

def create_start_script():
    """Create a simple start script"""
    script_content = '''#!/bin/bash
# Crypto Prediction Bot Startup Script

echo "🚀 Starting Crypto Prediction Bot..."
echo "📍 Directory: $(pwd)"
echo "🐍 Python: $(which python3)"

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    echo "📦 Activating virtual environment..."
    source venv/bin/activate
fi

# Keep the system awake and start monitoring in background
echo "☕ Starting system monitoring..."
caffeinate -dimsu -t 86400 python3 2.py --monitor > monitor.log 2>&1 &
MONITOR_PID=$!
echo "📊 Monitor started with PID: $MONITOR_PID"

# Wait a moment for model to load
sleep 5

# Start the Telegram bot
echo "🤖 Starting Telegram bot..."
python3 crypto_bot.py

# Cleanup on exit
echo "🛑 Stopping monitor..."
kill $MONITOR_PID 2>/dev/null || true
echo "✅ Shutdown complete"
'''
    
    with open("/Users/xfx/Desktop/trade/start_bot.sh", 'w') as f:
        f.write(script_content)
    
    # Make executable
    os.chmod("/Users/xfx/Desktop/trade/start_bot.sh", 0o755)
    print("✅ Created start_bot.sh")

def main():
    print("=" * 60)
    print("   🤖 CRYPTO PREDICTION TELEGRAM BOT SETUP")
    print("=" * 60)
    
    # Get configuration
    token = setup_bot_token()
    user_id = setup_user_id()
    
    # Update files
    print("\n🔧 Updating configuration...")
    update_bot_file(token, user_id)
    create_start_script()
    
    print("\n" + "=" * 60)
    print("✅ SETUP COMPLETE!")
    print("=" * 60)
    
    print("\n🚀 How to start your bot:")
    print("   Option 1 (Simple): python3 crypto_bot.py")
    print("   Option 2 (Full):   ./start_bot.sh")
    
    print("\n📱 How to use:")
    print("   1. Find your bot on Telegram")
    print("   2. Send /start to begin")
    print("   3. Use commands like /predict or tap buttons")
    
    print("\n🎯 Available commands:")
    print("   /predict    - Top 5 predictions")
    print("   /hot        - Coins with >5% predicted gains")
    print("   /moonshots  - Coins with >10% predicted gains")
    print("   /status     - Check bot health")
    
    print("\n⚠️  Remember:")
    print("   - Keep this terminal open while the bot runs")
    print("   - Press Ctrl+C to stop the bot")
    print("   - Bot predictions are not financial advice")
    
    print(f"\n🔐 Security: Only user {user_id} can use your bot")
    print("\n🎉 Happy trading!")

if __name__ == "__main__":
    main()
