#!/bin/bash
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
