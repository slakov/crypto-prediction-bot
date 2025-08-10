# 🤖 Crypto Prediction Telegram Bot

A powerful Telegram bot that brings the enhanced crypto prediction model to your phone! Get AI-powered cryptocurrency predictions with realistic market cap-based forecasts directly in Telegram.

## 🌟 Features

- **🧠 Advanced AI Model**: Uses ensemble of 4 ML models (Ridge, GBT, Random Forest, ElasticNet)
- **📊 Technical Analysis**: 14+ features including RSI, MACD, Bollinger Bands, momentum indicators
- **🎯 Realistic Predictions**: Market cap-based scaling for achievable forecasts
- **📱 Mobile-Ready**: Get predictions anywhere on your phone
- **🔄 Real-Time**: Fresh market data and predictions
- **⚡ Fast**: Quick responses with cached data
- **🔐 Secure**: User authorization system

## 📋 Setup Instructions

### 1. Create Telegram Bot

1. Message [@BotFather](https://t.me/BotFather) on Telegram
2. Send `/newbot` and follow instructions
3. Choose a name like "My Crypto Predictor Bot"
4. Choose a username like "my_crypto_predictor_bot"
5. Copy the bot token (looks like `123456789:ABCdef...`)

### 2. Get Your User ID

1. Message [@userinfobot](https://t.me/userinfobot) on Telegram
2. Copy your user ID (a number like `123456789`)

### 3. Install Dependencies

```bash
cd /Users/xfx/Desktop/trade
pip3 install -r requirements_bot.txt
```

### 4. Configure Bot

Edit `crypto_bot.py` and update:

```python
# Replace with your bot token from @BotFather
BOT_TOKEN = "YOUR_ACTUAL_BOT_TOKEN_HERE"

# Add your Telegram user ID for security
AUTHORIZED_USERS = [
    123456789,  # Replace with your actual user ID
    # Add more user IDs if needed
]
```

### 5. Run the Bot

```bash
python3 crypto_bot.py
```

You should see:
```
🚀 Starting Crypto Prediction Bot...
Press Ctrl+C to stop
```

### 6. Test on Telegram

1. Find your bot on Telegram (search for the username you created)
2. Send `/start` to begin
3. Try commands like `/predict` or use the inline buttons

## 🎮 Bot Commands

| Command | Description |
|---------|-------------|
| `/start` | Welcome message with quick action buttons |
| `/predict` | Get top 5 crypto predictions |
| `/predict_10` | Get top 10 predictions |
| `/hot` | Find coins with >5% predicted gains |
| `/moonshots` | Search for >10% predicted gains |
| `/status` | Check bot and model status |
| `/help` | Show detailed help |

## 🎯 Quick Actions (Inline Buttons)

- **🚀 Top 5 Predictions** - Best current opportunities
- **🔥 Hot Picks (>5%)** - Strong momentum plays  
- **🌕 Moonshots (>10%)** - High potential gains
- **📊 Model Status** - Check model health

## 📊 Sample Output

```
🚀 Crypto Predictions (2025-01-10 18:35:22)

🟡 1. PAXG (PAX Gold)
💰 $3,386.96
📈 Predicted: +2.96%
📊 24h: +0.02% | 1h: -0.05%
💼 Volume: $30M

🟡 2. RAY (Raydium)  
💰 $3.36
📈 Predicted: +2.85%
📊 24h: +10.45% | 1h: -0.32%
💼 Volume: $406M

⚠️ Not financial advice. Trade at your own risk.
```

## 🔧 Advanced Configuration

### Running as Service (Ubuntu/Linux)

Create `/etc/systemd/system/crypto-bot.service`:

```ini
[Unit]
Description=Crypto Prediction Telegram Bot
After=network.target

[Service]
Type=simple
User=your_username
WorkingDirectory=/Users/xfx/Desktop/trade
ExecStart=/usr/bin/python3 crypto_bot.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl enable crypto-bot
sudo systemctl start crypto-bot
sudo systemctl status crypto-bot
```

### Running in Background (macOS)

```bash
nohup python3 crypto_bot.py > bot.log 2>&1 &
```

### Environment Variables

For production, use environment variables:

```bash
export TELEGRAM_BOT_TOKEN="your_token_here"
export AUTHORIZED_USER_IDS="123456789,987654321"
python3 crypto_bot.py
```

## 🛠️ Troubleshooting

### Bot Not Responding
- Check bot token is correct
- Verify your user ID is in AUTHORIZED_USERS
- Check network connectivity
- Look at console output for errors

### "Prediction module not loaded"
- Ensure `2.py` is in the same directory
- Check all required packages are installed
- Verify no syntax errors in `2.py`

### Rate Limiting
- Bot respects API rate limits
- Wait if you see timeout errors
- Consider running your own CoinGecko API key

### Memory Issues
- Model uses ~200MB RAM
- Consider reducing universe size for limited memory

## 🚀 Usage Tips

1. **Best Times**: Use during active trading hours for fresh data
2. **Frequency**: Check predictions every few hours, not constantly  
3. **Combinations**: Use with technical analysis for better decisions
4. **Risk Management**: Never invest more than you can afford to lose
5. **Monitoring**: Check `/status` to ensure model is up-to-date

## 📈 Model Performance

- **Correlation**: 0.26 (3x improved from baseline)
- **Training MAE**: ~2.2-2.6 percentage points
- **Features**: 14+ technical and fundamental indicators
- **Update Frequency**: Model retrains every 4 hours
- **Prediction Horizon**: 24-hour price changes

## ⚠️ Disclaimer

This bot provides predictions for educational and research purposes only. Cryptocurrency trading carries significant risk. Always:

- Do your own research (DYOR)
- Never invest more than you can afford to lose
- Consider multiple sources of information
- Understand that past performance doesn't guarantee future results
- Be aware of high volatility in crypto markets

The predictions are based on historical patterns and may not reflect future market conditions.

## 🔄 Updates & Maintenance

The bot benefits from the continuous learning system in the main prediction model:

- **Automatic Model Updates**: Every 4 hours
- **Real-Time Data**: Fresh market data on each request
- **Ensemble Learning**: Multiple ML models working together
- **Performance Tracking**: Monitors and improves accuracy over time

Keep the main monitoring system running (`python3 2.py --monitor`) for best performance.

---

**Happy Trading! 🚀📱**
