#!/usr/bin/env python3
"""Test script for live predictions"""

from cloud_prediction_bot import SimpleCryptoPredictionEngine

def test_predictions():
    print("🔄 Testing live market data fetch...")
    
    engine = SimpleCryptoPredictionEngine()
    data = engine.get_predictions(top_n=5)
    
    print(f"📊 Results: {data.get('count', 0)} predictions")
    print(f"🕒 Status: {data.get('model_status', 'unknown')}")
    
    if data.get('predictions'):
        print("\n🚀 Top predictions:")
        for i, p in enumerate(data['predictions'][:3], 1):
            symbol = p['symbol']
            price = p['price']
            prediction = p['predicted_change']
            pc_24h = p['pc_24h']
            print(f"{i}. {symbol}: ${price:.4f} | Predicted: {prediction:+.1f}% | 24h: {pc_24h:+.1f}%")
    else:
        print("❌ No predictions generated")
        if 'error' in data:
            print(f"Error: {data['error']}")

if __name__ == "__main__":
    test_predictions()
