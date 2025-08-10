#!/usr/bin/env python3
"""Test script for production bot"""

from crypto_bot_production import CryptoPredictionEngine

def test_production():
    print("🔄 Testing production prediction engine...")
    
    engine = CryptoPredictionEngine()
    data = engine.get_predictions(top_n=10)
    
    print(f"📊 Results: {data.get('count', 0)} predictions")
    
    if data.get('predictions'):
        print("\n🚀 Top predictions (filtered for major exchanges):")
        for i, p in enumerate(data['predictions'], 1):
            symbol = p['symbol']
            price = p['price']
            prediction = p['predicted_change']
            pc_24h = p['pc_24h']
            print(f"{i}. {symbol}: ${price:.4f} | Predicted: {prediction:+.1f}% | 24h: {pc_24h:+.1f}%")
        
        # Check for excluded coins
        symbols = [p['symbol'] for p in data['predictions']]
        
        # Check for stablecoins
        stablecoins_found = [s for s in symbols if 'USD' in s or s in ['USDT', 'USDC', 'BUSD', 'DAI']]
        
        # Check for wrapped coins
        wrapped_found = [s for s in symbols if s.startswith('W') and s in ['WBTC', 'WETH', 'WBNB', 'WMATIC']]
        
        if stablecoins_found:
            print(f"\n⚠️ WARNING: Found stablecoins: {stablecoins_found}")
        elif wrapped_found:
            print(f"\n⚠️ WARNING: Found wrapped coins: {wrapped_found}")
        else:
            print("\n✅ SUCCESS: No stablecoins or wrapped coins found")
            print("✅ SUCCESS: All coins should be available on major exchanges")
    else:
        print(f"❌ Error: {data.get('error', 'Unknown')}")

if __name__ == "__main__":
    test_production()
