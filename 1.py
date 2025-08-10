import requests
import pandas as pd

# A manually curated list of tickers available on Crypto.com.
# This may not be exhaustive but covers most major coins.
CRYPTO_COM_TICKERS = {
    'BTC', 'ETH', 'CRO', 'USDT', 'USDC', 'BNB', 'XRP', 'ADA', 'SOL', 'DOGE',
    'DOT', 'AVAX', 'SHIB', 'MATIC', 'LTC', 'UNI', 'LINK', 'FTM', 'XLM', 'ATOM',
    'ALGO', 'VET', 'MANA', 'HBAR', 'SAND', 'THETA', 'XTZ', 'AXS', 'AAVE', 'EOS',
    'EGLD', 'KSM', 'FIL', 'IOTA', 'NEO', 'ZIL', 'ENJ', 'BAT', 'CHZ', 'GRT',
    'MKR', 'COMP', 'SNX', 'YFI', 'CRV', 'SUSHI', '1INCH', 'REN', 'KNC', 'LRC',
    'OMG', 'ZRX', 'ANKR', 'CELR', 'ONE', 'HOT', 'BTT', 'TRX', 'WAVES', 'ICX',
    'QTUM', 'ONT', 'DGB', 'SC', 'RVN', 'HNT', 'AR', 'GALA', 'PEPE', 'WIF', 'BONK'
}

def get_crypto_data():
    """Fetches market data for the top 250 coins from CoinGecko."""
    url = "https://api.coingecko.com/api/v3/coins/markets"
    params = {
        'vs_currency': 'usd',
        'order': 'market_cap_desc',
        'per_page': 250,
        'page': 1,
        'sparkline': 'false'
    }
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()  # Raises an exception for bad status codes
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data from CoinGecko API: {e}")
        return None

def find_best_candidate(data):
    """Analyzes the data to find the best candidate coin."""
    if not data:
        return None

    # Create a Pandas DataFrame for easier manipulation
    df = pd.DataFrame(data)
    
    # Select relevant columns
    df = df[[
        'symbol', 'name', 'current_price', 'total_volume', 
        'price_change_percentage_24h'
    ]]

    # --- Filtering Logic ---
    
    # 1. Filter for coins available on Crypto.com
    df['symbol_upper'] = df['symbol'].str.upper()
    df = df[df['symbol_upper'].isin(CRYPTO_COM_TICKERS)]

    # 2. Filter for coins with positive momentum (2% < change < 15%)
    df = df[df['price_change_percentage_24h'].between(2, 15)]

    if df.empty:
        return "No coins currently meet the specified momentum criteria."

    # 3. Sort by the highest trading volume to find the strongest trend
    df_sorted = df.sort_values(by='total_volume', ascending=False)
    
    # Select the top candidate
    candidate = df_sorted.iloc[0]
    
    return candidate

def main():
    """Main function to run the script."""
    print("--- Crypto Candidate Finder ---")
    print("Finding a coin on Crypto.com with positive momentum and high volume...")
    print("-" * 31)

    all_coins_data = get_crypto_data()
    candidate = find_best_candidate(all_coins_data)
    
    if isinstance(candidate, str):
        print(candidate)
    elif candidate is not None:
        print("🔥 Best Candidate Found 🔥\n")
        print(f"Coin: {candidate['name']} ({candidate['symbol'].upper()})")
        print(f"Current Price: ${candidate['current_price']:.4f}")
        print(f"24h Change: {candidate['price_change_percentage_24h']:.2f}%")
        print(f"24h Volume: ${candidate['total_volume']:,}")
        print("\nDisclaimer: This is not financial advice. Always do your own research.")
    else:
        print("Could not retrieve or process cryptocurrency data.")

if __name__ == "__main__":
    main()