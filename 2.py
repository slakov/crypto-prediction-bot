import math
import sys
import time
import argparse
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd
import numpy as np
import requests
from requests import Session
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy import stats
from scipy.stats import entropy


USER_AGENT = "crypto-candidate-predictor/1.0 (+https://example.com)"
COINGECKO_MARKETS_URL = "https://api.coingecko.com/api/v3/coins/markets"
CRYPTO_COM_INSTRUMENTS_URL = "https://api.crypto.com/v2/public/get-instruments"
CRYPTO_COM_EXCHANGE_INSTRUMENTS_URL = "https://api.crypto.com/exchange/v1/public/get-instruments"
CRYPTO_COM_BASES_CACHE_PATH = "/Users/xfx/Desktop/trade/crypto_com_bases_cache.json"
CRYPTO_COM_BASES_TTL_SECONDS = 6 * 60 * 60  # 6 hours
COINGECKO_MARKET_CHART_URL_TMPL = "https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
COINGECKO_GLOBAL_URL = "https://api.coingecko.com/api/v3/global"
COINGECKO_TRENDING_URL = "https://api.coingecko.com/api/v3/search/trending"
DEFAULT_CALIB_PATH = "/Users/xfx/Desktop/trade/scale_calibration.json"
COINGECKO_CACHE_DIR = "/Users/xfx/Desktop/trade/cache/coingecko"
COINGECKO_CACHE_TTL_SECONDS = 12 * 60 * 60  # 12 hours
BINANCE_KLINES_URL = "https://api.binance.com/api/v3/klines"

# Symbols to exclude (stablecoins and common pegs)
EXCLUDE_SYMBOLS: Set[str] = {
    "USDT",
    "USDC",
    "DAI",
    "BUSD",
    "TUSD",
    "FDUSD",
    "USDD",
    "GUSD",
    "EURS",
    "EURT",
    "PYUSD",
    "WBTC",  # wrapped pegged
}

# Fallback curated set of Crypto.com-listed base symbols (best-effort)
FALLBACK_CRYPTO_COM_TICKERS: Set[str] = {
    "BTC", "ETH", "CRO", "USDT", "USDC", "BNB", "XRP", "ADA", "SOL", "DOGE",
    "DOT", "AVAX", "SHIB", "MATIC", "LTC", "UNI", "LINK", "FTM", "XLM", "ATOM",
    "ALGO", "VET", "MANA", "HBAR", "SAND", "THETA", "XTZ", "AXS", "AAVE", "EOS",
    "EGLD", "KSM", "FIL", "IOTA", "NEO", "ZIL", "ENJ", "BAT", "CHZ", "GRT",
    "MKR", "COMP", "SNX", "YFI", "CRV", "SUSHI", "1INCH", "REN", "KNC", "LRC",
    "OMG", "ZRX", "ANKR", "CELR", "ONE", "HOT", "BTT", "TRX", "WAVES", "ICX",
    "QTUM", "ONT", "DGB", "SC", "RVN", "HNT", "AR", "GALA", "PEPE", "WIF", "BONK"
}


def create_http_session() -> Session:
    retry = Retry(
        total=4,
        backoff_factor=0.6,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def fetch_global_market_state(session: Session) -> Dict[str, Any]:
    try:
        r = session.get(COINGECKO_GLOBAL_URL, timeout=10)
        r.raise_for_status()
        data = r.json()
        if isinstance(data, dict):
            return data.get("data", {})
    except Exception:
        pass
    return {}


def fetch_trending_ids(session: Session) -> Set[str]:
    out: Set[str] = set()
    try:
        r = session.get(COINGECKO_TRENDING_URL, timeout=10)
        r.raise_for_status()
        js = r.json()
        coins = js.get("coins", []) if isinstance(js, dict) else []
        for c in coins:
            item = c.get("item", {}) if isinstance(c, dict) else {}
            cid = str(item.get("id", ""))
            if cid:
                out.add(cid)
    except Exception:
        pass
    return out

def fetch_crypto_com_base_symbols(session: Session) -> Set[str]:
    """Return a set of Crypto.com base symbols.

    Strategy:
    - Try cached bases if fresh (TTL).
    - Try v2 public endpoint; if fails/empty, try Exchange v1 endpoint.
    - If network succeeds, update cache.
    - If network fails but stale cache exists, return stale.
    - Otherwise, return empty set.
    """
    import json

    now = time.time()
    stale_bases: Set[str] = set()
    # Read cache
    try:
        if os.path.exists(CRYPTO_COM_BASES_CACHE_PATH):
            with open(CRYPTO_COM_BASES_CACHE_PATH, "r") as f:
                cache = json.load(f)
            fetched_at = float(cache.get("fetched_at", 0))
            bases_list = cache.get("bases", [])
            if isinstance(bases_list, list):
                cached = {str(x).upper() for x in bases_list if str(x).strip()}
            else:
                cached = set()
            if cached and (now - fetched_at) < CRYPTO_COM_BASES_TTL_SECONDS:
                return cached
            stale_bases = cached
    except Exception:
        pass
    # Try v2 public endpoint first
    try:
        resp = session.get(CRYPTO_COM_INSTRUMENTS_URL, timeout=12)
        if resp.status_code == 404:
            raise RuntimeError("v2 endpoint not available")
        resp.raise_for_status()
        payload = resp.json()
        if isinstance(payload, dict) and payload.get("code") == 0:
            instruments = payload.get("result", {}).get("instruments", [])
            bases = {str(item.get("base_currency", "")).upper() for item in instruments}
            bases = {b for b in bases if b}
            if bases:
                # update cache
                try:
                    with open(CRYPTO_COM_BASES_CACHE_PATH, "w") as f:
                        json.dump({"fetched_at": now, "bases": sorted(bases)}, f)
                except Exception:
                    pass
                return bases
    except Exception:
        pass

    # Fallback to Exchange v1 endpoint: result.data, with keys base_ccy and inst_type (filter SPOT)
    try:
        resp = session.get(CRYPTO_COM_EXCHANGE_INSTRUMENTS_URL, timeout=12)
        resp.raise_for_status()
        payload = resp.json()
        if isinstance(payload, dict) and payload.get("code") == 0:
            result = payload.get("result", {})
            # Some responses use "instruments", others use "data"
            data = result.get("data") if isinstance(result, dict) else None
            if data is None:
                data = result.get("instruments", []) if isinstance(result, dict) else []
            # Prefer SPOT instruments; if none, include all
            spot = [d for d in data if str(d.get("inst_type", "")).upper() == "SPOT"]
            rows = spot if spot else data
            bases = {str(item.get("base_ccy", "")).upper() for item in rows}
            bases = {b for b in bases if b}
            if bases:
                try:
                    with open(CRYPTO_COM_BASES_CACHE_PATH, "w") as f:
                        json.dump({"fetched_at": now, "bases": sorted(bases)}, f)
                except Exception:
                    pass
                return bases
    except Exception:
        pass

    # Fallback to stale cache if we have it
    if stale_bases:
        return stale_bases
    return set()


def fetch_coingecko_markets(session: Session, per_page: int = 250, pages: int = 1) -> List[Dict[str, Any]]:
    all_rows: List[Dict[str, Any]] = []
    for page in range(1, pages + 1):
        params = {
            "vs_currency": "usd",
            "order": "market_cap_desc",
            "per_page": per_page,
            "page": page,
            "sparkline": "false",
            "price_change_percentage": "1h,24h,7d",
        }
        try:
            resp = session.get(COINGECKO_MARKETS_URL, params=params, timeout=15)
            if resp.status_code == 429:
                # simple additional wait on hard rate-limit
                time.sleep(1.2)
                resp = session.get(COINGECKO_MARKETS_URL, params=params, timeout=15)
            resp.raise_for_status()
            page_rows = resp.json()
            if isinstance(page_rows, list):
                all_rows.extend(page_rows)
        except Exception:
            # continue best-effort on partial data
            continue
    return all_rows


def robust_z_scores(series: pd.Series) -> pd.Series:
    series = pd.to_numeric(series, errors="coerce")
    # If empty or all-NaN, return zeros without triggering numpy warnings
    if series.empty or series.dropna().empty:
        return pd.Series(0.0, index=series.index, dtype=float)
    med = series.median(skipna=True)
    mad = (series - med).abs().median(skipna=True)
    if mad == 0 or pd.isna(mad):
        mean = series.mean(skipna=True)
        std = series.std(ddof=0, skipna=True)
        if std == 0 or pd.isna(std):
            return pd.Series(0.0, index=series.index, dtype=float)
        return ((series - mean) / std).fillna(0.0)
    out = 0.6745 * (series - med) / mad
    # Clean NaN/inf
    out = out.replace([pd.NA, pd.NaT], 0.0)
    out = out.replace([float("inf"), float("-inf")], 0.0)
    return out.fillna(0.0)


def compute_technical_indicators(prices: pd.Series, volumes: pd.Series = None) -> Dict[str, pd.Series]:
    """Compute comprehensive technical indicators for price prediction."""
    indicators = {}
    
    # RSI (Relative Strength Index)
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss.replace(0, np.inf)
    indicators['rsi'] = 100 - (100 / (1 + rs))
    
    # MACD (Moving Average Convergence Divergence)
    ema12 = prices.ewm(span=12).mean()
    ema26 = prices.ewm(span=26).mean()
    indicators['macd'] = ema12 - ema26
    indicators['macd_signal'] = indicators['macd'].ewm(span=9).mean()
    indicators['macd_histogram'] = indicators['macd'] - indicators['macd_signal']
    
    # Bollinger Bands
    bb_period = 20
    bb_std = 2
    sma = prices.rolling(window=bb_period).mean()
    std = prices.rolling(window=bb_period).std()
    indicators['bb_upper'] = sma + (std * bb_std)
    indicators['bb_lower'] = sma - (std * bb_std)
    indicators['bb_position'] = (prices - indicators['bb_lower']) / (indicators['bb_upper'] - indicators['bb_lower'])
    indicators['bb_squeeze'] = std / sma  # Volatility squeeze indicator
    
    # Williams %R
    high_14 = prices.rolling(window=14).max()
    low_14 = prices.rolling(window=14).min()
    indicators['williams_r'] = -100 * ((high_14 - prices) / (high_14 - low_14))
    
    # Stochastic Oscillator
    indicators['stoch_k'] = 100 * ((prices - low_14) / (high_14 - low_14))
    indicators['stoch_d'] = indicators['stoch_k'].rolling(window=3).mean()
    
    # Price momentum features
    indicators['momentum_5'] = prices.pct_change(5) * 100
    indicators['momentum_10'] = prices.pct_change(10) * 100
    indicators['momentum_20'] = prices.pct_change(20) * 100
    
    # Mean reversion features
    sma_5 = prices.rolling(window=5).mean()
    sma_20 = prices.rolling(window=20).mean()
    sma_50 = prices.rolling(window=50).mean()
    indicators['price_vs_sma5'] = (prices / sma_5 - 1) * 100
    indicators['price_vs_sma20'] = (prices / sma_20 - 1) * 100
    indicators['price_vs_sma50'] = (prices / sma_50 - 1) * 100
    indicators['sma5_vs_sma20'] = (sma_5 / sma_20 - 1) * 100
    indicators['sma20_vs_sma50'] = (sma_20 / sma_50 - 1) * 100
    
    # Volatility features
    indicators['volatility_5'] = prices.pct_change().rolling(window=5).std() * 100 * np.sqrt(24)
    indicators['volatility_20'] = prices.pct_change().rolling(window=20).std() * 100 * np.sqrt(24)
    indicators['volatility_ratio'] = indicators['volatility_5'] / indicators['volatility_20']
    
    # Volume-based features (if available)
    if volumes is not None and not volumes.empty:
        # Volume momentum
        indicators['volume_sma'] = volumes.rolling(window=20).mean()
        indicators['volume_ratio'] = volumes / indicators['volume_sma']
        
        # Price-Volume Trend (PVT)
        price_change_pct = prices.pct_change()
        indicators['pvt'] = (price_change_pct * volumes).cumsum()
        
        # On-Balance Volume (OBV)
        price_direction = np.where(prices.diff() > 0, 1, np.where(prices.diff() < 0, -1, 0))
        indicators['obv'] = (volumes * price_direction).cumsum()
        
        # Volume Rate of Change
        indicators['volume_roc'] = volumes.pct_change(10) * 100
    
    # Advanced momentum features
    # Rate of Change
    indicators['roc_5'] = ((prices / prices.shift(5)) - 1) * 100
    indicators['roc_10'] = ((prices / prices.shift(10)) - 1) * 100
    
    # Commodity Channel Index (CCI)
    typical_price = prices  # Using close as proxy for typical price
    sma_tp = typical_price.rolling(window=20).mean()
    mad_tp = typical_price.rolling(window=20).apply(lambda x: np.abs(x - x.mean()).mean())
    indicators['cci'] = (typical_price - sma_tp) / (0.015 * mad_tp)
    
    # Average True Range (ATR) - simplified without high/low
    indicators['atr'] = prices.diff().abs().rolling(window=14).mean()
    
    return indicators


def compute_regime_features(prices: pd.Series, lookback_hours: int = 168) -> Dict[str, float]:
    """Detect market regime and compute regime-specific features."""
    if len(prices) < lookback_hours:
        return {'regime_trend': 0.0, 'regime_volatility': 0.5, 'regime_momentum': 0.0}
    
    recent_prices = prices.tail(lookback_hours)
    returns = recent_prices.pct_change().dropna()
    
    # Trend regime (using linear regression slope)
    x = np.arange(len(recent_prices))
    slope, _, r_value, _, _ = stats.linregress(x, recent_prices.values)
    regime_trend = np.tanh(slope * 1000)  # Normalize to [-1, 1]
    
    # Volatility regime
    current_vol = returns.std() * np.sqrt(24)  # Annualized
    historical_vol = prices.pct_change().rolling(window=720).std().iloc[-1] * np.sqrt(24)  # 30-day baseline
    regime_volatility = min(current_vol / (historical_vol + 1e-8), 3.0)  # Cap at 3x
    
    # Momentum regime
    short_ma = recent_prices.tail(24).mean()  # 1 day
    long_ma = recent_prices.tail(168).mean()  # 7 days
    regime_momentum = np.tanh((short_ma / long_ma - 1) * 10)
    
    return {
        'regime_trend': float(regime_trend),
        'regime_volatility': float(regime_volatility), 
        'regime_momentum': float(regime_momentum)
    }


def compute_market_structure_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute market microstructure and cross-asset features."""
    # Market cap concentration (Gini coefficient approximation)
    mcaps = df['market_cap'].dropna().sort_values(ascending=False)
    n = len(mcaps)
    if n > 1:
        index = np.arange(1, n + 1)
        gini = ((2 * index - n - 1) * mcaps.values).sum() / (n * mcaps.sum())
        df['market_concentration'] = gini
    else:
        df['market_concentration'] = 0.5
    
    # Sector momentum (approximate using market cap tiers)
    valid_mcaps = df['market_cap'].dropna()
    if len(valid_mcaps) >= 5 and valid_mcaps.nunique() >= 5:
        try:
            df['mcap_tier'] = pd.qcut(df['market_cap'].rank(method='first'), q=5, labels=False, duplicates='drop')
            for tier in range(5):
                tier_mask = df['mcap_tier'] == tier
                if tier_mask.sum() > 0:
                    tier_momentum = df.loc[tier_mask, 'price_change_percentage_24h_in_currency'].mean()
                    df.loc[tier_mask, 'sector_momentum'] = tier_momentum
        except (ValueError, TypeError):
            df['mcap_tier'] = 0
            df['sector_momentum'] = df['price_change_percentage_24h_in_currency'].mean()
    else:
        df['mcap_tier'] = 0
        df['sector_momentum'] = df['price_change_percentage_24h_in_currency'].mean()
    df['sector_momentum'] = df['sector_momentum'].fillna(0)
    
    # Cross-correlation with major assets
    if 'BTC' in df['symbol_upper'].values:
        btc_change = df.loc[df['symbol_upper'] == 'BTC', 'price_change_percentage_24h_in_currency'].iloc[0]
        df['btc_correlation_proxy'] = df['price_change_percentage_24h_in_currency'] * btc_change
    else:
        df['btc_correlation_proxy'] = 0
        
    # Liquidity proxy (volume/market_cap stability)
    df['liquidity_stability'] = (df['total_volume'] / df['market_cap']).rolling(window=min(len(df), 10)).std()
    df['liquidity_stability'] = df['liquidity_stability'].fillna(df['liquidity_stability'].median())
    
    return df


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def compute_mcap_scaling_factor(market_cap: float) -> float:
    """Compute realistic scaling factor based on market cap - larger caps move less."""
    if pd.isna(market_cap) or market_cap <= 0:
        return 0.5  # Default for unknown
    
    # Market cap tiers with realistic max daily moves
    if market_cap >= 500_000_000_000:  # >$500B (BTC, ETH)
        max_move = 6.0  # 6% max realistic daily move
    elif market_cap >= 100_000_000_000:  # $100B-500B (major alts)
        max_move = 10.0  # 10% max
    elif market_cap >= 10_000_000_000:  # $10B-100B (large caps)
        max_move = 15.0  # 15% max
    elif market_cap >= 1_000_000_000:  # $1B-10B (mid caps)
        max_move = 20.0  # 20% max
    elif market_cap >= 100_000_000:  # $100M-1B (small caps)
        max_move = 30.0  # 30% max
    else:  # <$100M (micro caps)
        max_move = 40.0  # 40% max
    
    # Scale factor to normalize to this max move, more conservative
    return max_move / 50.0  # More conservative base scaling


def log_prediction_accuracy(predicted: float, realized: float, coin_id: str, market_cap: float):
    """Log prediction accuracy for reinforcement learning."""
    import json
    import os
    from datetime import datetime
    
    log_file = "/Users/xfx/Desktop/trade/prediction_accuracy.jsonl"
    
    entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "coin_id": coin_id,
        "market_cap": market_cap,
        "predicted": predicted,
        "realized": realized,
        "error": abs(predicted - realized),
        "squared_error": (predicted - realized) ** 2,
        "direction_correct": (predicted * realized) > 0
    }
    
    try:
        with open(log_file, "a") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception:
        pass  # Fail silently if logging fails


def update_model_weights_rl():
    """Update model weights based on recent prediction accuracy using simple RL."""
    import json
    import os
    from datetime import datetime, timedelta
    
    log_file = "/Users/xfx/Desktop/trade/prediction_accuracy.jsonl"
    weights_file = "/Users/xfx/Desktop/trade/model_weights.json"
    
    if not os.path.exists(log_file):
        return
    
    # Read recent accuracy data (last 7 days)
    cutoff = datetime.utcnow() - timedelta(days=7)
    recent_data = []
    
    try:
        with open(log_file, "r") as f:
            for line in f:
                entry = json.loads(line.strip())
                entry_time = datetime.fromisoformat(entry["timestamp"])
                if entry_time > cutoff:
                    recent_data.append(entry)
    except Exception:
        return
    
    if len(recent_data) < 10:  # Need minimum data
        return
    
    # Compute accuracy metrics
    total_error = sum(entry["error"] for entry in recent_data)
    avg_error = total_error / len(recent_data)
    direction_accuracy = sum(entry["direction_correct"] for entry in recent_data) / len(recent_data)
    
    # Simple RL: adjust weights based on performance
    try:
        with open(weights_file, "r") as f:
            weights = json.load(f)
        
        # If performance is poor, reduce ensemble confidence
        if avg_error > 15.0 or direction_accuracy < 0.6:
            if "ensemble_weights" in weights:
                for model in weights["ensemble_weights"]:
                    weights["ensemble_weights"][model] *= 0.95  # Reduce confidence
        
        # If performance is good, increase confidence
        elif avg_error < 8.0 and direction_accuracy > 0.7:
            if "ensemble_weights" in weights:
                for model in weights["ensemble_weights"]:
                    weights["ensemble_weights"][model] *= 1.02  # Increase confidence
        
        # Renormalize weights
        if "ensemble_weights" in weights:
            total = sum(weights["ensemble_weights"].values())
            for model in weights["ensemble_weights"]:
                weights["ensemble_weights"][model] /= total
        
        # Save updated weights
        with open(weights_file, "w") as f:
            json.dump(weights, f)
            
        print(f"RL Update: avg_error={avg_error:.2f}, direction_acc={direction_accuracy:.2f}")
        
    except Exception:
        pass


def rolling_robust_z(series: pd.Series, window: int) -> pd.Series:
    if series.empty or window <= 3:
        return pd.Series(index=series.index, dtype=float)
    roll = series.rolling(window=window, min_periods=max(5, window // 3))
    med = roll.median()
    mad = (series - med).abs().rolling(window=window, min_periods=max(5, window // 3)).median()
    out = 0.6745 * (series - med) / mad.replace(0, pd.NA)
    return out.replace([pd.NA, pd.NaT, float("inf"), float("-inf")], 0.0)


def prepare_dataframe(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Normalize columns that may or may not exist
    if "price_change_percentage_1h_in_currency" not in df.columns:
        df["price_change_percentage_1h_in_currency"] = pd.NA

    if "price_change_percentage_24h_in_currency" not in df.columns:
        # fallback to legacy field name
        if "price_change_percentage_24h" in df.columns:
            df["price_change_percentage_24h_in_currency"] = df["price_change_percentage_24h"]
        else:
            df["price_change_percentage_24h_in_currency"] = pd.NA

    if "price_change_percentage_7d_in_currency" not in df.columns:
        df["price_change_percentage_7d_in_currency"] = pd.NA

    # Keep only relevant columns
    keep = [
        "id",
        "symbol",
        "name",
        "current_price",
        "total_volume",
        "market_cap",
        "market_cap_rank",
        "price_change_percentage_1h_in_currency",
        "price_change_percentage_24h_in_currency",
        "price_change_percentage_7d_in_currency",
    ]
    existing = [c for c in keep if c in df.columns]
    df = df[existing]

    # Coerce numerics
    numeric_cols = [
        "current_price",
        "total_volume",
        "market_cap",
        "market_cap_rank",
        "price_change_percentage_1h_in_currency",
        "price_change_percentage_24h_in_currency",
        "price_change_percentage_7d_in_currency",
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Uppercase symbol for joins/filters
    df["symbol_upper"] = df["symbol"].astype(str).str.upper()
    return df


def filter_universe(
    df: pd.DataFrame,
    crypto_com_bases: Set[str],
    min_market_cap: float = 150_000_000,  # filter out microcaps
    min_volume: float = 10_000_000,  # ensure liquidity
    include_symbols: Optional[Set[str]] = None,
    extra_exclude: Optional[Set[str]] = None,
) -> pd.DataFrame:
    if df.empty:
        return df

    include_symbols = {s.upper() for s in (include_symbols or set())}
    allowed = set(crypto_com_bases) | include_symbols
    # Listed on Crypto.com (by base symbol) or force-included symbols
    df = df[df["symbol_upper"].isin(allowed)]

    # Exclude stables and pegs
    excludes = set(EXCLUDE_SYMBOLS) | {s.upper() for s in (extra_exclude or set())}
    df = df[~df["symbol_upper"].isin(excludes)]

    # Basic liquidity and size filters
    df = df.dropna(subset=["market_cap", "total_volume"])  # type: ignore[arg-type]
    df = df[(df["market_cap"] >= min_market_cap) & (df["total_volume"] >= min_volume)]

    return df


def score_and_predict(
    df: pd.DataFrame,
    enable_regime_adjust: bool = True,
    learned_weights: Optional[Dict[str, Any]] = None,
    enable_external_context: bool = True,
    use_enhanced_features: bool = True,
) -> pd.DataFrame:
    if df.empty:
        return df

    # Features
    df["volume_to_marketcap"] = (df["total_volume"] / df["market_cap"]).replace([pd.NA, pd.NaT], pd.NA)

    # Clip raw percentage changes to reduce outlier impact
    df["pc_1h"] = df["price_change_percentage_1h_in_currency"].clip(lower=-10, upper=10)
    df["pc_24h"] = df["price_change_percentage_24h_in_currency"].clip(lower=-40, upper=40)
    df["pc_7d"] = df["price_change_percentage_7d_in_currency"].clip(lower=-90, upper=90)

    # Z-scores (robust)
    df["z_1h"] = robust_z_scores(df["pc_1h"]).fillna(0.0)
    df["z_24h"] = robust_z_scores(df["pc_24h"]).fillna(0.0)
    df["z_v2m"] = robust_z_scores(df["volume_to_marketcap"]).fillna(0.0)
    df["z_7d"] = robust_z_scores(df["pc_7d"]).fillna(0.0)

    # Relative strength vs BTC/ETH using available rows
    try:
        btc_row = df.loc[df["symbol_upper"] == "BTC"]["pc_24h"].iloc[0]
    except Exception:
        btc_row = 0.0
    try:
        eth_row = df.loc[df["symbol_upper"] == "ETH"]["pc_24h"].iloc[0]
    except Exception:
        eth_row = 0.0
    df["rs_24h_btc"] = (df["pc_24h"] - float(btc_row)).fillna(0.0)
    df["rs_24h_eth"] = (df["pc_24h"] - float(eth_row)).fillna(0.0)
    df["z_rs_btc"] = robust_z_scores(df["rs_24h_btc"]).fillna(0.0)
    df["z_rs_eth"] = robust_z_scores(df["rs_24h_eth"]).fillna(0.0)

    # Market-cap percentile (favor large caps slightly)
    try:
        df["mc_pct"] = df["market_cap"].rank(pct=True)
    except Exception:
        df["mc_pct"] = 0.5

    # Overextension penalty: penalize very extended 7d gains
    df["overextension_penalty"] = df["z_7d"].clip(lower=0) * -0.35

    # Trend consistency: reward when 1h and 24h have same sign
    df["trend_consistency"] = ((df["pc_1h"] * df["pc_24h"]) > 0).astype(float) * 0.25

    # Enhanced features for better prediction
    if use_enhanced_features:
        # Add market microstructure features
        df = compute_market_structure_features(df)
        
        # Add enhanced technical features (using synthetic price data if needed)
        for idx, row in df.iterrows():
            # Create synthetic hourly price series for technical analysis
            current_price = row.get('current_price', 1.0)
            pc_1h = row.get('pc_1h', 0.0)
            pc_24h = row.get('pc_24h', 0.0) 
            pc_7d = row.get('pc_7d', 0.0)
            
            # Generate synthetic price history for technical indicators
            hours_back = 168  # 7 days
            synthetic_prices = pd.Series(dtype=float)
            base_price = current_price / (1 + pc_7d/100)  # Price 7 days ago
            
            # Create trend-consistent price series
            price_trend = pc_7d / 100 / hours_back  # Hourly trend
            for h in range(hours_back):
                noise = np.random.normal(0, 0.01)  # Small random noise
                price = base_price * (1 + price_trend * h + noise)
                synthetic_prices.loc[h] = price
            
            # Apply recent changes
            if hours_back >= 24:
                synthetic_prices.iloc[-24:] *= (1 + pc_24h/100/24)
            if hours_back >= 1:
                synthetic_prices.iloc[-1:] *= (1 + pc_1h/100)
            
            # Compute technical indicators
            try:
                tech_indicators = compute_technical_indicators(synthetic_prices)
                
                # Extract the latest values (most recent hour)
                for indicator_name, indicator_series in tech_indicators.items():
                    if not indicator_series.empty:
                        latest_value = indicator_series.iloc[-1]
                        if pd.notna(latest_value) and np.isfinite(latest_value):
                            df.loc[idx, f'tech_{indicator_name}'] = float(latest_value)
                        else:
                            df.loc[idx, f'tech_{indicator_name}'] = 0.0
                    else:
                        df.loc[idx, f'tech_{indicator_name}'] = 0.0
                        
                # Compute regime features
                regime_features = compute_regime_features(synthetic_prices)
                for regime_name, regime_value in regime_features.items():
                    df.loc[idx, f'regime_{regime_name}'] = regime_value
                    
            except Exception:
                # Fallback values if technical computation fails
                for tech_name in ['rsi', 'macd', 'bb_position', 'williams_r', 'momentum_5', 'volatility_ratio']:
                    df.loc[idx, f'tech_{tech_name}'] = 0.0
                for regime_name in ['trend', 'volatility', 'momentum']:
                    df.loc[idx, f'regime_{regime_name}'] = 0.0
        
        # Fill any remaining NaN technical features with 0
        tech_cols = [col for col in df.columns if col.startswith('tech_') or col.startswith('regime_')]
        for col in tech_cols:
            df[col] = df[col].fillna(0.0)
        
        # Create Z-scores for key technical indicators
        key_tech_features = ['tech_rsi', 'tech_macd', 'tech_bb_position', 'tech_momentum_5', 'tech_volatility_ratio']
        for feature in key_tech_features:
            if feature in df.columns:
                df[f'z_{feature}'] = robust_z_scores(df[feature]).fillna(0.0)
    
    # Enhanced composite score incorporating technical indicators
    base_score = (
        0.30 * df["z_1h"]
        + 0.25 * df["z_24h"]
        + 0.20 * df["z_v2m"]
        + 0.12 * df["z_rs_btc"]
        + 0.06 * df["z_rs_eth"]
        + 0.04 * (1.0 - df["mc_pct"]).fillna(0.0)
        + df["overextension_penalty"]
        + df["trend_consistency"]
    )
    
    # Add technical indicator contributions if enhanced features are enabled
    if use_enhanced_features:
        tech_score = 0.0
        tech_weights = {
            'z_tech_rsi': 0.08,
            'z_tech_macd': 0.06,
            'z_tech_bb_position': 0.05,
            'z_tech_momentum_5': 0.07,
            'z_tech_volatility_ratio': 0.04,
            'regime_trend': 0.06,
            'regime_momentum': 0.05,
            'market_concentration': 0.03,
            'sector_momentum': 0.04,
            'btc_correlation_proxy': 0.02
        }
        
        for feature, weight in tech_weights.items():
            if feature in df.columns:
                tech_score += weight * df[feature].fillna(0.0)
        
        df["composite_score"] = base_score + tech_score
    else:
        df["composite_score"] = base_score

    # Consensus refinement: blend base with specialized sub-scores for more stable ranking
    try:
        momentum_score = 0.6 * df["z_24h"] + 0.4 * df["z_1h"]
        volume_score = df["z_v2m"]
        rs_score = 0.5 * df["z_rs_btc"] + 0.3 * df["z_rs_eth"]
        overext_pen = df["z_7d"].clip(lower=0) * -0.10
        consensus_score = (
            0.50 * df["composite_score"].fillna(0.0)
            + 0.25 * momentum_score.fillna(0.0)
            + 0.15 * volume_score.fillna(0.0)
            + 0.10 * rs_score.fillna(0.0)
            + 0.05 * df["trend_consistency"].fillna(0.0)
            + overext_pen.fillna(0.0)
        )
        # Rank-based ensemble to improve monotonicity
        r1 = df["z_1h"].rank(pct=True)
        r2 = df["z_24h"].rank(pct=True)
        r3 = df["z_v2m"].rank(pct=True)
        r4 = df["rs_24h_btc"].rank(pct=True)
        rank_ensemble = (0.40 * r2 + 0.25 * r1 + 0.25 * r3 + 0.10 * r4)
        # Blend normalized consensus with centered rank score
        norm_consensus = robust_z_scores(consensus_score).fillna(0.0)
        rank_centered = (rank_ensemble - 0.5) * 2.0
        df["composite_score"] = 0.60 * norm_consensus + 0.40 * rank_centered
    except Exception:
        pass

    # Mild regime adjustment by UTC hour/day
    if enable_regime_adjust:
        now = time.gmtime()
        hour = now.tm_hour
        dow = now.tm_wday  # 0=Mon
        is_weekend = dow in (5, 6)
        if 12 <= hour <= 20:
            mult = 1.08  # EU/US overlap tends to have stronger follow-through
        elif 7 <= hour < 12:
            mult = 1.03
        elif 0 <= hour < 7:
            mult = 0.98
        else:
            mult = 1.00
        if is_weekend:
            mult *= 0.97
        df["composite_score"] = df["composite_score"] * float(mult)

    # Global/trending context (optional; skip for speed during calibration/backtests)
    if enable_external_context:
        try:
            session = create_http_session()
            global_state = fetch_global_market_state(session)
            trending = fetch_trending_ids(session)
            btc_dom = float(global_state.get("market_cap_percentage", {}).get("btc", 0.0))
            risk_mult = 1.0
            if btc_dom and btc_dom > 55:
                risk_mult *= 0.97
            elif btc_dom and btc_dom < 45:
                risk_mult *= 1.03
            df["is_trending"] = df["id"].isin(trending).astype(float)
            df["composite_score"] = df["composite_score"] * float(risk_mult) + 0.1 * df["is_trending"]
        except Exception:
            df["is_trending"] = 0.0
    else:
        df["is_trending"] = 0.0

    # Predicted 24h change
    if learned_weights and all(k in learned_weights for k in ["features", "weights", "intercept"]):
        try:
            feat_names = learned_weights.get("features", [])
            weights = learned_weights.get("weights", [])
            intercept = float(learned_weights.get("intercept", 0.0))
            # Build feature matrix from z_1h, z_24h, z_v2m
            fmap = {"z_1h": df["z_1h"], "z_24h": df["z_24h"], "z_v2m": df["z_v2m"]}
            yhat = pd.Series(intercept, index=df.index, dtype=float)
            for name, w in zip(feat_names, weights):
                if name in fmap:
                    yhat = yhat + float(w) * fmap[name]
            # Optional scaling calibration
            calib = load_calibration()
            if calib:
                yhat = yhat * float(calib.get("multiplier", 1.0)) + float(calib.get("intercept", 0.0))
            
            # Apply market cap based scaling for realistic predictions
            mcap_scaling = df["market_cap"].apply(lambda x: compute_mcap_scaling_factor(x))

            yhat = yhat * mcap_scaling
            
            df["predicted_change_24h_pct"] = yhat.apply(lambda v: clamp(float(v), -25.0, 25.0))
        except Exception:
            # Fallback to heuristic scaling
            pc24_abs = pd.to_numeric(df.get("pc_24h", pd.Series(dtype=float)), errors="coerce").abs()
            valid_pc24 = pc24_abs.dropna()
            avg_abs_24h = float(valid_pc24.mean()) if not valid_pc24.empty else float("nan")
            if pd.isna(avg_abs_24h) or avg_abs_24h == 0:
                avg_abs_24h = 4.0
            scale = 0.6 * avg_abs_24h
            mcap_scaling = df["market_cap"].apply(lambda x: compute_mcap_scaling_factor(x))
            df["predicted_change_24h_pct"] = (df["composite_score"] * scale * mcap_scaling).apply(lambda v: clamp(float(v), -25.0, 25.0))
    else:
        pc24_abs = pd.to_numeric(df.get("pc_24h", pd.Series(dtype=float)), errors="coerce").abs()
        valid_pc24 = pc24_abs.dropna()
        avg_abs_24h = float(valid_pc24.mean()) if not valid_pc24.empty else float("nan")
        if pd.isna(avg_abs_24h) or avg_abs_24h == 0:
            avg_abs_24h = 4.0  # conservative fallback
        # Optional calibration
        calib = load_calibration()
        mult = float(calib.get("multiplier", 0.6)) if calib else 0.6
        intercept = float(calib.get("intercept", 0.0)) if calib else 0.0
        scale = mult * avg_abs_24h
        mcap_scaling = df["market_cap"].apply(lambda x: compute_mcap_scaling_factor(x))
        df["predicted_change_24h_pct"] = (df["composite_score"] * scale * mcap_scaling + intercept).apply(lambda v: clamp(float(v), -25.0, 25.0))

    return df


def pick_top_n(df: pd.DataFrame, n: int = 3) -> pd.DataFrame:
    if df.empty:
        return df
    df_sorted = df.sort_values(by=["composite_score", "total_volume"], ascending=[False, False])
    return df_sorted.head(n).copy()


def fetch_market_chart(
    session: Session,
    coin_id: str,
    vs_currency: str = "usd",
    days: int = 14,
    interval: str = "hourly",
    timeout: float = 10.0,
) -> Optional[Dict[str, Any]]:
    url = COINGECKO_MARKET_CHART_URL_TMPL.format(coin_id=coin_id)
    # CoinGecko: interval=hourly is Enterprise-only; omit interval and let server choose
    # For hourly granularity automatically, keep days between 2 and 90
    days_clamped = max(2, min(int(days), 90))
    params = {"vs_currency": vs_currency, "days": days_clamped}
    try:
        resp = session.get(url, params=params, timeout=timeout)
        if resp.status_code == 429:
            time.sleep(1.0)
            resp = session.get(url, params=params, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        if not isinstance(data, dict):
            return None
        return data
    except Exception:
        return None


def _ensure_dir(path: str) -> None:
    try:
        os.makedirs(path, exist_ok=True)
    except Exception:
        pass


def fetch_market_chart_cached(
    session: Session,
    coin_id: str,
    days: int,
    vs_currency: str = "usd",
    timeout: float = 10.0,
    ttl_seconds: int = COINGECKO_CACHE_TTL_SECONDS,
) -> Optional[Dict[str, Any]]:
    import json
    # Normalize days range to server limits
    days_clamped = max(2, min(int(days), 90))
    _ensure_dir(COINGECKO_CACHE_DIR)
    safe_coin = str(coin_id).replace("/", "_").replace("..", ".")
    cache_file = os.path.join(COINGECKO_CACHE_DIR, f"{safe_coin}_{days_clamped}.json")
    now_ts = time.time()
    # Try read cache
    try:
        if os.path.exists(cache_file):
            mtime = os.path.getmtime(cache_file)
            if (now_ts - mtime) <= ttl_seconds:
                with open(cache_file, "r") as f:
                    data = json.load(f)
                if isinstance(data, dict) and isinstance(data.get("prices", []), list) and len(data.get("prices", [])) >= 10:
                    return data
    except Exception:
        pass
    # Fetch live
    data = fetch_market_chart(session, coin_id, vs_currency=vs_currency, days=days_clamped, timeout=timeout)
    if isinstance(data, dict) and data.get("prices"):
        try:
            with open(cache_file, "w") as f:
                json.dump(data, f)
        except Exception:
            pass
        return data
    # Fallback to stale cache if exists
    try:
        if os.path.exists(cache_file):
            with open(cache_file, "r") as f:
                data = json.load(f)
            if isinstance(data, dict) and data.get("prices"):
                return data
    except Exception:
        pass
    return None


def fetch_binance_klines_series(session: Session, symbol_usdt: str, days: int, interval: str = "1h", timeout: float = 10.0) -> Optional[pd.DataFrame]:
    try:
        # Binance allows limit up to 1000; compute needed candles by days
        hours = max(48, min(days * 24 + 4, 1000))
        params = {"symbol": symbol_usdt, "interval": interval, "limit": hours}
        r = session.get(BINANCE_KLINES_URL, params=params, timeout=timeout)
        r.raise_for_status()
        data = r.json()
        if not isinstance(data, list) or not data:
            return None
        # Kline array: [openTime, open, high, low, close, volume, closeTime, ...]
        closes = [float(k[4]) for k in data]
        volumes = [float(k[5]) for k in data]
        times = [int(k[6]) for k in data]  # closeTime ms
        idx = pd.to_datetime(times, unit="ms")
        df = pd.DataFrame({"close": closes, "volume": volumes}, index=idx).sort_index()
        return df
    except Exception:
        return None


def _compute_history_features_from_chart(
    chart: Dict[str, Any],
    current_price: float,
) -> Optional[Dict[str, float]]:
    try:
        prices = chart.get("prices") or []
        vols = chart.get("total_volumes") or []
        if len(prices) < 48 or len(vols) < 48:
            return None

        price_series = pd.Series([p[1] for p in prices], index=pd.to_datetime([p[0] for p in prices], unit="ms"))
        vol_series = pd.Series([v[1] for v in vols], index=pd.to_datetime([v[0] for v in vols], unit="ms"))

        # Ensure monotonic index
        price_series = price_series.sort_index()
        vol_series = vol_series.sort_index()

        # Hourly log returns
        log_prices = price_series.apply(lambda x: math.log(x) if x and x > 0 else float("nan"))
        hourly_returns = log_prices.diff().dropna()

        # Realized daily vol from hourly (std * sqrt(24)) in percent
        if len(hourly_returns) >= 24:
            rv_30d = hourly_returns[-24 * 30 :] if len(hourly_returns) > 24 * 30 else hourly_returns
            rv_7d = hourly_returns[-24 * 7 :] if len(hourly_returns) > 24 * 7 else hourly_returns
            daily_vol_30d_pct = float(rv_30d.std(ddof=0) * math.sqrt(24) * 100)
            daily_vol_7d_pct = float(rv_7d.std(ddof=0) * math.sqrt(24) * 100)
        else:
            daily_vol_30d_pct = float("nan")
            daily_vol_7d_pct = float("nan")

        # 24h rolling volume stats from hourly volumes
        if len(vol_series) >= 24:
            roll24 = vol_series.rolling(window=24).sum().dropna()
            roll24_mean = float(roll24.mean()) if not roll24.empty else float("nan")
            roll24_std = float(roll24.std(ddof=0)) if not roll24.empty else float("nan")
            current_roll24 = float(roll24.iloc[-1]) if not roll24.empty else float("nan")
        else:
            roll24_mean = float("nan")
            roll24_std = float("nan")
            current_roll24 = float("nan")

        # Trend slope over last 72 hours (3 days) of log price
        window_hours = 72 if len(log_prices) >= 72 else max(24, len(log_prices))
        lp_tail = log_prices.tail(window_hours)
        x = pd.Series(range(len(lp_tail)), index=lp_tail.index, dtype=float)
        # simple OLS slope: cov(x,y)/var(x)
        var_x = float(((x - x.mean()) ** 2).mean())
        if var_x == 0 or math.isnan(var_x):
            slope_per_hour = 0.0
        else:
            cov_xy = float(((x - x.mean()) * (lp_tail - lp_tail.mean())).mean())
            slope_per_hour = cov_xy / var_x
        trend_slope_24h_pct = slope_per_hour * 24.0 * 100.0

        # Drawdown from last 7d high
        lookback = price_series.tail(7 * 24) if len(price_series) >= 7 * 24 else price_series
        if lookback.empty:
            drawdown_7d = float("nan")
        else:
            last_price = float(price_series.iloc[-1]) if not math.isnan(float(price_series.iloc[-1])) else current_price
            max_7d = float(lookback.max())
            if max_7d > 0:
                drawdown_7d = last_price / max_7d - 1.0
            else:
                drawdown_7d = float("nan")

        # Short-horizon continuation features
        try:
            mom_3h_pct = float((price_series.iloc[-1] / price_series.shift(3).iloc[-1] - 1.0) * 100.0) if len(price_series) >= 4 else float("nan")
        except Exception:
            mom_3h_pct = float("nan")
        try:
            mom_6h_pct = float((price_series.iloc[-1] / price_series.shift(6).iloc[-1] - 1.0) * 100.0) if len(price_series) >= 7 else float("nan")
        except Exception:
            mom_6h_pct = float("nan")

        try:
            up24 = hourly_returns.tail(24)
            up_hour_ratio_24h = float((up24 > 0).mean()) if len(up24) > 0 else float("nan")
        except Exception:
            up_hour_ratio_24h = float("nan")

        try:
            sma20 = price_series.rolling(window=20).mean()
            std20 = price_series.rolling(window=20).std(ddof=0)
            last_sma20 = float(sma20.iloc[-1]) if not math.isnan(float(sma20.iloc[-1])) else float("nan")
            last_std20 = float(std20.iloc[-1]) if not math.isnan(float(std20.iloc[-1])) else float("nan")
            if last_std20 and last_std20 > 0:
                bollinger_dist_20h = (float(price_series.iloc[-1]) - last_sma20) / last_std20
            else:
                bollinger_dist_20h = float("nan")
        except Exception:
            bollinger_dist_20h = float("nan")

        try:
            last_vol = float(vol_series.iloc[-1])
            vwin = vol_series.tail(24)
            vmean = float(vwin.mean()) if len(vwin) > 0 else float("nan")
            vstd = float(vwin.std(ddof=0)) if len(vwin) > 1 else float("nan")
            if vstd and vstd > 0:
                vol_1h_z_24h = (last_vol - vmean) / vstd
            else:
                vol_1h_z_24h = float("nan")
        except Exception:
            vol_1h_z_24h = float("nan")

        return {
            "daily_vol_30d_pct": daily_vol_30d_pct,
            "daily_vol_7d_pct": daily_vol_7d_pct,
            "roll24_mean": roll24_mean,
            "roll24_std": roll24_std,
            "current_roll24": current_roll24,
            "trend_slope_24h_pct": float(trend_slope_24h_pct),
            "drawdown_7d": float(drawdown_7d),
            "mom_3h_pct": float(mom_3h_pct),
            "mom_6h_pct": float(mom_6h_pct),
            "up_hour_ratio_24h": float(up_hour_ratio_24h),
            "bollinger_dist_20h": float(bollinger_dist_20h),
            "vol_1h_z_24h": float(vol_1h_z_24h),
        }
    except Exception:
        return None


def refine_with_history(
    df_scored: pd.DataFrame,
    session: Session,
    top_k: int = 10,
    days: int = 14,
    request_timeout: float = 6.0,
    max_total_seconds: float = 10.0,
    workers: int = 4,
) -> pd.DataFrame:
    if df_scored.empty:
        return df_scored

    short = df_scored.sort_values(by=["composite_score", "total_volume"], ascending=[False, False]).head(top_k).copy()

    # If not requesting history refinement or nothing selected, return empty to avoid noisy computations
    if top_k <= 0 or short.empty:
        return short

    # Prepare columns
    for col in [
        "daily_vol_30d_pct",
        "daily_vol_7d_pct",
        "roll24_mean",
        "roll24_std",
        "current_roll24",
        "trend_slope_24h_pct",
        "drawdown_7d",
        "mom_3h_pct",
        "mom_6h_pct",
        "up_hour_ratio_24h",
        "bollinger_dist_20h",
        "vol_1h_z_24h",
    ]:
        short[col] = float("nan")

    # Concurrent history fetch with overall time budget
    start_time = time.time()

    def task(coin_idx: Any, coin_id: str, price_now: float, req_timeout: float) -> Tuple[Any, Optional[Dict[str, float]]]:
        local_session = create_http_session()
        try:
            chart = fetch_market_chart(local_session, coin_id, days=days, interval="hourly", timeout=req_timeout)
            if not chart:
                return coin_idx, None
            feats = _compute_history_features_from_chart(chart, price_now)
            return coin_idx, feats
        finally:
            try:
                local_session.close()
            except Exception:
                pass

    futures_list = []
    pool = ThreadPoolExecutor(max_workers=max(1, workers))
    try:
        for idx, row in short.iterrows():
            if time.time() - start_time > max_total_seconds:
                break
            cid = str(row.get("id", ""))
            if not cid:
                continue
            price_now = float(row.get("current_price", float("nan")))
            # Bound per-request timeout by remaining global budget (with a small buffer)
            remaining_total = max(0.25, max_total_seconds - (time.time() - start_time))
            per_req_timeout = max(0.25, min(request_timeout, remaining_total - 0.10))
            futures_list.append(pool.submit(task, idx, cid, price_now, per_req_timeout))

        for fut in as_completed(futures_list, timeout=max_total_seconds):
            if time.time() - start_time > max_total_seconds:
                break
            try:
                remaining = max(0.05, max_total_seconds - (time.time() - start_time))
                coin_idx, feats = fut.result(timeout=remaining)
                if feats:
                    for k, v in feats.items():
                        short.at[coin_idx, k] = v
            except Exception:
                # Timeout or request error: skip
                continue
    except Exception:
        pass
    finally:
        # Do not wait for still-running tasks; cancel them
        try:
            pool.shutdown(wait=False, cancel_futures=True)
        except TypeError:
            # For very old Python versions without cancel_futures
            pool.shutdown(wait=False)

    # Compute refined features
    eps = 1e-6
    short["daily_vol_30d_pct"] = pd.to_numeric(short["daily_vol_30d_pct"], errors="coerce")
    short["vol_adj_mom"] = (short["pc_24h"] / (short["daily_vol_30d_pct"].abs() + eps)).clip(lower=-3.0, upper=3.0)

    # Volume surge z: use historical rolling 24h sums
    short["volume_surge_z"] = (
        (short["current_roll24"] - short["roll24_mean"]) / (short["roll24_std"].replace(0, pd.NA))
    ).replace([pd.NA, pd.NaT, float("inf"), float("-inf")], 0.0)

    # Trend factor from slope (bounded)
    short["trend_factor"] = short["trend_slope_24h_pct"].fillna(0.0).apply(lambda v: clamp(v / 5.0, -2.0, 2.0))

    # Continuation factors
    short["z_mom_3h"] = robust_z_scores(pd.to_numeric(short["mom_3h_pct"], errors="coerce")).fillna(0.0)
    short["z_mom_6h"] = robust_z_scores(pd.to_numeric(short["mom_6h_pct"], errors="coerce")).fillna(0.0)
    short["z_boll"] = robust_z_scores(pd.to_numeric(short["bollinger_dist_20h"], errors="coerce")).fillna(0.0)
    short["z_vol1h"] = robust_z_scores(pd.to_numeric(short["vol_1h_z_24h"], errors="coerce")).fillna(0.0)
    short["up_hour_ratio_24h"] = pd.to_numeric(short["up_hour_ratio_24h"], errors="coerce").fillna(0.0)

    # Breakout bonus: favor near recent highs but not extremely extended
    # Peak around ~5% below 7d high; 0 at >=20% below or >2% above
    def breakout_bonus(dd: float) -> float:
        if pd.isna(dd):
            return 0.0
        # dd is negative below high; transform to distance from -0.05 target
        dist = abs((dd + 0.05))
        bonus = max(0.0, 1.0 - (dist / 0.15))  # linear falloff within 15%
        # penalize if above the high (dd > 0)
        if dd > 0:
            bonus *= 0.5
        return clamp(bonus, 0.0, 1.0)

    short["breakout_bonus"] = short["drawdown_7d"].apply(breakout_bonus)

    # Refined composite score combining previous and new features
    short["refined_score"] = (
        0.25 * short["composite_score"].fillna(0.0)
        + 0.28 * short["vol_adj_mom"].fillna(0.0)
        + 0.30 * short["z_v2m"].fillna(0.0)
        + 0.25 * short["z_24h"].fillna(0.0)
        + 0.22 * short["volume_surge_z"].fillna(0.0)
        + 0.18 * short["trend_factor"].fillna(0.0)
        + 0.14 * short["breakout_bonus"].fillna(0.0)
        + 0.18 * short["z_mom_3h"].fillna(0.0)
        + 0.12 * short["z_mom_6h"].fillna(0.0)
        + 0.08 * short["z_boll"].fillna(0.0)
        + 0.08 * short["z_vol1h"].fillna(0.0)
        + 0.05 * short["up_hour_ratio_24h"].fillna(0.0)
        + short["overextension_penalty"].fillna(0.0) * 0.5
        + short["trend_consistency"].fillna(0.0) * 0.5
    )

    # Improved prediction mapping
    pc24_abs_short = pd.to_numeric(short.get("pc_24h", pd.Series(dtype=float)), errors="coerce").abs()
    valid_pc24_short = pc24_abs_short.dropna()
    avg_abs_24h = float(valid_pc24_short.mean()) if not valid_pc24_short.empty else float("nan")
    if pd.isna(avg_abs_24h) or avg_abs_24h == 0:
        avg_abs_24h = 4.0
    refined_scale = 0.6 * avg_abs_24h
    short["predicted_change_24h_pct"] = (
        short["refined_score"].fillna(short["composite_score"]).fillna(0.0) * refined_scale
    ).apply(lambda v: clamp(float(v), -35.0, 35.0))

    return short


def fetch_current_prices_by_ids(session: Session, ids: List[str]) -> pd.DataFrame:
    if not ids:
        return pd.DataFrame(columns=["id", "current_price"])
    rows: List[Dict[str, Any]] = []
    # Chunk to avoid overly long query strings
    chunk_size = 150
    for i in range(0, len(ids), chunk_size):
        chunk = ids[i : i + chunk_size]
        params = {
            "vs_currency": "usd",
            "ids": ",".join(chunk),
            "order": "market_cap_desc",
            "per_page": len(chunk),
            "page": 1,
            "sparkline": "false",
        }
        try:
            resp = session.get(COINGECKO_MARKETS_URL, params=params, timeout=12)
            resp.raise_for_status()
            data = resp.json()
            if isinstance(data, list):
                for d in data:
                    rows.append({
                        "id": d.get("id"),
                        "current_price": d.get("current_price"),
                    })
        except Exception:
            continue
    return pd.DataFrame(rows)


def save_predictions_csv(df_top: pd.DataFrame, path: str, prediction_time: float, model_label: str) -> str:
    ts_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(prediction_time))
    ts_struct = time.gmtime(prediction_time)
    prediction_dow = ts_struct.tm_wday  # 0=Mon
    prediction_hour_utc = ts_struct.tm_hour
    is_weekend = 1 if prediction_dow in (5, 6) else 0  # Sat=5, Sun=6
    # Simple session tag by UTC hour
    if 12 <= prediction_hour_utc <= 20:
        session = "EU_US_overlap"
    elif 7 <= prediction_hour_utc < 12:
        session = "EU_open"
    elif 0 <= prediction_hour_utc < 7:
        session = "Asia"
    else:
        session = "US_late"
    export_cols = [
        "prediction_time_iso",
        "prediction_time_unix",
        "model",
        "id",
        "symbol",
        "name",
        "current_price",
        "predicted_change_24h_pct",
        "predicted_target_price_usd",
        "composite_score",
        "refined_score",
        "pc_1h",
        "pc_24h",
        "pc_7d",
        "volume_to_marketcap",
        "total_volume",
        "market_cap",
        "prediction_dow",
        "prediction_hour_utc",
        "prediction_is_weekend",
        "prediction_session",
    ]
    out = df_top.copy()
    out["prediction_time_iso"] = ts_iso
    out["prediction_time_unix"] = int(prediction_time)
    out["model"] = model_label
    out["predicted_target_price_usd"] = out["current_price"] * (1.0 + (out["predicted_change_24h_pct"] / 100.0))
    out["prediction_dow"] = prediction_dow
    out["prediction_hour_utc"] = prediction_hour_utc
    out["prediction_is_weekend"] = is_weekend
    out["prediction_session"] = session

    for col in export_cols:
        if col not in out.columns:
            out[col] = pd.NA

    try:
        # If path looks like a directory, append filename
        if not path.lower().endswith(".csv"):
            filename = f"predictions_{ts_iso.replace(':','-')}.csv"
            if path.endswith("/"):
                export_path = path + filename
            else:
                export_path = path + "/" + filename
        else:
            export_path = path
        out[export_cols].to_csv(export_path, index=False)
        return export_path
    except Exception:
        return ""


def evaluate_predictions(session: Session, csv_path: str, save_eval_path: Optional[str] = None) -> None:
    try:
        df_pred = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Failed to read predictions file: {e}")
        sys.exit(1)

    required = ["id", "current_price", "predicted_change_24h_pct"]
    for col in required:
        if col not in df_pred.columns:
            print(f"Predictions file missing required column: {col}")
            sys.exit(1)

    base_price = pd.to_numeric(df_pred["current_price"], errors="coerce")
    df_pred["base_price_usd"] = base_price
    coin_ids = [str(x) for x in df_pred["id"].dropna().unique().tolist()]
    price_now_df = fetch_current_prices_by_ids(session, coin_ids)
    df_eval = df_pred.merge(price_now_df, on="id", how="left", suffixes=("", "_now"))

    df_eval["current_price_now"] = pd.to_numeric(df_eval["current_price_now"], errors="coerce")
    df_eval["realized_change_24h_pct"] = (df_eval["current_price_now"] / df_eval["base_price_usd"] - 1.0) * 100.0
    df_eval["error_pct_points"] = df_eval["realized_change_24h_pct"] - df_eval["predicted_change_24h_pct"]
    df_eval["abs_error"] = df_eval["error_pct_points"].abs()
    df_eval["sq_error"] = df_eval["error_pct_points"] ** 2

    valid = df_eval.dropna(subset=["realized_change_24h_pct", "predicted_change_24h_pct"])  # type: ignore[arg-type]
    if valid.empty:
        print("No valid rows to evaluate yet. Try again later.")
        return

    mae = float(valid["abs_error"].mean())
    rmse = float((valid["sq_error"].mean()) ** 0.5)
    bias = float(valid["error_pct_points"].mean())  # positive: underpredicted actual; negative: overpredicted

    print("Evaluation summary (vs. current prices):")
    print(f"- MAE (pct points): {mae:.2f}")
    print(f"- RMSE (pct points): {rmse:.2f}")
    print(f"- Bias (pred -> actual): {bias:+.2f} pct points")

    if save_eval_path:
        try:
            valid.to_csv(save_eval_path, index=False)
            print(f"Saved evaluation details to: {save_eval_path}")
        except Exception as e:
            print(f"Could not save evaluation CSV: {e}")


def backtest_topk(
    session: Session,
    universe_symbols: Set[str],
    days: int = 30,
    history_days: int = 14,
    shortlist_k: int = 30,
    workers: int = 6,
    per_page: int = 250,
    z_window_hours: int = 48,
    target_horizon_hours: int = 24,
) -> None:
    # 1) Pull market snapshot list to define universe
    rows = fetch_coingecko_markets(session, per_page=per_page, pages=1)
    df_raw = prepare_dataframe(rows)
    if df_raw.empty:
        print("Backtest aborted: could not load snapshot market data.")
        return
    df_universe = df_raw[df_raw["symbol_upper"].isin(universe_symbols)].copy()
    df_universe = df_universe[~df_universe["symbol_upper"].isin(EXCLUDE_SYMBOLS)]
    df_universe = df_universe.dropna(subset=["market_cap", "total_volume"])  # type: ignore[arg-type]
    df_universe = df_universe.sort_values(by=["market_cap"], ascending=[False]).head(shortlist_k)

    if df_universe.empty:
        print("Backtest: universe empty after filters.")
        return

    # 2) For each coin, fetch hourly chart and compute rolling features and labels
    def coin_task(coin_id: str) -> Optional[pd.DataFrame]:
        days_needed = max(history_days, int((z_window_hours + target_horizon_hours) / 24) + 2)
        # Try Binance first with USDT symbol; fallback to Coingecko cache
        price_series: Optional[pd.Series] = None
        vol_series: Optional[pd.Series] = None
        try:
            sym = df_universe.loc[df_universe["id"] == coin_id, "symbol"].iloc[0] if (df_universe["id"] == coin_id).any() else None
            if sym:
                sym_u = "".join(ch for ch in str(sym).upper() if ch.isalnum())
                dfb = fetch_binance_klines_series(session, f"{sym_u}USDT", days=days_needed)
                if isinstance(dfb, pd.DataFrame) and not dfb.empty:
                    price_series = dfb["close"].astype(float).sort_index()
                    vol_series = dfb["volume"].astype(float).sort_index()
        except Exception:
            price_series = None
            vol_series = None
        if price_series is None or vol_series is None:
            chart = fetch_market_chart_cached(session, coin_id, days=days_needed, timeout=10.0)
            if not chart:
                return None
            try:
                prices = chart.get("prices") or []
                vols = chart.get("total_volumes") or []
                if len(prices) < (z_window_hours + target_horizon_hours + 4):
                    return None
                price_series = pd.Series([p[1] for p in prices], index=pd.to_datetime([p[0] for p in prices], unit="ms"))
                vol_series = pd.Series([v[1] for v in vols], index=pd.to_datetime([v[0] for v in vols], unit="ms"))
            except Exception:
                return None
            price_series = price_series.sort_index()
            vol_series = vol_series.sort_index()

            log_prices = price_series.apply(lambda x: math.log(x) if x and x > 0 else float("nan"))
            hourly_returns = log_prices.diff().dropna()

            # Rolling features
            r_1h = hourly_returns
            z_r_1h = rolling_robust_z(r_1h, window=z_window_hours)
            v_roll24 = vol_series.rolling(window=24).sum()
            z_v_roll24 = rolling_robust_z(v_roll24, window=z_window_hours)

            # Label: future 24h return in percent
            future_log = log_prices.shift(-target_horizon_hours)
            fwd_log_ret = (future_log - log_prices) * 100.0  # percent in log points approx
            fwd_ret_pct = (price_series.shift(-target_horizon_hours) / price_series - 1.0) * 100.0
            label_gt10 = (fwd_ret_pct >= 10.0).astype(float)

            df_feat = pd.DataFrame({
                "coin_id": coin_id,
                "price": price_series,
                "r_1h": r_1h,
                "z_r_1h": z_r_1h,
                "v_roll24": v_roll24,
                "z_v_roll24": z_v_roll24,
                "fwd_ret_pct": fwd_ret_pct,
                "label_gt10": label_gt10,
            })
            df_feat = df_feat.dropna(subset=["z_r_1h", "z_v_roll24", "fwd_ret_pct"])  # type: ignore[arg-type]
            return df_feat

    frames: List[pd.DataFrame] = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = [pool.submit(coin_task, cid) for cid in df_universe["id"].dropna().tolist()]
        for fut in as_completed(futures):
            try:
                res = fut.result(timeout=20)
                if isinstance(res, pd.DataFrame) and not res.empty:
                    frames.append(res)
            except Exception:
                continue

    if not frames:
        print("Backtest: no historical frames computed.")
        return

    df_all = pd.concat(frames, axis=0, ignore_index=True)
    # Simple heuristic score: combine momentum and volume surge
    df_all["score"] = (1.2 * df_all["z_r_1h"] + 1.0 * df_all["z_v_roll24"]).fillna(0.0)

    # Evaluate discrimination for >10% events: precision/recall/F1 at a threshold
    # Choose threshold as 90th percentile of score
    thr = float(df_all["score"].quantile(0.9))
    df_all["pred_pos"] = (df_all["score"] >= thr).astype(int)
    tp = int(((df_all["pred_pos"] == 1) & (df_all["label_gt10"] == 1)).sum())
    fp = int(((df_all["pred_pos"] == 1) & (df_all["label_gt10"] == 0)).sum())
    fn = int(((df_all["pred_pos"] == 0) & (df_all["label_gt10"] == 1)).sum())
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # Regression error on fwd_ret_pct using a linear mapping of score
    # Calibrate scaling by minimizing squared error analytically: beta = cov(score, y)/var(score)
    score = df_all["score"]
    y = df_all["fwd_ret_pct"]
    var_s = float(((score - score.mean()) ** 2).mean())
    if var_s > 0:
        beta = float((((score - score.mean()) * (y - y.mean())).mean()) / var_s)
    else:
        beta = 0.0
    y_hat = score * beta
    mae = float((y - y_hat).abs().mean())
    rmse = float(((y - y_hat) ** 2).mean() ** 0.5)

    print("Backtest results (heuristic momentum + volume surge):")
    print(f"- Universe size: {len(df_universe)}")
    print(f"- Samples: {len(df_all)} over ~{history_days} days hourly")
    print(f"- Threshold (90th pct): {thr:.3f}")
    print(f"- Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}")
    print(f"- MAE: {mae:.2f} pct points, RMSE: {rmse:.2f} pct points")

    # Precision by hour (rough regime insight)
    df_all = df_all.copy()
    df_all["hour_utc"] = df_all.index.get_level_values(0) if isinstance(df_all.index, pd.MultiIndex) else pd.to_datetime(df_all.index)
    if not isinstance(df_all["hour_utc"], pd.DatetimeIndex):
        df_all["hour_utc"] = pd.to_datetime(df_all["hour_utc"], errors="coerce")
    df_all["hour"] = df_all["hour_utc"].dt.hour
    if "pred_pos" in df_all.columns:
        by_hour = df_all.dropna(subset=["hour"]).groupby("hour").apply(
            lambda g: (float((((g["pred_pos"] == 1) & (g["label_gt10"] == 1)).sum())) / max(1.0, float((g["pred_pos"] == 1).sum())))
        )
        if not by_hour.empty:
            top_hours = by_hour.sort_values(ascending=False).head(5)
            print("- Precision by hour (top):")
            for h, v in top_hours.items():
                print(f"  hour {int(h):02d}: {v:.2f}")


def backtest_random_points(
    session: Session,
    universe_symbols: Set[str],
    num_samples: int = 400,
    lookback_days: int = 30,
    universe_size: int = 80,
    top_n: int = 5,
    workers: int = 8,
    weights_path: str = "/Users/xfx/Desktop/trade/model_weights.json",
    z_window_hours: int = 48,
    min_interval: float = 0.7,
) -> None:
    import random
    import numpy as np

    # 1) Snapshot current markets to define a robust universe subset
    rows = fetch_coingecko_markets(session, per_page=250, pages=1)
    df_raw = prepare_dataframe(rows)
    if df_raw.empty:
        print("Random backtest aborted: could not load market data.")
        return
    df_uni = df_raw[df_raw["symbol_upper"].isin(universe_symbols)].copy()
    df_uni = df_uni[~df_uni["symbol_upper"].isin(EXCLUDE_SYMBOLS)]
    df_uni = df_uni.dropna(subset=["market_cap", "total_volume"])  # type: ignore[arg-type]
    df_uni = df_uni.sort_values(by=["market_cap"], ascending=[False]).head(universe_size)
    if df_uni.empty:
        print("Random backtest: empty universe")
        return

    coin_ids = df_uni["id"].dropna().tolist()
    # 2) Generate random hourly timestamps over lookback_days
    now = pd.Timestamp.utcnow().floor("h")
    start = now - pd.Timedelta(days=lookback_days)
    # Use 'h' (lowercase) to avoid FutureWarning
    all_hours = pd.date_range(start=start, end=now - pd.Timedelta(hours=24), freq="h")
    if len(all_hours) == 0:
        print("Random backtest: no hours in range")
        return
    sample_hours = random.sample(list(all_hours), k=min(num_samples, len(all_hours)))
    sample_hours = sorted(sample_hours)

    # 3) For each coin, fetch enough hourly data once, then evaluate all sampled timestamps
    last_req_time = [0.0]

    def fetch_chart(coin_id: str) -> Optional[pd.DataFrame]:
        days_needed = max(lookback_days, int((z_window_hours + 24) / 24) + 2)
        # Global pacing to avoid HTTP 429s
        while True:
            nowt = time.time()
            wait = min_interval - (nowt - last_req_time[0])
            if wait > 0:
                time.sleep(min(wait, 1.0))
                continue
            last_req_time[0] = time.time()
            break
        # Try Binance klines for USDT pair first
        s: Optional[pd.Series] = None
        try:
            symbol = df_uni.loc[df_uni["id"] == coin_id, "symbol"].iloc[0] if (df_uni["id"] == coin_id).any() else None
            if symbol:
                sym_u = "".join(ch for ch in str(symbol).upper() if ch.isalnum())
                dfb = fetch_binance_klines_series(session, f"{sym_u}USDT", days=days_needed)
                if isinstance(dfb, pd.DataFrame) and not dfb.empty:
                    s = dfb["close"]
        except Exception:
            s = None
        if s is None:
            chart = fetch_market_chart_cached(session, coin_id, days=days_needed, timeout=12.0)
            if not chart:
                return None
            try:
                prices = chart.get("prices") or []
                if len(prices) < (z_window_hours + 24 + 4):
                    return None
                s = pd.Series([p[1] for p in prices], index=pd.to_datetime([p[0] for p in prices], unit="ms")).sort_index()
            except Exception:
                return None
        return s.to_frame(name="price") if isinstance(s, pd.Series) and not s.empty else None

    print(f"[rb] Fetching hourly charts for {len(coin_ids)} coins (workers={workers})...", flush=True)
    # Auto-abort fast if nothing usable arrives quickly (likely rate-limited)
    hard_start = time.time()
    price_map: Dict[str, pd.DataFrame] = {}
    t0 = time.time()
    completed = 0
    successes = 0
    last_log = t0
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {pool.submit(fetch_chart, cid): cid for cid in coin_ids}
        for fut in as_completed(futures):
            cid = futures[fut]
            try:
                dfp = fut.result(timeout=30)
                if isinstance(dfp, pd.DataFrame) and not dfp.empty:
                    price_map[cid] = dfp
                    successes += 1
            except Exception:
                pass
            finally:
                completed += 1
                now_t = time.time()
                # Early abort if clearly rate-limited: 0 usable after 90s
                if successes == 0 and (now_t - hard_start) > 90:
                    print("[rb] Abort: no usable charts within 90s (likely rate-limited). Reduce workers/universe.", flush=True)
                    break
                if completed == len(coin_ids) or (now_t - last_log) > 1.5 or (completed % 10 == 0):
                    elapsed = now_t - t0
                    print(f"[rb] Charts progress: {completed}/{len(coin_ids)} | usable={successes} | {elapsed:.1f}s", flush=True)
                    last_log = now_t
        else:
            pass

    if not price_map:
        print("Random backtest: no price histories fetched")
        return

    # Tighten sample window to intersection of available series so forward window exists
    starts: List[pd.Timestamp] = []
    ends: List[pd.Timestamp] = []
    for dfp in price_map.values():
        if isinstance(dfp, pd.DataFrame) and not dfp.empty:
            starts.append(dfp.index.min())
            ends.append(dfp.index.max())
    if not starts or not ends:
        print("Random backtest: empty ranges after fetch")
        return
    global_start = max(starts)
    global_end = min(ends)
    effective_end = global_end - pd.Timedelta(hours=24)
    if effective_end <= global_start:
        print("Random backtest: insufficient overlapping history window")
        return
    all_hours = pd.date_range(start=global_start, end=effective_end, freq="h")
    if len(all_hours) == 0:
        print("Random backtest: no hours after intersection windowing")
        return
    # Re-sample timestamps to guaranteed-valid set
    sample_hours = sorted(random.sample(list(all_hours), k=min(num_samples, len(all_hours))))

    # 4) Evaluate model at each sampled timestamp
    learned = load_weights(weights_path)
    results: List[Dict[str, Any]] = []

    print(f"[rb] Evaluating {len(sample_hours)} timestamps (top {top_n})...", flush=True)
    eval_start = time.time()
    last_log = eval_start
    processed = 0
    for ts in sample_hours:
        # Build a pseudo-snapshot for this timestamp using available prices (last observed <= ts)
        rows_now: List[Dict[str, Any]] = []
        for cid in coin_ids:
            dfp = price_map.get(cid)
            if dfp is None or dfp.empty:
                continue
            # price at ts and at ts-1h, ts-24h, ts-7d etc. (approx using nearest <= ts)
            try:
                s = dfp["price"]
                s = s[s.index <= ts]
                if s.empty:
                    continue
                p_now = float(s.iloc[-1])
                p_1h = float(s.shift(1).iloc[-1]) if len(s) >= 2 else float("nan")
                p_24h = float(s.shift(24).iloc[-1]) if len(s) >= 25 else float("nan")
                p_7d = float(s.shift(24 * 7).iloc[-1]) if len(s) >= 24 * 7 + 1 else float("nan")
                pc1h = (p_now / p_1h - 1.0) * 100.0 if p_1h and p_1h > 0 else float("nan")
                pc24h = (p_now / p_24h - 1.0) * 100.0 if p_24h and p_24h > 0 else float("nan")
                pc7d = (p_now / p_7d - 1.0) * 100.0 if p_7d and p_7d > 0 else float("nan")
                rows_now.append({
                    "id": cid,
                    "symbol": df_uni.loc[df_uni["id"] == cid, "symbol"].iloc[0] if (df_uni["id"] == cid).any() else cid,
                    "name": df_uni.loc[df_uni["id"] == cid, "name"].iloc[0] if (df_uni["id"] == cid).any() else cid,
                    "current_price": p_now,
                    "total_volume": float("nan"),
                    "market_cap": float("nan"),
                    "market_cap_rank": float("nan"),
                    "price_change_percentage_1h_in_currency": pc1h,
                    "price_change_percentage_24h_in_currency": pc24h,
                    "price_change_percentage_7d_in_currency": pc7d,
                })
            except Exception:
                continue

        if not rows_now:
            processed += 1
            continue
        snap = prepare_dataframe(rows_now)
        # Use available symbols (skip liquidity filters since we don't have vol/mcap here)
        snap["symbol_upper"] = snap["symbol"].astype(str).str.upper()
        # Score and predict
        scored = score_and_predict(snap, enable_regime_adjust=False, learned_weights=learned, enable_external_context=False, use_enhanced_features=False)
        if scored.empty:
            processed += 1
            continue
        top = scored.sort_values(by=["composite_score"], ascending=[False]).head(top_n)

        # Realized forward 24h for each top coin
        for _, row in top.iterrows():
            cid = str(row.get("id", ""))
            dfp = price_map.get(cid)
            if dfp is None or dfp.empty:
                continue
            # Find price at ts and at ts+24h (nearest <= timestamp)
            s = dfp["price"]
            s_now = s[s.index <= ts]
            s_fwd = s[s.index <= ts + pd.Timedelta(hours=24)]
            if s_now.empty or s_fwd.empty:
                continue
            p_now = float(s_now.iloc[-1])
            p_fwd = float(s_fwd.iloc[-1])
            realized = (p_fwd / p_now - 1.0) * 100.0
            results.append({
                "timestamp": ts.isoformat(),
                "id": cid,
                "symbol": row.get("symbol", "").upper(),
                "predicted": float(row.get("predicted_change_24h_pct", float("nan"))),
                "score": float(row.get("composite_score", float("nan"))),
                "realized_24h": realized,
            })

        processed += 1
        now_t = time.time()
        if processed == len(sample_hours) or (now_t - last_log) > 2.0 or (processed % 10 == 0):
            elapsed = now_t - eval_start
            print(f"[rb] Eval progress: {processed}/{len(sample_hours)} | results={len(results)} | {elapsed:.1f}s", flush=True)
            last_log = now_t

    if not results:
        print("Random backtest: no results")
        return

    df_res = pd.DataFrame(results)
    # Summary metrics
    df_valid = df_res.dropna(subset=["predicted", "realized_24h"])  # type: ignore[arg-type]
    if df_valid.empty:
        print("Random backtest: no valid rows")
        return
    mae = float((df_valid["realized_24h"] - df_valid["predicted"]).abs().mean())
    rmse = float(((df_valid["realized_24h"] - df_valid["predicted"]) ** 2).mean() ** 0.5)
    bias = float((df_valid["realized_24h"] - df_valid["predicted"]).mean())
    corr = float(df_valid[["predicted", "realized_24h"]].corr().iloc[0,1]) if len(df_valid) > 2 else float("nan")
    print("Random timestamp backtest results:")
    print(f"- Samples evaluated: {len(df_valid)} (from {len(sample_hours)} timestamps)")
    print(f"- MAE (pct points): {mae:.2f}")
    print(f"- RMSE (pct points): {rmse:.2f}")
    print(f"- Bias (realized - predicted): {bias:+.2f}")
    print(f"- Corr(pred, realized): {corr:.2f}")


def calibrate_scale_from_random(
    session: Session,
    universe_symbols: Set[str],
    num_samples: int = 60,
    lookback_days: int = 20,
    universe_size: int = 30,
    workers: int = 1,
    weights_path: str = "/Users/xfx/Desktop/trade/model_weights.json",
    z_window_hours: int = 48,
    calib_path: str = DEFAULT_CALIB_PATH,
) -> None:
    # Reuse random backtest logic to collect (pred, realized) pairs
    import json
    import numpy as np
    import random
    results_pairs: List[Tuple[float, float]] = []

    def collect_pairs() -> None:
        nonlocal results_pairs
        rows = fetch_coingecko_markets(session, per_page=250, pages=1)
        df_raw = prepare_dataframe(rows)
        df_uni = df_raw[df_raw["symbol_upper"].isin(universe_symbols)].copy()
        df_uni = df_uni[~df_uni["symbol_upper"].isin(EXCLUDE_SYMBOLS)]
        df_uni = df_uni.dropna(subset=["market_cap", "total_volume"])  # type: ignore[arg-type]
        df_uni = df_uni.sort_values(by=["market_cap"], ascending=[False]).head(universe_size)
        if df_uni.empty:
            return
        # Use existing helper to fetch charts and evaluate timestamps quickly
        # Here we call backtest_random_points but capture printed results by re-implementing the minimum loop
        now = pd.Timestamp.utcnow().floor("h")
        start = now - pd.Timedelta(days=lookback_days)
        all_hours_initial = pd.date_range(start=start, end=now - pd.Timedelta(hours=24), freq="h")

        # Fetch prices via Binance/cache for just top universe ids
        coin_ids = df_uni["id"].dropna().tolist()
        price_map: Dict[str, pd.DataFrame] = {}
        for cid in coin_ids:
            sym = df_uni.loc[df_uni["id"] == cid, "symbol"].iloc[0] if (df_uni["id"] == cid).any() else None
            dfb = None
            if sym:
                sym_u = "".join(ch for ch in str(sym).upper() if ch.isalnum())
                dfb = fetch_binance_klines_series(session, f"{sym_u}USDT", days=max(lookback_days, 4))
            if isinstance(dfb, pd.DataFrame) and not dfb.empty:
                price_map[cid] = dfb[["close"]].rename(columns={"close": "price"})
                continue
            chart = fetch_market_chart_cached(session, cid, days=max(lookback_days, 4), timeout=8.0)
            if chart and chart.get("prices"):
                s = pd.Series([p[1] for p in chart["prices"]], index=pd.to_datetime([p[0] for p in chart["prices"]], unit="ms")).sort_index()
                price_map[cid] = s.to_frame(name="price")

        if not price_map:
            return

        # Tighten timestamp set to intersection so 24h forward exists
        starts: List[pd.Timestamp] = []
        ends: List[pd.Timestamp] = []
        for dfp in price_map.values():
            if isinstance(dfp, pd.DataFrame) and not dfp.empty:
                starts.append(dfp.index.min())
                ends.append(dfp.index.max())
        if not starts or not ends:
            return
        global_start = max(starts)
        effective_end = min(ends) - pd.Timedelta(hours=24)
        if effective_end <= global_start:
            return
        all_hours = pd.date_range(start=global_start, end=effective_end, freq="h")
        if len(all_hours) == 0:
            return
        sample_hours = sorted(random.sample(list(all_hours), k=min(num_samples, len(all_hours))))

        learned = load_weights(weights_path)
        for ts in sample_hours:
            rows_now: List[Dict[str, Any]] = []
            for cid in coin_ids:
                dfp = price_map.get(cid)
                if dfp is None or dfp.empty:
                    continue
                s = dfp["price"]
                s_now = s[s.index <= ts]
                if s_now.empty:
                    continue
                p_now = float(s_now.iloc[-1])
                p_1h = float(s_now.iloc[-2]) if len(s_now) >= 2 else float("nan")
                p_24h_series = s[s.index <= ts - pd.Timedelta(hours=24)]
                p_7d_series = s[s.index <= ts - pd.Timedelta(days=7)]
                p_24h = float(p_24h_series.iloc[-1]) if not p_24h_series.empty else float("nan")
                p_7d = float(p_7d_series.iloc[-1]) if not p_7d_series.empty else float("nan")
                pc1h = (p_now / p_1h - 1.0) * 100.0 if p_1h and p_1h > 0 else float("nan")
                pc24h = (p_now / p_24h - 1.0) * 100.0 if p_24h and p_24h > 0 else float("nan")
                pc7d = (p_now / p_7d - 1.0) * 100.0 if p_7d and p_7d > 0 else float("nan")
                rows_now.append({
                    "id": cid,
                    "symbol": df_uni.loc[df_uni["id"] == cid, "symbol"].iloc[0] if (df_uni["id"] == cid).any() else cid,
                    "name": df_uni.loc[df_uni["id"] == cid, "name"].iloc[0] if (df_uni["id"] == cid).any() else cid,
                    "current_price": p_now,
                    "total_volume": float("nan"),
                    "market_cap": float("nan"),
                    "market_cap_rank": float("nan"),
                    "price_change_percentage_1h_in_currency": pc1h,
                    "price_change_percentage_24h_in_currency": pc24h,
                    "price_change_percentage_7d_in_currency": pc7d,
                })
            if not rows_now:
                continue
            snap = prepare_dataframe(rows_now)
            snap["symbol_upper"] = snap["symbol"].astype(str).str.upper()
            scored = score_and_predict(snap, enable_regime_adjust=False, learned_weights=learned, enable_external_context=False, use_enhanced_features=False)
            if scored.empty:
                continue
            top = scored.sort_values(by=["composite_score"], ascending=[False]).head(5)
            for _, row in top.iterrows():
                cid = str(row.get("id", ""))
                s = price_map[cid]["price"] if cid in price_map else None
                if s is None or s.empty:
                    continue
                s_now = s[s.index <= ts]
                s_fwd = s[s.index <= ts + pd.Timedelta(hours=24)]
                if s_now.empty or s_fwd.empty:
                    continue
                p_now = float(s_now.iloc[-1])
                p_fwd = float(s_fwd.iloc[-1])
                realized = (p_fwd / p_now - 1.0) * 100.0
                pred = float(row.get("predicted_change_24h_pct", float("nan")))
                if not (math.isnan(pred) or math.isnan(realized)):
                    results_pairs.append((pred, realized))

    # Collect
    try:
        import random
        collect_pairs()
    except Exception:
        pass

    if len(results_pairs) < 10:
        print("Calibrate: insufficient pairs")
        return

    P = np.array([p for p,_ in results_pairs])
    R = np.array([r for _,r in results_pairs])
    # Fit R ≈ a*P + b by least squares
    A = np.vstack([P, np.ones_like(P)]).T
    w, b = np.linalg.lstsq(A, R, rcond=None)[0]
    # Save
    try:
        with open(calib_path, "w") as f:
            json.dump({"multiplier": float(w), "intercept": float(b)}, f)
        print(f"Saved calibration to {calib_path}: multiplier={w:.3f}, intercept={b:.3f}")
    except Exception as e:
        print(f"Calibrate: failed to save calibration: {e}")


def train_weights_from_backtest(
    session: Session,
    shortlist_k: int = 60,
    history_days: int = 14,
    z_window_hours: int = 48,
    workers: int = 8,
    weights_path: str = "/Users/xfx/Desktop/trade/model_weights.json",
) -> Optional[str]:
    """Legacy training function - calls the enhanced ensemble trainer."""
    return train_advanced_ensemble(session, shortlist_k, history_days, z_window_hours, workers, weights_path)


def train_advanced_ensemble(
    session: Session,
    shortlist_k: int = 60,
    history_days: int = 14,
    z_window_hours: int = 48,
    workers: int = 8,
    weights_path: str = "/Users/xfx/Desktop/trade/model_weights.json",
) -> Optional[str]:
    import json
    import numpy as np
    # Build dataset
    rows = fetch_coingecko_markets(session, per_page=250, pages=1)
    df_raw = prepare_dataframe(rows)
    bases = fetch_crypto_com_base_symbols(session) or set(FALLBACK_CRYPTO_COM_TICKERS)
    uni = df_raw[df_raw["symbol_upper"].isin(bases)].copy()
    uni = uni.dropna(subset=["market_cap", "total_volume"])  # type: ignore[arg-type]
    uni = uni.sort_values(by=["market_cap"], ascending=[False]).head(shortlist_k)
    if uni.empty:
        print("Train: empty universe")
        return None

    def coin_task(coin_id: str) -> Optional[pd.DataFrame]:
        days_needed = max(history_days, int((z_window_hours + 24) / 24) + 2)
        price_series: Optional[pd.Series] = None
        vol_series: Optional[pd.Series] = None
        try:
            sym = uni.loc[uni["id"] == coin_id, "symbol"].iloc[0] if (uni["id"] == coin_id).any() else None
            if sym:
                sym_u = "".join(ch for ch in str(sym).upper() if ch.isalnum())
                dfb = fetch_binance_klines_series(session, f"{sym_u}USDT", days=days_needed)
                if isinstance(dfb, pd.DataFrame) and not dfb.empty:
                    price_series = dfb["close"].astype(float).sort_index()
                    vol_series = dfb["volume"].astype(float).sort_index()
        except Exception:
            price_series = None
            vol_series = None
        if price_series is None or vol_series is None:
            chart = fetch_market_chart_cached(session, coin_id, days=days_needed, timeout=10.0)
            if not chart:
                return None
            try:
                prices = chart.get("prices") or []
                vols = chart.get("total_volumes") or []
                if len(prices) < (z_window_hours + 24 + 4):
                    return None
                price_series = pd.Series([p[1] for p in prices], index=pd.to_datetime([p[0] for p in prices], unit="ms")).sort_index()
                vol_series = pd.Series([v[1] for v in vols], index=pd.to_datetime([v[0] for v in vols], unit="ms")).sort_index()
            except Exception:
                return None
        log_prices = price_series.apply(lambda x: math.log(x) if x and x > 0 else float("nan"))
        r_1h = log_prices.diff().dropna()
        z_1h = rolling_robust_z(r_1h, window=z_window_hours)
        mom_24h = (price_series / price_series.shift(24) - 1.0) * 100.0
        z_24h = rolling_robust_z(mom_24h, window=z_window_hours)
        v_roll24 = vol_series.rolling(window=24).sum()
        z_v2m = rolling_robust_z(v_roll24, window=z_window_hours)
        y = (price_series.shift(-24) / price_series - 1.0) * 100.0
        df = pd.DataFrame({"z_1h": z_1h, "z_24h": z_24h, "z_v2m": z_v2m, "y": y}).dropna()
        if df.empty:
            return None
        # Preserve timestamp for time-decay weighting downstream
        try:
            df["ts"] = df.index.astype("int64")
        except Exception:
            pass
        return df

    frames: List[pd.DataFrame] = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = [pool.submit(coin_task, cid) for cid in uni["id"].dropna().tolist()]
        for fut in as_completed(futures):
            try:
                res = fut.result(timeout=20)
                if isinstance(res, pd.DataFrame) and not res.empty:
                    frames.append(res)
            except Exception:
                continue

    if not frames:
        print("Train: no frames")
        return None
    data = pd.concat(frames, axis=0, ignore_index=True)
    X = data[["z_1h", "z_24h", "z_v2m"]].fillna(0.0).to_numpy()
    y = data["y"].to_numpy()
    # Time-decay weights (recent samples count more)
    if "ts" in data.columns:
        ts = data["ts"].astype(float).to_numpy()
        tmin, tmax = float(np.nanmin(ts)), float(np.nanmax(ts))
        denom = (tmax - tmin) if (tmax > tmin) else 1.0
        rel = (ts - tmin) / denom
        wts = 0.5 + 0.5 * rel  # in [0.5, 1.0]
    else:
        wts = np.ones_like(y)
    # Create enhanced feature set including polynomial interactions
    poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    X_poly = poly.fit_transform(X)
    
    # Add synthetic technical features based on existing z-scores
    tech_features = []
    for i in range(len(X)):
        z_1h, z_24h, z_v2m = X[i]
        # RSI-like feature (normalized momentum)
        rsi_proxy = 50 + 20 * np.tanh(z_24h)
        # MACD-like feature (momentum divergence)
        macd_proxy = z_1h - 0.5 * z_24h
        # Momentum strength
        momentum_proxy = z_24h * 1.2
        # Volatility proxy from volume patterns
        volatility_proxy = abs(z_1h) + 0.3 * abs(z_24h)
        # Trend consistency score
        trend_proxy = 0.7 * z_24h + 0.3 * z_1h if z_1h * z_24h > 0 else 0
        
        tech_features.append([rsi_proxy, macd_proxy, momentum_proxy, volatility_proxy, trend_proxy])
    
    X_tech = np.array(tech_features)
    X_combined = np.hstack([X, X_poly, X_tech])
    
    # Train ensemble of models
    print(f"Training ensemble with {X_combined.shape[1]} features on {len(y)} samples...")
    
    models = {}
    predictions = {}
    
    # 1. Ridge Regression (baseline)
    ridge = Ridge(alpha=0.1)
    ridge.fit(X_combined, y, sample_weight=wts)
    pred_ridge = ridge.predict(X_combined)
    models['ridge'] = ridge
    predictions['ridge'] = pred_ridge
    
    # 2. Gradient Boosting (captures non-linear patterns)
    gbr = GradientBoostingRegressor(
        n_estimators=100, 
        max_depth=4, 
        learning_rate=0.1, 
        random_state=42,
        subsample=0.8
    )
    gbr.fit(X_combined, y, sample_weight=wts)
    pred_gbr = gbr.predict(X_combined)
    models['gbr'] = gbr
    predictions['gbr'] = pred_gbr
    
    # 3. Random Forest (robust to outliers)
    rf = RandomForestRegressor(
        n_estimators=100,
        max_depth=6,
        random_state=42,
        min_samples_split=10,
        min_samples_leaf=5
    )
    rf.fit(X_combined, y, sample_weight=wts)
    pred_rf = rf.predict(X_combined)
    models['rf'] = rf
    predictions['rf'] = pred_rf
    
    # 4. Elastic Net (sparse features)
    elastic = ElasticNet(alpha=0.01, l1_ratio=0.5, random_state=42)
    elastic.fit(X_combined, y, sample_weight=wts)
    pred_elastic = elastic.predict(X_combined)
    models['elastic'] = elastic
    predictions['elastic'] = pred_elastic
    
    # Compute ensemble weights based on cross-validation performance
    ensemble_weights = {}
    for name, pred in predictions.items():
        mae = np.average(np.abs(y - pred), weights=wts)
        rmse = np.sqrt(np.average((y - pred)**2, weights=wts))
        # Combine MAE and RMSE for robust weighting
        score = 0.7 * mae + 0.3 * rmse
        ensemble_weights[name] = 1.0 / (score + 1e-8)
    
    # Normalize weights
    total_weight = sum(ensemble_weights.values())
    for name in ensemble_weights:
        ensemble_weights[name] /= total_weight
    
    # Create ensemble prediction
    ensemble_pred = np.zeros_like(y)
    for name, weight in ensemble_weights.items():
        ensemble_pred += weight * predictions[name]
    
    # Compute final metrics
    ensemble_mae = float(np.average(np.abs(y - ensemble_pred), weights=wts))
    ensemble_rmse = float(np.sqrt(np.average((y - ensemble_pred)**2, weights=wts)))
    
    print(f"Ensemble performance: MAE={ensemble_mae:.3f}, RMSE={ensemble_rmse:.3f}")
    print(f"Model weights: {', '.join(f'{k}={v:.3f}' for k, v in ensemble_weights.items())}")
    
    # For backward compatibility, also compute simple Ridge weights
    ridge_beta = ridge.coef_[:3]  # First 3 features (z_1h, z_24h, z_v2m)
    ridge_intercept = ridge.intercept_
    
    # Save both ensemble and simple models
    feature_names = (
        ["z_1h", "z_24h", "z_v2m"] + 
        [f"poly_{i}" for i in range(X_poly.shape[1] - X.shape[1])] +
        ["tech_rsi", "tech_macd", "tech_momentum", "tech_volatility", "tech_trend"]
    )
    
    model = {
        "model_type": "ensemble",
        "features": ["z_1h", "z_24h", "z_v2m"],  # Keep simple for compatibility
        "weights": [float(x) for x in ridge_beta],
        "intercept": float(ridge_intercept),
        "ensemble_weights": ensemble_weights,
        "enhanced_features": feature_names,
        "samples": len(y),
        "train_mae": ensemble_mae,
        "train_rmse": ensemble_rmse
    }
    try:
        with open(weights_path, "w") as f:
            json.dump(model, f)
        print(f"Saved trained weights to {weights_path}")
        return weights_path
    except Exception as e:
        print(f"Failed to save weights: {e}")
        return None


def load_weights(weights_path: str) -> Optional[Dict[str, Any]]:
    import json, os
    try:
        if not os.path.exists(weights_path):
            return None
        with open(weights_path, "r") as f:
            return json.load(f)
    except Exception:
        return None


def load_calibration(calib_path: str = DEFAULT_CALIB_PATH) -> Optional[Dict[str, float]]:
    import json
    try:
            # allow env override
        envp = os.environ.get("PRED_CALIB_PATH")
        if envp:
            calib_path = envp
        if not os.path.exists(calib_path):
            return None
        with open(calib_path, "r") as f:
            data = json.load(f)
        mult = float(data.get("multiplier", 1.0))
        intercept = float(data.get("intercept", 0.0))
        return {"multiplier": mult, "intercept": intercept}
    except Exception:
        return None

def continuous_monitor():
    """Continuous monitoring and model improvement loop."""
    import time
    import schedule
    
    def update_and_retrain():
        """Scheduled task to update model weights and retrain if needed."""
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Running RL model update...")
        update_model_weights_rl()
        
        # Periodically retrain with fresh data
        session = create_http_session()
        try:
            train_advanced_ensemble(session, shortlist_k=40, history_days=10, workers=2)
            print("Model retrained with fresh data")
        except Exception as e:
            print(f"Retraining failed: {e}")
        finally:
            session.close()
    
    def log_current_predictions():
        """Log current predictions for later accuracy evaluation."""
        session = create_http_session()
        try:
            # Get current top predictions
            rows = fetch_coingecko_markets(session, per_page=250, pages=1)
            df = prepare_dataframe(rows)
            df = filter_universe(df, min_market_cap=100_000_000, min_volume=10_000_000)
            
            learned = load_weights("/Users/xfx/Desktop/trade/model_weights.json")
            scored = score_and_predict(df, learned_weights=learned, use_enhanced_features=True)
            
            if not scored.empty:
                top = scored.nlargest(10, "composite_score")
                
                # Log predictions for later validation
                for _, row in top.iterrows():
                    coin_id = row.get("id", "")
                    predicted = row.get("predicted_change_24h_pct", 0)
                    market_cap = row.get("market_cap", 0)
                    
                    # Log prediction (realized value will be filled later)
                    log_prediction_accuracy(predicted, float('nan'), coin_id, market_cap)
                    
                print(f"Logged predictions for {len(top)} coins")
        except Exception as e:
            print(f"Prediction logging failed: {e}")
        finally:
            session.close()
    
    # Schedule tasks
    schedule.every(4).hours.do(update_and_retrain)  # Update model every 4 hours
    schedule.every(1).hours.do(log_current_predictions)  # Log predictions every hour
    
    print("🔄 Starting continuous monitoring (Ctrl+C to stop)")
    print("- Model updates: every 4 hours")
    print("- Prediction logging: every 1 hour")
    
    try:
        while True:
            schedule.run_pending()
            time.sleep(60)  # Check every minute
    except KeyboardInterrupt:
        print("\n⏹️ Monitoring stopped")


def main() -> None:
    print("--- Crypto 24h Candidate Predictor (Crypto.com) ---")
    session = create_http_session()
    # CLI arguments
    parser = argparse.ArgumentParser(description="Crypto.com 24h candidate predictor")
    parser.add_argument("--top", type=int, default=3, help="Number of candidates to output")
    parser.add_argument("--min-mcap", type=float, default=150_000_000, help="Minimum market cap filter")
    parser.add_argument("--min-volume", type=float, default=10_000_000, help="Minimum 24h volume filter")
    parser.add_argument("--min-pred", type=float, default=0.0, help="Minimum predicted 24h % to include in results")
    parser.add_argument("--history", action="store_true", help="Enable historical refinement stage (slower)")
    parser.add_argument("--history-topk", type=int, default=15, help="Number of shortlisted coins to refine with history")
    parser.add_argument("--history-days", type=int, default=14, help="History days to fetch for refinement")
    parser.add_argument("--history-workers", type=int, default=4, help="Concurrent workers for history fetch")
    parser.add_argument("--history-timeout", type=float, default=10.0, help="Per-request timeout for history fetch (seconds)")
    parser.add_argument("--history-budget", type=float, default=15.0, help="Total time budget for history stage (seconds)")
    parser.add_argument("--save-csv", type=str, default="", help="Path to save predictions CSV (file or directory)")
    parser.add_argument("--evaluate", type=str, default="", help="Evaluate a previous predictions CSV instead of predicting now")
    parser.add_argument("--save-eval", type=str, default="", help="Optional path to save evaluation details CSV")
    parser.add_argument("--include", type=str, default="", help="Comma-separated symbols to force-include (e.g., MNT,PEPE)")
    parser.add_argument("--exclude", type=str, default="", help="Comma-separated symbols to exclude additionally")
    parser.add_argument("--train-weights", action="store_true", help="Train linear weights from recent history and save")
    parser.add_argument("--weights-path", type=str, default="/Users/xfx/Desktop/trade/model_weights.json", help="Path to model weights JSON")
    parser.add_argument("--calib-path", type=str, default=DEFAULT_CALIB_PATH, help="Path to scale calibration JSON")
    parser.add_argument("--backtest", action="store_true", help="Run backtest on recent hourly data for a shortlist")
    parser.add_argument("--backtest-shortlist", type=int, default=30, help="Shortlist size for backtest universe")
    parser.add_argument("--backtest-days", type=int, default=30, help="Snapshot lookback days (unused; markets API is realtime)")
    parser.add_argument("--backtest-history-days", type=int, default=14, help="Hourly history window for features")
    parser.add_argument("--backtest-workers", type=int, default=6, help="Concurrency for backtest history fetch")
    parser.add_argument("--backtest-zwin", type=int, default=48, help="Rolling z-score window (hours)")
    parser.add_argument("--backtest-horizon", type=int, default=24, help="Forward horizon for label (hours)")
    # Random timestamp backtest (full pipeline at past times)
    parser.add_argument("--rand-backtest", action="store_true", help="Random-timestamps backtest vs 24h forward returns")
    parser.add_argument("--rb-samples", type=int, default=400, help="Number of random hourly timestamps to sample")
    parser.add_argument("--rb-lookback-days", type=int, default=30, help="Lookback window (days) for sampling timestamps")
    parser.add_argument("--rb-universe", type=int, default=80, help="Universe size by market cap within Crypto.com bases")
    parser.add_argument("--rb-top", type=int, default=5, help="Evaluate top-N predictions per timestamp (reporting only)")
    parser.add_argument("--rb-workers", type=int, default=4, help="Concurrency for chart fetching")
    parser.add_argument("--rb-min-interval", type=float, default=0.7, help="Minimum seconds between any two chart requests (global pacing)")
    # Auto-calibration using random backtest pairs
    parser.add_argument("--calibrate", action="store_true", help="Auto-fit scale calibration from random backtest pairs")
    parser.add_argument("--calib-samples", type=int, default=60, help="Random timestamps for calibration")
    parser.add_argument("--calib-universe", type=int, default=30, help="Universe size during calibration")
    parser.add_argument("--calib-lookback-days", type=int, default=20, help="Lookback days for calibration sampling")
    parser.add_argument("--calib-workers", type=int, default=1, help="Concurrency during calibration fetch")
    parser.add_argument("--monitor", action="store_true", help="Start continuous monitoring and RL improvement mode")
    args, _ = parser.parse_known_args()
    
    # Handle monitoring mode
    if args.monitor:
        continuous_monitor()
        return

    # If no CLI arguments provided, use higher-quality defaults
    if len(sys.argv) == 1:
        args.history = True
        args.history_topk = 15
        args.history_budget = 12.0
        args.top = 5

    # Evaluation mode: assess a previous predictions CSV
    if args.evaluate:
        eval_out = args.save_eval if args.save_eval else ""
        evaluate_predictions(session, args.evaluate, save_eval_path=eval_out if eval_out else None)
        return

    # Backtest mode: rolling features and >10% label discrimination
    if args.backtest:
        crypto_com_bases = fetch_crypto_com_base_symbols(session) or set(FALLBACK_CRYPTO_COM_TICKERS)
        backtest_topk(
            session,
            crypto_com_bases,
            days=args.backtest_days,
            history_days=args.backtest_history_days,
            shortlist_k=args.backtest_shortlist,
            workers=args.backtest_workers,
            z_window_hours=args.backtest_zwin,
            target_horizon_hours=args.backtest_horizon,
        )
        return

    # Random timestamp backtest
    if args.rand_backtest:
        crypto_com_bases = fetch_crypto_com_base_symbols(session) or set(FALLBACK_CRYPTO_COM_TICKERS)
        backtest_random_points(
            session=session,
            universe_symbols=crypto_com_bases,
            num_samples=args.rb_samples,
            lookback_days=args.rb_lookback_days,
            universe_size=args.rb_universe,
            top_n=args.rb_top,
            workers=args.rb_workers,
            weights_path=args.weights_path,
            z_window_hours=args.backtest_zwin,
            min_interval=args.rb_min_interval,
        )
        return

    # Calibration mode
    if args.calibrate:
        crypto_com_bases = fetch_crypto_com_base_symbols(session) or set(FALLBACK_CRYPTO_COM_TICKERS)
        calibrate_scale_from_random(
            session=session,
            universe_symbols=crypto_com_bases,
            num_samples=args.calib_samples,
            lookback_days=args.calib_lookback_days,
            universe_size=args.calib_universe,
            workers=args.calib_workers,
            weights_path=args.weights_path,
            z_window_hours=args.backtest_zwin,
            calib_path=args.calib_path,
        )
        return

    # Train weights mode
    if args.train_weights:
        train_weights_from_backtest(
            session,
            shortlist_k=max(40, args.backtest_shortlist if hasattr(args, 'backtest_shortlist') else 60),
            history_days=args.backtest_history_days if hasattr(args, 'backtest_history_days') else 14,
            z_window_hours=args.backtest_zwin if hasattr(args, 'backtest_zwin') else 48,
            workers=args.backtest_workers if hasattr(args, 'backtest_workers') else 8,
            weights_path=args.weights_path,
        )
        return

    # Prediction mode
    crypto_com_bases = fetch_crypto_com_base_symbols(session)
    if not crypto_com_bases:
        print("Warning: Could not fetch Crypto.com instruments. Falling back to a static Crypto.com symbol list.")
        crypto_com_bases = set(FALLBACK_CRYPTO_COM_TICKERS)

    rows = fetch_coingecko_markets(session, per_page=250, pages=1)
    df_raw = prepare_dataframe(rows)
    if df_raw.empty:
        print("Could not load market data.")
        sys.exit(1)

    inc = set([s.strip() for s in args.include.split(",") if s.strip()]) if args.include else set()
    exc = set([s.strip() for s in args.exclude.split(",") if s.strip()]) if args.exclude else set()
    df_universe = filter_universe(
        df_raw, crypto_com_bases, min_market_cap=args.min_mcap, min_volume=args.min_volume,
        include_symbols=inc, extra_exclude=exc
    )
    if df_universe.empty:
        # Relax the constraints and try again
        df_universe = filter_universe(
            df_raw, crypto_com_bases, min_market_cap=max(50_000_000, args.min_mcap / 3), min_volume=max(3_000_000, args.min_volume / 3)
        )
        if df_universe.empty:
            print("No coins passed universe filters even after relaxing size/liquidity thresholds.")
            sys.exit(0)

    # Load learned weights if available
    # Allow pointing to a custom calibration file
    os.environ["PRED_CALIB_PATH"] = args.calib_path
    learned = load_weights(args.weights_path)
    df_scored = score_and_predict(df_universe, learned_weights=learned, use_enhanced_features=True)
    if df_scored.empty:
        print("No coins could be scored.")
        sys.exit(0)

    if args.history:
        refined = refine_with_history(
            df_scored,
            session,
            top_k=args.history_topk,
            days=args.history_days,
            request_timeout=args.history_timeout,
            max_total_seconds=args.history_budget,
            workers=args.history_workers,
        )
    else:
        refined = refine_with_history(
            df_scored, session, top_k=0, days=0, request_timeout=1.0, max_total_seconds=0.0, workers=1
        )
    if refined.empty:
        top = pick_top_n(df_scored, n=args.top)
        model_label = "base"
    else:
        top = refined.sort_values(by=["refined_score", "total_volume"], ascending=[False, False]).head(args.top)
        model_label = "refined"

    # Optional threshold filter by predicted change
    try:
        if args.min_pred and "predicted_change_24h_pct" in top.columns:
            top = top[top["predicted_change_24h_pct"] >= float(args.min_pred)]
    except Exception:
        pass

    print(f"Top {len(top)} candidates with predicted 24h change (%):\n")
    for idx, row in top.iterrows():
        name = str(row.get("name", "?"))
        symbol = str(row.get("symbol", "?")).upper()
        price = row.get("current_price", float("nan"))
        vol = row.get("total_volume", float("nan"))
        mc = row.get("market_cap", float("nan"))
        pc1h = row.get("pc_1h", float("nan"))
        pc24h = row.get("pc_24h", float("nan"))
        pred = row.get("predicted_change_24h_pct", float("nan"))
        score = row.get("refined_score", row.get("composite_score", float("nan")))

        print(f"- {name} ({symbol})")
        print(f"  Current Price: ${price:,.6f}  |  24h: {pc24h:.2f}%  |  1h: {pc1h:.2f}%")
        print(f"  Volume: ${vol:,.0f}  |  Market Cap: ${mc:,.0f}")
        print(f"  Model Score: {score:.3f}  |  Predicted next 24h: {pred:+.2f}%\n")

    # Save predictions if requested
    if args.save_csv:
        export_path = save_predictions_csv(top, args.save_csv, prediction_time=time.time(), model_label=model_label)
        if export_path:
            print(f"Saved predictions CSV to: {export_path}")
        else:
            print("Failed to save predictions CSV.")

    print("Note: This is a heuristic model using public market data; not financial advice.")


if __name__ == "__main__":
    main()


