#!/usr/bin/env python3
"""
Comprehensive Unit Tests for Crypto Prediction Telegram Bot
Tests all components: prediction engine, formatting, commands, callbacks
"""

import unittest
import asyncio
import json
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from datetime import datetime
import sys
import os

# Add the current directory to Python path for imports
sys.path.insert(0, '/Users/xfx/Desktop/trade')

from crypto_bot_production import (
    CryptoPredictionEngine, 
    format_prediction_message,
    start,
    button_callback,
    predict
)

class TestCryptoPredictionEngine(unittest.TestCase):
    """Test the core prediction engine"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.engine = CryptoPredictionEngine()
        
        # Sample market data for testing
        self.sample_coin_data = {
            "symbol": "btc",
            "name": "Bitcoin",
            "current_price": 45000.0,
            "total_volume": 25000000000,
            "market_cap": 875000000000,
            "price_change_percentage_1h_in_currency": 1.5,
            "price_change_percentage_24h_in_currency": 3.2,
            "market_cap_rank": 1
        }
        
        self.sample_market_response = [
            {
                "symbol": "btc",
                "name": "Bitcoin", 
                "current_price": 45000.0,
                "total_volume": 25000000000,
                "market_cap": 875000000000,
                "price_change_percentage_1h_in_currency": 1.5,
                "price_change_percentage_24h_in_currency": 3.2,
                "market_cap_rank": 1
            },
            {
                "symbol": "eth",
                "name": "Ethereum",
                "current_price": 2800.0, 
                "total_volume": 15000000000,
                "market_cap": 335000000000,
                "price_change_percentage_1h_in_currency": 0.8,
                "price_change_percentage_24h_in_currency": 2.1,
                "market_cap_rank": 2
            },
            {
                "symbol": "usdt",  # Should be excluded
                "name": "Tether",
                "current_price": 1.0,
                "total_volume": 50000000000,
                "market_cap": 75000000000,
                "price_change_percentage_1h_in_currency": 0.0,
                "price_change_percentage_24h_in_currency": 0.0,
                "market_cap_rank": 3
            }
        ]

    def test_prediction_algorithm_basic(self):
        """Test basic prediction algorithm functionality"""
        result = self.engine.prediction_algorithm(self.sample_coin_data)
        
        # Should return a numeric prediction
        self.assertIsInstance(result, (int, float))
        
        # Should be within reasonable bounds 
        self.assertGreaterEqual(result, -3.0)
        self.assertLessEqual(result, 18.0)

    def test_prediction_algorithm_high_momentum(self):
        """Test prediction with high momentum scenario"""
        high_momentum_data = self.sample_coin_data.copy()
        high_momentum_data.update({
            "price_change_percentage_1h_in_currency": 4.0,  # High 1h
            "price_change_percentage_24h_in_currency": 8.0,  # High 24h
            "total_volume": 50000000000  # Very high volume
        })
        
        result = self.engine.prediction_algorithm(high_momentum_data)
        
        # High momentum should yield positive prediction
        self.assertGreater(result, 2.0)

    def test_prediction_algorithm_oversold_bounce(self):
        """Test prediction for oversold bounce scenario"""
        oversold_data = self.sample_coin_data.copy()
        oversold_data.update({
            "price_change_percentage_1h_in_currency": 0.5,   # Slight recovery
            "price_change_percentage_24h_in_currency": -12.0, # Deep drop
            "total_volume": 30000000000  # High volume
        })
        
        result = self.engine.prediction_algorithm(oversold_data)
        
        # Oversold bounce should yield positive prediction
        self.assertGreater(result, 1.0)

    def test_prediction_algorithm_edge_cases(self):
        """Test prediction algorithm with edge cases"""
        # Test with zero/None values
        edge_case_data = {
            "current_price": 0,
            "total_volume": None,
            "market_cap": 0,
            "price_change_percentage_1h_in_currency": None,
            "price_change_percentage_24h_in_currency": None
        }
        
        result = self.engine.prediction_algorithm(edge_case_data)
        
        # Should handle gracefully and return reasonable value
        self.assertIsInstance(result, (int, float))
        self.assertGreaterEqual(result, -3.0)
        self.assertLessEqual(result, 18.0)

    @patch('crypto_bot_production.requests.get')
    def test_get_coingecko_data_success(self, mock_get):
        """Test successful CoinGecko API call"""
        # Mock successful response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = self.sample_market_response
        mock_get.return_value = mock_response
        
        result = self.engine.get_coingecko_data(3)
        
        # Should return the mocked data
        self.assertEqual(result, self.sample_market_response)
        
        # Should call with correct parameters
        mock_get.assert_called_once()
        args, kwargs = mock_get.call_args
        self.assertIn('vs_currency', kwargs['params'])
        self.assertEqual(kwargs['params']['per_page'], 3)

    @patch('crypto_bot_production.requests.get')
    def test_get_coingecko_data_failure(self, mock_get):
        """Test CoinGecko API failure handling"""
        # Mock failed response
        mock_response = Mock()
        mock_response.status_code = 429  # Rate limited
        mock_get.return_value = mock_response
        
        result = self.engine.get_coingecko_data(5)
        
        # Should return None on failure
        self.assertIsNone(result)

    @patch('crypto_bot_production.requests.get')
    def test_get_coingecko_data_exception(self, mock_get):
        """Test CoinGecko API exception handling"""
        # Mock exception
        mock_get.side_effect = Exception("Network error")
        
        result = self.engine.get_coingecko_data(5)
        
        # Should return None on exception
        self.assertIsNone(result)

    @patch.object(CryptoPredictionEngine, 'get_coingecko_data')
    def test_get_predictions_success(self, mock_get_data):
        """Test successful prediction generation"""
        mock_get_data.return_value = self.sample_market_response
        
        result = self.engine.get_predictions(top_n=5, min_pred=0.0)
        
        # Should return structured result
        self.assertIsInstance(result, dict)
        self.assertIn('count', result)
        self.assertIn('predictions', result)
        self.assertIn('timestamp', result)
        
        # Should exclude stablecoins
        symbols = [p['symbol'] for p in result['predictions']]
        self.assertNotIn('USDT', symbols)
        
        # Should include valid predictions
        self.assertIn('BTC', symbols)
        self.assertIn('ETH', symbols)
        
        # Each prediction should have required fields
        for pred in result['predictions']:
            self.assertIn('symbol', pred)
            self.assertIn('name', pred)
            self.assertIn('price', pred)
            self.assertIn('predicted_change', pred)
            self.assertIn('pc_24h', pred)
            self.assertIn('pc_1h', pred)

    @patch.object(CryptoPredictionEngine, 'get_coingecko_data')
    def test_get_predictions_filtering(self, mock_get_data):
        """Test prediction filtering functionality"""
        mock_get_data.return_value = self.sample_market_response
        
        # Test minimum prediction filtering
        result = self.engine.get_predictions(top_n=10, min_pred=5.0)
        
        # All predictions should meet minimum threshold
        for pred in result['predictions']:
            self.assertGreaterEqual(pred['predicted_change'], 5.0)

    @patch.object(CryptoPredictionEngine, 'get_coingecko_data')
    def test_get_predictions_caching(self, mock_get_data):
        """Test prediction caching mechanism"""
        mock_get_data.return_value = self.sample_market_response
        
        # First call
        result1 = self.engine.get_predictions(top_n=5)
        
        # Second call within cache window (should use cache)
        result2 = self.engine.get_predictions(top_n=5)
        
        # Should only call API once due to caching
        self.assertEqual(mock_get_data.call_count, 1)
        
        # Results should be similar (cache working)
        self.assertEqual(result1['count'], result2['count'])

    @patch.object(CryptoPredictionEngine, 'get_coingecko_data')
    def test_get_predictions_no_data(self, mock_get_data):
        """Test behavior when no market data available"""
        mock_get_data.return_value = None
        
        result = self.engine.get_predictions(top_n=5)
        
        # Should return error state
        self.assertEqual(result['count'], 0)
        self.assertIn('error', result)

    def test_exclusion_lists(self):
        """Test that exclusion lists work correctly"""
        # Test data with excluded coins
        test_data = [
            {"symbol": "btc", "name": "Bitcoin", "current_price": 45000},
            {"symbol": "usdt", "name": "Tether", "current_price": 1.0},  # Stablecoin
            {"symbol": "wbtc", "name": "Wrapped Bitcoin", "current_price": 45000}, # Wrapped
            {"symbol": "usd1", "name": "USD Coin 1", "current_price": 1.0}, # USD pattern
            {"symbol": "stable", "name": "Stable Coin", "current_price": 1.0} # Stable pattern
        ]
        
        with patch.object(self.engine, 'get_coingecko_data', return_value=test_data):
            result = self.engine.get_predictions(top_n=10)
            
            symbols = [p['symbol'] for p in result['predictions']]
            
            # Should exclude stablecoins and wrapped coins
            excluded = ['USDT', 'WBTC', 'USD1', 'STABLE']
            for exc in excluded:
                self.assertNotIn(exc, symbols)


class TestMessageFormatting(unittest.TestCase):
    """Test message formatting functions"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.sample_predictions = {
            "count": 2,
            "predictions": [
                {
                    "symbol": "BTC",
                    "name": "Bitcoin",
                    "price": 45000.0,
                    "predicted_change": 8.5,
                    "pc_24h": 3.2,
                    "pc_1h": 1.5,
                    "volume_24h": 25000000000,
                    "market_cap": 875000000000,
                    "rank": 1
                },
                {
                    "symbol": "ETH", 
                    "name": "Ethereum",
                    "price": 2800.0,
                    "predicted_change": 6.2,
                    "pc_24h": 2.1,
                    "pc_1h": 0.8,
                    "volume_24h": 15000000000,
                    "market_cap": 335000000000,
                    "rank": 2
                }
            ],
            "timestamp": "2025-08-11T10:30:00"
        }

    def test_format_prediction_message_success(self):
        """Test successful message formatting"""
        result = format_prediction_message(self.sample_predictions)
        
        # Should be a string
        self.assertIsInstance(result, str)
        
        # Should contain expected elements
        self.assertIn("Top Crypto Predictions", result)
        self.assertIn("BTC", result)
        self.assertIn("ETH", result)
        self.assertIn("Bitcoin", result)
        self.assertIn("Ethereum", result)
        
        # Should format prices correctly
        self.assertIn("$45,000.00", result)
        self.assertIn("$2,800.00", result)
        
        # Should show predictions
        self.assertIn("+8.5%", result)
        self.assertIn("+6.2%", result)
        
        # Should show 24h and 1h changes
        self.assertIn("+3.2%", result)
        self.assertIn("+2.1%", result)

    def test_format_prediction_message_error(self):
        """Test formatting with error data"""
        error_data = {
            "count": 0,
            "error": "API rate limit exceeded"
        }
        
        result = format_prediction_message(error_data)
        
        # Should contain error message
        self.assertIn("Error getting predictions", result)
        self.assertIn("API rate limit exceeded", result)

    def test_format_prediction_message_no_data(self):
        """Test formatting with no predictions"""
        no_data = {"count": 0}
        
        result = format_prediction_message(no_data)
        
        # Should show no predictions message
        self.assertIn("No predictions found", result)

    def test_format_prediction_message_emoji_assignment(self):
        """Test emoji assignment based on prediction values"""
        # Test different prediction ranges
        test_cases = [
            (15.0, "🌕"),  # Moonshot
            (7.0, "🔥"),   # Hot
            (3.0, "🟢"),  # Good
            (1.0, "🟡"),  # Mild
            (-1.0, "🔴")  # Negative
        ]
        
        for pred_value, expected_emoji in test_cases:
            test_data = {
                "count": 1,
                "predictions": [{
                    "symbol": "TEST",
                    "name": "Test Coin",
                    "price": 1.0,
                    "predicted_change": pred_value,
                    "pc_24h": 0.0,
                    "pc_1h": 0.0,
                    "volume_24h": 1000000,
                    "market_cap": 10000000,
                    "rank": 100
                }]
            }
            
            result = format_prediction_message(test_data)
            self.assertIn(expected_emoji, result)

    def test_format_prediction_message_volume_formatting(self):
        """Test volume and market cap formatting"""
        test_data = {
            "count": 1,
            "predictions": [{
                "symbol": "TEST",
                "name": "Test Coin", 
                "price": 1.0,
                "predicted_change": 5.0,
                "pc_24h": 2.0,
                "pc_1h": 1.0,
                "volume_24h": 1500000000,  # 1.5B
                "market_cap": 25000000000,  # 25B
                "rank": 50
            }]
        }
        
        result = format_prediction_message(test_data)
        
        # Should format large numbers correctly
        self.assertIn("$1.5B", result)  # Volume
        self.assertIn("$25.0B", result)  # Market cap

    def test_format_prediction_message_small_price(self):
        """Test formatting of small price values"""
        test_data = {
            "count": 1,
            "predictions": [{
                "symbol": "MICRO",
                "name": "Micro Coin",
                "price": 0.000123,  # Small price
                "predicted_change": 10.0,
                "pc_24h": 5.0,
                "pc_1h": 2.0,
                "volume_24h": 500000,
                "market_cap": 1000000,
                "rank": 500
            }]
        }
        
        result = format_prediction_message(test_data)
        
        # Should format small prices with 4 decimal places
        self.assertIn("$0.0001", result)


class TestTelegramCommands(unittest.TestCase):
    """Test Telegram bot command handlers"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.mock_update = Mock()
        self.mock_context = Mock()
        self.mock_message = Mock()
        self.mock_callback_query = Mock()
        
        # Configure mocks
        self.mock_update.message = self.mock_message
        self.mock_update.callback_query = self.mock_callback_query
        self.mock_message.reply_text = AsyncMock()
        self.mock_callback_query.answer = AsyncMock()
        self.mock_callback_query.edit_message_text = AsyncMock()

    async def test_start_command(self):
        """Test /start command handler"""
        await start(self.mock_update, self.mock_context)
        
        # Should send welcome message with keyboard
        self.mock_message.reply_text.assert_called_once()
        
        # Check call arguments
        call_args = self.mock_message.reply_text.call_args
        message_text = call_args[0][0]
        
        # Should contain welcome content
        self.assertIn("Crypto Prediction Bot", message_text)
        self.assertIn("Select what you're looking for", message_text)
        
        # Should have reply markup
        self.assertIn('reply_markup', call_args[1])

    async def test_predict_command(self):
        """Test /predict command handler"""
        with patch('crypto_bot_production.prediction_engine') as mock_engine:
            # Mock prediction data
            mock_engine.get_predictions.return_value = {
                "count": 1,
                "predictions": [{
                    "symbol": "BTC",
                    "name": "Bitcoin",
                    "price": 45000.0,
                    "predicted_change": 5.0,
                    "pc_24h": 2.0,
                    "pc_1h": 1.0,
                    "volume_24h": 1000000000,
                    "market_cap": 875000000000,
                    "rank": 1
                }]
            }
            
            await predict(self.mock_update, self.mock_context)
            
            # Should call prediction engine
            mock_engine.get_predictions.assert_called_once_with(top_n=5, min_pred=0.0)
            
            # Should send two messages (status + result)
            self.assertEqual(self.mock_message.reply_text.call_count, 2)

    async def test_button_callback_predict_5(self):
        """Test button callback for top 5 predictions"""
        self.mock_callback_query.data = "predict_5"
        
        with patch('crypto_bot_production.prediction_engine') as mock_engine:
            mock_engine.get_predictions.return_value = {
                "count": 2,
                "predictions": [
                    {"symbol": "BTC", "name": "Bitcoin", "price": 45000, 
                     "predicted_change": 5.0, "pc_24h": 2.0, "pc_1h": 1.0,
                     "volume_24h": 1000000000, "market_cap": 875000000000, "rank": 1},
                    {"symbol": "ETH", "name": "Ethereum", "price": 2800,
                     "predicted_change": 4.0, "pc_24h": 1.5, "pc_1h": 0.5, 
                     "volume_24h": 500000000, "market_cap": 335000000000, "rank": 2}
                ]
            }
            
            await button_callback(self.mock_update, self.mock_context)
            
            # Should answer callback
            self.mock_callback_query.answer.assert_called_once()
            
            # Should edit message twice (loading + result)
            self.assertEqual(self.mock_callback_query.edit_message_text.call_count, 2)
            
            # Should call engine with correct parameters
            mock_engine.get_predictions.assert_called_with(top_n=5, min_pred=0.0)

    async def test_button_callback_hot_picks(self):
        """Test button callback for hot picks"""
        self.mock_callback_query.data = "hot_picks"
        
        with patch('crypto_bot_production.prediction_engine') as mock_engine:
            mock_engine.get_predictions.return_value = {"count": 0}
            
            await button_callback(self.mock_update, self.mock_context)
            
            # Should call engine with hot picks parameters
            mock_engine.get_predictions.assert_called_with(top_n=20, min_pred=5.0)

    async def test_button_callback_moonshots(self):
        """Test button callback for moonshots"""
        self.mock_callback_query.data = "moonshots"
        
        with patch('crypto_bot_production.prediction_engine') as mock_engine:
            mock_engine.get_predictions.return_value = {"count": 0}
            
            await button_callback(self.mock_update, self.mock_context)
            
            # Should call engine with moonshot parameters  
            mock_engine.get_predictions.assert_called_with(top_n=25, min_pred=8.0)

    async def test_button_callback_status(self):
        """Test button callback for status check"""
        self.mock_callback_query.data = "status"
        
        with patch('crypto_bot_production.prediction_engine') as mock_engine:
            # Mock API test
            mock_engine.get_coingecko_data.return_value = [{"test": "data"}]
            
            await button_callback(self.mock_update, self.mock_context)
            
            # Should test API connection
            mock_engine.get_coingecko_data.assert_called_once_with(5)
            
            # Should send status message
            call_args = self.mock_callback_query.edit_message_text.call_args_list[-1]
            message_text = call_args[0][0]
            self.assertIn("Bot Status", message_text)
            self.assertIn("Connected", message_text)

    async def test_button_callback_help(self):
        """Test button callback for help"""
        self.mock_callback_query.data = "help"
        
        await button_callback(self.mock_update, self.mock_context)
        
        # Should send help message
        call_args = self.mock_callback_query.edit_message_text.call_args_list[-1]
        message_text = call_args[0][0]
        self.assertIn("Crypto Prediction Bot Help", message_text)
        self.assertIn("Quick Actions", message_text)

    async def test_button_callback_back_to_menu(self):
        """Test button callback for returning to menu"""
        self.mock_callback_query.data = "back_to_menu"
        
        await button_callback(self.mock_update, self.mock_context)
        
        # Should send main menu
        call_args = self.mock_callback_query.edit_message_text.call_args_list[-1]
        message_text = call_args[0][0]
        self.assertIn("Crypto Prediction Bot", message_text)
        self.assertIn("Select what you're looking for", message_text)

    async def test_button_callback_gainers_filtering(self):
        """Test gainers callback with specific filtering logic"""
        self.mock_callback_query.data = "gainers"
        
        with patch('crypto_bot_production.prediction_engine') as mock_engine:
            # Mock data with one gainer
            mock_engine.get_predictions.return_value = {
                "count": 2,
                "predictions": [
                    {"symbol": "GAINER", "pc_24h": 5.0, "predicted_change": 3.0},  # Qualifies
                    {"symbol": "STABLE", "pc_24h": 0.5, "predicted_change": 2.0}   # Doesn't qualify
                ]
            }
            
            await button_callback(self.mock_update, self.mock_context)
            
            # Should call with gainers parameters
            mock_engine.get_predictions.assert_called_with(top_n=30, min_pred=1.0)

    async def test_button_callback_value_picks_filtering(self):
        """Test value picks callback with specific filtering logic"""
        self.mock_callback_query.data = "value_picks"
        
        with patch('crypto_bot_production.prediction_engine') as mock_engine:
            # Mock data with one value pick
            mock_engine.get_predictions.return_value = {
                "count": 2,
                "predictions": [
                    {"symbol": "VALUE", "pc_24h": -2.0, "predicted_change": 5.0, "market_cap": 1000000000},  # Qualifies
                    {"symbol": "LARGE", "pc_24h": -1.0, "predicted_change": 4.0, "market_cap": 10000000000}  # Too large
                ]
            }
            
            await button_callback(self.mock_update, self.mock_context)
            
            # Should call with value picks parameters
            mock_engine.get_predictions.assert_called_with(top_n=30, min_pred=2.0)


class TestIntegrationScenarios(unittest.TestCase):
    """Test complete integration scenarios"""

    @patch('crypto_bot_production.requests.get')
    def test_full_prediction_flow(self, mock_get):
        """Test complete prediction flow from API to formatted output"""
        # Mock API response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            {
                "symbol": "btc",
                "name": "Bitcoin", 
                "current_price": 45000.0,
                "total_volume": 25000000000,
                "market_cap": 875000000000,
                "price_change_percentage_1h_in_currency": 2.0,
                "price_change_percentage_24h_in_currency": 5.0,
                "market_cap_rank": 1
            }
        ]
        mock_get.return_value = mock_response
        
        # Create engine and get predictions
        engine = CryptoPredictionEngine()
        result = engine.get_predictions(top_n=1)
        
        # Format message
        message = format_prediction_message(result)
        
        # Verify complete flow
        self.assertIn("BTC", message)
        self.assertIn("Bitcoin", message)
        self.assertIn("$45,000.00", message)
        self.assertIn("Top Crypto Predictions", message)

    def test_error_handling_resilience(self):
        """Test system resilience under various error conditions"""
        engine = CryptoPredictionEngine()
        
        # Test with completely invalid data
        with patch.object(engine, 'get_coingecko_data', return_value=[]):
            result = engine.get_predictions(top_n=5)
            self.assertEqual(result['count'], 0)
        
        # Test with None response
        with patch.object(engine, 'get_coingecko_data', return_value=None):
            result = engine.get_predictions(top_n=5)
            self.assertIn('error', result)
        
        # Test message formatting with error data
        error_result = {"count": 0, "error": "Test error"}
        message = format_prediction_message(error_result)
        self.assertIn("Error getting predictions", message)

    @patch('crypto_bot_production.prediction_engine')
    async def test_bot_command_integration(self, mock_engine):
        """Test integration between bot commands and prediction engine"""
        # Setup mock
        mock_engine.get_predictions.return_value = {
            "count": 1,
            "predictions": [{
                "symbol": "BTC",
                "name": "Bitcoin",
                "price": 45000.0,
                "predicted_change": 7.5,
                "pc_24h": 3.0,
                "pc_1h": 1.5,
                "volume_24h": 25000000000,
                "market_cap": 875000000000,
                "rank": 1
            }]
        }
        
        # Setup mocks
        mock_update = Mock()
        mock_context = Mock()
        mock_message = Mock()
        mock_update.message = mock_message
        mock_message.reply_text = AsyncMock()
        
        # Test predict command
        await predict(mock_update, mock_context)
        
        # Verify integration
        mock_engine.get_predictions.assert_called_once()
        self.assertEqual(mock_message.reply_text.call_count, 2)  # Status + result


async def run_async_tests():
    """Helper to run async tests"""
    test_loader = unittest.TestLoader()
    
    # Load async test methods
    async_test_cases = [
        TestTelegramCommands('test_start_command'),
        TestTelegramCommands('test_predict_command'),
        TestTelegramCommands('test_button_callback_predict_5'),
        TestTelegramCommands('test_button_callback_hot_picks'),
        TestTelegramCommands('test_button_callback_moonshots'),
        TestTelegramCommands('test_button_callback_status'),
        TestTelegramCommands('test_button_callback_help'),
        TestTelegramCommands('test_button_callback_back_to_menu'),
        TestTelegramCommands('test_button_callback_gainers_filtering'),
        TestTelegramCommands('test_button_callback_value_picks_filtering'),
        TestIntegrationScenarios('test_bot_command_integration')
    ]
    
    print("🧪 Running async tests...")
    for test_case in async_test_cases:
        try:
            await test_case._testMethodName()
            print(f"✅ {test_case._testMethodName}")
        except Exception as e:
            print(f"❌ {test_case._testMethodName}: {e}")


if __name__ == '__main__':
    print("🚀 Starting Crypto Bot Unit Tests")
    print("=" * 50)
    
    # Run synchronous tests
    print("\n📊 Running synchronous tests...")
    unittest.main(verbosity=2, exit=False)
    
    # Run async tests  
    print("\n🔄 Running asynchronous tests...")
    asyncio.run(run_async_tests())
    
    print("\n✅ All tests completed!")
