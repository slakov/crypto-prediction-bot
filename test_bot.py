#!/usr/bin/env python3
"""
Test script for the Telegram bot - checks if everything is working
"""

import sys
import os
sys.path.append('/Users/xfx/Desktop/trade')

def test_imports():
    """Test if all required modules can be imported"""
    print("🔄 Testing imports...")
    
    try:
        from telegram import Update
        from telegram.ext import Application
        print("✅ Telegram bot library imported successfully")
    except ImportError as e:
        print(f"❌ Telegram import failed: {e}")
        return False
    
    try:
        import pandas as pd
        import numpy as np
        print("✅ Data processing libraries imported")
    except ImportError as e:
        print(f"❌ Data libraries import failed: {e}")
        return False
    
    return True

def test_prediction_module():
    """Test if the prediction module can be loaded"""
    print("\n🔄 Testing prediction module...")
    
    try:
        # Try to import the prediction functions
        from importlib.util import spec_from_file_location, module_from_spec
        spec = spec_from_file_location("prediction", "/Users/xfx/Desktop/trade/2.py")
        prediction_module = module_from_spec(spec)
        spec.loader.exec_module(prediction_module)
        
        # Test key functions
        session = prediction_module.create_http_session()
        session.close()
        print("✅ Prediction module loaded successfully")
        print(f"✅ Available functions: {[name for name in dir(prediction_module) if not name.startswith('_')][:10]}...")
        return True
        
    except Exception as e:
        print(f"❌ Prediction module loading failed: {e}")
        return False

def test_bot_config():
    """Test bot configuration"""
    print("\n🔄 Testing bot configuration...")
    
    try:
        from crypto_bot import BOT_TOKEN, AUTHORIZED_USERS
        
        if BOT_TOKEN == "YOUR_BOT_TOKEN_HERE":
            print("⚠️  Bot token not configured - you need to set your actual token")
            print("   Get a token from @BotFather on Telegram")
            return False
        else:
            print("✅ Bot token is configured")
        
        if len(AUTHORIZED_USERS) == 0:
            print("⚠️  No authorized users configured - bot will accept all users")
        else:
            print(f"✅ Authorized users configured: {len(AUTHORIZED_USERS)} user(s)")
        
        return True
        
    except Exception as e:
        print(f"❌ Bot configuration test failed: {e}")
        return False

def main():
    print("🤖 Crypto Prediction Telegram Bot - Test Suite\n")
    
    all_passed = True
    
    # Test imports
    if not test_imports():
        all_passed = False
    
    # Test prediction module
    if not test_prediction_module():
        all_passed = False
    
    # Test bot config
    if not test_bot_config():
        all_passed = False
    
    print("\n" + "="*50)
    if all_passed:
        print("🎉 All tests passed! Bot should work correctly.")
        print("\nNext steps:")
        print("1. Set your bot token in crypto_bot.py")
        print("2. Add your Telegram user ID to AUTHORIZED_USERS")
        print("3. Run: python3 crypto_bot.py")
    else:
        print("❌ Some tests failed. Please fix the issues above.")
    
    print("\n💡 Need help?")
    print("- Check README_bot.md for detailed setup instructions")
    print("- Make sure all requirements are installed: pip3 install -r requirements_bot.txt")

if __name__ == "__main__":
    main()
