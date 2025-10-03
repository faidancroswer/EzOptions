#!/usr/bin/env python3
"""
Automated Trading System Launcher
Starts the complete trading system with ezOptions integration and MT5 execution
"""

import sys
import os
import logging
import json
import time
from datetime import datetime
from pathlib import Path

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from trading_system import AutomatedTradingSystem
    from ezoptions_connector import initialize_ezoptions_connection, get_ezoptions_status
    from vwap_bollinger_indicators import initialize_technical_indicators
    from trading_setups_corrected import trading_setups
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

class TradingSystemLauncher:
    """
    Launcher class for the automated trading system
    """

    def __init__(self, config_file: str = 'config.json'):
        self.config_file = config_file
        self.config = self._load_config()
        self.trading_system = None
        self.system_status = "initializing"

    def _load_config(self) -> dict:
        """Load configuration file"""
        try:
            with open(self.config_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Configuration file {self.config_file} not found.")
            print("Creating default configuration...")
            self._create_default_config()
            return self._load_config()
        except json.JSONDecodeError as e:
            print(f"Error parsing configuration file: {e}")
            sys.exit(1)

    def _create_default_config(self):
        """Create default configuration file"""
        default_config = {
            "ezoptions": {
                "base_url": "http://localhost:8501",
                "api_timeout": 30
            },
            "metatrader5": {
                "login": 123456,
                "password": "your_mt5_password",
                "server": "your_mt5_server"
            },
            "trading": {
                "ticker": "US100",
                "active_setups": [1, 2, 3, 4, 5, 6]
            }
        }

        with open(self.config_file, 'w') as f:
            json.dump(default_config, f, indent=2)

        print(f"Default configuration created: {self.config_file}")
        print("Please edit the configuration file with your actual credentials and settings.")

    def _setup_logging(self):
        """Setup logging configuration"""
        log_config = self.config.get('logging', {})

        # Create logs directory if it doesn't exist
        log_file = log_config.get('file', 'trading_system.log')
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)

        # Configure logging
        log_level = getattr(logging, log_config.get('level', 'INFO').upper())

        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout) if log_config.get('console_output', True) else None
            ]
        )

        # Remove None handlers
        logging.getLogger().handlers = [h for h in logging.getLogger().handlers if h is not None]

        logging.info("Logging system initialized")

    def _check_prerequisites(self) -> bool:
        """Check if all prerequisites are met"""
        print("Checking prerequisites...")

        # Check if ezOptions is running (simplified check)
        try:
            import requests
            ezoptions_url = self.config.get('ezoptions', {}).get('url', 'http://localhost:8501')
            # Try to connect to the base URL instead of /health endpoint
            response = requests.get(ezoptions_url, timeout=5)
            if response.status_code == 200:
                print("[OK] ezOptions is running")
            else:
                print("[WARNING] ezOptions returned non-200 status")
        except requests.exceptions.RequestException:
            print("[WARNING] Cannot connect to ezOptions. Make sure it's running on localhost:8501")
            print("  Start ezOptions with: streamlit run ezoptions.py --server.port 8501")
            # Don't return False here, just warn - the system can still work with yfinance fallback

        # Check MT5 configuration
        mt5_config = self.config.get('mt5', {})
        if not all([mt5_config.get('login'), mt5_config.get('password'), mt5_config.get('server')]):
            print("[WARNING] MT5 configuration incomplete. Please check config.json")
            print("  Required: login, password, server")
            # Don't return False, let the system continue with warnings
        else:
            print("[OK] MT5 configuration found")

        # Check required packages
        required_packages = ['pandas', 'numpy', 'requests']
        missing_packages = []

        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                missing_packages.append(package)

        if missing_packages:
            print(f"[ERROR] Missing required packages: {', '.join(missing_packages)}")
            print("  Install with: pip install -r requirements.txt")
            return False
        else:
            print("[OK] All required packages installed")

        print("All prerequisites checked!")
        return True

    def _initialize_components(self) -> bool:
        """Initialize all system components"""
        print("Initializing system components...")

        try:
            # Initialize ezOptions connection
            print("Initializing ezOptions connection...")
            try:
                if not initialize_ezoptions_connection():
                    print("[WARNING] Failed to initialize ezOptions connection")
                    # Continue anyway as we can use fallback methods
                else:
                    print("[OK] ezOptions connection initialized")
            except Exception as e:
                print(f"[WARNING] Error initializing ezOptions connection: {e}")
                # Continue anyway as we can use fallback methods

            # Initialize technical indicators
            print("Initializing technical indicators...")
            try:
                initialize_technical_indicators()
                print("[OK] Technical indicators initialized")
            except Exception as e:
                print(f"[WARNING] Error initializing technical indicators: {e}")
                # Continue anyway

            # Initialize trading system
            print("Initializing trading system...")
            try:
                self.trading_system = AutomatedTradingSystem(self.config_file)
                print("[OK] Trading system initialized")
            except Exception as e:
                print(f"[ERROR] Error initializing trading system: {e}")
                logging.error(f"Trading system initialization error: {e}")
                return False

            return True

        except Exception as e:
            print(f"[ERROR] Error initializing components: {e}")
            logging.error(f"Component initialization error: {e}")
            return False

    def _display_system_info(self):
        """Display system information and status"""
        print("\n" + "="*60)
        print("AUTOMATED TRADING SYSTEM - STATUS")
        print("="*60)

        # System info
        print(f"Python Version: {sys.version}")
        print(f"Working Directory: {os.getcwd()}")
        print(f"Configuration File: {self.config_file}")

        # Trading configuration
        trading_config = self.config.get('trading', {})
        print(f"\nTrading Configuration:")
        print(f"  Ticker: {trading_config.get('ticker', 'N/A')}")
        print(f"  Active Setups: {trading_config.get('active_setups', [])}")
        print(f"  Check Interval: {trading_config.get('check_interval', 'N/A')} seconds")

        # ezOptions status
        try:
            ez_status = get_ezoptions_status()
            print(f"\nEzOptions Status:")
            print(f"  Connection: {'[OK] Connected' if ez_status.get('data_manager_status', {}).get('connected') else '[ERROR] Disconnected'}")
            print(f"  Cache Size: {ez_status.get('data_manager_status', {}).get('cache_size', 0)} entries")
        except Exception as e:
            print(f"  EzOptions Status: Error - {e}")

        # MT5 status
        if self.trading_system and hasattr(self.trading_system, 'mt5_integration'):
            mt5_connected = self.trading_system.mt5_integration.connected
            print(f"\nMT5 Status:")
            print(f"  Connection: {'[OK] Connected' if mt5_connected else '[ERROR] Disconnected'}")
            if mt5_connected:
                account_info = self.trading_system.mt5_integration.get_account_info()
                print(f"  Balance: ${account_info.get('balance', 0):,.2f}")
                print(f"  Equity: ${account_info.get('equity', 0):,.2f}")

        print("\nAvailable Setups:")
        for setup_info in trading_setups.get_setup_summary():
            print(f"  Setup {setup_info['number']}: {setup_info['name']} ({setup_info['direction']})")

        print("="*60)

    def start_system(self):
        """Start the trading system"""
        print("Starting Automated Trading System...")

        # Setup logging
        self._setup_logging()

        # Check prerequisites
        if not self._check_prerequisites():
            print("Prerequisites not met. Exiting.")
            sys.exit(1)

        # Initialize components
        if not self._initialize_components():
            print("Failed to initialize components. Exiting.")
            sys.exit(1)

        # Display system info
        self._display_system_info()

        self.system_status = "running"

        try:
            # Start trading system
            print("Starting trading operations...")
            self.trading_system.start_trading()

            # Keep system running
            print("System is now running. Press Ctrl+C to stop.")

            while self.system_status == "running":
                time.sleep(1)

                # Periodic status check
                if int(time.time()) % 300 == 0:  # Every 5 minutes
                    self._display_system_info()

        except KeyboardInterrupt:
            print("\nShutdown signal received...")
            self.stop_system()

        except Exception as e:
            print(f"Error during system operation: {e}")
            logging.error(f"System operation error: {e}")
            self.stop_system()
            sys.exit(1)

    def stop_system(self):
        """Stop the trading system"""
        print("Stopping trading system...")

        self.system_status = "stopping"

        if self.trading_system:
            try:
                self.trading_system.stop_trading()
                print("[OK] Trading system stopped")
            except Exception as e:
                print(f"Error stopping trading system: {e}")

        print("System shutdown complete.")
        logging.info("System shutdown complete")

def main():
    """Main entry point"""
    print("Automated Trading System Launcher")
    print("=" * 40)

    # Check command line arguments
    config_file = 'config.json'
    if len(sys.argv) > 1:
        config_file = sys.argv[1]

    # Initialize launcher
    launcher = TradingSystemLauncher(config_file)

    # Start system
    launcher.start_system()

if __name__ == "__main__":
    main()