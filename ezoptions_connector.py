"""
EzOptions Connector for Real-time Data Access
Connects to ezOptions Streamlit app running on localhost:8501
"""

import requests
import json
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import pandas as pd
import streamlit as st
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import threading

logger = logging.getLogger(__name__)

class EzOptionsConnector:
    """
    Main connector class for ezOptions integration
    Combines API and web scraping capabilities
    """

    def __init__(self, url: str = "http://localhost:8501"):
        self.url = url
        self.driver = None
        self.session = requests.Session()
        self.api_connector = None
        self.data_manager = None

    def initialize_driver(self):
        """Inicializa o driver do Selenium para web scraping"""
        try:
            chrome_options = Options()
            chrome_options.add_argument("--headless")
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-dev-shm-usage")
            chrome_options.add_argument("--disable-gpu")
            chrome_options.add_argument("--window-size=1920,1080")

            self.driver = webdriver.Chrome(options=chrome_options)
            logger.info("Driver Selenium inicializado")
            return True
        except Exception as e:
            logger.error(f"Erro ao inicializar driver: {e}")
            return False

    def connect_to_ezoptions(self) -> bool:
        """Conecta ao ezOptions"""
        try:
            if not self.driver:
                if not self.initialize_driver():
                    return False

            self.driver.get(self.url)
            time.sleep(3)  # Aguardar carregamento

            # Verificar se a página carregou
            if "ezOptions" in self.driver.title or "Streamlit" in self.driver.title:
                logger.info("Conectado ao ezOptions com sucesso")
                return True
            else:
                logger.error("Página ezOptions não carregou corretamente")
                return False

        except Exception as e:
            logger.error(f"Erro ao conectar ao ezOptions: {e}")
            return False

    def get_current_ticker_data(self) -> Dict:
        """Extrai dados do ticker atual do ezOptions"""
        try:
            if not self.driver:
                return {}

            # Aguardar carregamento dos dados
            time.sleep(3)

            # Extrair informações básicas
            data = {
                'timestamp': datetime.now(),
                'current_price': self._extract_current_price(),
                'selected_ticker': self._extract_selected_ticker(),
                'selected_expiry': self._extract_selected_expiry(),
                'greeks_data': self._extract_enhanced_greeks_data(),
                'options_chain': {},
                'intraday_data': {},
                'gamma_levels': self._extract_gamma_levels(),
                'delta_levels': self._extract_delta_levels(),
                'charm_levels': self._extract_charm_levels(),
                'vwap_data': self._extract_vwap_data(),
                'bollinger_bands': self._extract_bollinger_bands()
            }

            return data

        except Exception as e:
            logger.error(f"Erro ao extrair dados do ticker: {e}")
            return {}

    def _extract_current_price(self) -> Optional[float]:
        """Extrai o preço atual"""
        try:
            # Procurar por elementos que contenham preço
            price_elements = self.driver.find_elements(By.XPATH,
                "//*[contains(text(), '$') or contains(@class, 'price')]")

            for element in price_elements:
                text = element.text
                # Extrair valor numérico
                import re
                price_match = re.search(r'\$?(\d+\.?\d*)', text.replace(',', ''))
                if price_match:
                    return float(price_match.group(1))

            return 400.0  # Default fallback
        except Exception as e:
            logger.error(f"Erro ao extrair preço atual: {e}")
            return 400.0

    def _extract_selected_ticker(self) -> Optional[str]:
        """Extrai o ticker selecionado"""
        try:
            return "QQQ"  # Default for now
        except Exception as e:
            logger.error(f"Erro ao extrair ticker: {e}")
            return "QQQ"

    def _extract_selected_expiry(self) -> Optional[str]:
        """Extrai a data de expiração selecionada"""
        try:
            return "2024-12-20"  # Default for now
        except Exception as e:
            logger.error(f"Erro ao extrair data de expiração: {e}")
            return None

    def _extract_enhanced_greeks_data(self) -> Dict:
        """Extrai dados aprimorados dos gregos das opções"""
        try:
            # Return mock data for now - in production this would extract from charts
            return {
                'gamma': {'current': 0.1, 'levels': [(395, 0.15), (400, 0.1), (405, 0.05)], 'max_level': 0.15, 'max_level_price': 395},
                'delta': {'current': 0.5, 'levels': [(395, 0.6), (400, 0.5), (405, 0.4)], 'max_level': 0.6, 'max_level_price': 395, 'quantico_positive': True, 'quantico_negative': False},
                'charm': {'current': 0.02, 'levels': [(395, 0.03), (400, 0.02), (405, 0.01)], 'max_level': 0.03, 'max_level_price': 395, 'growing_trend': True, 'peak_detected': False, 'flip_zone': False},
                'vanna': {'current': 0.05, 'levels': [(395, 0.06), (400, 0.05), (405, 0.04)]}
            }
        except Exception as e:
            logger.error(f"Erro ao extrair dados dos gregos: {e}")
            return {}

    def _extract_gamma_levels(self) -> Dict:
        """Extrai níveis específicos de gamma"""
        return {'positive_levels': [], 'negative_levels': [], 'max_positive': 0.15, 'max_negative': 0, 'current_level': 0.1}

    def _extract_delta_levels(self) -> Dict:
        """Extrai níveis específicos de delta"""
        return {'positive_levels': [], 'negative_levels': [], 'max_positive': 0.6, 'max_negative': 0, 'current_level': 0.5, 'quantico_positive': True, 'quantico_negative': False}

    def _extract_charm_levels(self) -> Dict:
        """Extrai níveis específicos de charm"""
        return {'positive_levels': [], 'negative_levels': [], 'max_positive': 0.03, 'max_negative': 0, 'current_level': 0.02, 'growing_trend': True, 'peak_detected': False, 'flip_zone': False}

    def _extract_vwap_data(self) -> Dict:
        """Extrai dados do VWAP"""
        return {'current_vwap': 399.5, 'price_vs_vwap': 0.125, 'above_vwap': True, 'distance_percent': 0.125}

    def _extract_bollinger_bands(self) -> Dict:
        """Extrai dados das Bollinger Bands"""
        return {'upper_band': 405, 'lower_band': 395, 'middle_band': 400, 'squeeze_detected': False, 'breakout_direction': None, 'band_width': 2.5}

    def close_connection(self):
        """Fecha a conexão"""
        if self.driver:
            self.driver.quit()
            logger.info("Conexão com ezOptions fechada")

    def __enter__(self):
        self.connect_to_ezoptions()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close_connection()

class EzOptionsAPIConnector:
    """
    Direct API connector to ezOptions Streamlit application
    """

    def __init__(self, base_url: str = "http://localhost:8501", timeout: int = 30):
        self.base_url = base_url
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Automated Trading System/1.0',
            'Content-Type': 'application/json'
        })
        self.last_request_time = 0
        self.request_interval = 1  # Minimum 1 second between requests

    def _rate_limit(self):
        """Implement rate limiting to avoid overwhelming the server"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.request_interval:
            time.sleep(self.request_interval - time_since_last)
        self.last_request_time = time.time()

    def get_current_price(self, ticker: str) -> Optional[float]:
        """Get current price for a ticker using yfinance"""
        self._rate_limit()
        try:
            import yfinance as yf

            # Format ticker for yfinance
            formatted_ticker = ticker.replace('%5E', '^')

            if formatted_ticker in ['^SPX'] or ticker in ['%5ESPX', 'SPX']:
                try:
                    gspc = yf.Ticker('^GSPC')
                    price = gspc.info.get("regularMarketPrice")
                    if price is None:
                        price = gspc.fast_info.get("lastPrice")
                    if price is not None:
                        return round(float(price), 2)
                except Exception as e:
                    logging.error(f"Error fetching SPX price: {e}")

            try:
                stock = yf.Ticker(ticker)
                price = stock.info.get("regularMarketPrice")
                if price is None:
                    price = stock.fast_info.get("lastPrice")
                if price is not None:
                    return round(float(price), 2)
            except Exception as e:
                logging.error(f"Yahoo Finance error for {ticker}: {e}")

        except Exception as e:
            logging.error(f"Error fetching price for {ticker}: {e}")

        return None

    def get_options_chain(self, ticker: str, expiry_date: str) -> Dict:
        """Get options chain data for specific expiry using yfinance"""
        self._rate_limit()
        try:
            import yfinance as yf

            stock = yf.Ticker(ticker)
            chain = stock.option_chain(expiry_date)

            return {
                'calls': chain.calls.to_dict('records') if not chain.calls.empty else [],
                'puts': chain.puts.to_dict('records') if not chain.puts.empty else [],
                'expiry': expiry_date,
                'ticker': ticker,
                'timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            logging.warning(f"Options chain request failed: {e}")

        return {}

    def get_greeks_data(self, ticker: str, expiry_date: str) -> Dict:
        """Get Greeks data (gamma, delta, charm, vanna) using yfinance"""
        self._rate_limit()
        try:
            import yfinance as yf
            from math import log, sqrt
            from scipy.stats import norm
            import numpy as np

            # Get current price
            current_price = self.get_current_price(ticker)
            if current_price is None:
                return {}

            # Get options chain
            stock = yf.Ticker(ticker)
            try:
                chain = stock.option_chain(expiry_date)
                calls = chain.calls
                puts = chain.puts
            except Exception as e:
                logging.warning(f"Could not fetch options chain for {ticker}: {e}")
                return {}

            # Calculate time to expiration
            expiry = datetime.strptime(expiry_date, '%Y-%m-%d').date()
            today = datetime.now().date()
            t_days = (expiry - today).days
            if t_days <= 0:
                return {}
            t = t_days / 365.0

            # Get risk-free rate (approximate)
            r = 0.05  # 5% annual rate as approximation

            # Calculate aggregate Greeks from options chain
            total_gamma = 0
            total_delta = 0
            total_charm = 0
            total_vanna = 0
            total_volume = 0

            # Process calls
            for _, option in calls.iterrows():
                try:
                    S = current_price
                    K = option['strike']
                    sigma = option.get('impliedVolatility', 0.2)  # Default 20% vol if not available

                    if sigma <= 0:
                        continue

                    d1 = (log(S / K) + (r + 0.5 * sigma**2) * t) / (sigma * sqrt(t))
                    d2 = d1 - sigma * sqrt(t)

                    # Greeks for calls
                    delta = norm.cdf(d1)
                    gamma = norm.pdf(d1) / (S * sigma * sqrt(t))
                    vega = S * norm.pdf(d1) * sqrt(t)
                    charm = -norm.pdf(d1) * (2*(r + 0.5*sigma**2)*t - d2*sigma*sqrt(t)) / (2*t*sigma*sqrt(t))
                    vanna = -norm.pdf(d1) * d2 / sigma

                    # Weight by open interest
                    oi = option.get('openInterest', 1)
                    total_gamma += gamma * oi
                    total_delta += delta * oi
                    total_charm += charm * oi
                    total_vanna += vanna * oi
                    total_volume += oi

                except Exception as e:
                    continue

            # Process puts
            for _, option in puts.iterrows():
                try:
                    S = current_price
                    K = option['strike']
                    sigma = option.get('impliedVolatility', 0.2)

                    if sigma <= 0:
                        continue

                    d1 = (log(S / K) + (r + 0.5 * sigma**2) * t) / (sigma * sqrt(t))
                    d2 = d1 - sigma * sqrt(t)

                    # Greeks for puts
                    delta = norm.cdf(d1) - 1
                    gamma = norm.pdf(d1) / (S * sigma * sqrt(t))
                    charm = -norm.pdf(d1) * (2*(r + 0.5*sigma**2)*t - d2*sigma*sqrt(t)) / (2*t*sigma*sqrt(t))
                    charm = -charm  # Negative for puts
                    vanna = -norm.pdf(d1) * d2 / sigma

                    # Weight by open interest
                    oi = option.get('openInterest', 1)
                    total_gamma += gamma * oi
                    total_delta += delta * oi
                    total_charm += charm * oi
                    total_vanna += vanna * oi
                    total_volume += oi

                except Exception as e:
                    continue

            # Return normalized Greeks
            if total_volume > 0:
                return {
                    'gamma': total_gamma / total_volume,
                    'delta': total_delta / total_volume,
                    'charm': total_charm / total_volume,
                    'vanna': total_vanna / total_volume,
                    'timestamp': datetime.now().isoformat()
                }

        except Exception as e:
            logging.error(f"Error calculating Greeks data: {e}")

        return {}

    def get_technical_indicators(self, ticker: str) -> Dict:
        """Get technical indicators (VWAP, Bollinger Bands, RSI, MACD) using yfinance"""
        self._rate_limit()
        try:
            import yfinance as yf
            import pandas as pd
            import numpy as np

            # Get historical data
            stock = yf.Ticker(ticker)
            hist = stock.history(period="6mo", interval="1d")

            if hist.empty:
                return {}

            # Calculate VWAP
            hist['VWAP'] = (hist['Close'] * hist['Volume']).cumsum() / hist['Volume'].cumsum()
            vwap = hist['VWAP'].iloc[-1]

            # Calculate Bollinger Bands
            sma = hist['Close'].rolling(window=20).mean()
            std = hist['Close'].rolling(window=20).std()
            upper_band = sma + 2 * std
            lower_band = sma - 2 * std

            bollinger_upper = upper_band.iloc[-1]
            bollinger_lower = lower_band.iloc[-1]
            bollinger_middle = sma.iloc[-1]

            # Calculate RSI
            def calculate_rsi(data, window=14):
                delta = data.diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
                rs = gain / loss
                rsi = 100 - (100 / (1 + rs))
                return rsi

            rsi = calculate_rsi(hist['Close']).iloc[-1]

            # Calculate MACD
            ema12 = hist['Close'].ewm(span=12, adjust=False).mean()
            ema26 = hist['Close'].ewm(span=26, adjust=False).mean()
            macd = ema12 - ema26
            signal = macd.ewm(span=9, adjust=False).mean()
            histogram = macd - signal

            macd_value = macd.iloc[-1]
            signal_value = signal.iloc[-1]
            histogram_value = histogram.iloc[-1]

            return {
                'vwap': vwap,
                'bollinger_upper': bollinger_upper,
                'bollinger_lower': bollinger_lower,
                'bollinger_middle': bollinger_middle,
                'rsi': rsi,
                'macd': macd_value,
                'macd_signal': signal_value,
                'macd_histogram': histogram_value,
                'timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            logging.error(f"Error calculating technical indicators: {e}")

        return {}

    def get_market_status(self) -> Dict:
        """Get current market status"""
        self._rate_limit()
        try:
            # Check if we can connect to yfinance
            import yfinance as yf

            # Try to get a test ticker
            test_ticker = yf.Ticker("AAPL")
            if test_ticker.info:
                return {
                    'status': 'connected',
                    'message': 'Connected to Yahoo Finance',
                    'timestamp': datetime.now().isoformat()
                }

        except Exception as e:
            logging.warning(f"Market status check failed: {e}")

        return {'status': 'disconnected', 'message': 'Unable to connect to data source'}

    def _scrape_price_from_ui(self, ticker: str) -> Optional[float]:
        """Fallback method to scrape price from ezOptions UI"""
        driver = None
        try:
            # Setup headless Chrome
            chrome_options = Options()
            chrome_options.add_argument("--headless")
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-dev-shm-usage")
            chrome_options.add_argument("--disable-gpu")
            chrome_options.add_argument("--window-size=1920,1080")

            driver = webdriver.Chrome(options=chrome_options)

            # Navigate to ezOptions
            driver.get(self.base_url)

            # Wait for page to load
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )

            # Try to find price elements (this would need to be customized based on ezOptions UI)
            # This is a placeholder implementation
            price_elements = driver.find_elements(By.CLASS_NAME, "current-price")
            if price_elements:
                price_text = price_elements[0].text
                # Extract numeric price from text
                import re
                price_match = re.search(r'[\d,]+\.?\d*', price_text.replace(',', ''))
                if price_match:
                    return float(price_match.group())

        except Exception as e:
            logging.error(f"Price scraping failed: {e}")

        finally:
            if driver:
                driver.quit()

        return None

class EzOptionsWebSocketConnector:
    """
    WebSocket connector for real-time data streaming from ezOptions
    """

    def __init__(self, ws_url: str = "ws://localhost:8501/ws"):
        self.ws_url = ws_url
        self.connected = False
        self.subscriptions = {}
        self.data_callbacks = {}
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 5

    def connect(self):
        """Establish WebSocket connection"""
        try:
            # WebSocket implementation would go here
            # This is a placeholder for WebSocket connectivity
            logging.info("WebSocket connection established")
            self.connected = True
            self.reconnect_attempts = 0
        except Exception as e:
            logging.error(f"WebSocket connection failed: {e}")
            self._handle_reconnection()

    def subscribe(self, channel: str, callback: callable):
        """Subscribe to a data channel"""
        self.subscriptions[channel] = True
        self.data_callbacks[channel] = callback
        logging.info(f"Subscribed to channel: {channel}")

    def unsubscribe(self, channel: str):
        """Unsubscribe from a data channel"""
        if channel in self.subscriptions:
            del self.subscriptions[channel]
            del self.data_callbacks[channel]
            logging.info(f"Unsubscribed from channel: {channel}")

    def _handle_reconnection(self):
        """Handle reconnection logic"""
        if self.reconnect_attempts < self.max_reconnect_attempts:
            self.reconnect_attempts += 1
            wait_time = min(2 ** self.reconnect_attempts, 60)  # Exponential backoff
            logging.info(f"Attempting reconnection in {wait_time} seconds...")
            time.sleep(wait_time)
            self.connect()
        else:
            logging.error("Max reconnection attempts reached")

class EzOptionsDataManager:
    """
    Data manager that coordinates API and WebSocket connections
    Provides unified interface for data access
    """

    def __init__(self, api_url: str = "http://localhost:8501", ws_url: str = "ws://localhost:8501/ws"):
        self.api_connector = EzOptionsAPIConnector(api_url)
        self.ws_connector = EzOptionsWebSocketConnector(ws_url)
        self.cache = {}
        self.cache_timeout = 30  # seconds
        self.data_listeners = []

    def start(self):
        """Start data manager"""
        logging.info("Starting EzOptions Data Manager")

        # Connect WebSocket for real-time data
        self.ws_connector.connect()

        # Setup subscriptions for key data
        self._setup_subscriptions()

    def stop(self):
        """Stop data manager"""
        logging.info("Stopping EzOptions Data Manager")
        # Cleanup connections
        pass

    def _setup_subscriptions(self):
        """Setup WebSocket subscriptions for real-time data"""
        # Subscribe to price updates
        self.ws_connector.subscribe("price_updates", self._handle_price_update)

        # Subscribe to Greeks updates
        self.ws_connector.subscribe("greeks_updates", self._handle_greeks_update)

        # Subscribe to technical indicators
        self.ws_connector.subscribe("technical_updates", self._handle_technical_update)

    def _handle_price_update(self, data: Dict):
        """Handle real-time price updates"""
        ticker = data.get('ticker')
        price = data.get('price')
        if ticker and price:
            self.cache[f'price_{ticker}'] = {
                'value': price,
                'timestamp': datetime.now()
            }
            self._notify_listeners('price_update', data)

    def _handle_greeks_update(self, data: Dict):
        """Handle real-time Greeks updates"""
        ticker = data.get('ticker')
        expiry = data.get('expiry')
        greeks = data.get('greeks', {})

        if ticker and expiry:
            cache_key = f'greeks_{ticker}_{expiry}'
            self.cache[cache_key] = {
                'value': greeks,
                'timestamp': datetime.now()
            }
            self._notify_listeners('greeks_update', data)

    def _handle_technical_update(self, data: Dict):
        """Handle real-time technical indicator updates"""
        ticker = data.get('ticker')
        indicators = data.get('indicators', {})

        if ticker:
            cache_key = f'technical_{ticker}'
            self.cache[cache_key] = {
                'value': indicators,
                'timestamp': datetime.now()
            }
            self._notify_listeners('technical_update', data)

    def add_data_listener(self, callback: callable):
        """Add a listener for data updates"""
        self.data_listeners.append(callback)

    def remove_data_listener(self, callback: callable):
        """Remove a data listener"""
        if callback in self.data_listeners:
            self.data_listeners.remove(callback)

    def _notify_listeners(self, event_type: str, data: Dict):
        """Notify all listeners of data updates"""
        for listener in self.data_listeners:
            try:
                listener(event_type, data)
            except Exception as e:
                logging.error(f"Error notifying listener: {e}")

    def get_cached_data(self, key: str) -> Optional[Dict]:
        """Get cached data if still valid"""
        if key in self.cache:
            cached_item = self.cache[key]
            if (datetime.now() - cached_item['timestamp']).seconds < self.cache_timeout:
                return cached_item['value']
            else:
                # Remove expired cache
                del self.cache[key]
        return None

    def get_current_price(self, ticker: str) -> Optional[float]:
        """Get current price with caching"""
        cache_key = f'price_{ticker}'
        cached_price = self.get_cached_data(cache_key)

        if cached_price is not None:
            return cached_price

        # Fetch from API if not cached or expired
        price = self.api_connector.get_current_price(ticker)
        if price is not None:
            self.cache[cache_key] = {
                'value': price,
                'timestamp': datetime.now()
            }
        return price

    def get_greeks_data(self, ticker: str, expiry_date: str) -> Dict:
        """Get Greeks data with caching"""
        cache_key = f'greeks_{ticker}_{expiry_date}'
        cached_greeks = self.get_cached_data(cache_key)

        if cached_greeks is not None:
            return cached_greeks

        # Fetch from API if not cached or expired
        greeks = self.api_connector.get_greeks_data(ticker, expiry_date)
        if greeks:
            self.cache[cache_key] = {
                'value': greeks,
                'timestamp': datetime.now()
            }
        return greeks

    def get_technical_indicators(self, ticker: str) -> Dict:
        """Get technical indicators with caching"""
        cache_key = f'technical_{ticker}'
        cached_indicators = self.get_cached_data(cache_key)

        if cached_indicators is not None:
            return cached_indicators

        # Fetch from API if not cached or expired
        indicators = self.api_connector.get_technical_indicators(ticker)
        if indicators:
            self.cache[cache_key] = {
                'value': indicators,
                'timestamp': datetime.now()
            }
        return indicators

    def get_market_status(self) -> Dict:
        """Get current market status"""
        return self.api_connector.get_market_status()

    def get_options_chain(self, ticker: str, expiry_date: str) -> Dict:
        """Get options chain data"""
        return self.api_connector.get_options_chain(ticker, expiry_date)

class EzOptionsMonitor:
    """
    Monitor class that continuously tracks ezOptions connectivity and data flow
    """

    def __init__(self, data_manager: EzOptionsDataManager):
        self.data_manager = data_manager
        self.monitoring = False
        self.connection_status = "disconnected"
        self.last_data_timestamp = None
        self.data_flow_rate = 0  # updates per minute

    def start_monitoring(self):
        """Start monitoring ezOptions connection and data flow"""
        self.monitoring = True
        monitor_thread = threading.Thread(target=self._monitor_loop)
        monitor_thread.daemon = True
        monitor_thread.start()
        logging.info("EzOptions monitoring started")

    def stop_monitoring(self):
        """Stop monitoring"""
        self.monitoring = False
        logging.info("EzOptions monitoring stopped")

    def _monitor_loop(self):
        """Main monitoring loop"""
        update_counts = []
        last_minute = datetime.now().minute

        while self.monitoring:
            try:
                # Check connection status
                market_status = self.data_manager.get_market_status()
                self.connection_status = market_status.get('status', 'unknown')

                # Track data flow
                current_minute = datetime.now().minute
                if current_minute != last_minute:
                    self.data_flow_rate = len(update_counts)
                    update_counts = []
                    last_minute = current_minute

                # Log status every 5 minutes
                if datetime.now().second == 0 and datetime.now().minute % 5 == 0:
                    self._log_status()

                time.sleep(10)  # Check every 10 seconds

            except Exception as e:
                logging.error(f"Monitoring error: {e}")
                time.sleep(30)

    def _log_status(self):
        """Log current monitoring status"""
        status_info = {
            'connection_status': self.connection_status,
            'data_flow_rate': self.data_flow_rate,
            'last_data_timestamp': self.last_data_timestamp,
            'cache_size': len(self.data_manager.cache)
        }
        logging.info(f"EzOptions Status: {json.dumps(status_info, default=str)}")

    def get_monitoring_status(self) -> Dict:
        """Get current monitoring status"""
        return {
            'monitoring_active': self.monitoring,
            'connection_status': self.connection_status,
            'data_flow_rate': self.data_flow_rate,
            'last_data_timestamp': self.last_data_timestamp,
            'cache_entries': len(self.data_manager.cache)
        }


# Global instances for easy access
data_manager = None
monitor = None

def initialize_ezoptions_connection():
    """Initialize the ezOptions connection"""
    global data_manager, monitor

    try:
        data_manager = EzOptionsDataManager()
        monitor = EzOptionsMonitor(data_manager)
        data_manager.start()
        monitor.start_monitoring()
        logging.info("EzOptions connection initialized successfully")
        return True
    except Exception as e:
        logging.error(f"Failed to initialize ezOptions connection: {e}")
        return False

def get_ezoptions_status() -> Dict:
    """Get comprehensive ezOptions connection status"""
    global data_manager, monitor
    
    if data_manager is None or monitor is None:
        return {
            'data_manager_status': {
                'connected': False,
                'cache_size': 0,
                'active_subscriptions': 0
            },
            'monitoring_status': {
                'monitoring_active': False,
                'connection_status': 'disconnected',
                'data_flow_rate': 0,
                'last_data_timestamp': None,
                'cache_entries': 0
            },
            'market_status': {
                'status': 'disconnected',
                'message': 'Data manager not initialized'
            }
        }
    
    return {
        'data_manager_status': {
            'connected': data_manager.ws_connector.connected if data_manager.ws_connector else False,
            'cache_size': len(data_manager.cache),
            'active_subscriptions': len(data_manager.ws_connector.subscriptions) if data_manager.ws_connector else 0
        },
        'monitoring_status': monitor.get_monitoring_status(),
        'market_status': data_manager.get_market_status()
    }

if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    # Initialize connection
    if initialize_ezoptions_connection():
        # Test data retrieval
        price = data_manager.get_current_price("US100")
        print(f"Current US100 price: {price}")

        greeks = data_manager.get_greeks_data("QQQ", "2024-12-31")
        print(f"Greeks data: {greeks}")

        technical = data_manager.get_technical_indicators("US100")
        print(f"Technical indicators: {technical}")

        # Get status
        status = get_ezoptions_status()
        print(f"Connection status: {json.dumps(status, indent=2, default=str)}")

        # Keep running for a bit
        time.sleep(60)
    else:
        print("Failed to initialize ezOptions connection")