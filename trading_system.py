"""
Automated Trading System Architecture
Integration between ezOptions data and MetaTrader 5 for NASDAQ-100 trading
"""

import threading
import time
import logging
import json
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
# from advanced_setups import AdvancedSetupDetector  # Módulo não encontrado - usando alternativa
from vwap_bollinger_indicators import VWAPBollingerIndicators
from trading_setups_corrected import TradingSetupsCorrected
import MetaTrader5 as mt5

logger = logging.getLogger(__name__)

class EzOptionsConnector:
    """
    Connector to fetch real-time data from ezOptions running on localhost:8501
    """

    def __init__(self, base_url: str = "http://localhost:8501"):
        self.base_url = base_url
        self.session = requests.Session()
        self.last_data = {}
        self.cache_timeout = 30  # seconds

    def get_current_price(self, ticker: str) -> Optional[float]:
        """Get current price for ticker using yfinance"""
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

    def get_options_greeks(self, ticker: str, expiry_date: str) -> Dict:
        """Get options Greeks data using yfinance"""
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

class MT5Integration:
    """
    MetaTrader 5 integration for automated trade execution
    """

    def __init__(self, account_config: Dict):
        self.account_config = account_config
        self.connected = False
        self.initialize_mt5()

    def initialize_mt5(self) -> bool:
        """Initialize MT5 connection"""
        try:
            if not mt5.initialize():
                logging.error("MT5 initialization failed")
                return False

            login = self.account_config.get('login')
            password = self.account_config.get('password')
            server = self.account_config.get('server')

            if not mt5.login(login, password, server):
                logging.error(f"MT5 login failed: {mt5.last_error()}")
                return False

            self.connected = True
            logging.info("MT5 connection established successfully")
            return True

        except Exception as e:
            logging.error(f"MT5 initialization error: {e}")
            return False

    def get_account_info(self) -> Dict:
        """Get current account information"""
        if not self.connected:
            return {}

        account_info = mt5.account_info()
        if account_info:
            return {
                'balance': account_info.balance,
                'equity': account_info.equity,
                'margin': account_info.margin,
                'margin_free': account_info.margin_free,
                'profit': account_info.profit
            }
        return {}

    def place_order(self, symbol: str, order_type: str, volume: float,
                   price: float = 0.0, sl: float = 0.0, tp: float = 0.0) -> Dict:
        """Place order in MT5"""
        if not self.connected:
            return {'success': False, 'error': 'MT5 not connected'}

        try:
            # Get symbol info for volume validation
            symbol_info = mt5.symbol_info(symbol)
            if not symbol_info:
                return {'success': False, 'error': f'Could not get symbol info for {symbol}'}

            # Get volume constraints
            min_volume = symbol_info.volume_min if symbol_info.volume_min > 0 else 0.01
            max_volume = symbol_info.volume_max if symbol_info.volume_max > 0 else 100.0
            volume_step = symbol_info.volume_step if symbol_info.volume_step > 0 else 0.01

            logging.info(f"Symbol {symbol} volume constraints - Min: {min_volume}, Max: {max_volume}, Step: {volume_step}")
            logging.info(f"Original volume requested: {volume}")

            # Adjust volume to be within limits
            if volume < min_volume:
                logging.warning(f"Volume {volume} below minimum {min_volume}, adjusting to {min_volume}")
                volume = min_volume
            elif volume > max_volume:
                logging.warning(f"Volume {volume} above maximum {max_volume}, adjusting to {max_volume}")
                volume = max_volume

            # Round volume to valid increment
            if volume_step > 0:
                volume = round(volume / volume_step) * volume_step
                volume = round(volume, 2)  # Avoid floating point precision issues
                logging.info(f"Volume after step rounding: {volume}")

            # Ensure volume is still within limits after rounding
            volume = max(min_volume, min(volume, max_volume))
            logging.info(f"Final volume after validation: {volume}")

            # Map order types
            order_type_map = {
                'BUY': mt5.ORDER_TYPE_BUY,
                'SELL': mt5.ORDER_TYPE_SELL,
                'BUY_LIMIT': mt5.ORDER_TYPE_BUY_LIMIT,
                'SELL_LIMIT': mt5.ORDER_TYPE_SELL_LIMIT,
                'BUY_STOP': mt5.ORDER_TYPE_BUY_STOP,
                'SELL_STOP': mt5.ORDER_TYPE_SELL_STOP
            }

            # Converter tipos numpy para Python nativos para evitar problemas com MT5 API
            def convert_numpy_types(value):
                import numpy as np
                if value is None:
                    return 0.0  # Return default for None values
                if hasattr(value, 'item'):  # É um tipo numpy
                    return value.item()
                return float(value) if value is not None else 0.0

            # Obter informações atualizadas do símbolo para garantir que está disponível
            symbol_info = mt5.symbol_info(symbol)
            if not symbol_info:
                logging.error(f"Símbolo {symbol} não encontrado ou indisponível")
                return {'success': False, 'error': f'Símbolo {symbol} não encontrado'}
            
            if not symbol_info.select:
                logging.error(f"Símbolo {symbol} não está selecionado para negociação")
                return {'success': False, 'error': f'Símbolo {symbol} não está selecionado'}
            
            if not symbol_info.visible:
                logging.error(f"Símbolo {symbol} não está visível")
                return {'success': False, 'error': f'Símbolo {symbol} não está visível'}

            # Obter o preço atual de mercado para usar na ordem e para validação
            current_tick = mt5.symbol_info_tick(symbol)
            if current_tick:
                market_price = current_tick.ask if 'BUY' in order_type else current_tick.bid
            else:
                logging.warning(f"Não foi possível obter preço atual para {symbol}, usando preço do sinal")
                # Se não conseguir preço atual, usar o preço original convertido
                market_price = convert_numpy_types(price)
            
            # Validar se o preço de referência é razoável para evitar valores estranhos
            if market_price <= 0 or market_price > 100000:  # Valor de segurança razoável para US100
                logging.error(f"Preço inválido detectado: {market_price}, usando preço de sinal convertido")
                market_price = convert_numpy_types(price)

            # Validar se o preço é válido baseado na posição de compra/venda usando o preço de mercado
            # Ajustar SL/TP baseado no preço de mercado (ask/bid) para garantir distâncias corretas
            if 'BUY' in order_type:
                # Para ordens de compra: SL deve estar abaixo, TP deve estar acima do preço de mercado
                if converted_sl != 0 and converted_sl >= market_price:
                    logging.warning(f"Stop loss ({converted_sl}) deve ser menor que preço de mercado ({market_price}) para ordem de compra")
                    # Ajustar SL para estar abaixo do preço de mercado
                    converted_sl = market_price * 0.995  # Ajustar para 0.5% abaixo (mais conservador)
                if converted_tp != 0 and converted_tp <= market_price:
                    logging.warning(f"Take profit ({converted_tp}) deve ser maior que preço de mercado ({market_price}) para ordem de compra")
                    # Ajustar TP para estar acima do preço de mercado
                    converted_tp = market_price * 1.01  # Ajustar para 1% acima (mais conservador)
            else:  # SELL
                # Para ordens de venda: SL deve estar acima, TP deve estar abaixo do preço de mercado
                if converted_sl != 0 and converted_sl <= market_price:
                    logging.warning(f"Stop loss ({converted_sl}) deve ser maior que preço de mercado ({market_price}) para ordem de venda")
                    # Ajustar SL para estar acima do preço de mercado
                    converted_sl = market_price * 1.005  # Ajustar para 0.5% acima (mais conservador)
                if converted_tp != 0 and converted_tp >= market_price:
                    logging.warning(f"Take profit ({converted_tp}) deve ser menor que preço de mercado ({market_price}) para ordem de venda")
                    # Ajustar TP para estar abaixo do preço de mercado
                    converted_tp = market_price * 0.99  # Ajustar para 1% abaixo (mais conservador)

            # Validação final: garantir distâncias mínimas para evitar erro 10016
            if converted_sl != 0:
                sl_distance = abs(converted_sl - market_price)
                if sl_distance < 5:  # Distância mínima de 5 pontos
                    if 'BUY' in order_type:
                        converted_sl = market_price - 5
                    else:
                        converted_sl = market_price + 5
                    logging.info(f"Ajustando SL para distância mínima: {converted_sl}")

            if converted_tp != 0:
                tp_distance = abs(converted_tp - market_price)
                if tp_distance < 5:  # Distância mínima de 5 pontos
                    if 'BUY' in order_type:
                        converted_tp = market_price + 5
                    else:
                        converted_tp = market_price - 5
                    logging.info(f"Ajustando TP para distância mínima: {converted_tp}")

            # Preparar request da ordem com validação extra
            converted_volume = convert_numpy_types(volume)
            converted_sl = convert_numpy_types(converted_sl) if converted_sl is not None and converted_sl != 0 else 0.0
            converted_tp = convert_numpy_types(converted_tp) if converted_tp is not None and converted_tp != 0 else 0.0

            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": converted_volume,
                "type": order_type_map.get(order_type, mt5.ORDER_TYPE_BUY),
                "price": market_price,
                "sl": converted_sl,
                "tp": converted_tp,
                "deviation": 20,  # Aumentar desvio para dar mais flexibilidade
                "magic": 123456,
                "comment": "Automated Trading System",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_FOK,  # Tentar FOK em vez de IOC
            }

            result = mt5.order_send(request)

            # Log detalhado do resultado
            logging.info(f"MT5 Order Result - retcode: {result.retcode}, comment: {result.comment}")

            if result.retcode == mt5.TRADE_RETCODE_DONE:
                return {
                    'success': True,
                    'order_id': result.order,
                    'price': result.price,
                    'volume': result.volume
                }
            else:
                # Log detalhado do erro
                error_details = {
                    'retcode': result.retcode,
                    'comment': result.comment,
                    'request': request,
                    'symbol_info': {
                        'trade_contract_size': symbol_info.trade_contract_size,
                        'volume_min': symbol_info.volume_min,
                        'volume_max': symbol_info.volume_max,
                        'volume_step': symbol_info.volume_step,
                        'point': symbol_info.point,
                        'digits': symbol_info.digits
                    }
                }
                logging.error(f"Order rejected: {result.retcode} - {result.comment}")
                logging.error(f"Error details: {error_details}")
                return {
                    'success': False,
                    'error': f"Order rejected: {result.retcode} - {result.comment}",
                    'retcode': result.retcode,
                    'details': error_details
                }

        except Exception as e:
            logging.error(f"Order placement error: {e}")
            return {'success': False, 'error': str(e)}

    def get_symbol_info(self, symbol: str = None) -> Dict:
        """Get symbol information including volume constraints"""
        if not self.connected:
            return {}
        
        target_symbol = symbol or 'US100'  # Default to US100 if no symbol specified
        symbol_info = mt5.symbol_info(target_symbol)
        
        if symbol_info:
            return {
                'symbol': symbol_info.name,
                'bid': symbol_info.bid,
                'ask': symbol_info.ask,
                'spread': symbol_info.spread,
                'digits': symbol_info.digits,
                'volume_min': symbol_info.volume_min,
                'volume_max': symbol_info.volume_max,
                'volume_step': symbol_info.volume_step
            }
        return {}

    def get_positions(self) -> List[Dict]:
        """Get current open positions"""
        if not self.connected:
            return []

        positions = mt5.positions_get()
        if positions:
            return [{
                'ticket': pos.ticket,
                'symbol': pos.symbol,
                'type': 'BUY' if pos.type == mt5.POSITION_TYPE_BUY else 'SELL',
                'volume': pos.volume,
                'price_open': pos.price_open,
                'price_current': pos.price_current,
                'profit': pos.profit,
                'sl': pos.sl,
                'tp': pos.tp
            } for pos in positions]
        return []

    def close_position(self, ticket: int) -> Dict:
        """Close specific position"""
        if not self.connected:
            return {'success': False, 'error': 'MT5 not connected'}

        try:
            position = mt5.positions_get(ticket=ticket)
            if not position:
                return {'success': False, 'error': 'Position not found'}

            pos = position[0]

            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "position": ticket,
                "symbol": pos.symbol,
                "volume": pos.volume,
                "type": mt5.ORDER_TYPE_SELL if pos.type == mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_BUY,
                "price": mt5.symbol_info_tick(pos.symbol).bid if pos.type == mt5.POSITION_TYPE_BUY else mt5.symbol_info_tick(pos.symbol).ask,
                "deviation": 10,
                "magic": 123456,
                "comment": "Close position",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }

            result = mt5.order_send(request)
            if result.retcode == mt5.TRADE_RETCODE_DONE:
                return {'success': True, 'closed_price': result.price}
            else:
                return {'success': False, 'error': f"Close failed: {result.retcode}"}

        except Exception as e:
            logging.error(f"Position close error: {e}")
            return {'success': False, 'error': str(e)}

class RiskManager:
    """
    Risk management system with stop-loss, take-profit, and position sizing
    """

    def __init__(self, config: Dict):
        self.config = config
        self.max_daily_loss = config.get('max_daily_loss', 1000)
        self.max_position_size = config.get('max_position_size', 0.1)  # 10% of account
        self.daily_pnl = 0.0
        self.open_positions = []

    def calculate_position_size(self, account_balance: float, risk_per_trade: float = 0.02) -> float:
        """Calculate position size based on risk management rules"""
        risk_amount = account_balance * risk_per_trade
        return min(risk_amount, account_balance * self.max_position_size)

    def validate_trade(self, trade_signal: Dict, account_info: Dict) -> Tuple[bool, str]:
        """Validate trade against risk management rules"""
        account_balance = account_info.get('balance', 0)

        # Check daily loss limit
        if self.daily_pnl <= -self.max_daily_loss:
            return False, "Daily loss limit reached"

        # Check account balance
        if account_balance < 1000:  # Minimum balance check
            return False, "Insufficient account balance"

        # Check position size
        position_size = trade_signal.get('volume', 0)
        max_allowed_size = self.calculate_position_size(account_balance)
        if position_size > max_allowed_size:
            return False, f"Position size {position_size} exceeds maximum allowed {max_allowed_size}"

        return True, "Trade validated"

    def update_daily_pnl(self, pnl_change: float):
        """Update daily P&L tracking"""
        self.daily_pnl += pnl_change

    def get_risk_metrics(self) -> Dict:
        """Get current risk metrics"""
        return {
            'daily_pnl': self.daily_pnl,
            'max_daily_loss': self.max_daily_loss,
            'daily_loss_percentage': (abs(self.daily_pnl) / self.max_daily_loss) * 100 if self.max_daily_loss > 0 else 0,
            'open_positions_count': len(self.open_positions)
        }

class SetupConfirmationEngine:
    """
    Engine to confirm trading setups based on multiple criteria
    """

    def __init__(self, ezoptions_connector: EzOptionsConnector):
        self.ezoptions_connector = ezoptions_connector
        self.setups = TradingSetupsCorrected()
        self.confirmation_history = []

    def confirm_setup(self, setup_number: int, ticker: str, expiry_date: str) -> Dict:
        """Confirm if a setup is valid for trading"""
        setup_config = self.setups.get_setup(setup_number)
        if not setup_config:
            return {'confirmed': False, 'reason': 'Setup not found'}

        # Get current market data
        market_data = self._get_market_data(ticker, expiry_date)
        if not market_data:
            return {'confirmed': False, 'reason': 'Unable to fetch market data'}

        # Validate basic conditions
        validation = self.setups.validate_setup_conditions(setup_number, market_data)

        if not validation['valid']:
            return {
                'confirmed': False,
                'reason': f"Setup conditions not met: {', '.join(validation['conditions_failed'])}",
                'confidence': validation['confidence_score']
            }

        # Check confirmation signals
        confirmation_signals = self._check_confirmation_signals(setup_config, market_data)

        # Calculate overall confidence
        overall_confidence = (validation['confidence_score'] + confirmation_signals['score']) / 2

        result = {
            'confirmed': overall_confidence >= 0.7,  # 70% confidence threshold
            'confidence_score': overall_confidence,
            'setup_name': setup_config['name'],
            'direction': setup_config['direction'],
            'risk_multiplier': setup_config['risk_multiplier'],
            'time_horizon': setup_config['time_horizon'],
            'market_data': market_data,
            'validation_details': validation,
            'confirmation_signals': confirmation_signals
        }

        # Store confirmation result
        self.confirmation_history.append({
            'timestamp': datetime.now(),
            'setup_number': setup_number,
            'result': result
        })

        return result

    def _get_market_data(self, ticker: str, expiry_date: str) -> Dict:
        """Get comprehensive market data for setup validation"""
        try:
            # Get Greeks data
            greeks_data = self.ezoptions_connector.get_options_greeks(ticker, expiry_date)

            # Get technical indicators
            tech_data = self.ezoptions_connector.get_technical_indicators(ticker)

            # Get current price
            current_price = self.ezoptions_connector.get_current_price(ticker)

            if not current_price:
                return {}

            return {
                'gamma': greeks_data.get('gamma', 0),
                'delta': greeks_data.get('delta', 0),
                'charm': greeks_data.get('charm', 0),
                'vanna': greeks_data.get('vanna', 0),
                'current_price': current_price,
                'vwap': tech_data.get('vwap'),
                'bollinger_upper': tech_data.get('bollinger_upper'),
                'bollinger_lower': tech_data.get('bollinger_lower'),
                'volume': greeks_data.get('volume', 0),
                'open_interest': greeks_data.get('open_interest', 0),
                'timestamp': datetime.now()
            }

        except Exception as e:
            logging.error(f"Error getting market data: {e}")
            return {}

    def _check_confirmation_signals(self, setup_config: Dict, market_data: Dict) -> Dict:
        """Check confirmation signals for the setup"""
        signals = setup_config.get('confirmation_signals', [])
        signal_results = {}
        score = 0.0

        for signal in signals:
            if signal == 'gamma_increasing':
                # Check if gamma is trending up
                signal_results[signal] = market_data.get('gamma', 0) > 0.1
                if signal_results[signal]:
                    score += 0.2

            elif signal == 'delta_stable':
                # Check if delta is relatively stable
                signal_results[signal] = abs(market_data.get('delta', 0)) < 0.15
                if signal_results[signal]:
                    score += 0.15

            elif signal == 'charm_accelerating':
                # Check if charm is accelerating
                signal_results[signal] = market_data.get('charm', 0) > 0.01
                if signal_results[signal]:
                    score += 0.15

            elif signal == 'vwap_rejection':
                # Check VWAP rejection for setup 5
                current_price = market_data.get('current_price', 0)
                vwap = market_data.get('vwap', 0)
                signal_results[signal] = abs(current_price - vwap) / vwap > 0.005  # 0.5% deviation
                if signal_results[signal]:
                    score += 0.2

            elif signal == 'bollinger_squeeze':
                # Check Bollinger Band squeeze
                upper = market_data.get('bollinger_upper', 0)
                lower = market_data.get('bollinger_lower', 0)
                current_price = market_data.get('current_price', 0)
                if upper > 0 and lower > 0:
                    bandwidth = (upper - lower) / current_price
                    signal_results[signal] = bandwidth < 0.02  # Tight bands
                    if signal_results[signal]:
                        score += 0.15

        return {
            'signals': signal_results,
            'score': min(score, 1.0)  # Cap at 1.0
        }

class AutomatedTradingSystem:
    """
    Sistema principal de negociação automatizada
    """

    def __init__(self, config_path: str = 'config.json'):
        self.config = self.load_config(config_path)
        self.mt5_connected = False
        self.ezoptions_data = {}
        self.active_positions = {}

        # Initialize advanced components
        self.ezoptions_connector = EzOptionsConnector(self.config.get("ezoptions", {}).get("base_url", "http://localhost:8501"))
        self.setup_detector = TradingSetupsCorrected(self.config)  # Usando alternativa existente
        self.vwap_bb_indicators = VWAPBollingerIndicators()

        # Initialize MT5 and other components
        self.mt5_integration = None
        self.risk_manager = None
        self.setup_engine = SetupConfirmationEngine(self.ezoptions_connector)
        self.trade_history = []
        self.running = False
        self.active_setups = self.config.get('trading_parameters', {}).get('active_setups', [])

        # Trading statistics
        self.trading_stats = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_profit': 0.0,
            'daily_trades': 0,
            'consecutive_losses': 0,
            'last_reset_date': datetime.now().date()
        }

        # Initialize MT5
        self.initialize_mt5()

    def load_config(self, config_path: str) -> Dict:
        """Carrega configuração do sistema"""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            # Configuração padrão
            return {
                "mt5": {
                    "server": "FBS-Real",
                    "login": 11655745,
                    "password": "Street@21",
                    "path": "C:\\Program Files\\MetaTrader 5\\terminal64.exe"
                },
                "ezoptions": {
                    "base_url": "http://localhost:8501",
                    "update_interval_seconds": 30,
                    "tickers": ["QQQ"]
                },
                "trading": {
                    "symbol": "US100",
                    "lot_size": 0.01,
                    "max_positions": 1,
                    "stop_loss_pips": 50,
                    "take_profit_pips": 100,
                    "max_drawdown_percent": 5.0
                },
                "risk_management": {
                    "max_daily_loss": 100.0,
                    "max_consecutive_losses": 3,
                    "daily_profit_target": 50.0
                },
                "trading_parameters": {
                    "ticker": "US100",
                    "expiry_date": "2024-12-20",
                    "active_setups": [1, 2, 3, 4, 5, 6],
                    "check_interval": 30
                }
            }
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing configuration file: {e}")
            # Return default configuration
            return {
                "mt5": {
                    "server": "FBS-Real",
                    "login": 11655745,
                    "password": "Street@21",
                    "path": "C:\\Program Files\\MetaTrader 5\\terminal64.exe"
                },
                "ezoptions": {
                    "base_url": "http://localhost:8501",
                    "update_interval_seconds": 30,
                    "tickers": ["QQQ"]
                },
                "trading": {
                    "symbol": "US100",
                    "lot_size": 0.01,
                    "max_positions": 1,
                    "stop_loss_pips": 50,
                    "take_profit_pips": 100,
                    "max_drawdown_percent": 5.0
                },
                "risk_management": {
                    "max_daily_loss": 100.0,
                    "max_consecutive_losses": 3,
                    "daily_profit_target": 50.0
                },
                "trading_parameters": {
                    "ticker": "US100",
                    "expiry_date": "2024-12-20",
                    "active_setups": [1, 2, 3, 4, 5, 6],
                    "check_interval": 30
                }
            }

    def initialize_mt5(self) -> bool:
        """Inicializa conexão com MT5"""
        try:
            # Initialize MT5Integration if not already initialized
            if not self.mt5_integration:
                mt5_config = self.config.get("mt5", {})
                self.mt5_integration = MT5Integration(mt5_config)
            
            # Check if already connected
            if self.mt5_integration.connected:
                logger.info("MT5 já está conectado")
                self.mt5_connected = True
                return True

            # Try to connect via MT5Integration
            if self.mt5_integration.initialize_mt5():
                self.mt5_connected = True
                logger.info("MT5 inicializado com sucesso via MT5Integration")

                # Log account info for debugging
                account_info = self.mt5_integration.get_account_info()
                if account_info:
                    logger.info(f"Account Info - Balance: {account_info.get('balance', 0)}, Equity: {account_info.get('equity', 0)}")
                else:
                    logger.warning("Não foi possível obter informações da conta MT5")

                return True
            else:
                logger.error("Falha ao inicializar MT5 via MT5Integration")
                return False

        except Exception as e:
            logger.error(f"Erro ao inicializar MT5: {e}")
            return False

    def shutdown(self):
        """Desliga o sistema"""
        if self.mt5_connected:
            mt5.shutdown()
            logger.info("MT5 desconectado")

        if hasattr(self, 'ezoptions_connector'):
            self.ezoptions_connector.close_connection()

        logger.info("Sistema de negociação encerrado")

        # Also log to console
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)

    def start_trading(self):
        """Start the automated trading system"""
        if self.running:
            logging.warning("Trading system is already running")
            return

        logging.info("Starting Automated Trading System...")
        self.running = True

        # Start trading loop in separate thread
        trading_thread = threading.Thread(target=self._trading_loop)
        trading_thread.daemon = True
        trading_thread.start()

        logging.info("Trading system started successfully")

    def stop_trading(self):
        """Stop the automated trading system"""
        logging.info("Stopping Automated Trading System...")
        self.running = False

        # Close all positions
        self._close_all_positions()

        logging.info("Trading system stopped")

    def _trading_loop(self):
        """Main trading loop"""
        check_interval = self.config.get('trading_parameters', {}).get('check_interval', 30)

        while self.running:
            try:
                self._check_and_execute_trades()
                time.sleep(check_interval)

            except Exception as e:
                logging.error(f"Error in trading loop: {e}")
                time.sleep(60)  # Wait longer on error

    def _check_and_execute_trades(self):
        """Check for trading opportunities and execute trades"""
        ticker = self.config.get('trading_parameters', {}).get('ticker', 'US100')
        expiry_date = self.config.get('trading_parameters', {}).get('expiry_date')
        active_setups = self.config.get('trading_parameters', {}).get('active_setups', [])

        if not expiry_date:
            logging.warning("No expiry date configured")
            return

        # Check each active setup
        for setup_number in active_setups:
            try:
                # Confirm setup
                confirmation = self.setup_engine.confirm_setup(setup_number, ticker, expiry_date)

                if confirmation['confirmed']:
                    logging.info(f"Setup {setup_number} confirmed: {confirmation['setup_name']}")

                    # Generate trade signal
                    trade_signal = self._generate_trade_signal(confirmation)

                    # Validate with risk manager
                    account_info = self.mt5_integration.get_account_info()
                    risk_validation = self.risk_manager.validate_trade(trade_signal, account_info)

                    if risk_validation[0]:  # Trade validated
                        # Execute trade
                        result = self.mt5_integration.place_order(
                            symbol=ticker,
                            order_type=trade_signal['order_type'],
                            volume=trade_signal['volume'],
                            sl=trade_signal['stop_loss'],
                            tp=trade_signal['take_profit']
                        )

                        if result['success']:
                            logging.info(f"Trade executed successfully: {result}")
                            self.trade_history.append({
                                'timestamp': datetime.now(),
                                'setup_number': setup_number,
                                'trade_result': result,
                                'confirmation': confirmation
                            })
                        else:
                            logging.error(f"Trade execution failed: {result}")
                    else:
                        logging.warning(f"Trade rejected by risk manager: {risk_validation[1]}")

                else:
                    logging.debug(f"Setup {setup_number} not confirmed: {confirmation.get('reason', 'Unknown')}")

            except Exception as e:
                logging.error(f"Error processing setup {setup_number}: {e}")

    def _generate_trade_signal(self, confirmation: Dict) -> Dict:
        """Generate trade signal from setup confirmation"""
        direction = confirmation['direction']
        risk_multiplier = confirmation['risk_multiplier']

        # Base position size calculation
        account_info = self.mt5_integration.get_account_info()
        base_size = self.risk_manager.calculate_position_size(account_info.get('balance', 10000))

        # Adjust for risk multiplier
        position_size = base_size * risk_multiplier

        # Determine order type
        if direction == 'bullish':
            order_type = 'BUY'
        elif direction == 'bearish':
            order_type = 'SELL'
        else:
            order_type = 'BUY'  # Default for reversal setups

        # Calculate stop loss and take profit
        current_price = confirmation['market_data']['current_price']
        stop_loss_distance = current_price * 0.005 * risk_multiplier  # 0.5% base SL adjusted by risk
        take_profit_distance = current_price * 0.01 * risk_multiplier  # 1% base TP adjusted by risk

        if direction == 'bullish':
            stop_loss = current_price - stop_loss_distance
            take_profit = current_price + take_profit_distance
        else:
            stop_loss = current_price + stop_loss_distance
            take_profit = current_price - take_profit_distance

        return {
            'order_type': order_type,
            'volume': position_size,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'symbol': 'US100',
            'confirmation': confirmation
        }

    def _close_all_positions(self):
        """Close all open positions"""
        positions = self.mt5_integration.get_positions()

        for position in positions:
            result = self.mt5_integration.close_position(position['ticket'])
            if result['success']:
                logging.info(f"Closed position {position['ticket']}: {result}")
            else:
                logging.error(f"Failed to close position {position['ticket']}: {result}")

    def get_system_status(self) -> Dict:
        """Get current system status"""
        return {
            'running': self.running,
            'mt5_connected': self.mt5_integration.connected if self.mt5_integration else False,
            'ezoptions_connected': self.ezoptions_connector.session is not None if self.ezoptions_connector else False,
            'account_info': self.mt5_integration.get_account_info() if self.mt5_integration else {},
            'risk_metrics': self.risk_manager.get_risk_metrics() if self.risk_manager else {},
            'active_setups': self.active_setups,
            'recent_trades': self.trade_history[-10:] if self.trade_history else []
        }


if __name__ == "__main__":
    # Example usage
    system = AutomatedTradingSystem()

    # Print system status
    status = system.get_system_status()
    print("System Status:")
    print(json.dumps(status, indent=2, default=str))