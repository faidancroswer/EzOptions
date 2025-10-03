"""
API REST para EzOptions - Fornece dados reais para o sistema de trading
Roda em paralelo com o dashboard Streamlit
"""

from flask import Flask, jsonify, request
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import threading
import logging
from typing import Dict, List, Optional

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

class EzOptionsAPI:
    """API para fornecer dados reais do ezOptions"""

    def __init__(self):
        self.cache = {}
        self.cache_timeout = 30  # segundos

    def get_qqq_data(self) -> Dict:
        """Obtém dados reais do QQQ para análise de trading"""
        try:
            logger.info("Buscando dados reais do QQQ...")

            # Obter dados do QQQ
            ticker = yf.Ticker("QQQ")

            # Dados básicos
            info = ticker.info
            history = ticker.history(period="5d", interval="5m")

            if history.empty:
                return {"error": "No data available"}

            current_price = info.get('regularMarketPrice', history['Close'].iloc[-1])
            previous_close = info.get('previousClose', current_price)

            # Dados intraday
            intraday = history.tail(50)  # Últimas 50 candles de 5 minutos

            # Calcular VWAP e Bandas de Bollinger simplificadas
            typical_price = (intraday['High'] + intraday['Low'] + intraday['Close']) / 3
            vwap = (typical_price * intraday['Volume']).sum() / intraday['Volume'].sum()

            # Bandas de Bollinger (20 períodos, 2 desvios)
            bb_period = 20
            bb_std = 2
            rolling_mean = intraday['Close'].rolling(window=bb_period).mean()
            rolling_std = intraday['Close'].rolling(window=bb_period).std()
            bb_upper = rolling_mean.iloc[-1] + (bb_std * rolling_std.iloc[-1])
            bb_lower = rolling_mean.iloc[-1] - (bb_std * rolling_std.iloc[-1])

            # Obter dados de opções
            try:
                # Obter expirações disponíveis
                expirations = ticker.options
                if expirations:
                    # Usar a próxima expiração
                    next_exp = expirations[0]
                    options_chain = ticker.option_chain(next_exp)
                    calls = options_chain.calls
                    puts = options_chain.puts

                    # Filtrar opções com volume significativo
                    calls = calls[calls['volume'] > 0].sort_values('volume', ascending=False)
                    puts = puts[puts['volume'] > 0].sort_values('volume', ascending=False)
                else:
                    calls = pd.DataFrame()
                    puts = pd.DataFrame()
            except Exception as e:
                logger.warning(f"Erro ao obter dados de opções: {e}")
                calls = pd.DataFrame()
                puts = pd.DataFrame()

            # Calcular Greeks simplificados (simulação baseada no preço)
            delta_call = 0.5 if current_price > 400 else 0.6
            delta_put = -0.5 if current_price > 400 else -0.4
            gamma = 0.01
            charm = 0.001

            # Calcular indicadores técnicos
            rsi = self._calculate_rsi(intraday['Close'])
            macd = self._calculate_macd(intraday['Close'])

            # Análise de volume
            total_volume = intraday['Volume'].sum()
            avg_volume = info.get('averageVolume', total_volume)
            volume_ratio = total_volume / avg_volume if avg_volume > 0 else 1

            response_data = {
                'current_price': current_price,
                'previous_close': previous_close,
                'price_change': current_price - previous_close,
                'price_change_percent': ((current_price - previous_close) / previous_close) * 100,
                'greeks': {
                    'delta': delta_call,
                    'gamma': gamma,
                    'charm': charm,
                    'delta_call': delta_call,
                    'delta_put': delta_put
                },
                'vwap': vwap,
                'bollinger': {
                    'upper': bb_upper,
                    'middle': rolling_mean.iloc[-1],
                    'lower': bb_lower,
                    'position': (current_price - bb_lower) / (bb_upper - bb_lower) if bb_upper != bb_lower else 0.5
                },
                'volume': {
                    'total': total_volume,
                    'average': avg_volume,
                    'ratio': volume_ratio,
                    'call_volume': calls['volume'].sum() if not calls.empty else 0,
                    'put_volume': puts['volume'].sum() if not puts.empty else 0,
                    'put_call_ratio': puts['volume'].sum() / calls['volume'].sum() if calls['volume'].sum() > 0 else 1.0
                },
                'options': {
                    'calls': calls.head(10).to_dict('records') if not calls.empty else [],
                    'puts': puts.head(10).to_dict('records') if not puts.empty else [],
                    'most_active_call': calls.iloc[0].to_dict() if not calls.empty else None,
                    'most_active_put': puts.iloc[0].to_dict() if not puts.empty else None
                },
                'technical_indicators': {
                    'rsi': rsi,
                    'macd': macd,
                    'price_above_vwap': current_price > vwap,
                    'price_above_bb_middle': current_price > rolling_mean.iloc[-1]
                },
                'timestamp': datetime.now().isoformat(),
                'market_status': 'open' if self._is_market_open() else 'closed'
            }

            logger.info(f"Dados QQQ obtidos com sucesso: preço=${current_price:.2f}")
            return response_data

        except Exception as e:
            logger.error(f"Erro ao obter dados do QQQ: {e}")
            return {"error": str(e)}

    def _calculate_rsi(self, prices, period=14):
        """Calcula RSI"""
        if len(prices) < period:
            return 50

        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50

    def _calculate_macd(self, prices, fast=12, slow=26, signal=9):
        """Calcula MACD"""
        if len(prices) < slow:
            return {'macd': 0, 'signal': 0, 'histogram': 0}

        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        histogram = macd_line - signal_line

        return {
            'macd': macd_line.iloc[-1] if not pd.isna(macd_line.iloc[-1]) else 0,
            'signal': signal_line.iloc[-1] if not pd.isna(signal_line.iloc[-1]) else 0,
            'histogram': histogram.iloc[-1] if not pd.isna(histogram.iloc[-1]) else 0
        }

    def _is_market_open(self):
        """Verifica se o mercado está aberto"""
        now = datetime.now()
        # Mercado abre às 9:30 e fecha às 16:00, horário de Nova York
        market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)

        # Considerar timezone
        from pytz import timezone
        ny_time = now.astimezone(timezone('America/New_York'))

        # Verificar se é dia de semana (segunda-sexta)
        if ny_time.weekday() > 4:  # Sábado=5, Domingo=6
            return False

        # Verificar horário de mercado
        return market_open <= ny_time <= market_close

# Instanciar API
ez_api = EzOptionsAPI()

@app.route('/api/qqq/data', methods=['GET'])
def get_qqq_data():
    """Endpoint para obter dados do QQQ"""
    try:
        data = ez_api.get_qqq_data()
        return jsonify(data)
    except Exception as e:
        logger.error(f"Erro no endpoint /api/qqq/data: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/status', methods=['GET'])
def get_status():
    """Endpoint para verificar status da API"""
    return jsonify({
        "status": "running",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0"
    })

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "healthy"}), 200

def run_api_server(port=8504):
    """Inicia o servidor da API"""
    logger.info(f"Iniciando API EzOptions na porta {port}")
    app.run(host='localhost', port=port, debug=False, threaded=True)

if __name__ == '__main__':
    run_api_server()