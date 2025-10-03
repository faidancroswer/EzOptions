"""
VWAP and Bollinger Bands Indicators for ezOptions
Enhances the ezOptions chart with additional technical indicators
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Tuple, Optional

logger = logging.getLogger(__name__)

class VWAPBollingerIndicators:
    """
    Classe para calcular e plotar VWAP e Bollinger Bands
    """

    def __init__(self):
        self.vwap_data = {}
        self.bb_data = {}

    def calculate_vwap(self, df: pd.DataFrame) -> pd.Series:
        """
        Calcula VWAP (Volume Weighted Average Price)
        """
        try:
            if df.empty or 'Volume' not in df.columns:
                return pd.Series(index=df.index, dtype=float)

            # Typical Price = (High + Low + Close) / 3
            typical_price = (df['High'] + df['Low'] + df['Close']) / 3

            # VWAP = Cumulative(Typical Price * Volume) / Cumulative(Volume)
            cumulative_pv = (typical_price * df['Volume']).cumsum()
            cumulative_volume = df['Volume'].cumsum()

            # Evitar divisão por zero
            vwap = cumulative_pv / cumulative_volume.replace(0, np.nan)

            return vwap.fillna(method='ffill')

        except Exception as e:
            logger.error(f"Erro ao calcular VWAP: {e}")
            return pd.Series(index=df.index, dtype=float)

    def calculate_bollinger_bands(self, df: pd.DataFrame, period: int = 20, std_dev: float = 2.0) -> Dict:
        """
        Calcula Bollinger Bands
        """
        try:
            if df.empty or 'Close' not in df.columns:
                return {'upper': pd.Series(dtype=float), 'middle': pd.Series(dtype=float), 'lower': pd.Series(dtype=float)}

            # Média móvel simples
            sma = df['Close'].rolling(window=period).mean()

            # Desvio padrão
            std = df['Close'].rolling(window=period).std()

            # Bandas
            upper_band = sma + (std * std_dev)
            lower_band = sma - (std * std_dev)

            return {
                'upper': upper_band,
                'middle': sma,
                'lower': lower_band,
                'squeeze': self._detect_squeeze(upper_band, lower_band, sma)
            }

        except Exception as e:
            logger.error(f"Erro ao calcular Bollinger Bands: {e}")
            return {'upper': pd.Series(dtype=float), 'middle': pd.Series(dtype=float), 'lower': pd.Series(dtype=float)}

    def _detect_squeeze(self, upper: pd.Series, lower: pd.Series, middle: pd.Series) -> pd.Series:
        """
        Detecta squeeze nas Bollinger Bands
        """
        try:
            # Calcular largura das bandas
            band_width = (upper - lower) / middle

            # Squeeze quando a largura está abaixo da média histórica
            avg_width = band_width.rolling(window=50).mean()
            squeeze = band_width < avg_width * 0.8  # 20% abaixo da média

            return squeeze

        except Exception as e:
            logger.error(f"Erro ao detectar squeeze: {e}")
            return pd.Series(dtype=bool)

    def fetch_intraday_data(self, ticker: str, period: str = "1d", interval: str = "1m") -> pd.DataFrame:
        """
        Busca dados intraday para cálculo dos indicadores
        """
        try:
            stock = yf.Ticker(ticker)
            data = stock.history(period=period, interval=interval)

            if data.empty:
                logger.warning(f"Nenhum dado encontrado para {ticker}")
                return pd.DataFrame()

            return data

        except Exception as e:
            logger.error(f"Erro ao buscar dados intraday para {ticker}: {e}")
            return pd.DataFrame()

    def calculate_all_indicators(self, ticker: str) -> Dict:
        """
        Calcula todos os indicadores para um ticker
        """
        try:
            # Buscar dados intraday
            df = self.fetch_intraday_data(ticker)

            if df.empty:
                return {}

            # Calcular VWAP
            vwap = self.calculate_vwap(df)

            # Calcular Bollinger Bands
            bb = self.calculate_bollinger_bands(df)

            # Dados atuais
            current_price = df['Close'].iloc[-1] if not df.empty else 0
            current_vwap = vwap.iloc[-1] if not vwap.empty else 0

            result = {
                'timestamp': datetime.now(),
                'ticker': ticker,
                'current_price': current_price,
                'vwap': {
                    'current': current_vwap,
                    'series': vwap,
                },
                'bollinger_bands': {
                    'upper': bb['upper'],
                    'middle': bb['middle'],
                    'lower': bb['lower'],
                    'squeeze': bb.get('squeeze', pd.Series(dtype=bool)),
                },
                'raw_data': df
            }

            return result

        except Exception as e:
            logger.error(f"Erro ao calcular indicadores para {ticker}: {e}")
            return {}

def initialize_technical_indicators():
    """Initialize technical indicators system"""
    try:
        global indicators
        indicators = VWAPBollingerIndicators()
        logger.info("Technical indicators initialized successfully")
        print("[OK] Technical indicators initialized")
        return True
    except Exception as e:
        logger.error(f"Error initializing technical indicators: {e}")
        print(f"[ERROR] Error initializing technical indicators: {e}")
        return False


if __name__ == "__main__":
    # Test the VWAP and Bollinger Bands indicators
    indicators = VWAPBollingerIndicators()

    # Test with QQQ data
    result = indicators.calculate_all_indicators("QQQ")

    if result:
        print("VWAP and Bollinger Bands calculated successfully")
        print(f"Current VWAP: {result.get('vwap', {}).get('current', 'N/A')}")
        print(f"Current Price: {result.get('current_price', 'N/A')}")
    else:
        print("No indicator data returned - check market hours")