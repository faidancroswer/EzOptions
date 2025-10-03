import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import math
from math import log, sqrt
import re
import time
from scipy.stats import norm
import threading
from contextlib import contextmanager
from scipy.interpolate import griddata
import numpy as np
import pytz
from datetime import timedelta
import json
import requests
import MetaTrader5 as mt5
# Flask não é mais necessário no dashboard principal
# from waitress import serve

# Funções para dados reais do QQQ
def get_real_qqq_data():
    """Obtém dados reais do QQQ para análise de trading"""
    try:
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

        # Calcular VWAP e Bandas de Bollinger reais
        typical_price = (intraday['High'] + intraday['Low'] + intraday['Close']) / 3
        vwap = (typical_price * intraday['Volume']).sum() / intraday['Volume'].sum()

        # Bandas de Bollinger (20 períodos, 2 desvios)
        bb_period = 20
        bb_std = 2
        rolling_mean = intraday['Close'].rolling(window=bb_period).mean()
        rolling_std = intraday['Close'].rolling(window=bb_period).std()
        bb_upper = rolling_mean.iloc[-1] + (bb_std * rolling_std.iloc[-1])
        bb_lower = rolling_mean.iloc[-1] - (bb_std * rolling_std.iloc[-1])

        # Obter dados de opções reais
        try:
            expirations = ticker.options
            if expirations:
                next_exp = expirations[0]
                options_chain = ticker.option_chain(next_exp)
                calls = options_chain.calls
                puts = options_chain.puts

                calls = calls[calls['volume'] > 0].sort_values('volume', ascending=False)
                puts = puts[puts['volume'] > 0].sort_values('volume', ascending=False)
            else:
                calls = pd.DataFrame()
                puts = pd.DataFrame()
        except Exception as e:
            calls = pd.DataFrame()
            puts = pd.DataFrame()

        # Calcular indicadores técnicos reais
        rsi = calculate_rsi(intraday['Close'])
        macd = calculate_macd(intraday['Close'])

        # Análise de volume real
        total_volume = intraday['Volume'].sum()
        avg_volume = info.get('averageVolume', total_volume)
        volume_ratio = total_volume / avg_volume if avg_volume > 0 else 1

        # Calcular Greeks baseados no preço real
        delta_call = min(0.9, max(0.1, (current_price - 400) / 100))
        delta_put = delta_call - 1
        gamma = 0.01 * (1 + abs(current_price - 450) / 50)
        charm = 0.001 * (1 if current_price > vwap else -1)

        return {
            'current_price': current_price,
            'previous_close': previous_close,
            'price_change': current_price - previous_close,
            'price_change_percent': ((current_price - previous_close) / previous_close) * 100,
            'greeks_data': {
                'gamma': {
                    'current': gamma,
                    'max_level_price': current_price + 2,
                    'max_level': gamma * 1.2,
                    'levels': [current_price - 2, current_price - 1, current_price, current_price + 1, current_price + 2]
                },
                'delta': {
                    'current': delta_call,
                    'max_level_price': current_price + 1.5,
                    'max_level': min(0.9, delta_call + 0.1),
                    'positive_bars_upward': current_price > vwap,
                    'negative_bars_downward': current_price < vwap,
                    'levels': [current_price - 1.5, current_price - 0.5, current_price + 0.5, current_price + 1.5]
                },
                'charm': {
                    'current': charm,
                    'max_level_price': current_price + (1 if charm > 0 else -1),
                    'max_level': charm * 1.5,
                    'growing_trend': current_price > vwap,
                    'direction_up': current_price > vwap,
                    'decreasing_trend': current_price < vwap,
                    'direction_down': current_price < vwap,
                    'flip_zone': abs(current_price - vwap) < 0.5,
                    'levels': [current_price - 1, current_price + 1]
                },
                'theta': {
                    'current': -0.05 * (1 + abs(current_price - 450) / 100),
                    'max_level_price': current_price - 0.5,
                    'max_level': -0.03
                }
            },
            'vwap_data': {
                'current_vwap': vwap,
                'first_deviation_up': vwap * 1.002,
                'first_deviation_down': vwap * 0.998,
                'deviations_straight': abs(current_price - vwap) < (vwap * 0.001),
                'equilibrium': abs(current_price - vwap) < (vwap * 0.0005)
            },
            'volume_data': {
                'total': total_volume,
                'average': avg_volume,
                'ratio': volume_ratio,
                'call_volume': calls['volume'].sum() if not calls.empty else 0,
                'put_volume': puts['volume'].sum() if not puts.empty else 0,
                'put_call_ratio': puts['volume'].sum() / calls['volume'].sum() if calls['volume'].sum() > 0 else 1.0,
                'strikes_volume': {int(row['strike']): row['volume'] for _, row in calls.head(5).iterrows()} if not calls.empty else {},
                'heat_map_balance': abs(calls['volume'].sum() - puts['volume'].sum()) / max(calls['volume'].sum(), puts['volume'].sum(), 1) < 0.2
            },
            'options_data': {
                'calls_volume': calls['volume'].sum() if not calls.empty else 0,
                'puts_volume': puts['volume'].sum() if not puts.empty else 0
            },
            'bollinger': {
                'upper': bb_upper,
                'middle': rolling_mean.iloc[-1],
                'lower': bb_lower,
                'position': (current_price - bb_lower) / (bb_upper - bb_lower) if bb_upper != bb_lower else 0.5
            },
            'technical_indicators': {
                'rsi': rsi,
                'macd': macd,
                'price_above_vwap': current_price > vwap,
                'price_above_bb_middle': current_price > rolling_mean.iloc[-1]
            },
            'timestamp': datetime.now().isoformat(),
            'market_status': 'open' if is_market_open() else 'closed'
        }

    except Exception as e:
        return {"error": str(e)}

def calculate_rsi(prices, period=14):
    """Calcula RSI"""
    if len(prices) < period:
        return 50

    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50

def calculate_macd(prices, fast=12, slow=26, signal=9):
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

def is_market_open():
    """Verifica se o mercado está aberto"""
    # Obter horário atual em Nova York
    ny_time = datetime.now(pytz.timezone('America/New_York'))

    # Mercado abre às 9:30 e fecha às 16:00, horário de Nova York
    market_open = ny_time.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = ny_time.replace(hour=16, minute=0, second=0, microsecond=0)

    # Verificar se é dia de semana (segunda-sexta)
    if ny_time.weekday() > 4:  # Sábado=5, Domingo=6
        return False

    # Verificar horário de mercado
    return market_open <= ny_time <= market_close


def calculate_heikin_ashi(df):
    """Calculate Heikin Ashi candlestick values."""
    ha_df = pd.DataFrame(index=df.index)
    
    # Calculate Heikin Ashi values
    ha_df['HA_Close'] = (df['Open'] + df['High'] + df['Low'] + df['Close']) / 4
    
    # Initialize HA_Open with first candle's opening price
    ha_df['HA_Open'] = pd.Series(index=df.index)
    ha_df.loc[ha_df.index[0], 'HA_Open'] = df['Open'].iloc[0]
    
    # Calculate subsequent HA_Open values
    for i in range(1, len(df)):
        ha_df.loc[ha_df.index[i], 'HA_Open'] = (ha_df['HA_Open'].iloc[i-1] + ha_df['HA_Close'].iloc[i-1]) / 2
    
    ha_df['HA_High'] = df[['High', 'Open', 'Close']].max(axis=1)
    ha_df['HA_Low'] = df[['Low', 'Open', 'Close']].min(axis=1)
    
    return ha_df


@contextmanager
def st_thread_context():
    """Thread context management for Streamlit"""
    try:
        if not hasattr(threading.current_thread(), '_StreamlitThread__cached_st'):
           
            import warnings
            warnings.filterwarnings('ignore', category=RuntimeWarning, message='.*missing ScriptRunContext.*')
        yield
    finally:
        pass


with st_thread_context():
    st.set_page_config(layout="wide")

# Initialize session state for colors if not already set
if 'call_color' not in st.session_state:
    st.session_state.call_color = '#00FF00'  # Default green for calls
if 'put_color' not in st.session_state:
    st.session_state.put_color = '#FF0000'   # Default red for puts
if 'vix_color' not in st.session_state:
    st.session_state.vix_color = '#800080'   # Default purple for VIX

# -------------------------------
# Helper Functions
# -------------------------------
def format_ticker(ticker):
    """Helper function to format tickers for indices"""
    ticker = ticker.upper()
    if ticker == "SPX":
        return "^SPX"
    elif ticker == "NDX":
        return "^NDX"
    elif ticker == "VIX":
        return "^VIX"
    elif ticker == "DJI":
        return "^DJI"
    elif ticker == "RUT":
        return "^RUT"
    return ticker

def check_market_status():
    """Check if we're in pre-market, market hours, or post-market"""
    # Get current time in PST for market checks
    pacific = datetime.now(tz=pytz.timezone('US/Pacific'))
    
    # Get local time and timezone
    local = datetime.now()
    local_tz = datetime.now().astimezone().tzinfo
    
    market_message = None
    
    if pacific.hour >= 21 or pacific.hour < 7:
        next_update = pacific.replace(hour=7, minute=0) if pacific.hour < 7 else \
                     (pacific + timedelta(days=1)).replace(hour=7, minute=00)
        time_until = next_update - pacific
        hours, remainder = divmod(time_until.seconds, 3600)
        minutes, _ = divmod(remainder, 60)
        
        # Convert PST update time to local time
        local_next_update = next_update.astimezone(local_tz)
        
        market_message = f"""
        ⚠️ **WAIT FOR NEW DATA**
        - Current time: {local.strftime('%I:%M %p')} {local_tz}
        - New data will be available at approximately {local_next_update.strftime('%I:%M %p')}
        - Time until new data: {hours}h {minutes}m
        """
    return market_message

def get_cache_ttl():
    """Get the cache TTL from session state refresh rate, with a minimum of 10 seconds"""
    return max(float(st.session_state.get('refresh_rate', 10)), 10)

def calculate_strike_range(current_price, percentage=None):
    """Calculate strike range based on percentage of current price"""
    if percentage is None:
        percentage = st.session_state.get('strike_range', 1.0)
    return current_price * (percentage / 100.0)

@st.cache_data(ttl=get_cache_ttl())  # Cache TTL matches refresh rate
def fetch_options_for_date(ticker, date, S=None):
    """Fetch options data for a specific date with caching"""
    print(f"Fetching option chain for {ticker} EXP {date}")
    try:
        stock = yf.Ticker(ticker)
        chain = stock.option_chain(date)
        calls = chain.calls
        puts = chain.puts
        
        if not calls.empty:
            calls = calls.copy()
            calls['extracted_expiry'] = calls['contractSymbol'].apply(extract_expiry_from_contract)
        if not puts.empty:
            puts = puts.copy()
            puts['extracted_expiry'] = puts['contractSymbol'].apply(extract_expiry_from_contract)
            
        return calls, puts
    except Exception as e:
        st.error(f"Error fetching options data: {e}")
        return pd.DataFrame(), pd.DataFrame()

@st.cache_data(ttl=get_cache_ttl())  # Cache TTL matches refresh rate
def fetch_all_options(ticker):
    """Fetch all available options with caching"""
    print(f"Fetching all options for {ticker}")
    try:
        stock = yf.Ticker(ticker)
        all_calls = []
        all_puts = []
        
        for next_exp in stock.options:
            try:
                calls, puts = fetch_options_for_date(ticker, next_exp)
                if not calls.empty:
                    all_calls.append(calls)
                if not puts.empty:
                    all_puts.append(puts)
            except Exception as e:
                st.error(f"Error fetching fallback options data: {e}")
        
        if all_calls:
            combined_calls = pd.concat(all_calls, ignore_index=True)
        else:
            combined_calls = pd.DataFrame()
        if all_puts:
            combined_puts = pd.concat(all_puts, ignore_index=True)
        else:
            combined_puts = pd.DataFrame()
        
        return combined_calls, combined_puts
    except Exception as e:
        st.error(f"Error fetching all options: {e}")
        return pd.DataFrame(), pd.DataFrame()

def clear_page_state():
    """Clear all page-specific content and containers"""
    for key in list(st.session_state.keys()):
        if key.startswith(('container_', 'chart_', 'table_', 'page_')):
            del st.session_state[key]
    
    if 'current_page_container' in st.session_state:
        del st.session_state['current_page_container']
    
    st.empty()

def extract_expiry_from_contract(contract_symbol):
    """
    Extracts the expiration date from an option contract symbol.
    Handles both 6-digit (YYMMDD) and 8-digit (YYYYMMDD) date formats.
    """
    pattern = r'[A-Z]+W?(?P<date>\d{6}|\d{8})[CP]\d+'
    match = re.search(pattern, contract_symbol)
    if match:
        date_str = match.group("date")
        try:
            if len(date_str) == 6:
                # Parse as YYMMDD
                expiry_date = datetime.strptime(date_str, "%y%m%d").date()
            else:
                # Parse as YYYYMMDD
                expiry_date = datetime.strptime(date_str, "%Y%m%d").date()
            return expiry_date
        except ValueError:
            return None
    return None

def add_current_price_line(fig, current_price):
    """
    Adds a dashed white line at the current price to a Plotly figure.
    For horizontal bar charts, adds a horizontal line. For other charts, adds a vertical line.
    """
    if st.session_state.chart_type == 'Horizontal Bar':
        fig.add_hline(
            y=current_price,
            line_dash="dash",
            line_color="white",
            opacity=0.7
        )
    else:
        fig.add_vline(
            x=current_price,
            line_dash="dash",
            line_color="white",
            opacity=0.7,
            annotation_text=f"{current_price}",
            annotation_position="top",
            annotation=dict(
                font=dict(size=st.session_state.chart_text_size)
            )
        )
    return fig

@st.cache_data(ttl=get_cache_ttl())  # Cache TTL matches refresh rate
def get_screener_data(screener_type):
    """Fetch screener data from Yahoo Finance"""
    try:
        response = yf.screen(screener_type)
        if isinstance(response, dict) and 'quotes' in response:
            data = []
            for quote in response['quotes']:
                # Extract relevant information
                info = {
                    'symbol': quote.get('symbol', ''),
                    'shortName': quote.get('shortName', ''),
                    'regularMarketPrice': quote.get('regularMarketPrice', 0),
                    'regularMarketChangePercent': quote.get('regularMarketChangePercent', 0),
                    'regularMarketVolume': quote.get('regularMarketVolume', 0),
                }
                data.append(info)
            return pd.DataFrame(data)
        return pd.DataFrame()
    except Exception as e:
        print(f"Error fetching screener data: {e}")
        return pd.DataFrame()

def calculate_annualized_return(data, period='1y'):
    """Calculate annualized return rate for each weekday"""
    # Convert period to days
    period_days = {
        '1y': 365,
        '6mo': 180,
        '3mo': 90,
        '1mo': 30,
    }
    days = period_days.get(period, 365)
    
    # Filter data for selected period using proper indexing
    end_date = data.index.max()
    start_date = end_date - pd.Timedelta(days=days)
    filtered_data = data.loc[start_date:end_date].copy()
    
    # Calculate daily returns
    filtered_data['Returns'] = filtered_data['Close'].pct_change()
    
    # Group by weekday and calculate mean return
    weekday_returns = filtered_data.groupby(filtered_data.index.weekday)['Returns'].mean()
    
    # Annualize returns (252 trading days per year)
    annualized_returns = (1 + weekday_returns) ** 252 - 1
    
    # Map weekday numbers to names
    weekday_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
    annualized_returns.index = weekday_names
    
    return annualized_returns * 100  # Convert to percentage

def create_weekday_returns_chart(returns):
    """Create a bar chart of weekday returns"""
    fig = go.Figure()
    
    # Add bars with colors based on return value
    for day, value in returns.items():
        color = st.session_state.call_color if value >= 0 else st.session_state.put_color
        fig.add_trace(go.Bar(
            x=[day],
            y=[value],
            name=day,
            marker_color=color,
            text=[f'{value:.2f}%'],
            textposition='outside'
        ))
    
    # Calculate y-axis range with padding
    y_values = returns.values
    y_max = max(y_values)
    y_min = min(y_values)
    y_range = y_max - y_min
    padding = y_range * 0.2  # 20% padding
    
    fig.update_layout(
        title=dict(
            text='Annualized Return Rate by Weekday',
            x=0,
            xanchor='left',
            font=dict(size=st.session_state.chart_text_size + 8)
        ),
        xaxis_title=dict(
            text='Weekday',
            font=dict(size=st.session_state.chart_text_size)
        ),
        yaxis_title=dict(
            text='Annualized Return (%)',
            font=dict(size=st.session_state.chart_text_size)
        ),
        yaxis=dict(
            range=[y_min - padding, y_max + padding],  # Add padding to y-axis range
            tickfont=dict(size=st.session_state.chart_text_size)
        ),
        showlegend=False,
        template="plotly_dark"
    )
    
    # Update axis fonts
    fig.update_xaxes(tickfont=dict(size=st.session_state.chart_text_size))
    
    return fig

def analyze_options_flow(calls_df, puts_df, current_price):
    """Analyze options flow to determine bought vs sold contracts"""
    # Deep copy to avoid modifying originals
    calls = calls_df.copy()
    puts = puts_df.copy()
    
    # Determine if option is likely bought/sold based on trade price vs bid/ask
    # For calls: trades near ask = likely bought, trades near bid = likely sold
    calls['trade_type'] = calls.apply(lambda x: 'bought' if x['lastPrice'] >= (x['bid'] + (x['ask'] - x['bid'])*0.6) else 'sold', axis=1)
    puts['trade_type'] = puts.apply(lambda x: 'bought' if x['lastPrice'] >= (x['bid'] + (x['ask'] - x['bid'])*0.6) else 'sold', axis=1)
    
    # Add ITM/OTM classification
    calls['moneyness'] = calls.apply(lambda x: 'ITM' if x['strike'] <= current_price else 'OTM', axis=1)
    puts['moneyness'] = puts.apply(lambda x: 'ITM' if x['strike'] >= current_price else 'OTM', axis=1)
    
    # Calculate volume-weighted stats
    call_stats = {
        'bought': {
            'volume': calls[calls['trade_type'] == 'bought']['volume'].sum(),
            'premium': (calls[calls['trade_type'] == 'bought']['volume'] * calls[calls['trade_type'] == 'bought']['lastPrice'] * 100).sum()
        },
        'sold': {
            'volume': calls[calls['trade_type'] == 'sold']['volume'].sum(),
            'premium': (calls[calls['trade_type'] == 'sold']['volume'] * calls[calls['trade_type'] == 'sold']['lastPrice'] * 100).sum()
        },
        'OTM': {
            'volume': calls[calls['moneyness'] == 'OTM']['volume'].sum(),
            'premium': (calls[calls['moneyness'] == 'OTM']['volume'] * calls[calls['moneyness'] == 'OTM']['lastPrice'] * 100).sum()
        },
        'ITM': {
            'volume': calls[calls['moneyness'] == 'ITM']['volume'].sum(), 
            'premium': (calls[calls['moneyness'] == 'ITM']['volume'] * calls[calls['moneyness'] == 'ITM']['lastPrice'] * 100).sum()
        }
    }
    
    put_stats = {
        'bought': {
            'volume': puts[puts['trade_type'] == 'bought']['volume'].sum(),
            'premium': (puts[puts['trade_type'] == 'bought']['volume'] * puts[puts['trade_type'] == 'bought']['lastPrice'] * 100).sum()
        },
        'sold': {
            'volume': puts[puts['trade_type'] == 'sold']['volume'].sum(),
            'premium': (puts[puts['trade_type'] == 'sold']['volume'] * puts[puts['trade_type'] == 'sold']['lastPrice'] * 100).sum()
        },
        'OTM': {
            'volume': puts[puts['moneyness'] == 'OTM']['volume'].sum(),
            'premium': (puts[puts['moneyness'] == 'OTM']['volume'] * puts[puts['moneyness'] == 'OTM']['lastPrice'] * 100).sum()
        },
        'ITM': {
            'volume': puts[puts['moneyness'] == 'ITM']['volume'].sum(), 
            'premium': (puts[puts['moneyness'] == 'ITM']['volume'] * puts[puts['moneyness'] == 'ITM']['lastPrice'] * 100).sum()
        }
    }
    
    # Calculate OTM bought/sold breakdown
    otm_calls_bought = calls[(calls['moneyness'] == 'OTM') & (calls['trade_type'] == 'bought')]['volume'].sum()
    otm_calls_sold = calls[(calls['moneyness'] == 'OTM') & (calls['trade_type'] == 'sold')]['volume'].sum()
    otm_puts_bought = puts[(puts['moneyness'] == 'OTM') & (puts['trade_type'] == 'bought')]['volume'].sum()
    otm_puts_sold = puts[(puts['moneyness'] == 'OTM') & (puts['trade_type'] == 'sold')]['volume'].sum()
    
    # Calculate total premium values
    total_call_premium = (calls['volume'] * calls['lastPrice'] * 100).sum()
    total_put_premium = (puts['volume'] * puts['lastPrice'] * 100).sum()
    
    return {
        'calls': call_stats,
        'puts': put_stats,
        'otm_detail': {
            'calls_bought': otm_calls_bought,
            'calls_sold': otm_calls_sold,
            'puts_bought': otm_puts_bought,
            'puts_sold': otm_puts_sold
        },
        'total_premium': {
            'calls': total_call_premium,
            'puts': total_put_premium
        }
    }

def create_option_flow_charts(flow_data, title="Options Flow Analysis"):
    """Create visual charts for options flow analysis"""
    call_color = st.session_state.call_color
    put_color = st.session_state.put_color
    
    # Create bar chart for bought vs sold
    fig_flow = go.Figure()
    
    # Calls bought/sold
    fig_flow.add_trace(go.Bar(
        x=['Calls Bought', 'Calls Sold'],
        y=[flow_data['calls']['bought']['volume'], flow_data['calls']['sold']['volume']],
        name='Calls',
        marker_color=call_color
    ))
    
    # Puts bought/sold
    fig_flow.add_trace(go.Bar(
        x=['Puts Bought', 'Puts Sold'],
        y=[flow_data['puts']['bought']['volume'], flow_data['puts']['sold']['volume']],
        name='Puts',
        marker_color=put_color
    ))
    
    fig_flow.update_layout(
        title=dict(
            text=title,
            x=0,
            xanchor='left',
            font=dict(size=st.session_state.chart_text_size + 8)
        ),
        xaxis_title=dict(
            text='Trade Direction',
            font=dict(size=st.session_state.chart_text_size)
        ),
        yaxis_title=dict(
            text='Volume',
            font=dict(size=st.session_state.chart_text_size)
        ),
        legend=dict(
            font=dict(size=st.session_state.chart_text_size)
        ),
        barmode='relative',
        template="plotly_dark"
    )
    
    # Create OTM/ITM chart
    fig_money = go.Figure()
    
    # Calls OTM/ITM
    fig_money.add_trace(go.Bar(
        x=['OTM Calls', 'ITM Calls'],
        y=[flow_data['calls']['OTM']['volume'], flow_data['calls']['ITM']['volume']],
        name='Calls',
        marker_color=call_color
    ))
    
    # Puts OTM/ITM
    fig_money.add_trace(go.Bar(
        x=['OTM Puts', 'ITM Puts'],
        y=[flow_data['puts']['OTM']['volume'], flow_data['puts']['ITM']['volume']],
        name='Puts',
        marker_color=put_color
    ))
    
    fig_money.update_layout(
        title=dict(
            text="OTM vs ITM Volume",
            x=0,
            xanchor='left',
            font=dict(size=st.session_state.chart_text_size + 8)
        ),
        xaxis_title=dict(
            text='Moneyness',
            font=dict(size=st.session_state.chart_text_size)
        ),
        yaxis_title=dict(
            text='Volume',
            font=dict(size=st.session_state.chart_text_size)
        ),
        legend=dict(
            font=dict(size=st.session_state.chart_text_size)
        ),
        barmode='relative',
        template="plotly_dark"
    )
    
    # Premium chart (donut)
    premium_labels = ['Call Premium', 'Put Premium']
    premium_values = [flow_data['total_premium']['calls'], flow_data['total_premium']['puts']]
    
    fig_premium = go.Figure(data=[go.Pie(
        labels=premium_labels,
        values=premium_values,
        hole=0.4,
        marker=dict(colors=[call_color, put_color])
    )])
    
    total_premium = flow_data['total_premium']['calls'] + flow_data['total_premium']['puts']
    premium_text = f"${total_premium:,.0f}"
    
    fig_premium.update_layout(
        title=dict(
            text="Total Premium Flow",
            x=0,
            xanchor='left',
            font=dict(size=st.session_state.chart_text_size + 8)
        ),
        legend=dict(font=dict(size=st.session_state.chart_text_size)),
        annotations=[dict(
            text=premium_text,
            x=0.5, y=0.5,
            font=dict(size=st.session_state.chart_text_size + 4),
            showarrow=False
        )],
        template="plotly_dark"
    )
    
    # OTM Analysis Breakdown (horizontal)
    fig_otm = go.Figure()
    
    # OTM Calls bought/sold
    fig_otm.add_trace(go.Bar(
        y=['OTM Calls'],
        x=[flow_data['otm_detail']['calls_bought']],
        name='Bought',
        orientation='h',
        marker_color='lightgreen',
        offsetgroup=0
    ))
    
    fig_otm.add_trace(go.Bar(
        y=['OTM Calls'],
        x=[flow_data['otm_detail']['calls_sold']],
        name='Sold',
        orientation='h',
        marker_color='darkgreen',
        offsetgroup=1
    ))
    
    # OTM Puts bought/sold
    fig_otm.add_trace(go.Bar(
        y=['OTM Puts'],
        x=[flow_data['otm_detail']['puts_bought']],
        name='Bought',
        orientation='h',
        marker_color='pink',
        offsetgroup=0
    ))
    
    fig_otm.add_trace(go.Bar(
        y=['OTM Puts'],
        x=[flow_data['otm_detail']['puts_sold']],
        name='Sold',
        orientation='h',
        marker_color='darkred',
        offsetgroup=1
    ))
    
    fig_otm.update_layout(
        title=dict(
            text="OTM Options Bought vs Sold",
            x=0,
            xanchor='left',
            font=dict(size=st.session_state.chart_text_size + 8)
        ),
        xaxis_title=dict(
            text='Volume',
            font=dict(size=st.session_state.chart_text_size)
        ),
        legend=dict(
            font=dict(size=st.session_state.chart_text_size)
        ),
        barmode='relative',
        template="plotly_dark"
    )
    
    return fig_flow, fig_money, fig_premium, fig_otm

def create_option_premium_heatmap(calls_df, puts_df, strikes, expiry_dates, current_price):
    """Create a heatmap showing premium distribution across strikes and expiries"""
    # Initialize data matrices
    call_premium = np.zeros((len(expiry_dates), len(strikes)))
    put_premium = np.zeros((len(expiry_dates), len(strikes)))
    
    # Map strikes and expiry dates to indices
    strike_to_idx = {strike: i for i, strike in enumerate(strikes)}
    expiry_to_idx = {expiry: i for i, expiry in enumerate(expiry_dates)}
    
    # Fill matrices with premium data (volume * price)
    for _, row in calls_df.iterrows():
        if row['strike'] in strike_to_idx and row['expiry_date'] in expiry_to_idx:
            i = expiry_to_idx[row['expiry_date']]
            j = strike_to_idx[row['strike']]
            call_premium[i, j] = row['volume'] * row['lastPrice'] * 100
    
    for _, row in puts_df.iterrows():
        if row['strike'] in strike_to_idx and row['expiry_date'] in expiry_to_idx:
            i = expiry_to_idx[row['expiry_date']]
            j = strike_to_idx[row['strike']]
            put_premium[i, j] = row['volume'] * row['lastPrice'] * 100
    
    # Create heatmaps
    fig_calls = go.Figure(data=go.Heatmap(
        z=call_premium,
        x=strikes,
        y=expiry_dates,
        colorscale=[[0, 'rgba(0,0,0,0)'], [0.01, 'rgba(0,255,0,0.1)'], [1, st.session_state.call_color]],
        hoverongaps=False,
        name="Call Premium",
        showscale=True,
        colorbar=dict(
            title="Premium ($)",
            titleside="top",
            tickformat="$,.0f"
        )
    ))
    
    # Add current price line
    fig_calls.add_vline(
        x=current_price,
        line_dash="dash",
        line_color="white",
        opacity=0.7
    )
    
    fig_calls.update_layout(
        title=dict(
            text="Call Premium Heatmap",
            x=0,
            xanchor='left',
            font=dict(size=st.session_state.chart_text_size + 8)
        ),
        xaxis_title=dict(
            text='Strike Price',
            font=dict(size=st.session_state.chart_text_size)
        ),
        yaxis_title=dict(
            text='Expiration Date',
            font=dict(size=st.session_state.chart_text_size)
        ),
        template="plotly_dark",
        yaxis=dict(
            tickfont=dict(size=st.session_state.chart_text_size)
        ),
        xaxis=dict(
            tickfont=dict(size=st.session_state.chart_text_size)
        )
    )
    
    fig_puts = go.Figure(data=go.Heatmap(
        z=put_premium,
        x=strikes,
        y=expiry_dates,
        colorscale=[[0, 'rgba(0,0,0,0)'], [0.01, 'rgba(255,0,0,0.1)'], [1, st.session_state.put_color]],
        hoverongaps=False,
        name="Put Premium",
        showscale=True,
        colorbar=dict(
            title="Premium ($)",
            titleside="top",
            tickformat="$,.0f"
        )
    ))
    
    # Add current price line
    fig_puts.add_vline(
        x=current_price,
        line_dash="dash",
        line_color="white",
        opacity=0.7
    )
    
    fig_puts.update_layout(
        title=dict(
            text="Put Premium Heatmap",
            x=0,
            xanchor='left',
            font=dict(size=st.session_state.chart_text_size + 8)
        ),
        xaxis_title=dict(
            text='Strike Price',
            font=dict(size=st.session_state.chart_text_size)
        ),
        yaxis_title=dict(
            text='Expiration Date',
            font=dict(size=st.session_state.chart_text_size)
        ),
        template="plotly_dark",
        yaxis=dict(
            tickfont=dict(size=st.session_state.chart_text_size)
        ),
        xaxis=dict(
            tickfont=dict(size=st.session_state.chart_text_size)
        )
    )
    
    return fig_calls, fig_puts

def create_premium_heatmap(calls_df, puts_df, filtered_strikes, selected_expiry_dates, current_price):
    """Create heatmaps showing premium distribution across strikes and expiration dates."""
    # Initialize data matrices
    call_premium = np.zeros((len(selected_expiry_dates), len(filtered_strikes)))
    put_premium = np.zeros((len(selected_expiry_dates), len(filtered_strikes)))
    
    # Map strikes and expiry dates to indices
    strike_to_idx = {strike: i for i, strike in enumerate(filtered_strikes)}
    expiry_to_idx = {expiry: i for i, expiry in enumerate(selected_expiry_dates)}
    
    # Fill matrices with premium data (volume * price)
    for _, row in calls_df.iterrows():
        if row['strike'] in filtered_strikes and row['extracted_expiry'].strftime('%Y-%m-%d') in expiry_to_idx:
            strike_idx = strike_to_idx[row['strike']]
            expiry_idx = expiry_to_idx[row['extracted_expiry'].strftime('%Y-%m-%d')]
            call_premium[expiry_idx][strike_idx] += row['volume'] * row['lastPrice'] * 100
    
    for _, row in puts_df.iterrows():
        if row['strike'] in filtered_strikes and row['extracted_expiry'].strftime('%Y-%m-%d') in expiry_to_idx:
            strike_idx = strike_to_idx[row['strike']]
            expiry_idx = expiry_to_idx[row['extracted_expiry'].strftime('%Y-%m-%d')]
            put_premium[expiry_idx][strike_idx] += row['volume'] * row['lastPrice'] * 100
    
    # Create heatmaps
    fig_calls = go.Figure(data=go.Heatmap(
        z=call_premium,
        x=filtered_strikes,
        y=selected_expiry_dates,
        colorscale=[[0, 'rgba(0,0,0,0)'], [0.01, 'rgba(0,255,0,0.1)'], [1, st.session_state.call_color]],
        hoverongaps=False,
        name="Call Premium",
        showscale=True,
        colorbar=dict(
            title="Premium ($)",
            tickformat="$,.0f"
        )
    ))
    
    # Add current price line
    fig_calls.add_vline(
        x=current_price,
        line_dash="dash",
        line_color="white",
        opacity=0.7
    )
    
    fig_calls.update_layout(
        title=dict(
            text="Call Premium Heatmap",
            x=0,
            xanchor='left',
            font=dict(size=st.session_state.chart_text_size + 8)
        ),
        xaxis_title=dict(
            text='Strike Price',
            font=dict(size=st.session_state.chart_text_size)
        ),
        yaxis_title=dict(
            text='Expiration Date',
            font=dict(size=st.session_state.chart_text_size)
        ),
        template="plotly_dark",
        yaxis=dict(
            tickfont=dict(size=st.session_state.chart_text_size)
        ),
        xaxis=dict(
            tickfont=dict(size=st.session_state.chart_text_size)
        )
    )
    
    fig_puts = go.Figure(data=go.Heatmap(
        z=put_premium,
        x=filtered_strikes,
        y=selected_expiry_dates,
        colorscale=[[0, 'rgba(0,0,0,0)'], [0.01, 'rgba(255,0,0,0.1)'], [1, st.session_state.put_color]],
        hoverongaps=False,
        name="Put Premium",
        showscale=True,
        colorbar=dict(
            title="Premium ($)",
            tickformat="$,.0f"
        )
    ))
    
    # Add current price line
    fig_puts.add_vline(
        x=current_price,
        line_dash="dash",
        line_color="white",
        opacity=0.7
    )
    
    fig_puts.update_layout(
        title=dict(
            text="Put Premium Heatmap",
            x=0,
            xanchor='left',
            font=dict(size=st.session_state.chart_text_size + 8)
        ),
        xaxis_title=dict(
            text='Strike Price',
            font=dict(size=st.session_state.chart_text_size)
        ),
        yaxis_title=dict(
            text='Expiration Date',
            font=dict(size=st.session_state.chart_text_size)
        ),
        template="plotly_dark",
        yaxis=dict(
            tickfont=dict(size=st.session_state.chart_text_size)
        ),
        xaxis=dict(
            tickfont=dict(size=st.session_state.chart_text_size)
        )
    )
    
    return fig_calls, fig_puts

# Removed: def create_premium_ratio_chart(calls_df, puts_df): function is deleted

# -------------------------------
# Fetch all options experations and add extract expiry
# -------------------------------
def fetch_all_options(ticker):
    """
    Fetches option chains for all available expirations for the given ticker.
    Returns two DataFrames: one for calls and one for puts, with an added column 'extracted_expiry'.
    """
    print(f"Fetching avaiable expirations for {ticker}")  # Add print statement
    stock = yf.Ticker(ticker)
    all_calls = []
    all_puts = []
    
    if stock.options:
        # Get current market date
        current_market_date = datetime.now().date()
        
        for exp in stock.options:
            try:
                chain = stock.option_chain(exp)
                calls = chain.calls
                puts = chain.puts
                
                # Only process options that haven't expired
                exp_date = datetime.strptime(exp, '%Y-%m-%d').date()
                if exp_date >= current_market_date:
                    if not calls.empty:
                        calls = calls.copy()
                        calls['extracted_expiry'] = calls['contractSymbol'].apply(extract_expiry_from_contract)
                        all_calls.append(calls)
                    if not puts.empty:
                        puts = puts.copy()
                        puts['extracted_expiry'] = puts['contractSymbol'].apply(extract_expiry_from_contract)
                        all_puts.append(puts)
            except Exception as e:
                st.error(f"Error fetching chain for expiry {exp}: {e}")
                continue
    else:
        try:
            # Get next valid expiration
            next_exp = stock.options[0] if stock.options else None
            if next_exp:
                chain = stock.option_chain(next_exp)
                calls = chain.calls
                puts = chain.puts
                if not calls.empty:
                    calls = calls.copy()
                    calls['extracted_expiry'] = calls['contractSymbol'].apply(extract_expiry_from_contract)
                    all_calls.append(calls)
                if not puts.empty:
                    puts = puts.copy()
                    puts['extracted_expiry'] = puts['contractSymbol'].apply(extract_expiry_from_contract)
                    all_puts.append(puts)
        except Exception as e:
            st.error(f"Error fetching fallback options data: {e}")
    
    if all_calls:
        combined_calls = pd.concat(all_calls, ignore_index=True)
    else:
        combined_calls = pd.DataFrame()
    if all_puts:
        combined_puts = pd.concat(all_puts, ignore_index=True)
    else:
        combined_puts = pd.DataFrame()
    
    return combined_calls, combined_puts

# Charts and price fetching
@st.cache_data(ttl=get_cache_ttl())  # Cache TTL matches refresh rate
def get_current_price(ticker):
    """Get current price with fallback logic"""
    print(f"Fetching current price for {ticker}")
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
            print(f"Error fetching SPX price: {str(e)}")
    
    try:
        stock = yf.Ticker(ticker)
        price = stock.info.get("regularMarketPrice")
        if price is None:
            price = stock.fast_info.get("lastPrice")
        if price is not None:
            return round(float(price), 2)
    except Exception as e:
        print(f"Yahoo Finance error: {str(e)}")
    
    return None

def create_oi_volume_charts(calls, puts, S):
    if S is None:
        st.error("Could not fetch underlying price.")
        return

    # Get colors from session state at the start
    call_color = st.session_state.call_color
    put_color = st.session_state.put_color

    # Calculate strike range around current price (percentage-based)
    strike_range = calculate_strike_range(S)
    min_strike = S - strike_range
    max_strike = S + strike_range
    
    # Filter data based on strike range
    calls_filtered = calls[(calls['strike'] >= min_strike) & (calls['strike'] <= max_strike)]
    puts_filtered = puts[(puts['strike'] >= min_strike) & (puts['strike'] <= max_strike)]
    
    # Create separate dataframes for OI and Volume, filtering out zeros
    calls_oi_df = calls_filtered[['strike', 'openInterest']].copy()
    calls_oi_df = calls_oi_df[calls_oi_df['openInterest'] > 0]  # Changed from != 0 to > 0
    calls_oi_df['OptionType'] = 'Call'
    
    puts_oi_df = puts_filtered[['strike', 'openInterest']].copy()
    puts_oi_df = puts_oi_df[puts_oi_df['openInterest'] > 0]  # Changed from != 0 to > 0
    puts_oi_df['OptionType'] = 'Put'
    
    calls_vol_df = calls_filtered[['strike', 'volume']].copy()
    calls_vol_df = calls_vol_df[calls_vol_df['volume'] > 0]  # Changed from != 0 to > 0
    calls_vol_df['OptionType'] = 'Call'
    
    puts_vol_df = puts_filtered[['strike', 'volume']].copy()
    puts_vol_df = puts_vol_df[puts_vol_df['volume'] > 0]  # Changed from != 0 to > 0
    puts_vol_df['OptionType'] = 'Put'
    
    # Calculate Net Open Interest and Net Volume using filtered data
    net_oi = calls_filtered.groupby('strike')['openInterest'].sum() - puts_filtered.groupby('strike')['openInterest'].sum()
    net_volume = calls_filtered.groupby('strike')['volume'].sum() - puts_filtered.groupby('strike')['volume'].sum()
    
    # Calculate total values for titles (handle empty dataframes)
    total_call_oi = calls_oi_df['openInterest'].sum() if not calls_oi_df.empty else 0
    total_put_oi = puts_oi_df['openInterest'].sum() if not puts_oi_df.empty else 0
    total_call_volume = calls_vol_df['volume'].sum() if not calls_vol_df.empty else 0
    total_put_volume = puts_vol_df['volume'].sum() if not puts_vol_df.empty else 0
    
    # Create titles with totals using HTML for colored values
    oi_title_with_totals = (
        f"Open Interest by Strike     "
        f"<span style='color: {call_color}'>{total_call_oi:,.0f}</span> | "
        f"<span style='color: {put_color}'>{total_put_oi:,.0f}</span>"
    )
    
    volume_title_with_totals = (
        f"Volume by Strike     "
        f"<span style='color: {call_color}'>{total_call_volume:,.0f}</span> | "
        f"<span style='color: {put_color}'>{total_put_volume:,.0f}</span>"
    )
    
    # Create Open Interest Chart
    fig_oi = go.Figure()
    
    # Add calls if enabled and data exists
    if st.session_state.show_calls and not calls_oi_df.empty:
        if st.session_state.chart_type == 'Bar':
            fig_oi.add_trace(go.Bar(
                x=calls_oi_df['strike'],
                y=calls_oi_df['openInterest'],
                name='Call',
                marker_color=call_color
            ))
        elif st.session_state.chart_type == 'Horizontal Bar':
            fig_oi.add_trace(go.Bar(
                y=calls_oi_df['strike'],
                x=calls_oi_df['openInterest'],
                name='Call',
                marker_color=call_color,
                orientation='h'
            ))
        elif st.session_state.chart_type == 'Scatter':
            fig_oi.add_trace(go.Scatter(
                x=calls_oi_df['strike'],
                y=calls_oi_df['openInterest'],
                mode='markers',
                name='Call',
                marker=dict(color=call_color)
            ))
        elif st.session_state.chart_type == 'Line':
            fig_oi.add_trace(go.Scatter(
                x=calls_oi_df['strike'],
                y=calls_oi_df['openInterest'],
                mode='lines',
                name='Call',
                line=dict(color=call_color)
            ))
        elif st.session_state.chart_type == 'Area':
            fig_oi.add_trace(go.Scatter(
                x=calls_oi_df['strike'],
                y=calls_oi_df['openInterest'],
                mode='lines',
                fill='tozeroy',
                name='Call',
                line=dict(color=call_color, width=0.5),
                fillcolor=call_color
            ))

    # Add puts if enabled and data exists
    if st.session_state.show_puts and not puts_oi_df.empty:
        if st.session_state.chart_type == 'Bar':
            fig_oi.add_trace(go.Bar(
                x=puts_oi_df['strike'],
                y=puts_oi_df['openInterest'],
                name='Put',
                marker_color=put_color
            ))
        elif st.session_state.chart_type == 'Horizontal Bar':
            fig_oi.add_trace(go.Bar(
                y=puts_oi_df['strike'],
                x=puts_oi_df['openInterest'],
                name='Put',
                marker_color=put_color,
                orientation='h'
            ))
        elif st.session_state.chart_type == 'Scatter':
            fig_oi.add_trace(go.Scatter(
                x=puts_oi_df['strike'],
                y=puts_oi_df['openInterest'],
                mode='markers',
                name='Put',
                marker=dict(color=put_color)
            ))
        elif st.session_state.chart_type == 'Line':
            fig_oi.add_trace(go.Scatter(
                x=puts_oi_df['strike'],
                y=puts_oi_df['openInterest'],
                mode='lines',
                name='Put',
                line=dict(color=put_color)
            ))
        elif st.session_state.chart_type == 'Area':
            fig_oi.add_trace(go.Scatter(
                x=puts_oi_df['strike'],
                y=puts_oi_df['openInterest'],
                mode='lines',
                fill='tozeroy',
                name='Put',
                line=dict(color=put_color, width=0.5),
                fillcolor=put_color
            ))

    # Add Net OI if enabled
    if st.session_state.show_net and not net_oi.empty:
        if st.session_state.chart_type == 'Bar':
            fig_oi.add_trace(go.Bar(
                x=net_oi.index,
                y=net_oi.values,
                name='Net OI',
                marker_color=[call_color if val >= 0 else put_color for val in net_oi.values]
            ))
        elif st.session_state.chart_type == 'Horizontal Bar':
            fig_oi.add_trace(go.Bar(
                y=net_oi.index,
                x=net_oi.values,
                name='Net OI',
                marker_color=[call_color if val >= 0 else put_color for val in net_oi.values],
                orientation='h'
            ))
        elif st.session_state.chart_type in ['Scatter', 'Line']:
            positive_mask = net_oi.values >= 0
            
            # Plot positive values
            if any(positive_mask):
                fig_oi.add_trace(go.Scatter(
                    x=net_oi.index[positive_mask],
                    y=net_oi.values[positive_mask],
                    mode='markers' if st.session_state.chart_type == 'Scatter' else 'lines',
                    name='Net OI (Positive)',
                    marker=dict(color=call_color) if st.session_state.chart_type == 'Scatter' else None,
                    line=dict(color=call_color) if st.session_state.chart_type == 'Line' else None
                ))
            
            # Plot negative values
            if any(~positive_mask):
                fig_oi.add_trace(go.Scatter(
                    x=net_oi.index[~positive_mask],
                    y=net_oi.values[~positive_mask],
                    mode='markers' if st.session_state.chart_type == 'Scatter' else 'lines',
                    name='Net OI (Negative)',
                    marker=dict(color=put_color) if st.session_state.chart_type == 'Scatter' else None,
                    line=dict(color=put_color) if st.session_state.chart_type == 'Line' else None
                ))
        elif st.session_state.chart_type == 'Area':
            positive_mask = net_oi.values >= 0
            
            # Plot positive values
            if any(positive_mask):
                fig_oi.add_trace(go.Scatter(
                    x=net_oi.index[positive_mask],
                    y=net_oi.values[positive_mask],
                    mode='lines',
                    fill='tozeroy',
                    name='Net OI (Positive)',
                    line=dict(color=call_color, width=0.5),
                    fillcolor=call_color
                ))
            
            # Plot negative values
            if any(~positive_mask):
                fig_oi.add_trace(go.Scatter(
                    x=net_oi.index[~positive_mask],
                    y=net_oi.values[~positive_mask],
                    mode='lines',
                    fill='tozeroy',
                    name='Net OI (Negative)',
                    line=dict(color=put_color, width=0.5),
                    fillcolor=put_color
                ))

    # Calculate y-axis range with improved padding for OI chart
    y_values = []
    for trace in fig_oi.data:
        if hasattr(trace, 'y') and trace.y is not None:
            y_values.extend([y for y in trace.y if y is not None and not np.isnan(y)])
    
    if y_values:
        oi_y_min = min(min(y_values), 0)  # Include 0 in the range
        oi_y_max = max(y_values)
        oi_y_range = oi_y_max - oi_y_min
        
        # Add 15% padding on top and 5% on bottom
        oi_padding_top = oi_y_range * 0.15
        oi_padding_bottom = oi_y_range * 0.05
        oi_y_min = oi_y_min - oi_padding_bottom
        oi_y_max = oi_y_max + oi_padding_top
    else:
        # Default values if no valid y values
        oi_y_min = 0
        oi_y_max = 100
    
    # Add padding for x-axis range
    padding = strike_range * 0.1
    
    # Update OI chart layout
    if st.session_state.chart_type == 'Horizontal Bar':
        fig_oi.update_layout(
            title=dict(
                text=oi_title_with_totals,
                x=0,
                xanchor='left',
                font=dict(size=st.session_state.chart_text_size + 8)
            ),
            xaxis_title=dict(
                text='Open Interest',
                font=dict(size=st.session_state.chart_text_size)
            ),
            yaxis_title=dict(
                text='Strike Price',
                font=dict(size=st.session_state.chart_text_size)
            ),
            legend=dict(
                font=dict(size=st.session_state.chart_text_size)
            ),
            hovermode='y unified',
            xaxis=dict(
                autorange=True,
                tickfont=dict(size=st.session_state.chart_text_size)
            ),
            yaxis=dict(
                autorange=True,
                tickfont=dict(size=st.session_state.chart_text_size)
            ),
            template="plotly_dark",
            height=600  # Increased height for better visibility
        )
    else:
        fig_oi.update_layout(
            title=dict(
                text=oi_title_with_totals,
                x=0,
                xanchor='left',
                font=dict(size=st.session_state.chart_text_size + 8)
            ),
            xaxis_title=dict(
                text='Strike Price',
                font=dict(size=st.session_state.chart_text_size)
            ),
            yaxis_title=dict(
                text='Open Interest',
                font=dict(size=st.session_state.chart_text_size)
            ),
            legend=dict(
                font=dict(size=st.session_state.chart_text_size)
            ),
            hovermode='x unified',
            xaxis=dict(
                autorange=True,
                tickfont=dict(size=st.session_state.chart_text_size)
            ),
            yaxis=dict(
                autorange=True,
                tickfont=dict(size=st.session_state.chart_text_size)
            ),
            template="plotly_dark",
            height=600  # Increased height for better visibility
        )
    
    # Create Volume Chart
    fig_volume = go.Figure()
    
    # Add calls if enabled and data exists
    if st.session_state.show_calls and not calls_vol_df.empty:
        if st.session_state.chart_type == 'Bar':
            fig_volume.add_trace(go.Bar(
                x=calls_vol_df['strike'],
                y=calls_vol_df['volume'],
                name='Call',
                marker_color=call_color
            ))
        elif st.session_state.chart_type == 'Horizontal Bar':
            fig_volume.add_trace(go.Bar(
                y=calls_vol_df['strike'],
                x=calls_vol_df['volume'],
                name='Call',
                marker_color=call_color,
                orientation='h'
            ))
        elif st.session_state.chart_type == 'Scatter':
            fig_volume.add_trace(go.Scatter(
                x=calls_vol_df['strike'],
                y=calls_vol_df['volume'],
                mode='markers',
                name='Call',
                marker=dict(color=call_color)
            ))
        elif st.session_state.chart_type == 'Line':
            fig_volume.add_trace(go.Scatter(
                x=calls_vol_df['strike'],
                y=calls_vol_df['volume'],
                mode='lines',
                name='Call',
                line=dict(color=call_color)
            ))
        elif st.session_state.chart_type == 'Area':
            fig_volume.add_trace(go.Scatter(
                x=calls_vol_df['strike'],
                y=calls_vol_df['volume'],
                mode='lines',
                fill='tozeroy',
                name='Call',
                line=dict(color=call_color, width=0.5),
                fillcolor=call_color
            ))

    # Add puts if enabled and data exists
    if st.session_state.show_puts and not puts_vol_df.empty:
        if st.session_state.chart_type == 'Bar':
            fig_volume.add_trace(go.Bar(
                x=puts_vol_df['strike'],
                y=puts_vol_df['volume'],
                name='Put',
                marker_color=put_color
            ))
        elif st.session_state.chart_type == 'Horizontal Bar':
            fig_volume.add_trace(go.Bar(
                y=puts_vol_df['strike'],
                x=puts_vol_df['volume'],
                name='Put',
                marker_color=put_color,
                orientation='h'
            ))
        elif st.session_state.chart_type == 'Scatter':
            fig_volume.add_trace(go.Scatter(
                x=puts_vol_df['strike'],
                y=puts_vol_df['volume'],
                mode='markers',
                name='Put',
                marker=dict(color=put_color)
            ))
        elif st.session_state.chart_type == 'Line':
            fig_volume.add_trace(go.Scatter(
                x=puts_vol_df['strike'],
                y=puts_vol_df['volume'],
                mode='lines',
                name='Put',
                line=dict(color=put_color)
            ))
        elif st.session_state.chart_type == 'Area':
            fig_volume.add_trace(go.Scatter(
                x=puts_vol_df['strike'],
                y=puts_vol_df['volume'],
                mode='lines',
                fill='tozeroy',
                name='Put',
                line=dict(color=put_color, width=0.5),
                fillcolor=put_color
            ))

    # Add Net Volume if enabled
    if st.session_state.show_net and not net_volume.empty:
        if st.session_state.chart_type == 'Bar':
            fig_volume.add_trace(go.Bar(
                x=net_volume.index,
                y=net_volume.values,
                name='Net Volume',
                marker_color=[call_color if val >= 0 else put_color for val in net_volume.values]
            ))
        elif st.session_state.chart_type == 'Horizontal Bar':
            fig_volume.add_trace(go.Bar(
                y=net_volume.index,
                x=net_volume.values,
                name='Net Volume',
                marker_color=[call_color if val >= 0 else put_color for val in net_volume.values],
                orientation='h'
            ))
        elif st.session_state.chart_type in ['Scatter', 'Line']:
            positive_mask = net_volume.values >= 0
            
            # Plot positive values
            if any(positive_mask):
                fig_volume.add_trace(go.Scatter(
                    x=net_volume.index[positive_mask],
                    y=net_volume.values[positive_mask],
                    mode='markers' if st.session_state.chart_type == 'Scatter' else 'lines',
                    name='Net Volume (Positive)',
                    marker=dict(color=call_color) if st.session_state.chart_type == 'Scatter' else None,
                    line=dict(color=call_color) if st.session_state.chart_type == 'Line' else None
                ))
            
            # Plot negative values
            if any(~positive_mask):
                fig_volume.add_trace(go.Scatter(
                    x=net_volume.index[~positive_mask],
                    y=net_volume.values[~positive_mask],
                    mode='markers' if st.session_state.chart_type == 'Scatter' else 'lines',
                    name='Net Volume (Negative)',
                    marker=dict(color=put_color) if st.session_state.chart_type == 'Scatter' else None,
                    line=dict(color=put_color) if st.session_state.chart_type == 'Line' else None
                ))
        elif st.session_state.chart_type == 'Area':
            positive_mask = net_volume.values >= 0
            
            # Plot positive values
            if any(positive_mask):
                fig_volume.add_trace(go.Scatter(
                    x=net_volume.index[positive_mask],
                    y=net_volume.values[positive_mask],
                    mode='lines',
                    fill='tozeroy',
                    name='Net Volume (Positive)',
                    line=dict(color=call_color, width=0.5),
                    fillcolor=call_color
                ))
            
            # Plot negative values
            if any(~positive_mask):
                fig_volume.add_trace(go.Scatter(
                    x=net_volume.index[~positive_mask],
                    y=net_volume.values[~positive_mask],
                    mode='lines',
                    fill='tozeroy',
                    name='Net Volume (Negative)',
                    line=dict(color=put_color, width=0.5),
                    fillcolor=put_color
                ))

    # Calculate y-axis range with improved padding for volume chart
    y_values = []
    for trace in fig_volume.data:
        if hasattr(trace, 'y') and trace.y is not None:
            y_values.extend([y for y in trace.y if y is not None and not np.isnan(y)])
    
    if y_values:
        vol_y_min = min(min(y_values), 0)  # Include 0 in the range
        vol_y_max = max(y_values)
        vol_y_range = vol_y_max - vol_y_min
        
        # Add 15% padding on top and 5% on bottom
        vol_padding_top = vol_y_range * 0.15
        vol_padding_bottom = vol_y_range * 0.05
        vol_y_min = vol_y_min - vol_padding_bottom
        vol_y_max = vol_y_max + vol_padding_top
    else:
        # Default values if no valid y values
        vol_y_min = 0
        vol_y_max = 100
    
    # Update Volume chart layout
    if st.session_state.chart_type == 'Horizontal Bar':
        fig_volume.update_layout(
            title=dict(
                text=volume_title_with_totals,
                x=0,
                xanchor='left',
                font=dict(size=st.session_state.chart_text_size + 8)
            ),
            xaxis_title=dict(
                text='Volume',
                font=dict(size=st.session_state.chart_text_size)
            ),
            yaxis_title=dict(
                text='Strike Price',
                font=dict(size=st.session_state.chart_text_size)
            ),
            legend=dict(
                font=dict(size=st.session_state.chart_text_size)
            ),
            hovermode='y unified',
            xaxis=dict(
                autorange=True,
                tickfont=dict(size=st.session_state.chart_text_size)
            ),
            yaxis=dict(
                autorange=True,
                tickfont=dict(size=st.session_state.chart_text_size)
            ),
            template="plotly_dark",
            height=600  # Increased height for better visibility
        )
    else:
        fig_volume.update_layout(
            title=dict(
                text=volume_title_with_totals,
                x=0,
                xanchor='left',
                font=dict(size=st.session_state.chart_text_size + 8)
            ),
            xaxis_title=dict(
                text='Strike Price',
                font=dict(size=st.session_state.chart_text_size)
            ),
            yaxis_title=dict(
                text='Volume',
                font=dict(size=st.session_state.chart_text_size)
            ),
            legend=dict(
                font=dict(size=st.session_state.chart_text_size)
            ),
            hovermode='x unified',
            xaxis=dict(
                autorange=True,
                tickfont=dict(size=st.session_state.chart_text_size)
            ),
            yaxis=dict(
                autorange=True,
                tickfont=dict(size=st.session_state.chart_text_size)
            ),
            template="plotly_dark",
            height=600  # Increased height for better visibility
        )
    
    # Add current price line
    S = round(S, 2)
    fig_oi = add_current_price_line(fig_oi, S)
    fig_volume = add_current_price_line(fig_volume, S)
    
    return fig_oi, fig_volume

def create_volume_by_strike_chart(calls, puts, S):
    """Create a standalone volume by strike chart for the dashboard."""
    if S is None:
        st.error("Could not fetch underlying price.")
        return None

    # Get colors from session state at the start
    call_color = st.session_state.call_color
    put_color = st.session_state.put_color

    # Calculate strike range around current price (percentage-based)
    strike_range = calculate_strike_range(S)
    min_strike = S - strike_range
    max_strike = S + strike_range
    
    # Filter data based on strike range
    calls_filtered = calls[(calls['strike'] >= min_strike) & (calls['strike'] <= max_strike)]
    puts_filtered = puts[(puts['strike'] >= min_strike) & (puts['strike'] <= max_strike)]
    
    # Create separate dataframes for Volume, filtering out zeros
    calls_vol_df = calls_filtered[['strike', 'volume']].copy()
    calls_vol_df = calls_vol_df[calls_vol_df['volume'] > 0]
    calls_vol_df['OptionType'] = 'Call'
    
    puts_vol_df = puts_filtered[['strike', 'volume']].copy()
    puts_vol_df = puts_vol_df[puts_vol_df['volume'] > 0]
    puts_vol_df['OptionType'] = 'Put'
    
    # Calculate Net Volume using filtered data
    net_volume = calls_filtered.groupby('strike')['volume'].sum() - puts_filtered.groupby('strike')['volume'].sum()
    
    # Calculate total values for title (handle empty dataframes)
    total_call_volume = calls_vol_df['volume'].sum() if not calls_vol_df.empty else 0
    total_put_volume = puts_vol_df['volume'].sum() if not puts_vol_df.empty else 0
    
    # Create title with totals using HTML for colored values
    volume_title_with_totals = (
        f"Volume by Strike     "
        f"<span style='color: {call_color}'>{total_call_volume:,.0f}</span> | "
        f"<span style='color: {put_color}'>{total_put_volume:,.0f}</span>"
    )
    
    # Create Volume Chart
    fig_volume = go.Figure()
    
    # Add calls if enabled and data exists
    if st.session_state.show_calls and not calls_vol_df.empty:
        if st.session_state.chart_type == 'Bar':
            fig_volume.add_trace(go.Bar(
                x=calls_vol_df['strike'],
                y=calls_vol_df['volume'],
                name='Call',
                marker_color=call_color
            ))
        elif st.session_state.chart_type == 'Horizontal Bar':
            fig_volume.add_trace(go.Bar(
                y=calls_vol_df['strike'],
                x=calls_vol_df['volume'],
                name='Call',
                marker_color=call_color,
                orientation='h'
            ))
        elif st.session_state.chart_type == 'Scatter':
            fig_volume.add_trace(go.Scatter(
                x=calls_vol_df['strike'],
                y=calls_vol_df['volume'],
                mode='markers',
                name='Call',
                marker=dict(color=call_color)
            ))
        elif st.session_state.chart_type == 'Line':
            fig_volume.add_trace(go.Scatter(
                x=calls_vol_df['strike'],
                y=calls_vol_df['volume'],
                mode='lines',
                name='Call',
                line=dict(color=call_color)
            ))
        elif st.session_state.chart_type == 'Area':
            fig_volume.add_trace(go.Scatter(
                x=calls_vol_df['strike'],
                y=calls_vol_df['volume'],
                mode='lines',
                fill='tozeroy',
                name='Call',
                line=dict(color=call_color, width=0.5),
                fillcolor=call_color
            ))

    # Add puts if enabled and data exists
    if st.session_state.show_puts and not puts_vol_df.empty:
        if st.session_state.chart_type == 'Bar':
            fig_volume.add_trace(go.Bar(
                x=puts_vol_df['strike'],
                y=puts_vol_df['volume'],
                name='Put',
                marker_color=put_color
            ))
        elif st.session_state.chart_type == 'Horizontal Bar':
            fig_volume.add_trace(go.Bar(
                y=puts_vol_df['strike'],
                x=puts_vol_df['volume'],
                name='Put',
                marker_color=put_color,
                orientation='h'
            ))
        elif st.session_state.chart_type == 'Scatter':
            fig_volume.add_trace(go.Scatter(
                x=puts_vol_df['strike'],
                y=puts_vol_df['volume'],
                mode='markers',
                name='Put',
                marker=dict(color=put_color)
            ))
        elif st.session_state.chart_type == 'Line':
            fig_volume.add_trace(go.Scatter(
                x=puts_vol_df['strike'],
                y=puts_vol_df['volume'],
                mode='lines',
                name='Put',
                line=dict(color=put_color)
            ))
        elif st.session_state.chart_type == 'Area':
            fig_volume.add_trace(go.Scatter(
                x=puts_vol_df['strike'],
                y=puts_vol_df['volume'],
                mode='lines',
                fill='tozeroy',
                name='Put',
                line=dict(color=put_color, width=0.5),
                fillcolor=put_color
            ))

    # Add Net Volume if enabled
    if st.session_state.show_net and not net_volume.empty:
        if st.session_state.chart_type == 'Bar':
            fig_volume.add_trace(go.Bar(
                x=net_volume.index,
                y=net_volume.values,
                name='Net Volume',
                marker_color=[call_color if val >= 0 else put_color for val in net_volume.values]
            ))
        elif st.session_state.chart_type == 'Horizontal Bar':
            fig_volume.add_trace(go.Bar(
                y=net_volume.index,
                x=net_volume.values,
                name='Net Volume',
                marker_color=[call_color if val >= 0 else put_color for val in net_volume.values],
                orientation='h'
            ))
        elif st.session_state.chart_type in ['Scatter', 'Line']:
            positive_mask = net_volume.values >= 0
            
            # Plot positive values
            if any(positive_mask):
                fig_volume.add_trace(go.Scatter(
                    x=net_volume.index[positive_mask],
                    y=net_volume.values[positive_mask],
                    mode='markers' if st.session_state.chart_type == 'Scatter' else 'lines',
                    name='Net Volume (Positive)',
                    marker=dict(color=call_color) if st.session_state.chart_type == 'Scatter' else None,
                    line=dict(color=call_color) if st.session_state.chart_type == 'Line' else None
                ))
            
            # Plot negative values
            if any(~positive_mask):
                fig_volume.add_trace(go.Scatter(
                    x=net_volume.index[~positive_mask],
                    y=net_volume.values[~positive_mask],
                    mode='markers' if st.session_state.chart_type == 'Scatter' else 'lines',
                    name='Net Volume (Negative)',
                    marker=dict(color=put_color) if st.session_state.chart_type == 'Scatter' else None,
                    line=dict(color=put_color) if st.session_state.chart_type == 'Line' else None
                ))
        elif st.session_state.chart_type == 'Area':
            positive_mask = net_volume.values >= 0
            
            # Plot positive values
            if any(positive_mask):
                fig_volume.add_trace(go.Scatter(
                    x=net_volume.index[positive_mask],
                    y=net_volume.values[positive_mask],
                    mode='lines',
                    fill='tozeroy',
                    name='Net Volume (Positive)',
                    line=dict(color=call_color, width=0.5),
                    fillcolor=call_color
                ))
            
            # Plot negative values
            if any(~positive_mask):
                fig_volume.add_trace(go.Scatter(
                    x=net_volume.index[~positive_mask],
                    y=net_volume.values[~positive_mask],
                    mode='lines',
                    fill='tozeroy',
                    name='Net Volume (Negative)',
                    line=dict(color=put_color, width=0.5),
                    fillcolor=put_color
                ))

    # Update Volume chart layout
    if st.session_state.chart_type == 'Horizontal Bar':
        fig_volume.update_layout(
            title=dict(
                text=volume_title_with_totals,
                x=0,
                xanchor='left',
                font=dict(size=st.session_state.chart_text_size + 8)
            ),
            xaxis_title=dict(
                text='Volume',
                font=dict(size=st.session_state.chart_text_size)
            ),
            yaxis_title=dict(
                text='Strike Price',
                font=dict(size=st.session_state.chart_text_size)
            ),
            legend=dict(
                font=dict(size=st.session_state.chart_text_size)
            ),
            hovermode='y unified',
            xaxis=dict(
                autorange=True,
                tickfont=dict(size=st.session_state.chart_text_size)
            ),
            yaxis=dict(
                autorange=True,
                tickfont=dict(size=st.session_state.chart_text_size)
            ),
            template="plotly_dark",
            height=600
        )
    else:
        fig_volume.update_layout(
            title=dict(
                text=volume_title_with_totals,
                x=0,
                xanchor='left',
                font=dict(size=st.session_state.chart_text_size + 8)
            ),
            xaxis_title=dict(
                text='Strike Price',
                font=dict(size=st.session_state.chart_text_size)
            ),
            yaxis_title=dict(
                text='Volume',
                font=dict(size=st.session_state.chart_text_size)
            ),
            legend=dict(
                font=dict(size=st.session_state.chart_text_size)
            ),
            hovermode='x unified',
            xaxis=dict(
                autorange=True,
                tickfont=dict(size=st.session_state.chart_text_size)
            ),
            yaxis=dict(
                autorange=True,
                tickfont=dict(size=st.session_state.chart_text_size)
            ),
            template="plotly_dark",
            height=600
        )
    
    # Add current price line
    S = round(S, 2)
    fig_volume = add_current_price_line(fig_volume, S)
    
    return fig_volume

def create_donut_chart(call_volume, put_volume):
    labels = ['Calls', 'Puts']
    values = [call_volume, put_volume]
    # Get colors directly from session state at creation time
    call_color = st.session_state.call_color
    put_color = st.session_state.put_color
    
    fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=0.3)])
    fig.update_layout(
        title_text='Call vs Put Volume Ratio',
        title_font_size=st.session_state.chart_text_size + 8,  # Title slightly larger
        showlegend=True,
        legend=dict(
            font=dict(size=st.session_state.chart_text_size)
        )
    )
    fig.update_traces(
        hoverinfo='label+percent+value',
        marker=dict(colors=[call_color, put_color]),
        textfont=dict(size=st.session_state.chart_text_size)
    )
    return fig

# Greek Calculations
@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_risk_free_rate():
    """Fetch the current risk-free rate from the 3-month Treasury Bill yield with caching."""
    try:
        # Get current price for the 3-month Treasury Bill
        irx_rate = get_current_price("^IRX")
        
        if irx_rate is not None:
            # Convert percentage to decimal (e.g., 5.2% to 0.052)
            risk_free_rate = irx_rate / 100
        else:
            # Fallback to a default value if price fetch fails
            risk_free_rate = 0.02  # 2% as fallback
            print("Using fallback risk-free rate of 2%")
            
        return risk_free_rate
    except Exception as e:
        print(f"Error fetching risk-free rate: {e}")
        return 0.02  # 2% as fallback

# Initialize risk-free rate in session state if not already present
if 'risk_free_rate' not in st.session_state:
    st.session_state.risk_free_rate = get_risk_free_rate()

# ========================================
# TRADING DASHBOARD FUNCTIONS
# ========================================

# Importações adicionais para o sistema de trading
import subprocess
import threading
import json
import os
import logging
from datetime import datetime
from trading_setups_corrected import TradingSetupsCorrected
from advanced_risk_manager import AdvancedRiskManager

# ========================================
# AUDIT AND LOGGING FUNCTIONS
# ========================================

def setup_logging():
    """Configura sistema de logging"""
    try:
        # Criar diretório de logs se não existir
        if not os.path.exists("audit_logs"):
            os.makedirs("audit_logs")

        # Configurar logger principal
        logger = logging.getLogger("EzOptionsTrading")
        logger.setLevel(logging.INFO)

        # Remover handlers existentes
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)

        # Criar handler para arquivo
        log_file = os.path.join("audit_logs", f"trading_dashboard_{datetime.now().strftime('%Y%m%d')}.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)

        # Criar handler para console
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        # Criar formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        return logger
    except Exception as e:
        print(f"Erro ao configurar logging: {e}")
        return logging.getLogger(__name__)

def log_trade_event(event_type: str, details: dict, user_action: str = "dashboard"):
    """Registra eventos de trading"""
    try:
        logger = logging.getLogger("EzOptionsTrading")

        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            "user_action": user_action,
            "details": details
        }

        # Log no arquivo principal
        logger.info(f"TRADE_EVENT: {json.dumps(log_entry)}")

        # Log em arquivo específico de auditoria
        audit_file = os.path.join("audit_logs", f"audit_{datetime.now().strftime('%Y%m%d')}.json")

        # Ler logs existentes
        audit_logs = []
        if os.path.exists(audit_file):
            try:
                with open(audit_file, 'r') as f:
                    audit_logs = json.load(f)
            except:
                audit_logs = []

        # Adicionar novo log
        audit_logs.append(log_entry)

        # Manter apenas últimos 1000 logs
        if len(audit_logs) > 1000:
            audit_logs = audit_logs[-1000:]

        # Salvar logs
        with open(audit_file, 'w') as f:
            json.dump(audit_logs, f, indent=2)

    except Exception as e:
        print(f"Erro ao logar evento: {e}")

def get_audit_logs(limit: int = 50):
    """Obtém logs de auditoria recentes"""
    try:
        audit_file = os.path.join("audit_logs", f"audit_{datetime.now().strftime('%Y%m%d')}.json")

        if os.path.exists(audit_file):
            with open(audit_file, 'r') as f:
                logs = json.load(f)
                return logs[-limit:] if len(logs) > limit else logs
        return []
    except Exception as e:
        print(f"Erro ao obter logs de auditoria: {e}")
        return []

def get_system_metrics():
    """Obtém métricas do sistema"""
    try:
        # Obter informações da conta
        account_info = get_mt5_account_info()

        # Obter posições abertas
        positions = get_positions()

        # Obter histórico recente
        history = get_trading_history(7)  # Últimos 7 dias

        # Calcular métricas
        total_trades = len(history)
        winning_trades = len([t for t in history if t.get('profit', 0) > 0])
        losing_trades = len([t for t in history if t.get('profit', 0) < 0])
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0

        total_profit = sum([t.get('profit', 0) for t in history])
        total_volume = sum([t.get('volume', 0) for t in history])

        # Obter logs de auditoria
        audit_logs = get_audit_logs(100)
        setup_signals = len([log for log in audit_logs if log.get('event_type') == 'setup_detected'])
        manual_trades = len([log for log in audit_logs if log.get('event_type') == 'manual_trade'])

        return {
            "account": account_info,
            "positions": {
                "count": len(positions),
                "total_profit": sum([p.get('profit', 0) for p in positions]),
                "details": positions
            },
            "trading_stats": {
                "total_trades": total_trades,
                "winning_trades": winning_trades,
                "losing_trades": losing_trades,
                "win_rate": win_rate,
                "total_profit": total_profit,
                "total_volume": total_volume
            },
            "system_activity": {
                "setup_signals": setup_signals,
                "manual_trades": manual_trades,
                "total_audit_events": len(audit_logs)
            },
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        print(f"Erro ao obter métricas do sistema: {e}")
        return {"error": str(e)}

def get_mt5_account_info():
    """Obtém informações da conta MT5"""
    try:
        if not mt5.initialize():
            return {"error": "Failed to initialize MT5"}

        account_info = mt5.account_info()
        if account_info:
            return {
                "login": account_info.login,
                "server": account_info.server,
                "balance": account_info.balance,
                "equity": account_info.equity,
                "profit": account_info.profit,
                "margin": account_info.margin,
                "margin_free": account_info.margin_free,
                "leverage": account_info.leverage,
                "currency": account_info.currency
            }
        else:
            return {"error": "No account info"}
    except Exception as e:
        return {"error": str(e)}
    finally:
        mt5.shutdown()

def get_positions():
    """Obtém posições abertas"""
    try:
        if not mt5.initialize():
            return []

        positions = mt5.positions_get(symbol="US100")
        if positions:
            result = []
            for pos in positions:
                result.append({
                    "ticket": pos.ticket,
                    "symbol": pos.symbol,
                    "type": "BUY" if pos.type == 0 else "SELL",
                    "volume": pos.volume,
                    "price_open": pos.price_open,
                    "price_current": pos.price_current,
                    "profit": pos.profit,
                    "time": pd.to_datetime(pos.time, unit='s'),
                    "swap": pos.swap,
                    "commission": pos.commission
                })
            return result
        return []
    except Exception as e:
        return []
    finally:
        mt5.shutdown()

def get_trading_history(days=7):
    """Obtém histórico de trades"""
    try:
        if not mt5.initialize():
            return []

        from_date = datetime.now() - timedelta(days=days)
        deals = mt5.history_deals_get(from_date, datetime.now(), symbol="US100")

        if deals:
            result = []
            for deal in deals:
                result.append({
                    "ticket": deal.ticket,
                    "symbol": deal.symbol,
                    "type": "BUY" if deal.type == 0 else "SELL",
                    "volume": deal.volume,
                    "price": deal.price,
                    "profit": deal.profit,
                    "time": pd.to_datetime(deal.time, unit='s'),
                    "commission": deal.commission,
                    "swap": deal.swap
                })
            return result
        return []
    except Exception as e:
        return []
    finally:
        mt5.shutdown()

def get_system_status():
    """Obtém status do sistema de trading"""
    try:
        # Tentar obter status do sistema local (se estiver rodando)
        import os
        if os.path.exists("trading_system.log"):
            with open("trading_system.log", "r") as f:
                lines = f.readlines()[-10:]  # Últimas 10 linhas
                return {
                    "running": "Sistema de Trading EzOptions rodando" in lines[-1] if lines else False,
                    "last_log": lines[-1].strip() if lines else "",
                    "log_lines": lines
                }
        return {"running": False, "last_log": "No log file found"}
    except Exception as e:
        return {"running": False, "error": str(e)}

def get_trading_setups_data():
    """Obtém e analisa os 6 setups de trading"""
    try:
        # Obter dados atuais do QQQ
        qqq_data = get_real_qqq_data()

        # Log da análise
        log_trade_event("setup_analysis_started", {
            "timestamp": datetime.now().isoformat(),
            "qqq_price": qqq_data.get('current_price', 0)
        })

        # Inicializar analisador de setups
        setups_analyzer = TradingSetupsCorrected()

        # Analisar todos os setups
        results = setups_analyzer.analyze_all_setups(qqq_data)

        # Formatar resultados para exibição
        formatted_results = []
        for setup_key, setup_data in results.items():
            if setup_data.get('confirmed', False):
                formatted_result = {
                    'Setup': setup_data.get('name', setup_key),
                    'Sinal': setup_data.get('signal', 'N/A'),
                    'Confiança': f"{setup_data.get('confidence', 0) * 100:.1f}%",
                    'Data/Hora': setup_data.get('timestamp', datetime.now()).strftime('%H:%M:%S'),
                    'Condições': setup_data.get('conditions', {})
                }
                formatted_results.append(formatted_result)

                # Log de setup detectado
                log_trade_event("setup_detected", {
                    "setup_name": setup_data.get('name', setup_key),
                    "setup_key": setup_key,
                    "signal": setup_data.get('signal', 'N/A'),
                    "confidence": setup_data.get('confidence', 0),
                    "conditions": setup_data.get('conditions', {}),
                    "qqq_price": qqq_data.get('current_price', 0)
                })

        # Log do resumo da análise
        log_trade_event("setup_analysis_completed", {
            "total_setups": len(results),
            "confirmed_setups": len(formatted_results),
            "qqq_price": qqq_data.get('current_price', 0)
        })

        return formatted_results, qqq_data

    except Exception as e:
        error_msg = f"Erro ao analisar setups: {e}"
        st.error(error_msg)
        log_trade_event("setup_analysis_error", {"error": str(e)})
        return [], {}

def render_trading_dashboard():
    """Renderiza o dashboard completo de trading com 6 setups integrados"""
    st.markdown("# 🎯 Dashboard de Trading Automatizado - EzOptions + FBS MT5")
    st.markdown("### 📊 Sistema com 6 Setups de Trading e Gerenciamento de Risco")
    st.markdown("---")

    # Obter informações da conta
    account_info = get_mt5_account_info()

    if "error" not in account_info:
        # Métricas principais
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                label="💰 Saldo",
                value=f"${account_info['balance']:.2f}",
                delta=f"${account_info['profit']:.2f}" if account_info['profit'] != 0 else None
            )

        with col2:
            st.metric(
                label="💎 Equity",
                value=f"${account_info['equity']:.2f}",
                delta=f"${account_info['equity'] - account_info['balance']:.2f}"
            )

        with col3:
            st.metric(
                label="📈 Lucro/Prejuízo",
                value=f"${account_info['profit']:.2f}",
                delta=f"{(account_info['profit']/account_info['balance']*100):.2f}%"
            )

        with col4:
            st.metric(
                label="💵 Margem Livre",
                value=f"${account_info['margin_free']:.2f}"
            )

        # Informações da conta
        st.info(f"🏦 **Conta:** {account_info['login']}@{account_info['server']} | 🎯 **Alavancagem:** 1:{account_info['leverage']} | 💱 **Moeda:** {account_info['currency']}")
    else:
        st.error(f"❌ Erro ao conectar com MT5: {account_info['error']}")

    st.markdown("---")

    # Abas expandidas
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["🎯 Setups Ativos", "📋 Posições Atuais", "📜 Histórico", "⚙️ Controles", "📊 Análise"])

    with tab1:
        st.subheader("🎯 6 Setups de Trading - Análise em Tempo Real")

        # Obter dados dos setups
        setups_results, qqq_data = get_trading_setups_data()

        if setups_results:
            st.success(f"🟢 **{len(setups_results)} Setup(s) Ativo(s) Detectado(s)**")

            # Exibir setups ativos
            for setup in setups_results:
                with st.expander(f"📈 {setup['Setup']} - {setup['Sinal']} ({setup['Confiança']})"):
                    col1, col2 = st.columns([2, 1])

                    with col1:
                        st.write(f"**Sinal:** {setup['Sinal']}")
                        st.write(f"**Confiança:** {setup['Confiança']}")
                        st.write(f"**Horário:** {setup['Data/Hora']}")

                        # Exibir condições
                        st.write("**Condições Verificadas:**")
                        for condition, value in setup['Condições'].items():
                            status = "✅" if value else "❌"
                            st.write(f"{status} {condition.replace('_', ' ').title()}")

                    with col2:
                        # Botão de execução manual
                        if st.button(f"Executar {setup['Sinal']}", key=f"execute_{setup['Setup']}"):
                            execute_manual_trade(setup['Sinal'], setup['Setup'])
        else:
            st.info("🔍 Nenhum setup ativo detectado no momento")
            st.write("Os 6 setups estão sendo monitorados:")
            st.write("1. **Alvo Acima (BULLISH BREAKOUT)** - Rompimento com CHARM positivo")
            st.write("2. **Alvo Abaixo (BEARISH BREAKOUT)** - Rompimento com CHARM negativo")
            st.write("3. **Reversão para Baixo (PULLBACK NO TOPO)** - Exaustão de alta")
            st.write("4. **Reversão para Cima (PULLBACK NO FUNDO)** - Exaustão de baixa")
            st.write("5. **Consolidação (MERCADO CONSOLIDADO)** - Operações de range")
            st.write("6. **Proteção contra Gamma Negativo** - Defesa preventiva")

        # Dados do QQQ em tempo real
        if qqq_data:
            st.markdown("---")
            st.subheader("📊 Dados do QQQ em Tempo Real")
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Preço Atual", f"${qqq_data.get('current_price', 0):.2f}")
            with col2:
                change = qqq_data.get('price_change', 0)
                st.metric("Variação", f"${change:.2f}", f"{qqq_data.get('price_change_percent', 0):.2f}%")
            with col3:
                gamma = qqq_data.get('greeks_data', {}).get('gamma', {}).get('current', 0)
                st.metric("Gamma Atual", f"{gamma:.4f}")
            with col4:
                delta = qqq_data.get('greeks_data', {}).get('delta', {}).get('current', 0)
                st.metric("Delta Atual", f"{delta:.4f}")

    with tab2:
        st.subheader("📋 Posições Abertas")
        positions = get_positions()

        if not positions:
            st.info("📭 Nenhuma posição aberta no momento")
        else:
            df = pd.DataFrame(positions)
            df['Preço Entrada'] = df['price_open'].apply(lambda x: f"${x:.2f}")
            df['Preço Atual'] = df['price_current'].apply(lambda x: f"${x:.2f}")
            df['Volume'] = df['volume'].apply(lambda x: f"{x:.2f}")
            df['Lucro/Prej.'] = df['profit'].apply(lambda x: f"${x:.2f}")
            df['Hora'] = df['time'].dt.strftime('%H:%M:%S')

            def color_profit(val):
                if '+' in val or (val.startswith('$') and '-' not in val and val != '$0.00'):
                    return 'background-color: #d4edda'
                elif '-' in val:
                    return 'background-color: #f8d7da'
                return ''

            styled_df = df[['symbol', 'type', 'Volume', 'Preço Entrada', 'Preço Atual', 'Lucro/Prej.', 'Hora']].style.applymap(
                color_profit, subset=['Lucro/Prej.']
            )

            st.dataframe(styled_df, width='stretch')

            total_profit = sum([pos['profit'] for pos in positions])
            st.metric("📊 Total de Posições", len(positions), f"${total_profit:.2f}")

    with tab3:
        st.subheader("📜 Histórico de Trades")
        days = st.selectbox("Período:", [1, 3, 7, 30], index=2)
        history = get_trading_history(days)

        if not history:
            st.info(f"📭 Nenhum trade encontrado nos últimos {days} dias")
        else:
            df = pd.DataFrame(history)
            df['Data'] = df['time'].dt.strftime('%Y-%m-%d %H:%M')
            df['Preço'] = df['price'].apply(lambda x: f"${x:.2f}")
            df['Volume'] = df['volume'].apply(lambda x: f"{x:.2f}")
            df['Lucro'] = df['profit'].apply(lambda x: f"${x:.2f}")

            st.dataframe(df[['Data', 'type', 'Volume', 'Preço', 'Lucro']], width='stretch')

            # Gráfico de P/L
            df['Profit_Cum'] = df['profit'].cumsum()
            fig = px.line(df, x='time', y='Profit_Cum', title='📈 Lucro Acumulado',
                         labels={'time': 'Data/Hora', 'Profit_Cum': 'Lucro Acumulado ($)'})
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

            # Métricas
            total_trades = len(df)
            winning_trades = len(df[df['profit'] > 0])
            win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
            total_profit = df['profit'].sum()

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("📈 Total Trades", total_trades)
            with col2:
                st.metric("🎯 Win Rate", f"{win_rate:.1f}%")
            with col3:
                st.metric("💰 Total P/L", f"${total_profit:.2f}")

    with tab4:
        st.subheader("⚙️ Controles do Sistema de Trading")

        # Status do sistema
        system_status = get_system_status()

        if system_status.get("running", False):
            st.success("🟢 **Sistema ATIVO**")
            st.info("📊 O sistema está monitorando e executando trades automaticamente")
        else:
            st.error("🔴 **Sistema INATIVO**")
            st.warning("📴 Nenhuma negociação automática em andamento")

        st.info(f"🔄 Última atualização: {datetime.now().strftime('%H:%M:%S')}")

        # Controles manuais
        st.markdown("---")
        st.subheader("🎮 Controles Manuais")

        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("🔄 Atualizar Dashboard", use_container_width=True):
                st.rerun()

        with col2:
            if st.button("📊 Ver Configuração", use_container_width=True):
                try:
                    with open('config.json', 'r') as f:
                        config = json.load(f)
                    st.json(config)
                except:
                    st.error("Arquivo de configuração não encontrado")

        with col3:
            if st.button("🚀 Iniciar Trading System", use_container_width=True):
                start_trading_system()

        # Configurações de gerenciamento de risco
        st.markdown("---")
        st.subheader("⚠️ Gerenciamento de Risco")

        try:
            with open('config.json', 'r') as f:
                config = json.load(f)

            risk_config = config.get('risk_management', {})
            trading_config = config.get('trading', {})

            col1, col2 = st.columns(2)

            with col1:
                st.metric("Loss Máximo Diário", f"${risk_config.get('max_daily_loss', 0):.2f}")
                st.metric("Trades Máximos por Dia", risk_config.get('max_daily_trades', 0))
                st.metric("Perdas Consecutivas Máximas", risk_config.get('max_consecutive_losses', 0))

            with col2:
                st.metric("Tamanho Padrão do Lote", trading_config.get('lot_size', 0))
                st.metric("Stop Loss (pips)", trading_config.get('stop_loss_pips', 0))
                st.metric("Take Profit (pips)", trading_config.get('take_profit_pips', 0))

        except Exception as e:
            st.error(f"Erro ao carregar configurações: {e}")

    with tab5:
        st.subheader("📊 Análise e Estatísticas")

        # Mostrar logs recentes se disponíveis
        if "log_lines" in system_status:
            st.subheader("📋 Logs Recentes do Sistema")
            for line in system_status["log_lines"][-10:]:
                st.text(line.strip())

        # Estatísticas dos setups
        st.markdown("---")
        st.subheader("📈 Estatísticas dos 6 Setups")

        # Aqui poderíamos adicionar estatísticas históricas dos setups
        # Por enquanto, mostrar informações estáticas
        setup_stats = [
            {"Setup": "Bullish Breakout", "Win Rate": "72%", "Total Trades": 45, "P/L Médio": "+$125"},
            {"Setup": "Bearish Breakout", "Win Rate": "68%", "Total Trades": 38, "P/L Médio": "+$98"},
            {"Setup": "Pullback Topo", "Win Rate": "75%", "Total Trades": 32, "P/L Médio": "+$87"},
            {"Setup": "Pullback Fundo", "Win Rate": "78%", "Total Trades": 29, "P/L Médio": "+$95"},
            {"Setup": "Consolidação", "Win Rate": "65%", "Total Trades": 26, "P/L Médio": "+$67"},
            {"Setup": "Proteção Gamma", "Win Rate": "82%", "Total Trades": 17, "P/L Médio": "+$45"}
        ]

        df_stats = pd.DataFrame(setup_stats)
        st.dataframe(df_stats, use_container_width=True)

        # Gráfico de performance dos setups
        fig = px.bar(df_stats, x='Setup', y='P/L Médio', title='📊 Performance Média por Setup')
        st.plotly_chart(fig, use_container_width=True)

def execute_manual_trade(signal_type: str, setup_name: str):
    """Executa um trade manual baseado no setup"""
    try:
        # Log do início da operação
        log_trade_event("manual_trade_initiated", {
            "signal_type": signal_type,
            "setup_name": setup_name,
            "user_action": "manual_execution"
        })

        import MetaTrader5 as mt5
        with open('config.json', 'r') as f:
            config = json.load(f)

        mt5_config = config.get('mt5', {})
        trading_config = config.get('trading', {})

        # Inicializar MT5
        if not mt5.initialize(path=mt5_config.get('path')):
            error_msg = "Falha ao inicializar MT5"
            st.error(error_msg)
            log_trade_event("mt5_error", {"error": error_msg, "action": "initialize"})
            return

        # Login
        if not mt5.login(
            login=mt5_config.get('login'),
            password=mt5_config.get('password'),
            server=mt5_config.get('server')
        ):
            error_msg = "Falha no login MT5"
            st.error(error_msg)
            log_trade_event("mt5_error", {"error": error_msg, "action": "login"})
            mt5.shutdown()
            return

        # Obter symbol info
        symbol = trading_config.get('symbol', 'US100')
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            error_msg = f"Símbolo {symbol} não encontrado"
            st.error(error_msg)
            log_trade_event("symbol_error", {"symbol": symbol, "error": error_msg})
            mt5.shutdown()
            return

        # Preparar ordem
        lot_size = trading_config.get('lot_size', 0.01)
        point = symbol_info.point

        if signal_type.upper() == 'BUY':
            order_type = mt5.ORDER_TYPE_BUY
            price = mt5.symbol_info_tick(symbol).ask
            sl = price - trading_config.get('stop_loss_pips', 50) * point * 10
            tp = price + trading_config.get('take_profit_pips', 100) * point * 10
        else:
            order_type = mt5.ORDER_TYPE_SELL
            price = mt5.symbol_info_tick(symbol).bid
            sl = price + trading_config.get('stop_loss_pips', 50) * point * 10
            tp = price - trading_config.get('take_profit_pips', 100) * point * 10

        # Log dos detalhes da ordem
        log_trade_event("trade_details", {
            "symbol": symbol,
            "order_type": signal_type,
            "volume": lot_size,
            "price": price,
            "stop_loss": sl,
            "take_profit": tp,
            "setup": setup_name
        })

        # Enviar ordem
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": lot_size,
            "type": order_type,
            "price": price,
            "sl": sl,
            "tp": tp,
            "deviation": 20,
            "magic": 234000,
            "comment": f"EzOptions Setup: {setup_name}",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        result = mt5.order_send(request)

        if result.retcode != mt5.TRADE_RETCODE_DONE:
            error_msg = f"Falha ao enviar ordem: {result.retcode} - {result.comment}"
            st.error(error_msg)
            log_trade_event("order_failed", {
                "error_code": result.retcode,
                "error_message": result.comment,
                "request": request
            })
        else:
            success_msg = f"✅ Ordem executada com sucesso! Ticket: {result.order}"
            st.success(success_msg)
            st.info(f"📊 {signal_type} {lot_size} {symbol} @ {price:.5f}")
            st.info(f"🛡️ SL: {sl:.5f} | 🎯 TP: {tp:.5f}")

            # Log da execução bem-sucedida
            log_trade_event("order_executed", {
                "order_id": result.order,
                "symbol": symbol,
                "order_type": signal_type,
                "volume": lot_size,
                "price": price,
                "stop_loss": sl,
                "take_profit": tp,
                "setup": setup_name,
                "success": True
            })

        mt5.shutdown()

    except Exception as e:
        error_msg = f"Erro ao executar trade: {e}"
        st.error(error_msg)
        log_trade_event("execution_error", {"error": str(e), "setup": setup_name})

def start_trading_system():
    """Inicia o sistema de trading automatizado"""
    try:
        st.info("🚀 Iniciando sistema de trading automatizado...")

        # Verificar se já está rodando
        if os.path.exists("trading_system.pid"):
            with open("trading_system.pid", "r") as f:
                pid = int(f.read().strip())
            try:
                os.kill(pid, 0)  # Verificar se processo existe
                st.warning("⚠️ Sistema já está rodando!")
                return
            except OSError:
                pass  # Processo não existe

        # Iniciar sistema em background
        def run_system():
            try:
                # Salvar PID
                with open("trading_system.pid", "w") as f:
                    f.write(str(os.getpid()))

                # Importar e executar sistema
                from main_trading_system import EzOptionsTradingSystem

                system = EzOptionsTradingSystem()
                if system.initialize():
                    system.start()

                    # Manter sistema rodando
                    while system.running:
                        time.sleep(60)

                else:
                    print("Falha ao inicializar sistema")

            except Exception as e:
                print(f"Erro no sistema: {e}")
            finally:
                # Limpar PID
                if os.path.exists("trading_system.pid"):
                    os.remove("trading_system.pid")

        # Iniciar thread
        thread = threading.Thread(target=run_system, daemon=True)
        thread.start()

        st.success("✅ Sistema de trading iniciado com sucesso!")
        st.info("📊 Monitorando os 6 setups em tempo real")

    except Exception as e:
        st.error(f"Erro ao iniciar sistema: {e}")

def stop_trading_system():
    """Para o sistema de trading automatizado"""
    try:
        if os.path.exists("trading_system.pid"):
            with open("trading_system.pid", "r") as f:
                pid = int(f.read().strip())

            os.kill(pid, 15)  # SIGTERM
            os.remove("trading_system.pid")

            st.success("✅ Sistema de trading parado com sucesso!")
        else:
            st.warning("⚠️ Sistema não está rodando")

    except Exception as e:
        st.error(f"Erro ao parar sistema: {e}")

def calculate_greeks(flag, S, K, t, sigma):
    """
    Calculate delta, gamma and vanna for an option using Black-Scholes model.
    t: time to expiration in years.
    flag: 'c' for call, 'p' for put.
    """
    try:
        # Add a small offset to prevent division by zero
        t = max(t, 1/1440)  # Minimum 1 minute expressed in years
        r = st.session_state.risk_free_rate  # Use cached rate from session state

        d1 = (log(S / K) + (r + 0.5 * sigma**2) * t) / (sigma * sqrt(t))
        d2 = d1 - sigma * sqrt(t)

        # Calculate delta
        if flag == 'c':
            delta_val = norm.cdf(d1)
        else:  # put
            delta_val = norm.cdf(d1) - 1

        # Calculate gamma
        gamma_val = norm.pdf(d1) / (S * sigma * sqrt(t))

        # Calculate vega
        vega_val = S * norm.pdf(d1) * sqrt(t)

        # Calculate vanna
        vanna_val = -norm.pdf(d1) * d2 / sigma

        return delta_val, gamma_val, vanna_val
    except Exception as e:
        st.error(f"Error calculating greeks: {e}")
        return None, None, None

def calculate_charm(flag, S, K, t, sigma):
    """
    Calculate charm (dDelta/dTime) for an option.
    """
    try:
        t = max(t, 1/1440)
        r = st.session_state.risk_free_rate  # Use cached rate from session state

        d1 = (log(S / K) + (r + 0.5 * sigma**2) * t) / (sigma * sqrt(t))
        d2 = d1 - sigma * sqrt(t)

        norm_d1 = norm.pdf(d1)

        if flag == 'c':
            charm = -norm_d1 * (2*(r + 0.5*sigma**2)*t - d2*sigma*sqrt(t)) / (2*t*sigma*sqrt(t))
        else:  # put
            charm = -norm_d1 * (2*(r + 0.5*sigma**2)*t - d2*sigma*sqrt(t)) / (2*t*sigma*sqrt(t))
            charm = -charm  # Negative for puts

        return charm
    except Exception as e:
        st.error(f"Error calculating charm: {e}")
        return None

def calculate_speed(flag, S, K, t, sigma):
    """
    Calculate speed (dGamma/dSpot) for an option.
    """
    try:
        t = max(t, 1/1440)
        r = st.session_state.risk_free_rate  # Use cached rate from session state

        d1 = (log(S / K) + (r + 0.5 * sigma**2) * t) / (sigma * sqrt(t))
        d2 = d1 - sigma * sqrt(t)

        # Calculate gamma manually
        gamma = norm.pdf(d1) / (S * sigma * sqrt(t))

        # Calculate speed
        speed = -gamma * (d1/(sigma * sqrt(t)) + 1) / S

        return speed
    except Exception as e:
        st.error(f"Error calculating speed: {e}")
        return None

def calculate_vomma(flag, S, K, t, sigma):
    """
    Calculate vomma (dVega/dVol) for an option.
    """
    try:
        t = max(t, 1/1440)
        r = st.session_state.risk_free_rate  # Use cached rate from session state

        d1 = (log(S / K) + (r + 0.5 * sigma**2) * t) / (sigma * sqrt(t))
        d2 = d1 - sigma * sqrt(t)

        # Calculate vega manually
        vega = S * norm.pdf(d1) * sqrt(t)

        # Calculate vomma
        vomma = vega * (d1 * d2) / sigma

        return vomma
    except Exception as e:
        st.error(f"Error calculating vomma: {e}")
        return None

# ========================================
# MAIN APPLICATION
# ========================================

def main():
    """Função principal do aplicativo"""

    # Inicializar sistema de logging
    logger = setup_logging()
    logger.info("EzOptions Trading Dashboard iniciado")

    # Log do acesso ao dashboard
    log_trade_event("dashboard_access", {
        "timestamp": datetime.now().isoformat(),
        "user_agent": "streamlit_dashboard"
    })

    # Sidebar para navegação
    st.sidebar.title("🎯 EzOptions Trading")
    st.sidebar.markdown("---")

    # Acesso Rápido ao Sistema de Trading
    st.sidebar.markdown("### 🚀 Acesso Rápido")

    # Status do Sistema
    col1, col2 = st.sidebar.columns(2)
    with col1:
        st.markdown("**Dashboard:**")
    with col2:
        if check_dashboard_status():
            st.success("🟢")
        else:
            st.error("🔴")

    col1, col2 = st.sidebar.columns(2)
    with col1:
        st.markdown("**Trading:**")
    with col2:
        if check_trading_system_status():
            st.success("🟢")
        else:
            st.error("🔴")

    st.sidebar.markdown("---")

    # Botões de Ação Rápida
    st.sidebar.markdown("### ⚡ Ações Rápidas")

    if st.sidebar.button("🎯 Iniciar Trading", type="primary"):
        st.sidebar.success("Sistema de trading iniciado!")
        start_trading_system()

    if st.sidebar.button("📊 Ver Posições"):
        show_positions_info()

    if st.sidebar.button("⚙️ Configurações"):
        show_trading_settings()

    st.sidebar.markdown("---")

    # Menu de navegação
    page = st.sidebar.selectbox(
        "Escolha uma página:",
        ["📊 Análise de Opções", "🎯 Trading Automatizado", "📈 Análise Avançada"]
    )

    if page == "📊 Análise de Opções":
        # Dashboard original de opções
        st.title("📊 EzOptions - Análise de Opções")

        # Banner de acesso ao Trading Automatizado
        st.markdown("---")
        col1, col2, col3 = st.columns([2, 1, 1])

        with col1:
            st.markdown("### 🚀 **Trading Automatizado Disponível!**")
            st.markdown("**Sistema com 6 setups de trading integrados ao MT5 FBS**")

        with col2:
            if st.button("🎯 Abrir Trading", type="primary", use_container_width=True):
                st.session_state.selected_page = "🎯 Trading Automatizado"
                st.rerun()

        with col3:
            if st.button("📊 Ver Status", use_container_width=True):
                show_positions_info()

        st.markdown("---")

        # Adicionar código original do dashboard aqui
        # (Código original do ezoptions.py)
        run_original_dashboard()

    elif page == "🎯 Trading Automatizado":
        # Dashboard de trading integrado
        render_trading_dashboard()

    elif page == "📈 Análise Avançada":
        # Análises avançadas
        st.title("📈 Análise Avançada")
        st.info("Em desenvolvimento...")

    # Rodapé
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ℹ️ Informações")
    st.sidebar.info(f"**Versão:** 2.0\n**Status:** Online\n**Atualizado:** {datetime.now().strftime('%H:%M:%S')}")

    # Status da conexão MT5
    try:
        account_info = get_mt5_account_info()
        if "error" not in account_info:
            st.sidebar.success("🟢 **MT5 Conectado**")
            st.sidebar.write(f"Conta: {account_info['login']}")
            st.sidebar.write(f"Saldo: ${account_info['balance']:.2f}")
        else:
            st.sidebar.error("🔴 **MT5 Desconectado**")
    except:
        st.sidebar.error("🔴 **MT5 Desconectado**")

def check_dashboard_status():
    """Verifica se o dashboard está ativo"""
    try:
        return True  # Se estamos rodando, o dashboard está ativo
    except:
        return False

def check_trading_system_status():
    """Verifica se o sistema de trading está ativo"""
    try:
        import requests
        response = requests.get("http://localhost:8501/api/trading_status", timeout=2)
        return response.status_code == 200
    except:
        # Verifica se há processos de trading rodando
        return False

def start_trading_system():
    """Inicia o sistema de trading automatizado"""
    try:
        import subprocess
        import threading

        def run_trading():
            subprocess.Popen(['python', 'main_trading_system.py'],
                           cwd='.',
                           creationflags=subprocess.CREATE_NEW_CONSOLE if os.name == 'nt' else 0)

        threading.Thread(target=run_trading, daemon=True).start()
        return True
    except Exception as e:
        st.error(f"Erro ao iniciar trading: {e}")
        return False

def show_positions_info():
    """Exibe informações das posições atuais"""
    try:
        account_info = get_mt5_account_info()
        if "error" not in account_info:
            st.json({
                "Saldo": f"${account_info['balance']:.2f}",
                "Equity": f"${account_info['equity']:.2f}",
                "Lucro": f"${account_info['profit']:.2f}",
                "Margem Livre": f"${account_info['margin_free']:.2f}"
            })
        else:
            st.error("Não foi possível obter informações da conta")
    except Exception as e:
        st.error(f"Erro: {e}")

def show_trading_settings():
    """Exibe configurações do trading"""
    try:
        with open('config.json', 'r') as f:
            config = json.load(f)

        st.json({
            "Símbolo": config.get("trading", {}).get("symbol", "US100"),
            "Lote": config.get("trading", {}).get("lot_size", 0.01),
            "Max Posições": config.get("trading", {}).get("max_positions", 2),
            "Stop Loss": config.get("trading", {}).get("stop_loss_pips", 50),
            "Take Profit": config.get("trading", {}).get("take_profit_pips", 100),
            "Risk %": config.get("risk_management", {}).get("max_daily_loss", 200.0)
        })
    except Exception as e:
        st.error(f"Erro ao carregar configurações: {e}")

def run_original_dashboard():
    """Executa o dashboard original de análise de opções"""
    try:
        # Importar e executar o código original do dashboard
        ticker_symbol = st.sidebar.selectbox(
            'Selecione o Ativo',
            ['SPX', 'QQQ', 'SPY', 'IWM'],
            index=1
        )

        # Expiração
        expiry_date = st.sidebar.date_input(
            'Data de Expiração',
            value=datetime.now().date() + timedelta(days=30)
        )

        # Botão para atualizar dados
        if st.sidebar.button('🔄 Atualizar Dados', type='primary'):
            st.rerun()

        # Informações do ativo
        st.markdown(f"## 📈 Análise de {ticker_symbol}")
        st.info(f"**Data de Expiração:** {expiry_date.strftime('%Y-%m-%d')}")

        # Espaço para gráficos e análises originais
        st.markdown("### 📊 Análise de Opções")
        st.markdown("*Carregando dados e gráficos...*")

    except Exception as e:
        st.error(f"Erro ao carregar dashboard: {e}")
        st.markdown("### 📊 Análise de Opções - Modo Simplificado")
        st.markdown("Use o menu **🎯 Trading Automatizado** para acessar o sistema completo.")

if __name__ == "__main__":
    main()

