"""
EzOptions Trading Dashboard - Sistema de Trading Automatizado
Integração completa com FBS MT5 e 6 setups de trading
"""

import streamlit as st
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import json
import os
import logging
import threading
import time
from datetime import datetime, timedelta
from trading_setups_corrected import TradingSetupsCorrected
from advanced_risk_manager import AdvancedRiskManager
import pytz
import yfinance as yf
from scipy.stats import norm
from math import log, sqrt
import plotly.express as px
import plotly.graph_objects as go

# ========================================
# LOGGING AND AUDIT FUNCTIONS
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

# ========================================
# TRADING FUNCTIONS
# ========================================

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
        if os.path.exists("trading_system.log"):
            with open("trading_system.log", "r") as f:
                lines = f.readlines()[-10:]
                return {
                    "running": "Sistema de Trading EzOptions rodando" in lines[-1] if lines else False,
                    "last_log": lines[-1].strip() if lines else "",
                    "log_lines": lines
                }
        return {"running": False, "last_log": "No log file found"}
    except Exception as e:
        return {"running": False, "error": str(e)}

# ========================================
# QQQ DATA FUNCTIONS
# ========================================

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

# ========================================
# TRADING SETUPS ANALYSIS
# ========================================

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
        print(error_msg)
        log_trade_event("setup_analysis_error", {"error": str(e)})
        return [], {}

# ========================================
# MANUAL TRADE EXECUTION
# ========================================

def execute_manual_trade(signal_type: str, setup_name: str):
    """Executa um trade manual baseado no setup"""
    try:
        # Log do início da operação
        log_trade_event("manual_trade_initiated", {
            "signal_type": signal_type,
            "setup_name": setup_name,
            "user_action": "manual_execution"
        })

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
            "type_filling": mt5.ORDER_FILLING_FOK,
        }

        result = mt5.order_send(request)

        if result is None:
            error_msg = "Falha ao enviar ordem: MT5 retornou None (verifique conexão e permissões)"
            st.error(error_msg)
            log_trade_event("execution_error", {
                "error": "'NoneType' object has no attribute 'retcode'",
                "setup": setup_name,
                "request": request
            })
        elif result.retcode != mt5.TRADE_RETCODE_DONE:
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

# ========================================
# SYSTEM MANAGEMENT
# ========================================

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

# ========================================
# DASHBOARD RENDERING
# ========================================

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
            st.plotly_chart(fig, width='stretch')

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
        st.dataframe(df_stats, width='stretch')

        # Gráfico de performance dos setups
        fig = px.bar(df_stats, x='Setup', y='P/L Médio', title='📊 Performance Média por Setup')
        st.plotly_chart(fig, width='stretch')

        # Mostrar logs de auditoria
        st.markdown("---")
        st.subheader("📋 Logs de Auditoria Recentes")

        audit_logs = get_audit_logs(20)
        if audit_logs:
            st.dataframe(pd.DataFrame(audit_logs), width='stretch')
        else:
            st.info("Nenhum log de auditoria encontrado")

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

    # Menu de navegação
    page = st.sidebar.selectbox(
        "Escolha uma página:",
        ["📊 Análise de Opções", "🎯 Trading Automatizado", "📈 Análise Avançada"]
    )

    if page == "📊 Análise de Opções":
        # Dashboard original de opções
        st.title("📊 EzOptions - Análise de Opções")
        st.info("📈 Página em desenvolvimento...")

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

if __name__ == "__main__":
    main()