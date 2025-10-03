"""
Módulo de Dashboard de Trading para EzOptions
Fornece painel completo de acompanhamento e controle de negociações
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import json
import requests
import MetaTrader5 as mt5
import time
from typing import Dict, List, Optional

class TradingDashboard:
    """Dashboard completo para gerenciamento de trading automatizado"""

    def __init__(self):
        self.config = self._load_config()
        self.api_url = "http://localhost:8000"  # API do sistema de trading

    def _load_config(self) -> Dict:
        """Carrega configuração do sistema"""
        try:
            with open('config.json', 'r') as f:
                return json.load(f)
        except:
            return {}

    def get_mt5_account_info(self) -> Dict:
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

    def get_positions(self) -> List[Dict]:
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

    def get_trading_history(self, days: int = 7) -> List[Dict]:
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

    def get_system_status(self) -> Dict:
        """Obtém status do sistema de trading"""
        try:
            response = requests.get(f"{self.api_url}/status", timeout=5)
            if response.status_code == 200:
                return response.json()
            return {"error": "API not responding"}
        except:
            return {"error": "Cannot connect to trading system"}

    def start_trading_system(self) -> Dict:
        """Inicia sistema de trading"""
        try:
            response = requests.post(f"{self.api_url}/start", timeout=5)
            if response.status_code == 200:
                return response.json()
            return {"error": "Failed to start trading"}
        except:
            return {"error": "Cannot connect to trading system"}

    def stop_trading_system(self) -> Dict:
        """Para sistema de trading"""
        try:
            response = requests.post(f"{self.api_url}/stop", timeout=5)
            if response.status_code == 200:
                return response.json()
            return {"error": "Failed to stop trading"}
        except:
            return {"error": "Cannot connect to trading system"}

    def render_account_overview(self):
        """Renderiza painel de visão geral da conta"""
        st.subheader("📊 Visão Geral da Conta")

        account_info = self.get_mt5_account_info()

        if "error" in account_info:
            st.error(f"❌ Erro ao conectar com MT5: {account_info['error']}")
            return

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
                value=f"${account_info['margin_free']:.2f}",
                delta=None
            )

        # Informações adicionais
        col1, col2, col3 = st.columns(3)
        with col1:
            st.info(f"🏦 **Conta:** {account_info['login']}@{account_info['server']}")
        with col2:
            st.info(f"🎯 **Alavancagem:** 1:{account_info['leverage']}")
        with col3:
            st.info(f"💱 **Moeda:** {account_info['currency']}")

    def render_positions_table(self):
        """Renderiza tabela de posições abertas"""
        st.subheader("📋 Posições Abertas")

        positions = self.get_positions()

        if not positions:
            st.info("📭 Nenhuma posição aberta no momento")
            return

        # Converter para DataFrame
        df = pd.DataFrame(positions)

        # Formatar colunas
        df['Preço Entrada'] = df['price_open'].apply(lambda x: f"${x:.2f}")
        df['Preço Atual'] = df['price_current'].apply(lambda x: f"${x:.2f}")
        df['Volume'] = df['volume'].apply(lambda x: f"{x:.2f}")
        df['Lucro/Prej.'] = df['profit'].apply(lambda x: f"${x:.2f}")
        df['Hora'] = df['time'].dt.strftime('%H:%M:%S')

        # Colorir lucro/prejuízo
        def color_profit(val):
            if '+' in val:
                return 'background-color: #d4edda'
            elif '-' in val:
                return 'background-color: #f8d7da'
            return ''

        styled_df = df[['symbol', 'type', 'Volume', 'Preço Entrada', 'Preço Atual', 'Lucro/Prej.', 'Hora']].style.applymap(
            color_profit, subset=['Lucro/Prej.']
        )

        st.dataframe(styled_df, use_container_width=True)

        # Resumo
        total_profit = sum([pos['profit'] for pos in positions])
        st.metric("📊 Total de Posições", len(positions), f"${total_profit:.2f}")

    def render_trading_history(self):
        """Renderiza histórico de trades"""
        st.subheader("📜 Histórico de Trades")

        days = st.selectbox("Período:", [1, 3, 7, 30], index=2)

        history = self.get_trading_history(days)

        if not history:
            st.info(f"📭 Nenhum trade encontrado nos últimos {days} dias")
            return

        # Converter para DataFrame
        df = pd.DataFrame(history)
        df['Data'] = df['time'].dt.strftime('%Y-%m-%d %H:%M')
        df['Preço'] = df['price'].apply(lambda x: f"${x:.2f}")
        df['Volume'] = df['volume'].apply(lambda x: f"{x:.2f}")
        df['Lucro'] = df['profit'].apply(lambda x: f"${x:.2f}")

        st.dataframe(
            df[['Data', 'type', 'Volume', 'Preço', 'Lucro']],
            use_container_width=True
        )

        # Gráfico de P/L
        df['Profit_Cum'] = df['profit'].cumsum()
        fig = px.line(
            df,
            x='time',
            y='Profit_Cum',
            title='📈 Lucro Acumulado',
            labels={'time': 'Data/Hora', 'Profit_Cum': 'Lucro Acumulado ($)'}
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    def render_control_panel(self):
        """Renderiza painel de controle do sistema"""
        st.subheader("🎮 Controle do Sistema")

        # Status do sistema
        system_status = self.get_system_status()

        col1, col2 = st.columns(2)

        with col1:
            if system_status.get("running", False):
                st.success("🟢 **Sistema ATIVO**")
                st.info(f"📊 Sinais detectados: {system_status.get('signals_detected', 0)}")
                st.info(f"✅ Sinais executados: {system_status.get('signals_executed', 0)}")
            else:
                st.error("🔴 **Sistema INATIVO**")
                st.warning("📴 Nenhuma negociação sendo executada")

        with col2:
            st.info(f"🔄 Última atualização: {datetime.now().strftime('%H:%M:%S')}")
            if 'integration_score' in system_status:
                st.info(f"📈 Score Integração: {system_status['integration_score']}%")

        # Botões de controle
        st.markdown("---")

        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("🚀 Iniciar Trading", type="primary", use_container_width=True):
                result = self.start_trading_system()
                if "error" not in result:
                    st.success("✅ Sistema de trading iniciado com sucesso!")
                    st.rerun()
                else:
                    st.error(f"❌ Erro: {result['error']}")

        with col2:
            if st.button("⏹️ Parar Trading", type="secondary", use_container_width=True):
                result = self.stop_trading_system()
                if "error" not in result:
                    st.success("✅ Sistema de trading parado com sucesso!")
                    st.rerun()
                else:
                    st.error(f"❌ Erro: {result['error']}")

        with col3:
            if st.button("🔄 Atualizar", use_container_width=True):
                st.rerun()

    def render_performance_metrics(self):
        """Renderiza métricas de performance"""
        st.subheader("📊 Métricas de Performance")

        # Obter dados dos últimos 7 dias
        history = self.get_trading_history(7)

        if not history:
            st.info("📭 Sem dados suficientes para análise")
            return

        df = pd.DataFrame(history)

        # Métricas básicas
        total_trades = len(df)
        winning_trades = len(df[df['profit'] > 0])
        losing_trades = len(df[df['profit'] < 0])
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0

        total_profit = df['profit'].sum()
        avg_profit = df['profit'].mean()
        max_profit = df['profit'].max()
        max_loss = df['profit'].min()

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("📈 Total Trades", total_trades)
            st.metric("✅ Wins", winning_trades)
            st.metric("❌ Losses", losing_trades)

        with col2:
            st.metric("🎯 Win Rate", f"{win_rate:.1f}%")
            st.metric("💰 Total P/L", f"${total_profit:.2f}")
            st.metric("📊 Média P/L", f"${avg_profit:.2f}")

        with col3:
            st.metric("🏆 Max Profit", f"${max_profit:.2f}")
            st.metric("⚠️ Max Loss", f"${max_loss:.2f}")
            profit_factor = abs(df[df['profit'] > 0]['profit'].sum() / df[df['profit'] < 0]['profit'].sum()) if losing_trades > 0 else 0
            st.metric("📐 Profit Factor", f"{profit_factor:.2f}")

        with col4:
            # Gráfico de pizza Win/Loss
            fig = go.Figure(data=[go.Pie(labels=['Wins', 'Losses'],
                                        values=[winning_trades, losing_trades],
                                        hole=0.3)])
            fig.update_layout(title="Win/Loss Ratio")
            st.plotly_chart(fig, use_container_width=True)

    def render(self):
        """Renderiza dashboard completo"""
        st.markdown("# 🎯 Dashboard de Trading Automatizado")
        st.markdown("---")

        # Criar abas
        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 Visão Geral",
            "📋 Posições",
            "📜 Histórico",
            "⚙️ Controle"
        ])

        with tab1:
            self.render_account_overview()
            self.render_performance_metrics()

        with tab2:
            self.render_positions_table()

        with tab3:
            self.render_trading_history()

        with tab4:
            self.render_control_panel()

            # Configurações
            st.markdown("---")
            st.subheader("⚙️ Configurações")

            st.json(self.config)