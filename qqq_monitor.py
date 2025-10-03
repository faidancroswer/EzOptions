"""
Módulo de Monitoramento QQQ
Responsável por detectar sinais válidos no módulo 'qqq' do ezoptionsverifiar 
e executar negociações automaticamente no MT5 para o ativo US100
"""

import logging
import time
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import threading
import requests
from dataclasses import dataclass, asdict
from enum import Enum
import hashlib

logger = logging.getLogger(__name__)

def get_next_option_expiry():
    """Calcula a próxima data de expiração válida para opções QQQ"""
    import yfinance as yf
    try:
        # Obter datas de expiração disponíveis
        ticker = yf.Ticker("QQQ")
        expirations = ticker.options

        if not expirations:
            # Fallback: usar data fixa válida (hoje + 7 dias, ou próxima sexta-feira)
            target_date = datetime.now() + timedelta(days=7)
            while target_date.weekday() != 4:  # 4 = Friday
                target_date += timedelta(days=1)
            return target_date.strftime('%Y-%m-%d')

        # Encontrar a próxima expiração válida
        today = datetime.now().date()
        for exp_str in expirations:
            exp_date = datetime.strptime(exp_str, '%Y-%m-%d').date()
            if exp_date > today:
                return exp_str

        # Se nenhuma encontrada, usar a primeira disponível
        return expirations[0] if expirations else (datetime.now() + timedelta(days=7)).strftime('%Y-%m-%d')

    except Exception as e:
        logger.warning(f"Erro ao obter datas de expiração: {e}")
        # Fallback seguro - usar data mais curta para evitar expirações inexistentes
        target_date = datetime.now() + timedelta(days=7)
        while target_date.weekday() != 4:
            target_date += timedelta(days=1)
        return target_date.strftime('%Y-%m-%d')

class SignalStatus(Enum):
    """Status dos sinais detectados"""
    PENDING = "pending"
    CONFIRMED = "confirmed"
    EXECUTED = "executed"
    EXPIRED = "expired"
    CANCELLED = "cancelled"

@dataclass
class TradingSignal:
    """Estrutura de sinal de negociação"""
    setup_number: int
    signal_type: str  # 'BUY' ou 'SELL'
    confidence: float
    timestamp: datetime
    expiry_time: datetime
    status: SignalStatus
    market_data: Dict
    execution_price: Optional[float] = None
    order_id: Optional[str] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    risk_reward_ratio: Optional[float] = None

    def to_dict(self) -> Dict:
        """Converte sinal para dicionário"""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        data['expiry_time'] = self.expiry_time.isoformat()
        data['status'] = self.status.value
        return data

class QQQMonitor:
    """
    Monitor do módulo QQQ para detectar sinais válidos
    """

    def __init__(self, config: Dict):
        self.config = config
        self.ezoptions_url = config.get('ezoptions', {}).get('url', 'http://localhost:8501')
        self.ticker = config.get('trading_parameters', {}).get('ticker', 'US100')
        self.active_setups = config.get('trading_parameters', {}).get('active_setups', [1, 2, 3, 4, 5, 6])

        # Configurações de monitoramento
        self.check_interval = config.get('trading_parameters', {}).get('check_interval', 30)
        self.signal_expiry_minutes = config.get('signal_expiry_minutes', 15)
        self.min_confidence_threshold = config.get('min_confidence_threshold', 0.8)

        # Estado do monitor
        self.running = False
        self.signals_history: List[TradingSignal] = []
        self.active_signals: Dict[str, TradingSignal] = {}
        self.last_check = None
        self.session = requests.Session()

        # Callbacks
        self.on_signal_detected = None
        self.on_signal_confirmed = None
        self.on_signal_executed = None

        # Lock para thread safety
        self._lock = threading.Lock()

    def start_monitoring(self):
        """Inicia monitoramento do módulo QQQ"""
        if self.running:
            logger.warning("Monitor QQQ já está rodando")
            return

        logger.info("Iniciando monitoramento do módulo QQQ...")
        self.running = True

        # Iniciar thread de monitoramento
        monitor_thread = threading.Thread(target=self._monitoring_loop)
        monitor_thread.daemon = True
        monitor_thread.start()

        logger.info("Monitor QQQ iniciado com sucesso")

    def stop_monitoring(self):
        """Para monitoramento do módulo QQQ"""
        logger.info("Parando monitoramento do módulo QQQ...")
        self.running = False

        # Cancelar sinais ativos
        with self._lock:
            for signal_id, signal in self.active_signals.items():
                if signal.status in [SignalStatus.PENDING, SignalStatus.CONFIRMED]:
                    signal.status = SignalStatus.CANCELLED
                    logger.info(f"Sinal {signal_id} cancelado")

        logger.info("Monitor QQQ parado")

    def _monitoring_loop(self):
        """Loop principal de monitoramento"""
        while self.running:
            try:
                # Verificar sinais
                self._check_for_signals()

                # Limpar sinais expirados
                self._cleanup_expired_signals()

                # Aguardar próximo ciclo
                time.sleep(self.check_interval)

            except Exception as e:
                logger.error(f"Erro no loop de monitoramento: {e}")
                time.sleep(self.check_interval * 2)  # Aguardar mais tempo em caso de erro

    def _check_for_signals(self):
        """Verifica módulo QQQ por sinais válidos"""
        try:
            # Buscar dados atuais do ezoptions
            market_data = self._fetch_ezoptions_data()
            if not market_data:
                logger.warning("Não foi possível obter dados do ezoptions")
                return

            # Verificar cada setup ativo
            for setup_number in self.active_setups:
                try:
                    # Analisar setup específico
                    signal_detected = self._analyze_setup_for_signal(setup_number, market_data)

                    if signal_detected:
                        logger.info(f"Sinal detectado no Setup {setup_number}: {signal_detected.get('signal_type', 'UNKNOWN')}")

                        # Criar objeto de sinal
                        signal = self._create_trading_signal(setup_number, signal_detected, market_data)

                        # Adicionar aos sinais ativos
                        with self._lock:
                            self.active_signals[signal_id] = signal

                        # Callback para sinal detectado
                        if self.on_signal_detected:
                            self.on_signal_detected(signal)

                except Exception as e:
                    logger.error(f"Erro ao analisar Setup {setup_number}: {e}")

        except Exception as e:
            logger.error(f"Erro ao verificar sinais: {e}")

    def _fetch_ezoptions_data(self) -> Optional[Dict]:
        """Busca dados diretos do Yahoo Finance (API externa original do ezOptions)"""
        try:
            # Usar o EzOptionsAPIConnector para acessar Yahoo Finance diretamente
            from ezoptions_connector import EzOptionsAPIConnector

            # Inicializar conector com Yahoo Finance
            api_connector = EzOptionsAPIConnector()

            # Obter dados do QQQ do Yahoo Finance
            current_price = api_connector.get_current_price("QQQ")
            if current_price is None:
                logger.error("Não foi possível obter preço atual do QQQ")
                return None

            # Obter dados de Greeks (calculados a partir das opções do Yahoo Finance)
            expiry_date = get_next_option_expiry()
            greeks_data = api_connector.get_greeks_data("QQQ", expiry_date)

            # Obter indicadores técnicos
            technical_indicators = api_connector.get_technical_indicators("QQQ")

            # Obter cadeia de opções
            options_chain = api_connector.get_options_chain("QQQ", expiry_date)

            # Processar dados de volume
            volume_data = {
                'total': options_chain.get('calls', [{}])[0].get('volume', 0) +
                        options_chain.get('puts', [{}])[0].get('volume', 0),
                'call_volume': sum(call.get('volume', 0) for call in options_chain.get('calls', [])),
                'put_volume': sum(put.get('volume', 0) for put in options_chain.get('puts', [])),
            }
            volume_data['put_call_ratio'] = (volume_data['put_volume'] /
                                           volume_data['call_volume']
                                           if volume_data['call_volume'] > 0 else 1.0)

            # Formatar dados de VWAP e Bollinger Bands
            vwap_data = {
                'vwap': technical_indicators.get('vwap', current_price),
                'upper_band': technical_indicators.get('bollinger_upper', current_price * 1.02),
                'lower_band': technical_indicators.get('bollinger_lower', current_price * 0.98),
                'middle_band': technical_indicators.get('bollinger_middle', current_price),
                'position': (current_price - technical_indicators.get('bollinger_lower', current_price * 0.98)) /
                           (technical_indicators.get('bollinger_upper', current_price * 1.02) -
                            technical_indicators.get('bollinger_lower', current_price * 0.98))
            }

            # Formatar Greeks para o formato esperado
            formatted_greeks = {
                'gamma': {
                    'current': greeks_data.get('gamma', 0.01),
                    'max_level': greeks_data.get('gamma', 0.01) * 1.2,
                    'max_level_price': current_price + 2,
                    'levels': [current_price - 2, current_price - 1, current_price, current_price + 1, current_price + 2]
                },
                'delta': {
                    'current': greeks_data.get('delta', 0.5),
                    'max_level': min(0.9, greeks_data.get('delta', 0.5) + 0.1),
                    'max_level_price': current_price + 1.5,
                    'positive_bars_upward': current_price > technical_indicators.get('vwap', current_price),
                    'negative_bars_downward': current_price < technical_indicators.get('vwap', current_price),
                    'levels': [current_price - 1.5, current_price - 0.5, current_price + 0.5, current_price + 1.5]
                },
                'charm': {
                    'current': greeks_data.get('charm', 0.001),
                    'max_level': greeks_data.get('charm', 0.001) * 1.5,
                    'max_level_price': current_price + (1 if greeks_data.get('charm', 0.001) > 0 else -1),
                    'growing_trend': current_price > technical_indicators.get('vwap', current_price),
                    'direction_up': current_price > technical_indicators.get('vwap', current_price),
                    'decreasing_trend': current_price < technical_indicators.get('vwap', current_price),
                    'direction_down': current_price < technical_indicators.get('vwap', current_price),
                    'flip_zone': abs(current_price - technical_indicators.get('vwap', current_price)) < 0.5,
                    'levels': [current_price - 1, current_price + 1]
                },
                'theta': {
                    'current': -0.05 * (1 + abs(current_price - 450) / 100),
                    'max_level': -0.03,
                    'max_level_price': current_price - 0.5
                }
            }

            # Retornar dados no formato esperado pelo sistema
            return {
                'current_price': current_price,
                'greeks_data': formatted_greeks,
                'vwap_data': vwap_data,
                'volume_data': volume_data,
                'options_data': options_chain,
                'technical_indicators': technical_indicators,
                'timestamp': datetime.now()
            }

        except Exception as e:
            logger.error(f"Erro ao buscar dados do Yahoo Finance: {e}")
            return None

    def _analyze_setup_for_signal(self, setup_number: int, market_data: Dict) -> Optional[Dict]:
        """Analisa setup específico em busca de sinais"""
        try:
            from trading_setups_corrected import TradingSetupsCorrected

            # Inicializar detector de setups
            setup_detector = TradingSetupsCorrected()

            # Mapear setups para métodos de análise
            analysis_methods = {
                1: setup_detector.analyze_bullish_breakout,
                2: setup_detector.analyze_bearish_breakout,
                3: setup_detector.analyze_pullback_top,
                4: setup_detector.analyze_pullback_bottom,
                5: setup_detector.analyze_consolidated_market,
                6: setup_detector.analyze_negative_gamma_protection
            }

            if setup_number not in analysis_methods:
                logger.warning(f"Setup {setup_number} não encontrado")
                return None

            # Executar análise
            result = analysis_methods[setup_number](market_data)

            # Verificar se sinal foi confirmado
            if result.get('confirmed', False):
                confidence = result.get('confidence', 0)

                # Verificar threshold de confiança
                if confidence >= self.min_confidence_threshold:
                    return {
                        'signal_type': result.get('signal', 'BUY'),
                        'confidence': confidence,
                        'setup_name': result.get('name', f'Setup {setup_number}'),
                        'analysis_result': result
                    }

            return None

        except Exception as e:
            logger.error(f"Erro ao analisar setup {setup_number}: {e}")
            return None

    def _create_trading_signal(self, setup_number: int, signal_detected: Dict, market_data: Dict) -> TradingSignal:
        """Cria objeto TradingSignal"""
        signal_id = self._generate_signal_id(setup_number, signal_detected, market_data)

        # Calcular tempo de expiração
        expiry_time = datetime.now() + timedelta(minutes=self.signal_expiry_minutes)

        # Calcular níveis de stop loss e take profit
        current_price = market_data.get('current_price', 0)
        stop_loss, take_profit = self._calculate_sl_tp(
            setup_number,
            signal_detected['signal_type'],
            current_price,
            signal_detected['confidence']
        )

        # Calcular relação risco-recompensa
        risk_reward_ratio = self._calculate_risk_reward_ratio(
            current_price, stop_loss, take_profit, signal_detected['signal_type']
        )

        return TradingSignal(
            setup_number=setup_number,
            signal_type=signal_detected['signal_type'],
            confidence=signal_detected['confidence'],
            timestamp=datetime.now(),
            expiry_time=expiry_time,
            status=SignalStatus.PENDING,
            market_data=market_data,
            stop_loss=stop_loss,
            take_profit=take_profit,
            risk_reward_ratio=risk_reward_ratio
        )

    def _generate_signal_id(self, setup_number: int, signal_detected: Dict, market_data: Dict) -> str:
        """Gera ID único para o sinal"""
        # Criar hash baseado nos dados do sinal para evitar duplicatas
        signal_data = f"{setup_number}_{signal_detected['signal_type']}_{market_data.get('current_price', 0)}_{datetime.now().strftime('%Y%m%d%H%M')}"

        # Usar apenas primeiros 60 minutos do dia para agrupar sinais similares
        current_minute = datetime.now().minute
        grouped_minute = (current_minute // 15) * 15  # Agrupar em blocos de 15 minutos
        signal_data = signal_data[:signal_data.rfind('_') + 1] + f"{grouped_minute:02d}"

        return hashlib.md5(signal_data.encode()).hexdigest()[:8]

    def _calculate_sl_tp(self, setup_number: int, signal_type: str, current_price: float, confidence: float) -> Tuple[float, float]:
        """Calcula níveis de stop loss e take profit"""
        try:
            # Base de cálculo: 1% do preço atual
            base_distance = current_price * 0.01

            # Ajustar baseado na confiança
            confidence_multiplier = 1 + (confidence - 0.8) * 2.5  # Multiplicador de 1x a 3.75x

            # Ajustar baseado no setup
            setup_multipliers = {
                1: 1.0,  # Bullish Breakout
                2: 1.2,  # Bearish Breakout
                3: 0.8,  # Pullback Topo
                4: 0.8,  # Pullback Fundo
                5: 0.6,  # Consolidação
                6: 1.5   # Proteção
            }

            setup_multiplier = setup_multipliers.get(setup_number, 1.0)
            total_multiplier = confidence_multiplier * setup_multiplier

            # Calcular distância ajustada
            adjusted_distance = base_distance * total_multiplier

            if signal_type == 'BUY':
                stop_loss = current_price - adjusted_distance
                take_profit = current_price + (adjusted_distance * 2)  # 2:1 risk-reward mínimo
            else:  # SELL
                stop_loss = current_price + adjusted_distance
                take_profit = current_price - (adjusted_distance * 2)

            return round(stop_loss, 2), round(take_profit, 2)

        except Exception as e:
            logger.error(f"Erro ao calcular SL/TP: {e}")
            # Valores padrão
            if signal_type == 'BUY':
                return current_price * 0.99, current_price * 1.02
            else:
                return current_price * 1.01, current_price * 0.98

    def _calculate_risk_reward_ratio(self, entry_price: float, stop_loss: float, take_profit: float, signal_type: str) -> float:
        """Calcula relação risco-recompensa"""
        try:
            if signal_type == 'BUY':
                risk = entry_price - stop_loss
                reward = take_profit - entry_price
            else:
                risk = stop_loss - entry_price
                reward = entry_price - take_profit

            if risk == 0:
                return 0

            return reward / risk

        except:
            return 2.0  # Valor padrão

    def _cleanup_expired_signals(self):
        """Remove sinais expirados"""
        current_time = datetime.now()

        with self._lock:
            expired_signals = []

            for signal_id, signal in self.active_signals.items():
                if signal.status in [SignalStatus.PENDING, SignalStatus.CONFIRMED]:
                    if current_time > signal.expiry_time:
                        signal.status = SignalStatus.EXPIRED
                        expired_signals.append(signal_id)
                        logger.info(f"Sinal {signal_id} expirado")

            # Remover sinais expirados
            for signal_id in expired_signals:
                self.active_signals.pop(signal_id, None)

    def get_active_signals(self) -> List[Dict]:
        """Retorna sinais ativos"""
        with self._lock:
            return [signal.to_dict() for signal in self.active_signals.values()]

    def get_signal_history(self, limit: int = 100) -> List[Dict]:
        """Retorna histórico de sinais"""
        return [signal.to_dict() for signal in self.signals_history[-limit:]]

    def confirm_signal(self, signal_id: str) -> bool:
        """Confirma sinal para execução"""
        with self._lock:
            if signal_id in self.active_signals:
                signal = self.active_signals[signal_id]

                if signal.status == SignalStatus.PENDING:
                    signal.status = SignalStatus.CONFIRMED

                    # Callback para sinal confirmado
                    if self.on_signal_confirmed:
                        self.on_signal_confirmed(signal)

                    logger.info(f"Sinal {signal_id} confirmado para execução")
                    return True

        return False

    def mark_signal_executed(self, signal_id: str, order_id: str, execution_price: float) -> bool:
        """Marca sinal como executado"""
        with self._lock:
            if signal_id in self.active_signals:
                signal = self.active_signals[signal_id]

                if signal.status == SignalStatus.CONFIRMED:
                    signal.status = SignalStatus.EXECUTED
                    signal.order_id = order_id
                    signal.execution_price = execution_price

                    # Mover para histórico
                    self.signals_history.append(signal)
                    if len(self.signals_history) > 1000:
                        self.signals_history = self.signals_history[-1000:]

                    # Remover dos ativos
                    del self.active_signals[signal_id]

                    # Callback para sinal executado
                    if self.on_signal_executed:
                        self.on_signal_executed(signal)

                    logger.info(f"Sinal {signal_id} executado - Ordem: {order_id}")
                    return True

        return False

    def get_monitoring_status(self) -> Dict:
        """Retorna status do monitoramento"""
        return {
            'running': self.running,
            'last_check': self.last_check.isoformat() if self.last_check else None,
            'active_signals_count': len(self.active_signals),
            'total_signals_today': len([s for s in self.signals_history if s.timestamp.date() == datetime.now().date()]),
            'ticker': self.ticker,
            'active_setups': self.active_setups,
            'check_interval': self.check_interval
        }

class SignalExecutor:
    """
    Executor de sinais para integração com MT5
    """

    def __init__(self, mt5_integration, risk_manager, config: Dict):
        self.mt5_integration = mt5_integration
        self.risk_manager = risk_manager
        self.config = config

        # Configurações de execução
        self.default_lot_size = config.get('trading', {}).get('lot_size', 0.01)
        self.max_positions = config.get('trading', {}).get('max_positions', 1)

        # Estado de execução
        self.executed_orders = {}
        self.execution_log = []

    def execute_signal(self, signal: TradingSignal) -> Dict:
        """Executa sinal de negociação"""
        try:
            # Verificar se já foi executado
            if signal.order_id:
                return {'success': False, 'error': 'Sinal já executado'}

            # Verificar se sinal ainda é válido
            if signal.status != SignalStatus.CONFIRMED:
                return {'success': False, 'error': 'Sinal não confirmado'}

            # Verificar expiração
            if datetime.now() > signal.expiry_time:
                return {'success': False, 'error': 'Sinal expirado'}

            # Verificar limite de posições
            current_positions = self.mt5_integration.get_positions()
            if len(current_positions) >= self.max_positions:
                return {'success': False, 'error': 'Limite máximo de posições atingido'}

            # Calcular tamanho da posição baseado no risco
            account_info = self.mt5_integration.get_account_info()
            position_size = self._calculate_position_size(signal, account_info)

            # Preparar parâmetros da ordem
            current_price = signal.market_data.get('current_price', 0)

            order_params = {
                'symbol': self.config.get('trading', {}).get('symbol', 'US100'),
                'order_type': signal.signal_type,
                'volume': position_size,
                'price': current_price,
                'sl': signal.stop_loss,
                'tp': signal.take_profit
            }

            # Executar ordem
            result = self.mt5_integration.place_order(**order_params)

            if result['success']:
                # Marcar sinal como executado
                signal.order_id = result['order_id']
                signal.execution_price = result.get('price', current_price)

                # Registrar execução
                self._log_execution(signal, result)

                logger.info(f"Sinal executado com sucesso: {result}")
                return result
            else:
                logger.error(f"Falha na execução do sinal: {result}")
                return result

        except Exception as e:
            error_msg = f"Erro ao executar sinal: {e}"
            logger.error(error_msg)
            return {'success': False, 'error': error_msg}

    def _calculate_position_size(self, signal: TradingSignal, account_info: Dict) -> float:
        """Calcula tamanho da posição baseado no gerenciamento de risco"""
        try:
            account_balance = account_info.get('balance', 10000)

            # Tamanho base
            base_size = self.default_lot_size

            # Ajustar baseado na confiança
            confidence_multiplier = 1 + (signal.confidence - 0.8) * 1.5  # 1x a 2.5x

            # Ajustar baseado no setup
            setup_multipliers = {
                1: 1.0,  # Bullish Breakout
                2: 1.2,  # Bearish Breakout
                3: 0.8,  # Pullback Topo
                4: 0.8,  # Pullback Fundo
                5: 0.6,  # Consolidação
                6: 1.5   # Proteção
            }

            setup_multiplier = setup_multipliers.get(signal.setup_number, 1.0)
            total_multiplier = confidence_multiplier * setup_multiplier

            # Calcular tamanho final
            position_size = base_size * total_multiplier

            # Limitar a 5% do capital total
            max_size = (account_balance * 0.05) / 1000  # Assumindo contrato de $100k
            position_size = min(position_size, max_size)

            return round(position_size, 2)

        except Exception as e:
            logger.error(f"Erro ao calcular tamanho da posição: {e}")
            return self.default_lot_size

    def _log_execution(self, signal: TradingSignal, result: Dict):
        """Registra execução para auditoria"""
        try:
            execution_record = {
                'timestamp': datetime.now(),
                'signal_id': id(signal),
                'setup_number': signal.setup_number,
                'signal_type': signal.signal_type,
                'confidence': signal.confidence,
                'execution_price': signal.execution_price,
                'volume': result.get('volume', 0),
                'order_id': result.get('order_id'),
                'stop_loss': signal.stop_loss,
                'take_profit': signal.take_profit,
                'risk_reward_ratio': signal.risk_reward_ratio
            }

            self.execution_log.append(execution_record)

            # Manter apenas últimos 1000 registros
            if len(self.execution_log) > 1000:
                self.execution_log = self.execution_log[-1000:]

        except Exception as e:
            logger.error(f"Erro ao registrar execução: {e}")

    def get_execution_summary(self) -> Dict:
        """Retorna resumo das execuções"""
        try:
            if not self.execution_log:
                return {'total_executions': 0}

            # Calcular estatísticas
            total_executions = len(self.execution_log)
            executions_today = len([e for e in self.execution_log
                                  if e['timestamp'].date() == datetime.now().date()])

            # Estatísticas por tipo de sinal
            buy_executions = len([e for e in self.execution_log if e['signal_type'] == 'BUY'])
            sell_executions = len([e for e in self.execution_log if e['signal_type'] == 'SELL'])

            # Estatísticas por setup
            setup_stats = {}
            for execution in self.execution_log:
                setup = execution['setup_number']
                if setup not in setup_stats:
                    setup_stats[setup] = 0
                setup_stats[setup] += 1

            return {
                'total_executions': total_executions,
                'executions_today': executions_today,
                'buy_executions': buy_executions,
                'sell_executions': sell_executions,
                'setup_distribution': setup_stats,
                'last_execution': self.execution_log[-1] if self.execution_log else None
            }

        except Exception as e:
            logger.error(f"Erro ao gerar resumo de execuções: {e}")
            return {'error': str(e)}

# Função para criar e configurar o sistema completo
def create_qqq_monitoring_system(config_path: str = 'config.json'):
    """Cria sistema completo de monitoramento QQQ"""
    try:
        # Carregar configuração
        with open(config_path, 'r') as f:
            config = json.load(f)

        # Inicializar componentes
        from trading_system import MT5Integration, RiskManager

        mt5_integration = MT5Integration(config.get('mt5', {}))
        risk_manager = RiskManager(config.get('risk_management', {}))
        signal_executor = SignalExecutor(mt5_integration, risk_manager, config)

        # Inicializar monitor QQQ
        qqq_monitor = QQQMonitor(config)

        # Configurar callbacks
        def on_signal_detected(signal):
            logger.info(f"Sinal detectado: Setup {signal.setup_number}, {signal.signal_type}, Confiança: {signal.confidence:.2f}")

        def on_signal_confirmed(signal):
            logger.info(f"Sinal confirmado para execução: {signal.signal_type} no preço {signal.market_data.get('current_price', 0)}")

            # Tentar executar automaticamente
            result = signal_executor.execute_signal(signal)
            if result['success']:
                logger.info(f"Sinal executado automaticamente: Ordem {result.get('order_id')}")
            else:
                logger.error(f"Falha na execução automática: {result.get('error')}")

        def on_signal_executed(signal):
            logger.info(f"Sinal executado: Ordem {signal.order_id} no preço {signal.execution_price}")

        qqq_monitor.on_signal_detected = on_signal_detected
        qqq_monitor.on_signal_confirmed = on_signal_confirmed
        qqq_monitor.on_signal_executed = on_signal_executed

        return {
            'qqq_monitor': qqq_monitor,
            'signal_executor': signal_executor,
            'mt5_integration': mt5_integration,
            'risk_manager': risk_manager
        }

    except Exception as e:
        logger.error(f"Erro ao criar sistema de monitoramento QQQ: {e}")
        return None

if __name__ == "__main__":
    # Exemplo de uso
    system = create_qqq_monitoring_system()

    if system:
        qqq_monitor = system['qqq_monitor']

        # Iniciar monitoramento
        qqq_monitor.start_monitoring()

        try:
            # Manter sistema rodando
            while True:
                time.sleep(60)

                # Mostrar status
                status = qqq_monitor.get_monitoring_status()
                print(f"Status: {status}")

        except KeyboardInterrupt:
            print("Parando sistema...")
            qqq_monitor.stop_monitoring()
    else:
        print("Erro ao inicializar sistema")