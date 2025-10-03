"""
Sistema Avançado de Gerenciamento de Risco
Implementa regras específicas: 1% max stop loss, 2:1 risk-reward, 5% max exposição
"""

import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import threading
from dataclasses import dataclass, asdict
from enum import Enum

logger = logging.getLogger(__name__)

class RiskStatus(Enum):
    """Status de risco"""
    SAFE = "safe"
    WARNING = "warning"
    DANGER = "danger"
    CRITICAL = "critical"

@dataclass
class RiskMetrics:
    """Métricas de risco"""
    account_balance: float
    current_exposure: float
    daily_pnl: float
    daily_trades: int
    consecutive_losses: int
    max_drawdown: float
    risk_status: RiskStatus
    timestamp: datetime

    def to_dict(self) -> Dict:
        """Converte métricas para dicionário"""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        data['risk_status'] = self.risk_status.value
        return data

class AdvancedRiskManager:
    """
    Sistema avançado de gerenciamento de risco com regras específicas
    """

    def __init__(self, config: Dict):
        self.config = config

        # Regras específicas solicitadas
        self.max_stop_loss_percent = 1.0  # 1% máximo do saldo
        self.min_risk_reward_ratio = 2.0   # 2:1 mínimo
        self.max_exposure_percent = 5.0    # 5% máximo do capital
        self.max_daily_loss_percent = 2.0  # 2% perda diária máxima

        # Configurações adicionais
        self.max_consecutive_losses = config.get('risk_management', {}).get('max_consecutive_losses', 3)
        self.max_daily_trades = config.get('risk_management', {}).get('max_daily_trades', 10)
        self.profit_target_percent = config.get('risk_management', {}).get('daily_profit_target', 1.0)

        # Estado do sistema
        self.account_balance = 10000.0  # Valor padrão
        self.current_exposure = 0.0
        self.daily_pnl = 0.0
        self.daily_trades = 0
        self.consecutive_losses = 0
        self.max_drawdown = 0.0
        self.peak_balance = 10000.0

        # Histórico
        self.risk_history: List[RiskMetrics] = []
        self.trade_log: List[Dict] = []

        # Controle de exposição por setup
        self.setup_exposure = {}
        self.symbol_exposure = {}

        # Estado de trading
        self.trading_enabled = True
        self.last_reset_date = datetime.now().date()

        # Lock para thread safety
        self._lock = threading.Lock()

    def validate_signal(self, signal: Dict) -> Tuple[bool, str, Dict]:
        """
        Valida sinal contra regras de risco específicas

        Args:
            signal: Sinal de trading com parâmetros completos

        Returns:
            Tuple (válido, razão, detalhes_risco)
        """
        try:
            with self._lock:
                # Reset diário se necessário
                self._check_daily_reset()

                # 1. Verificar se trading está habilitado
                if not self.trading_enabled:
                    return False, "Trading desabilitado por regras de risco", {}

                # 2. Verificar perda diária máxima (2%)
                max_daily_loss = self.account_balance * (self.max_daily_loss_percent / 100)
                if self.daily_pnl <= -max_daily_loss:
                    self.trading_enabled = False
                    return False, f"Limite de perda diária atingido: ${self.daily_pnl:.2f}", {
                        'rule': 'max_daily_loss',
                        'current_loss': self.daily_pnl,
                        'max_allowed': max_daily_loss
                    }

                # 3. Verificar número máximo de trades diários
                if self.daily_trades >= self.max_daily_trades:
                    return False, f"Limite de trades diários atingido: {self.daily_trades}", {
                        'rule': 'max_daily_trades',
                        'current_trades': self.daily_trades,
                        'max_allowed': self.max_daily_trades
                    }

                # 4. Verificar exposição máxima (5% do capital)
                signal_exposure = self._calculate_signal_exposure(signal)
                max_exposure = self.account_balance * (self.max_exposure_percent / 100)

                if self.current_exposure + signal_exposure > max_exposure:
                    return False, f"Exposição máxima seria excedida: ${self.current_exposure + signal_exposure:.2f}", {
                        'rule': 'max_exposure',
                        'current_exposure': self.current_exposure,
                        'signal_exposure': signal_exposure,
                        'max_allowed': max_exposure,
                        'would_be_exposure': self.current_exposure + signal_exposure
                    }

                # 5. Verificar stop loss máximo (1% do saldo)
                stop_loss_distance = self._calculate_stop_loss_distance(signal)
                max_stop_loss = self.account_balance * (self.max_stop_loss_percent / 100)

                if stop_loss_distance > max_stop_loss:
                    return False, f"Stop loss excede máximo permitido: ${stop_loss_distance:.2f}", {
                        'rule': 'max_stop_loss',
                        'stop_loss_distance': stop_loss_distance,
                        'max_allowed': max_stop_loss
                    }

                # 6. Verificar relação risco-recompensa mínima (2:1)
                risk_reward_ratio = self._calculate_risk_reward_ratio(signal)

                if risk_reward_ratio < self.min_risk_reward_ratio:
                    return False, f"Relação risco-recompensa abaixo do mínimo: {risk_reward_ratio:.2f}:1", {
                        'rule': 'min_risk_reward',
                        'current_ratio': risk_reward_ratio,
                        'min_required': self.min_risk_reward_ratio
                    }

                # 7. Verificar exposição por setup
                setup_number = signal.get('setup_number')
                if setup_number:
                    setup_max_exposure = self.account_balance * 0.02  # 2% por setup
                    current_setup_exposure = self.setup_exposure.get(setup_number, 0)

                    if current_setup_exposure + signal_exposure > setup_max_exposure:
                        return False, f"Exposição máxima por setup excedida para Setup {setup_number}", {
                            'rule': 'max_setup_exposure',
                            'setup': setup_number,
                            'current_setup_exposure': current_setup_exposure,
                            'signal_exposure': signal_exposure,
                            'max_allowed': setup_max_exposure
                        }

                # Todas as validações passaram
                risk_details = {
                    'signal_exposure': signal_exposure,
                    'new_total_exposure': self.current_exposure + signal_exposure,
                    'risk_reward_ratio': risk_reward_ratio,
                    'stop_loss_distance': stop_loss_distance,
                    'validation_passed': True
                }

                return True, "Sinal validado com sucesso", risk_details

        except Exception as e:
            error_msg = f"Erro na validação de risco: {e}"
            logger.error(error_msg)
            return False, error_msg, {}

    def _calculate_signal_exposure(self, signal: Dict) -> float:
        """Calcula exposição do sinal em dólares"""
        try:
            entry_price = signal.get('entry_price', 0)
            stop_loss = signal.get('stop_loss', 0)
            volume = signal.get('volume', 0.01)

            # Calcular distância do stop loss
            stop_distance = abs(entry_price - stop_loss)

            # Para US100 (NASDAQ-100), assumindo contrato de $100k
            # Exposição = distância do stop * volume * valor do contrato
            contract_value = 100000
            exposure = stop_distance * volume * contract_value

            return exposure

        except Exception as e:
            logger.error(f"Erro ao calcular exposição do sinal: {e}")
            return 0.0

    def _calculate_stop_loss_distance(self, signal: Dict) -> float:
        """Calcula distância do stop loss em dólares"""
        try:
            entry_price = signal.get('entry_price', 0)
            stop_loss = signal.get('stop_loss', 0)

            return abs(entry_price - stop_loss)

        except:
            return 0.0

    def _calculate_risk_reward_ratio(self, signal: Dict) -> float:
        """Calcula relação risco-recompensa"""
        try:
            entry_price = signal.get('entry_price', 0)
            stop_loss = signal.get('stop_loss', 0)
            take_profit = signal.get('take_profit', 0)

            # Calcular risco e recompensa
            risk = abs(entry_price - stop_loss)
            reward = abs(take_profit - entry_price)

            if risk == 0:
                return 0.0

            return reward / risk

        except:
            return 0.0

    def _check_daily_reset(self):
        """Verifica se precisa resetar contadores diários"""
        current_date = datetime.now().date()

        if current_date != self.last_reset_date:
            logger.info("Resetando contadores diários")

            # Salvar métricas finais do dia anterior
            self._record_risk_metrics()

            # Resetar contadores
            self.daily_pnl = 0.0
            self.daily_trades = 0
            self.last_reset_date = current_date

            # Reabilitar trading se estava desabilitado
            if not self.trading_enabled:
                self.trading_enabled = True
                logger.info("Trading reabilitado após reset diário")

    def _record_risk_metrics(self):
        """Registra métricas de risco para análise"""
        try:
            # Calcular drawdown
            current_drawdown = ((self.peak_balance - self.account_balance) / self.peak_balance) * 100

            # Determinar status de risco
            if current_drawdown > 5.0 or self.consecutive_losses >= 3:
                risk_status = RiskStatus.CRITICAL
            elif current_drawdown > 2.0 or self.daily_pnl < -100:
                risk_status = RiskStatus.DANGER
            elif current_drawdown > 1.0 or self.consecutive_losses >= 2:
                risk_status = RiskStatus.WARNING
            else:
                risk_status = RiskStatus.SAFE

            metrics = RiskMetrics(
                account_balance=self.account_balance,
                current_exposure=self.current_exposure,
                daily_pnl=self.daily_pnl,
                daily_trades=self.daily_trades,
                consecutive_losses=self.consecutive_losses,
                max_drawdown=current_drawdown,
                risk_status=risk_status,
                timestamp=datetime.now()
            )

            self.risk_history.append(metrics)

            # Manter apenas últimos 30 dias
            if len(self.risk_history) > 30:
                self.risk_history = self.risk_history[-30:]

        except Exception as e:
            logger.error(f"Erro ao registrar métricas de risco: {e}")

    def update_account_balance(self, new_balance: float):
        """Atualiza saldo da conta"""
        with self._lock:
            try:
                # Atualizar pico se necessário
                if new_balance > self.peak_balance:
                    self.peak_balance = new_balance

                self.account_balance = new_balance

                # Calcular novo drawdown
                if self.peak_balance > 0:
                    self.max_drawdown = ((self.peak_balance - self.account_balance) / self.peak_balance) * 100

                logger.info(f"Saldo atualizado: ${new_balance:.2f}")

            except Exception as e:
                logger.error(f"Erro ao atualizar saldo: {e}")

    def record_trade(self, signal: Dict, execution_result: Dict, pnl: float = None):
        """Registra execução de trade"""
        with self._lock:
            try:
                self.daily_trades += 1

                # Atualizar exposição
                if execution_result.get('success', False):
                    signal_exposure = self._calculate_signal_exposure(signal)
                    self.current_exposure += signal_exposure

                    # Atualizar exposição por setup
                    setup_number = signal.get('setup_number')
                    if setup_number:
                        self.setup_exposure[setup_number] = self.setup_exposure.get(setup_number, 0) + signal_exposure

                    # Atualizar exposição por símbolo
                    symbol = signal.get('symbol', 'US100')
                    self.symbol_exposure[symbol] = self.symbol_exposure.get(symbol, 0) + signal_exposure

                # Atualizar P&L diário
                if pnl is not None:
                    self.daily_pnl += pnl

                    # Verificar se foi gain ou loss
                    is_win = pnl > 0

                    if is_win:
                        self.consecutive_losses = 0
                    else:
                        self.consecutive_losses += 1

                        # Desabilitar trading após perdas consecutivas
                        if self.consecutive_losses >= self.max_consecutive_losses:
                            self.trading_enabled = False
                            logger.warning(f"Trading desabilitado após {self.consecutive_losses} perdas consecutivas")

                # Registrar no log
                trade_record = {
                    'timestamp': datetime.now(),
                    'setup_number': signal.get('setup_number'),
                    'signal_type': signal.get('signal_type'),
                    'volume': signal.get('volume'),
                    'entry_price': signal.get('entry_price'),
                    'stop_loss': signal.get('stop_loss'),
                    'take_profit': signal.get('take_profit'),
                    'execution_success': execution_result.get('success', False),
                    'pnl': pnl,
                    'account_balance': self.account_balance,
                    'exposure': self.current_exposure
                }

                self.trade_log.append(trade_record)

                # Manter apenas últimos 1000 trades
                if len(self.trade_log) > 1000:
                    self.trade_log = self.trade_log[-1000:]

                logger.info(f"Trade registrado - P&L: ${pnl}, Exposição: ${self.current_exposure:.2f}")

            except Exception as e:
                logger.error(f"Erro ao registrar trade: {e}")

    def close_position_update(self, pnl: float, symbol: str = 'US100'):
        """Atualiza métricas quando posição é fechada"""
        with self._lock:
            try:
                # Atualizar P&L diário
                self.daily_pnl += pnl

                # Reduzir exposição
                position_exposure = self.symbol_exposure.get(symbol, 0)

                # Estimar exposição da posição fechada (aproximado)
                if pnl != 0:
                    # Calcular exposição baseada no P&L
                    # Esta é uma aproximação - em produção seria melhor rastrear exposição exata por posição
                    estimated_exposure = abs(pnl) * 10  # Estimativa conservadora
                    self.current_exposure = max(0, self.current_exposure - estimated_exposure)

                    # Atualizar exposição do símbolo
                    self.symbol_exposure[symbol] = max(0, self.symbol_exposure.get(symbol, 0) - estimated_exposure)

                # Verificar se foi gain ou loss
                is_win = pnl > 0

                if is_win:
                    self.consecutive_losses = 0
                else:
                    self.consecutive_losses += 1

                    # Desabilitar trading após perdas consecutivas
                    if self.consecutive_losses >= self.max_consecutive_losses:
                        self.trading_enabled = False
                        logger.warning(f"Trading desabilitado após {self.consecutive_losses} perdas consecutivas")

                logger.info(f"Posição fechada - P&L: ${pnl:.2f}, Nova exposição: ${self.current_exposure:.2f}")

            except Exception as e:
                logger.error(f"Erro ao atualizar fechamento de posição: {e}")

    def get_risk_status(self) -> Dict:
        """Retorna status completo de risco"""
        with self._lock:
            try:
                # Calcular métricas atuais
                max_daily_loss = self.account_balance * (self.max_daily_loss_percent / 100)
                max_exposure = self.account_balance * (self.max_exposure_percent / 100)

                # Determinar status de risco
                if not self.trading_enabled:
                    risk_status = RiskStatus.CRITICAL
                elif self.daily_pnl <= -max_daily_loss * 0.8:  # 80% do limite
                    risk_status = RiskStatus.DANGER
                elif self.current_exposure >= max_exposure * 0.8:  # 80% da exposição máxima
                    risk_status = RiskStatus.WARNING
                else:
                    risk_status = RiskStatus.SAFE

                return {
                    'trading_enabled': self.trading_enabled,
                    'account_balance': self.account_balance,
                    'current_exposure': self.current_exposure,
                    'daily_pnl': self.daily_pnl,
                    'daily_trades': self.daily_trades,
                    'consecutive_losses': self.consecutive_losses,
                    'max_drawdown': self.max_drawdown,
                    'peak_balance': self.peak_balance,
                    'risk_status': risk_status.value,
                    'risk_metrics': {
                        'max_stop_loss_percent': self.max_stop_loss_percent,
                        'min_risk_reward_ratio': self.min_risk_reward_ratio,
                        'max_exposure_percent': self.max_exposure_percent,
                        'max_daily_loss': max_daily_loss,
                        'max_exposure': max_exposure,
                        'setup_exposure': self.setup_exposure,
                        'symbol_exposure': self.symbol_exposure
                    },
                    'last_update': datetime.now().isoformat()
                }

            except Exception as e:
                logger.error(f"Erro ao obter status de risco: {e}")
                return {'error': str(e)}

    def force_enable_trading(self):
        """Força habilitação de trading (uso administrativo)"""
        with self._lock:
            self.trading_enabled = True
            logger.warning("Trading habilitado manualmente")

    def force_disable_trading(self):
        """Força desabilitação de trading"""
        with self._lock:
            self.trading_enabled = False
            logger.warning("Trading desabilitado manualmente")

    def get_risk_report(self) -> Dict:
        """Gera relatório detalhado de risco"""
        try:
            status = self.get_risk_status()

            # Estatísticas adicionais
            if self.trade_log:
                total_trades = len(self.trade_log)
                winning_trades = len([t for t in self.trade_log if t.get('pnl', 0) > 0])
                losing_trades = total_trades - winning_trades
                win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0

                total_pnl = sum(t.get('pnl', 0) for t in self.trade_log)
                avg_trade_pnl = total_pnl / total_trades if total_trades > 0 else 0

                # Estatísticas por setup
                setup_stats = {}
                for trade in self.trade_log:
                    setup = trade.get('setup_number', 'unknown')
                    if setup not in setup_stats:
                        setup_stats[setup] = {'count': 0, 'pnl': 0, 'wins': 0}

                    setup_stats[setup]['count'] += 1
                    setup_stats[setup]['pnl'] += trade.get('pnl', 0)
                    if trade.get('pnl', 0) > 0:
                        setup_stats[setup]['wins'] += 1

            else:
                total_trades = 0
                winning_trades = 0
                losing_trades = 0
                win_rate = 0
                total_pnl = 0
                avg_trade_pnl = 0
                setup_stats = {}

            return {
                'summary': status,
                'statistics': {
                    'total_trades': total_trades,
                    'winning_trades': winning_trades,
                    'losing_trades': losing_trades,
                    'win_rate': win_rate,
                    'total_pnl': total_pnl,
                    'average_trade_pnl': avg_trade_pnl,
                    'setup_performance': setup_stats
                },
                'risk_limits': {
                    'max_stop_loss_percent': self.max_stop_loss_percent,
                    'min_risk_reward_ratio': self.min_risk_reward_ratio,
                    'max_exposure_percent': self.max_exposure_percent,
                    'max_daily_loss_percent': self.max_daily_loss_percent,
                    'max_consecutive_losses': self.max_consecutive_losses,
                    'max_daily_trades': self.max_daily_trades
                },
                'generated_at': datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Erro ao gerar relatório de risco: {e}")
            return {'error': str(e)}

class PositionTracker:
    """
    Rastreamento detalhado de posições para controle de exposição
    """

    def __init__(self):
        self.positions = {}
        self._lock = threading.Lock()

    def add_position(self, order_id: str, position_data: Dict):
        """Adiciona posição para rastreamento"""
        with self._lock:
            self.positions[order_id] = {
                'order_id': order_id,
                'symbol': position_data.get('symbol', 'US100'),
                'entry_price': position_data.get('entry_price', 0),
                'volume': position_data.get('volume', 0),
                'stop_loss': position_data.get('stop_loss', 0),
                'take_profit': position_data.get('take_profit', 0),
                'setup_number': position_data.get('setup_number'),
                'opened_at': datetime.now(),
                'current_pnl': 0.0,
                'status': 'open'
            }

    def update_position_pnl(self, order_id: str, current_pnl: float):
        """Atualiza P&L atual da posição"""
        with self._lock:
            if order_id in self.positions:
                self.positions[order_id]['current_pnl'] = current_pnl

    def close_position(self, order_id: str, close_price: float, final_pnl: float):
        """Fecha posição do rastreamento"""
        with self._lock:
            if order_id in self.positions:
                position = self.positions[order_id]
                position['close_price'] = close_price
                position['final_pnl'] = final_pnl
                position['closed_at'] = datetime.now()
                position['status'] = 'closed'

                # Manter no dicionário para histórico
                logger.info(f"Posição fechada no rastreamento: {order_id} - P&L: ${final_pnl:.2f}")

    def get_open_positions(self) -> List[Dict]:
        """Retorna posições abertas"""
        with self._lock:
            return [pos for pos in self.positions.values() if pos['status'] == 'open']

    def get_total_exposure(self) -> float:
        """Calcula exposição total das posições abertas"""
        try:
            open_positions = self.get_open_positions()

            total_exposure = 0.0
            for pos in open_positions:
                # Calcular exposição baseada no risco máximo (stop loss)
                risk_distance = abs(pos['entry_price'] - pos['stop_loss'])
                exposure = risk_distance * pos['volume'] * 100000  # Contrato US100
                total_exposure += exposure

            return total_exposure

        except Exception as e:
            logger.error(f"Erro ao calcular exposição total: {e}")
            return 0.0

# Função para criar sistema completo de gerenciamento de risco
def create_risk_management_system(config_path: str = 'config.json'):
    """Cria sistema completo de gerenciamento de risco"""
    try:
        # Carregar configuração
        with open(config_path, 'r') as f:
            config = json.load(f)

        # Inicializar componentes
        risk_manager = AdvancedRiskManager(config)
        position_tracker = PositionTracker()

        logger.info("Sistema de gerenciamento de risco inicializado")
        logger.info(f"Regras aplicadas: SL max 1%, RR min 2:1, Exposição max 5%")

        return {
            'risk_manager': risk_manager,
            'position_tracker': position_tracker,
            'config': config
        }

    except Exception as e:
        logger.error(f"Erro ao criar sistema de gerenciamento de risco: {e}")
        return None

if __name__ == "__main__":
    # Exemplo de uso
    system = create_risk_management_system()

    if system:
        risk_manager = system['risk_manager']
        position_tracker = system['position_tracker']

        # Exemplo de sinal para validação
        test_signal = {
            'setup_number': 1,
            'signal_type': 'BUY',
            'entry_price': 15000.0,
            'stop_loss': 14900.0,  # 100 pontos = 1% aproximadamente
            'take_profit': 15200.0,  # 200 pontos = 2:1 ratio
            'volume': 0.01,
            'symbol': 'US100'
        }

        # Validar sinal
        is_valid, reason, details = risk_manager.validate_signal(test_signal)

        print(f"Sinal válido: {is_valid}")
        print(f"Razão: {reason}")
        print(f"Detalhes: {details}")

        # Mostrar status de risco
        status = risk_manager.get_risk_status()
        print(f"Status de risco: {status}")
    else:
        print("Erro ao inicializar sistema de gerenciamento de risco")