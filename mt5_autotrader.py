"""
Sistema de Execução Automática MT5 para US100
Integração específica com FBS para execução de sinais do módulo QQQ
"""

import logging
import time
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import threading
import MetaTrader5 as mt5
from dataclasses import dataclass, asdict
from enum import Enum

logger = logging.getLogger(__name__)

class OrderStatus(Enum):
    """Status das ordens"""
    PENDING = "pending"
    FILLED = "filled"
    PARTIAL = "partial"
    CANCELLED = "cancelled"
    REJECTED = "rejected"

@dataclass
class TradeOrder:
    """Ordem de negociação"""
    order_id: str
    symbol: str
    order_type: str  # 'BUY' ou 'SELL'
    volume: float
    entry_price: float
    stop_loss: float
    take_profit: float
    status: OrderStatus
    timestamp: datetime
    fill_price: Optional[float] = None
    filled_volume: float = 0.0
    pnl: float = 0.0
    setup_number: Optional[int] = None
    signal_confidence: Optional[float] = None

    def to_dict(self) -> Dict:
        """Converte ordem para dicionário"""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        data['status'] = self.status.value
        return data

class MT5AutoTrader:
    """
    Sistema automático de execução de ordens no MT5 para US100
    """

    def __init__(self, config: Dict):
        self.config = config

        # Configurações MT5
        self.mt5_config = config.get('mt5', {})
        self.server = self.mt5_config.get('server', 'FBS-Real')
        self.login = self.mt5_config.get('login', 11655745)
        self.password = self.mt5_config.get('password', 'Street@21')
        self.path = self.mt5_config.get('path', 'C:\\Program Files\\MetaTrader 5\\terminal64.exe')

        # Configurações de trading
        self.symbol = config.get('trading', {}).get('symbol', 'US100')
        self.default_lot_size = config.get('trading', {}).get('lot_size', 0.01)
        self.max_positions = config.get('trading', {}).get('max_positions', 2)

        # Estado do sistema
        self.connected = False
        self.account_info = {}
        self.positions = []
        self.orders = []
        self.running = False

        # Controle de execução
        self._lock = threading.Lock()
        self._position_monitor_thread = None

        # Callbacks
        self.on_order_filled = None
        self.on_position_closed = None
        self.on_connection_lost = None

    def initialize(self) -> bool:
        """Inicializa conexão com MT5"""
        try:
            logger.info("Inicializando MT5...")

            # Inicializar MT5
            if not mt5.initialize(path=self.path):
                logger.error("Falha ao inicializar MT5")
                return False

            # Fazer login
            if not mt5.login(
                login=self.login,
                password=self.password,
                server=self.server
            ):
                error = mt5.last_error()
                logger.error(f"Falha no login MT5: {error}")
                return False

            self.connected = True
            logger.info("MT5 conectado com sucesso")

            # Obter informações da conta
            self._update_account_info()

            # Iniciar monitoramento de posições
            self._start_position_monitoring()

            return True

        except Exception as e:
            logger.error(f"Erro ao inicializar MT5: {e}")
            return False

    def shutdown(self):
        """Encerra conexão com MT5"""
        try:
            logger.info("Encerrando conexão MT5...")

            self.running = False

            # Parar monitoramento
            if self._position_monitor_thread and self._position_monitor_thread.is_alive():
                self._position_monitor_thread.join(timeout=5)

            # Fechar MT5
            mt5.shutdown()

            self.connected = False
            logger.info("MT5 desconectado")

        except Exception as e:
            logger.error(f"Erro ao encerrar MT5: {e}")

    def _update_account_info(self) -> bool:
        """Atualiza informações da conta"""
        try:
            account_info = mt5.account_info()
            if account_info:
                self.account_info = {
                    'balance': account_info.balance,
                    'equity': account_info.equity,
                    'margin': account_info.margin,
                    'margin_free': account_info.margin_free,
                    'margin_level': account_info.margin_level,
                    'profit': account_info.profit,
                    'currency': account_info.currency
                }
                return True
            return False

        except Exception as e:
            logger.error(f"Erro ao atualizar informações da conta: {e}")
            return False

    def _start_position_monitoring(self):
        """Inicia monitoramento de posições"""
        if self._position_monitor_thread and self._position_monitor_thread.is_alive():
            return

        self._position_monitor_thread = threading.Thread(target=self._position_monitoring_loop)
        self._position_monitor_thread.daemon = True
        self._position_monitor_thread.start()
        logger.info("Monitoramento de posições iniciado")

    def _position_monitoring_loop(self):
        """Loop de monitoramento de posições"""
        while self.running and self.connected:
            try:
                # Atualizar posições
                self._update_positions()

                # Atualizar informações da conta
                self._update_account_info()

                # Aguardar próximo ciclo
                time.sleep(1)  # Verificar a cada segundo

            except Exception as e:
                logger.error(f"Erro no monitoramento de posições: {e}")
                time.sleep(5)

    def _update_positions(self):
        """Atualiza lista de posições"""
        try:
            if not self.connected:
                return

            positions = mt5.positions_get(symbol=self.symbol)
            if positions:
                self.positions = []
                for pos in positions:
                    position_data = {
                        'ticket': pos.ticket,
                        'symbol': pos.symbol,
                        'type': 'BUY' if pos.type == mt5.POSITION_TYPE_BUY else 'SELL',
                        'volume': pos.volume,
                        'price_open': pos.price_open,
                        'price_current': pos.price_current,
                        'profit': pos.profit,
                        'sl': pos.sl,
                        'tp': pos.tp,
                        'time': datetime.fromtimestamp(pos.time)
                    }
                    self.positions.append(position_data)

        except Exception as e:
            logger.error(f"Erro ao atualizar posições: {e}")

    def place_order(self, order_type: str, volume: float, entry_price: float,
                   stop_loss: float, take_profit: float, setup_number: int = None,
                   confidence: float = None) -> Dict:
        """
        Executa ordem no MT5

        Args:
            order_type: 'BUY' ou 'SELL'
            volume: Tamanho da posição em lotes
            entry_price: Preço de entrada
            stop_loss: Stop loss
            take_profit: Take profit
            setup_number: Número do setup (opcional)
            confidence: Confiança do sinal (opcional)
        """
        try:
            if not self.connected:
                return {'success': False, 'error': 'MT5 não conectado'}

            # Verificar se já temos posições máximas
            if len(self.positions) >= self.max_positions:
                return {'success': False, 'error': f'Limite de posições atingido ({self.max_positions})'}

            # Preparar request da ordem
            order_request = {
                'action': mt5.TRADE_ACTION_DEAL,
                'symbol': self.symbol,
                'volume': volume,
                'type': mt5.ORDER_TYPE_BUY if order_type == 'BUY' else mt5.ORDER_TYPE_SELL,
                'price': entry_price,
                'sl': stop_loss,
                'tp': take_profit,
                'deviation': 10,
                'magic': 123456,
                'comment': f"Setup {setup_number} - Conf: {confidence:.2f}" if setup_number else "Automated",
                'type_time': mt5.ORDER_TIME_GTC,
                'type_filling': mt5.ORDER_FILLING_IOC,
            }

            # Enviar ordem
            result = mt5.order_send(order_request)

            if result.retcode == mt5.TRADE_RETCODE_DONE:
                # Ordem executada com sucesso
                order_id = str(result.order)

                # Criar objeto da ordem
                order = TradeOrder(
                    order_id=order_id,
                    symbol=self.symbol,
                    order_type=order_type,
                    volume=volume,
                    entry_price=entry_price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    status=OrderStatus.FILLED,
                    timestamp=datetime.now(),
                    fill_price=result.price,
                    filled_volume=result.volume,
                    setup_number=setup_number,
                    signal_confidence=confidence
                )

                # Adicionar à lista de ordens
                with self._lock:
                    self.orders.append(order)

                logger.info(f"Ordem executada: {order_id} - {order_type} {volume} lotes a {result.price}")
                return {
                    'success': True,
                    'order_id': order_id,
                    'price': result.price,
                    'volume': result.volume
                }
            else:
                # Ordem falhou
                error_msg = f"Ordem rejeitada: {result.retcode}"
                logger.error(error_msg)
                return {
                    'success': False,
                    'error': error_msg,
                    'retcode': result.retcode
                }

        except Exception as e:
            error_msg = f"Erro ao executar ordem: {e}"
            logger.error(error_msg)
            return {'success': False, 'error': error_msg}

    def close_position(self, ticket: int, volume: float = None) -> Dict:
        """Fecha posição específica"""
        try:
            if not self.connected:
                return {'success': False, 'error': 'MT5 não conectado'}

            # Buscar posição
            position = mt5.positions_get(ticket=ticket)
            if not position:
                return {'success': False, 'error': 'Posição não encontrada'}

            pos = position[0]

            # Se volume não especificado, fechar posição completa
            if volume is None:
                volume = pos.volume

            # Determinar tipo de ordem para fechar
            if pos.type == mt5.POSITION_TYPE_BUY:
                order_type = mt5.ORDER_TYPE_SELL
                price = mt5.symbol_info_tick(self.symbol).bid
            else:
                order_type = mt5.ORDER_TYPE_BUY
                price = mt5.symbol_info_tick(self.symbol).ask

            # Preparar request
            close_request = {
                'action': mt5.TRADE_ACTION_DEAL,
                'position': ticket,
                'symbol': self.symbol,
                'volume': volume,
                'type': order_type,
                'price': price,
                'deviation': 10,
                'magic': 123456,
                'comment': "Close position",
                'type_time': mt5.ORDER_TIME_GTC,
                'type_filling': mt5.ORDER_FILLING_IOC,
            }

            # Executar fechamento
            result = mt5.order_send(close_request)

            if result.retcode == mt5.TRADE_RETCODE_DONE:
                logger.info(f"Posição fechada: {ticket} - P&L: {pos.profit:.2f}")

                # Callback para posição fechada
                if self.on_position_closed:
                    self.on_position_closed(ticket, pos.profit)

                return {
                    'success': True,
                    'closed_price': result.price,
                    'pnl': pos.profit
                }
            else:
                error_msg = f"Falha ao fechar posição: {result.retcode}"
                logger.error(error_msg)
                return {'success': False, 'error': error_msg}

        except Exception as e:
            error_msg = f"Erro ao fechar posição: {e}"
            logger.error(error_msg)
            return {'success': False, 'error': error_msg}

    def close_all_positions(self) -> Dict:
        """Fecha todas as posições"""
        try:
            results = []

            for position in self.positions:
                result = self.close_position(position['ticket'])
                results.append({
                    'ticket': position['ticket'],
                    'result': result
                })

            success_count = len([r for r in results if r['result']['success']])

            return {
                'success': success_count == len(results),
                'total_positions': len(results),
                'closed_positions': success_count,
                'results': results
            }

        except Exception as e:
            error_msg = f"Erro ao fechar todas as posições: {e}"
            logger.error(error_msg)
            return {'success': False, 'error': error_msg}

    def modify_position(self, ticket: int, stop_loss: float = None, take_profit: float = None) -> Dict:
        """Modifica stop loss e/ou take profit de posição"""
        try:
            if not self.connected:
                return {'success': False, 'error': 'MT5 não conectado'}

            # Buscar posição atual
            position = mt5.positions_get(ticket=ticket)
            if not position:
                return {'success': False, 'error': 'Posição não encontrada'}

            pos = position[0]

            # Preparar modificação
            modify_request = {
                'action': mt5.TRADE_ACTION_SLTP,
                'position': ticket,
                'symbol': self.symbol,
                'sl': stop_loss if stop_loss is not None else pos.sl,
                'tp': take_profit if take_profit is not None else pos.tp,
                'magic': 123456,
            }

            # Executar modificação
            result = mt5.order_send(modify_request)

            if result.retcode == mt5.TRADE_RETCODE_DONE:
                logger.info(f"Posição modificada: {ticket}")
                return {'success': True}
            else:
                error_msg = f"Falha ao modificar posição: {result.retcode}"
                logger.error(error_msg)
                return {'success': False, 'error': error_msg}

        except Exception as e:
            error_msg = f"Erro ao modificar posição: {e}"
            logger.error(error_msg)
            return {'success': False, 'error': error_msg}

    def get_symbol_info(self) -> Dict:
        """Obtém informações do símbolo"""
        try:
            if not self.connected:
                return {}

            symbol_info = mt5.symbol_info(self.symbol)
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

        except Exception as e:
            logger.error(f"Erro ao obter informações do símbolo: {e}")
            return {}

    def calculate_position_size(self, risk_percent: float = 1.0) -> float:
        """
        Calcula tamanho da posição baseado no risco

        Args:
            risk_percent: Percentual de risco (1% = 1.0)
        """
        try:
            if not self.account_info:
                return self.default_lot_size

            account_balance = self.account_info.get('balance', 10000)

            # Calcular valor de risco em dólares
            risk_amount = account_balance * (risk_percent / 100)

            # Obter informações do símbolo para calcular valor por lote
            symbol_info = self.get_symbol_info()
            if not symbol_info:
                return self.default_lot_size

            # Para US100 (NASDAQ-100), assumindo contrato de aproximadamente $100k
            contract_value = 100000  # Valor aproximado do contrato

            # Calcular número de lotes baseado no risco
            lot_size = risk_amount / contract_value

            # Respeitar limites do símbolo
            lot_size = max(lot_size, symbol_info.get('volume_min', 0.01))
            lot_size = min(lot_size, symbol_info.get('volume_max', 100.0))

            return round(lot_size, 2)

        except Exception as e:
            logger.error(f"Erro ao calcular tamanho da posição: {e}")
            return self.default_lot_size

    def get_trading_status(self) -> Dict:
        """Retorna status completo do sistema de trading"""
        try:
            return {
                'connected': self.connected,
                'account_info': self.account_info,
                'positions_count': len(self.positions),
                'orders_count': len(self.orders),
                'symbol': self.symbol,
                'positions': self.positions,
                'recent_orders': [order.to_dict() for order in self.orders[-10:]],
                'symbol_info': self.get_symbol_info(),
                'last_update': datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Erro ao obter status de trading: {e}")
            return {'error': str(e)}

class RiskManager:
    """
    Gerenciamento avançado de risco para operações US100
    """

    def __init__(self, config: Dict):
        self.config = config

        # Configurações de risco
        self.max_daily_loss = config.get('risk_management', {}).get('max_daily_loss', 200.0)
        self.max_consecutive_losses = config.get('risk_management', {}).get('max_consecutive_losses', 3)
        self.daily_profit_target = config.get('risk_management', {}).get('daily_profit_target', 100.0)
        self.max_daily_trades = config.get('risk_management', {}).get('max_daily_trades', 10)

        # Estado de risco
        self.daily_pnl = 0.0
        self.consecutive_losses = 0
        self.daily_trades = 0
        self.last_reset_date = datetime.now().date()

        # Controle de exposição
        self.max_exposure_percent = 5.0  # Máximo 5% do capital
        self.current_exposure = 0.0

    def validate_trade(self, signal: Dict, account_info: Dict) -> Tuple[bool, str]:
        """
        Valida se trade está dentro dos parâmetros de risco

        Args:
            signal: Sinal de trading com parâmetros
            account_info: Informações da conta
        """
        try:
            # Reset diário se necessário
            self._check_daily_reset()

            account_balance = account_info.get('balance', 10000)

            # 1. Verificar perda diária máxima
            if self.daily_pnl <= -self.max_daily_loss:
                return False, f"Limite de perda diária atingido: ${self.daily_pnl:.2f}"

            # 2. Verificar número máximo de trades diários
            if self.daily_trades >= self.max_daily_trades:
                return False, f"Limite de trades diários atingido: {self.daily_trades}"

            # 3. Verificar exposição máxima
            signal_exposure = self._calculate_signal_exposure(signal, account_balance)
            if self.current_exposure + signal_exposure > (account_balance * self.max_exposure_percent / 100):
                return False, f"Exposição máxima seria excedida: {self.current_exposure + signal_exposure:.2f}"

            # 4. Verificar relação risco-recompensa mínima (2:1)
            risk_reward_ratio = signal.get('risk_reward_ratio', 0)
            if risk_reward_ratio < 2.0:
                return False, f"Relação risco-recompensa abaixo do mínimo: {risk_reward_ratio:.2f}"

            # 5. Verificar stop loss máximo (1% do saldo)
            stop_loss_distance = abs(signal.get('stop_loss', 0) - signal.get('entry_price', 0))
            max_stop_loss = account_balance * 0.01

            if stop_loss_distance > max_stop_loss:
                return False, f"Stop loss excede máximo permitido: ${stop_loss_distance:.2f} > ${max_stop_loss:.2f}"

            return True, "Trade validado"

        except Exception as e:
            return False, f"Erro na validação de risco: {e}"

    def _calculate_signal_exposure(self, signal: Dict, account_balance: float) -> float:
        """Calcula exposição do sinal"""
        try:
            entry_price = signal.get('entry_price', 0)
            stop_loss = signal.get('stop_loss', 0)
            volume = signal.get('volume', 0.01)

            # Calcular risco em dólares
            risk_distance = abs(entry_price - stop_loss)
            exposure = risk_distance * volume * 100000  # Assumindo contrato de $100k

            return exposure

        except:
            return 0.0

    def _check_daily_reset(self):
        """Verifica se precisa resetar contadores diários"""
        current_date = datetime.now().date()

        if current_date != self.last_reset_date:
            self.daily_pnl = 0.0
            self.daily_trades = 0
            self.last_reset_date = current_date
            logger.info("Contadores diários resetados")

    def update_trade_result(self, pnl: float, is_win: bool):
        """Atualiza estatísticas após resultado de trade"""
        try:
            self.daily_pnl += pnl
            self.daily_trades += 1

            if is_win:
                self.consecutive_losses = 0
            else:
                self.consecutive_losses += 1

            logger.info(f"Estatísticas atualizadas - P&L Diário: ${self.daily_pnl:.2f}, Trades: {self.daily_trades}")

        except Exception as e:
            logger.error(f"Erro ao atualizar resultado do trade: {e}")

    def get_risk_status(self) -> Dict:
        """Retorna status atual de risco"""
        return {
            'daily_pnl': self.daily_pnl,
            'daily_trades': self.daily_trades,
            'consecutive_losses': self.consecutive_losses,
            'current_exposure': self.current_exposure,
            'max_daily_loss': self.max_daily_loss,
            'max_daily_trades': self.max_daily_trades,
            'risk_ok': self.daily_pnl > -self.max_daily_loss and self.daily_trades < self.max_daily_trades,
            'last_reset': self.last_reset_date.isoformat()
        }

# Função para criar sistema completo
def create_autotrading_system(config_path: str = 'config.json'):
    """Cria sistema completo de autotrading"""
    try:
        # Carregar configuração
        with open(config_path, 'r') as f:
            config = json.load(f)

        # Inicializar componentes
        autotrader = MT5AutoTrader(config)
        risk_manager = RiskManager(config)

        # Inicializar MT5
        if not autotrader.initialize():
            logger.error("Falha ao inicializar sistema de autotrading")
            return None

        return {
            'autotrader': autotrader,
            'risk_manager': risk_manager,
            'config': config
        }

    except Exception as e:
        logger.error(f"Erro ao criar sistema de autotrading: {e}")
        return None

if __name__ == "__main__":
    # Exemplo de uso
    system = create_autotrading_system()

    if system:
        autotrader = system['autotrader']
        risk_manager = system['risk_manager']

        # Manter sistema rodando
        try:
            while True:
                # Mostrar status a cada 30 segundos
                status = autotrader.get_trading_status()
                risk_status = risk_manager.get_risk_status()

                print(f"Status MT5: {status['connected']}")
                print(f"Saldo: ${status.get('account_info', {}).get('balance', 0):.2f}")
                print(f"Posições: {status['positions_count']}")
                print(f"P&L Diário: ${risk_status['daily_pnl']:.2f}")
                print(f"Trades Hoje: {risk_status['daily_trades']}")
                print("---")

                time.sleep(30)

        except KeyboardInterrupt:
            print("Encerrando sistema...")
            autotrader.shutdown()
    else:
        print("Erro ao inicializar sistema")