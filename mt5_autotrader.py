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
        self.login = self.mt5_config.get('login', 111655745)
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

    def place_order(self, order_type: str = None, volume: float = None, entry_price: float = None,
                   stop_loss: float = None, take_profit: float = None, symbol: str = None,
                   price: float = None, sl: float = None, tp: float = None, 
                   setup_number: int = None, confidence: float = None) -> Dict:
        # Handle parameter aliases
        entry_price = price if price is not None else entry_price
        stop_loss = sl if sl is not None else stop_loss
        take_profit = tp if tp is not None else take_profit
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

            # Use provided symbol or default
            target_symbol = symbol if symbol else self.symbol

            # Obter informações do símbolo para validar volume
            symbol_info = self.get_symbol_info()
            if not symbol_info:
                return {'success': False, 'error': f'Não foi possível obter informações do símbolo {target_symbol}'}

            # Validar volume contra limites do símbolo
            min_volume = symbol_info.get('volume_min', 0.01)
            max_volume = symbol_info.get('volume_max', 100.0)
            volume_step = symbol_info.get('volume_step', 0.01)
            
            logger.info(f"Symbol volume constraints - Min: {min_volume}, Max: {max_volume}, Step: {volume_step}")
            logger.info(f"Original volume requested: {volume}")

            # Ajustar volume para estar dentro dos limites
            if volume < min_volume:
                logger.warning(f"Volume {volume} abaixo do mínimo {min_volume}, ajustando para {min_volume}")
                volume = min_volume
            elif volume > max_volume:
                logger.warning(f"Volume {volume} acima do máximo {max_volume}, ajustando para {max_volume}")
                volume = max_volume

            # Arredondar volume para o incremento válido mais próximo
            if volume_step > 0:
                volume = round(volume / volume_step) * volume_step
                volume = round(volume, 2)  # Evitar problemas de precisão de ponto flutuante
                logger.info(f"Volume after step rounding: {volume}")

            # Verificar se volume ainda está dentro dos limites após arredondamento
            volume = max(min_volume, min(volume, max_volume))
            logger.info(f"Final volume after validation: {volume}")

            # Verificar se já temos posições máximas
            if len(self.positions) >= self.max_positions:
                return {'success': False, 'error': f'Limite de posições atingido ({self.max_positions})'}

            # Converter tipos numpy para Python nativos para evitar problemas com MT5 API
            def convert_numpy_types(value):
                import numpy as np
                if value is None:
                    return 0.0  # Return default for None values
                if hasattr(value, 'item'):  # É um tipo numpy
                    return value.item()
                return float(value) if value is not None else 0.0

            # Preparar request da ordem com validação extra
            converted_volume = convert_numpy_types(volume)
            converted_price = convert_numpy_types(entry_price)
            converted_sl = convert_numpy_types(stop_loss) if stop_loss is not None and stop_loss != 0 else 0.0
            converted_tp = convert_numpy_types(take_profit) if take_profit is not None and take_profit != 0 else 0.0

            # Obter informações atualizadas do símbolo para garantir que está disponível
            symbol_info = mt5.symbol_info(target_symbol)
            if not symbol_info:
                logger.error(f"Símbolo {target_symbol} não encontrado ou indisponível")
                return {'success': False, 'error': f'Símbolo {target_symbol} não encontrado'}
            
            if not symbol_info.select:
                logger.error(f"Símbolo {target_symbol} não está selecionado para negociação")
                return {'success': False, 'error': f'Símbolo {target_symbol} não está selecionado'}
            
            if not symbol_info.visible:
                logger.error(f"Símbolo {target_symbol} não está visível")
                return {'success': False, 'error': f'Símbolo {target_symbol} não está visível'}

            # Validar se o preço é válido baseado na posição de compra/venda
            current_tick = mt5.symbol_info_tick(target_symbol)
            if not current_tick:
                logger.error(f"Não foi possível obter tick atual para {target_symbol}")
                return {'success': False, 'error': f'Não foi possível obter dados do símbolo'}

            # Obter o preço atual de mercado para usar na ordem e para validação
            current_tick = mt5.symbol_info_tick(target_symbol)
            if current_tick:
                market_price = current_tick.ask if order_type == 'BUY' else current_tick.bid
            else:
                logger.warning(f"Não foi possível obter preço atual para {target_symbol}, usando preço do sinal")
                # Se não conseguir preço atual, usar o preço original convertido
                market_price = convert_numpy_types(entry_price)
            
            # Validar se o preço de referência é razoável para evitar valores estranhos
            if market_price <= 0 or market_price > 100000:  # Valor de segurança razoável para US100
                logger.error(f"Preço inválido detectado: {market_price}, usando preço de sinal convertido")
                market_price = convert_numpy_types(entry_price)

            # Validar se o preço é válido baseado na posição de compra/venda usando o preço original de validação
            validation_price = converted_price  # Preço original convertido para validação
            if order_type == 'BUY':
                # O preço de entrada deve ser próximo do ask, e SL deve ser abaixo do preço
                if converted_sl != 0 and converted_sl >= validation_price:
                    logger.warning(f"Stop loss ({converted_sl}) deve ser menor que preço de entrada ({validation_price}) para ordem de compra")
                    # Ajustar SL para ser abaixo do preço
                    converted_sl = validation_price * 0.99  # Ajustar para 1% abaixo
                if converted_tp != 0 and converted_tp <= validation_price:
                    logger.warning(f"Take profit ({converted_tp}) deve ser maior que preço de entrada ({validation_price}) para ordem de compra")
                    # Ajustar TP para ser acima do preço
                    converted_tp = validation_price * 1.02  # Ajustar para 2% acima
            else:  # SELL
                # Para ordens de venda, SL deve ser acima do preço, TP deve ser abaixo
                if converted_sl != 0 and converted_sl <= validation_price:
                    logger.warning(f"Stop loss ({converted_sl}) deve ser maior que preço de entrada ({validation_price}) para ordem de venda")
                    # Ajustar SL para ser acima do preço
                    converted_sl = validation_price * 1.01  # Ajustar para 1% acima
                if converted_tp != 0 and converted_tp >= validation_price:
                    logger.warning(f"Take profit ({converted_tp}) deve ser menor que preço de entrada ({validation_price}) para ordem de venda")
                    # Ajustar TP para ser abaixo do preço
                    converted_tp = validation_price * 0.98  # Ajustar para 2% abaixo

            # Preparar request da ordem - usar o preço de mercado para a ordem
            order_request = {
                'action': mt5.TRADE_ACTION_DEAL,
                'symbol': target_symbol,
                'volume': converted_volume,
                'type': mt5.ORDER_TYPE_BUY if order_type == 'BUY' else mt5.ORDER_TYPE_SELL,
                'price': market_price,
                'sl': converted_sl,
                'tp': converted_tp,
                'deviation': 20,  # Aumentar desvio para dar mais flexibilidade
                'magic': 123456,
                'comment': f"Setup {setup_number} - Conf: {confidence:.2f}" if setup_number else "Automated",
                'type_time': mt5.ORDER_TIME_GTC,
                'type_filling': mt5.ORDER_FILLING_FOK,  # Tentar FOK (Fill or Kill) em vez de IOC
            }

            logger.info(f"Enviando ordem: {order_request}")
            
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

    def get_positions(self):
        """Retorna lista de posições ativas"""
        return self.positions

    def get_account_info(self):
        """Retorna informações da conta"""
        return self.account_info

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
            min_volume = symbol_info.get('volume_min', 0.01)
            max_volume = symbol_info.get('volume_max', 100.0)
            volume_step = symbol_info.get('volume_step', 0.01)

            # Ajustar tamanho da posição dentro dos limites
            lot_size = max(lot_size, min_volume)
            lot_size = min(lot_size, max_volume)

            # Arredondar para o passo de volume válido
            if volume_step > 0:
                lot_size = round(lot_size / volume_step) * volume_step
                lot_size = round(lot_size, 2)  # Evitar problemas de precisão de ponto flutuante

            return lot_size

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