"""
Sistema Principal de Trading - EzOptions QQQ Integration
Integração completa de todos os componentes para execução automática
"""

import logging
import json
import time
import threading
import signal
import sys
from datetime import datetime
from typing import Dict, Optional

# Importar componentes do sistema
from trading_setups_corrected import TradingSetupsCorrected
from qqq_monitor import QQQMonitor, SignalExecutor
from mt5_autotrader import MT5AutoTrader
from advanced_risk_manager import AdvancedRiskManager, PositionTracker
from audit_logger import AuditLogger, initialize_audit_logger
from duplicate_checker import SignalDeduplicator
from integration_test import IntegrationTester

logger = logging.getLogger(__name__)

class EzOptionsTradingSystem:
    """
    Sistema principal de trading que integra todos os componentes
    """

    def __init__(self, config_path: str = 'config.json'):
        self.config_path = config_path
        self.config = self._load_config()

        # Estado do sistema
        self.running = False
        self.initialized = False

        # Componentes principais
        self.audit_logger = None
        self.qqq_monitor = None
        self.mt5_trader = None
        self.risk_manager = None
        self.position_tracker = None
        self.deduplicator = None

        # Threads de monitoramento
        self.monitoring_threads = []

        # Métricas de performance
        self.start_time = None
        self.signals_detected = 0
        self.signals_executed = 0
        self.total_pnl = 0.0

    def _load_config(self) -> Dict:
        """Carrega configuração do sistema"""
        try:
            with open(self.config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.error(f"Arquivo de configuração não encontrado: {self.config_path}")
            return self._get_default_config()
        except json.JSONDecodeError as e:
            logger.error(f"Erro no arquivo de configuração: {e}")
            return self._get_default_config()

    def _get_default_config(self) -> Dict:
        """Retorna configuração padrão"""
        return {
            "mt5": {
                "server": "FBS-Real",
                "login": 11655745,
                "password": "Street@21",
                "path": "C:\\Program Files\\MetaTrader 5\\terminal64.exe"
            },
            "ezoptions": {
                "url": "http://localhost:8501",
                "update_interval_seconds": 30,
                "tickers": ["QQQ"]
            },
            "trading": {
                "symbol": "US100",
                "lot_size": 0.01,
                "max_positions": 2,
                "stop_loss_pips": 50,
                "take_profit_pips": 100,
                "max_drawdown_percent": 5.0
            },
            "risk_management": {
                "max_daily_loss": 200.0,
                "max_consecutive_losses": 3,
                "daily_profit_target": 100.0,
                "max_daily_trades": 10
            },
            "trading_parameters": {
                "ticker": "US100",
                "expiry_date": "2025-12-20",
                "active_setups": [1, 2, 3, 4, 5, 6],
                "check_interval": 30
            },
            "logging": {
                "level": "INFO",
                "audit_dir": "audit_logs",
                "max_file_size_mb": 100,
                "backup_count": 30
            }
        }

    def initialize(self) -> bool:
        """
        Inicializa todos os componentes do sistema

        Returns:
            True se inicialização foi bem-sucedida
        """
        try:
            logger.info("Inicializando Sistema de Trading EzOptions...")

            # 1. Inicializar sistema de auditoria
            self.audit_logger = initialize_audit_logger(self.config_path)
            if not self.audit_logger:
                logger.error("Falha ao inicializar sistema de auditoria")
                return False

            self.audit_logger.log_system_start("MainSystem", self.config)

            # 2. Inicializar componentes principais
            try:
                # Risk Manager e Position Tracker
                self.risk_manager = AdvancedRiskManager(self.config)
                self.position_tracker = PositionTracker()

                # MT5 AutoTrader
                self.mt5_trader = MT5AutoTrader(self.config)
                if not self.mt5_trader.initialize():
                    logger.error("Falha ao inicializar MT5")
                    return False

                # QQQ Monitor
                self.qqq_monitor = QQQMonitor(self.config)

                # Signal Deduplicator
                self.deduplicator = SignalDeduplicator(self.config)

                # Signal Executor
                signal_executor = SignalExecutor(
                    self.mt5_trader,
                    self.risk_manager,
                    self.config
                )

                # Configurar callbacks do QQQ Monitor
                def on_signal_detected(signal):
                    self.signals_detected += 1
                    self.audit_logger.log_signal_detected("QQQMonitor", signal.to_dict())

                    logger.info(f"Sinal detectado: Setup {signal.setup_number}, {signal.signal_type}")

                    # Verificar duplicatas
                    should_process, reason, details = self.deduplicator.should_process_signal(signal.to_dict())

                    if should_process:
                        logger.info(f"Sinal aprovado para execução: {reason}")
                        signal.status = signal.status.CONFIRMED

                        # Executar sinal
                        result = signal_executor.execute_signal(signal)
                        if result['success']:
                            self.signals_executed += 1
                            self.audit_logger.log_signal_executed("SignalExecutor", result)
                            logger.info(f"Sinal executado: {result['order_id']}")
                        else:
                            self.audit_logger.log_error("SignalExecutor", Exception(result['error']))
                    else:
                        logger.warning(f"Sinal rejeitado: {reason}")
                        self.audit_logger.log_warning("Deduplicator", reason, details)

                def on_signal_confirmed(signal):
                    self.audit_logger.log_signal_confirmed("QQQMonitor", signal.to_dict())

                def on_signal_executed(signal):
                    self.audit_logger.log_signal_executed("SignalExecutor", {
                        'order_id': signal.order_id,
                        'execution_price': signal.execution_price,
                        'volume': signal.market_data.get('volume', 0)
                    })

                self.qqq_monitor.on_signal_detected = on_signal_detected
                self.qqq_monitor.on_signal_confirmed = on_signal_confirmed
                self.qqq_monitor.on_signal_executed = on_signal_executed

                # 3. Executar teste de integração
                logger.info("Executando teste de integração...")
                tester = IntegrationTester(self.config)
                test_results = tester.run_full_integration_test()

                if test_results.get('integration_score', 0) < 80:
                    logger.warning(f"Score de integração baixo: {test_results['integration_score']:.1f}%")
                    logger.warning("Sistema pode ter problemas de integração")

                    # Mostrar recomendações
                    for rec in test_results.get('recommendations', []):
                        logger.warning(f"Recomendação: {rec}")

                self.initialized = True
                logger.info("Sistema inicializado com sucesso")

                return True

            except Exception as e:
                logger.error(f"Erro ao inicializar componentes: {e}")
                self.audit_logger.log_error("MainSystem", e, {'component': 'initialization'})
                return False

        except Exception as e:
            logger.error(f"Erro geral na inicialização: {e}")
            return False

    def start(self):
        """Inicia sistema de trading"""
        if not self.initialized:
            logger.error("Sistema não inicializado. Execute initialize() primeiro.")
            return False

        if self.running:
            logger.warning("Sistema já está rodando")
            return True

        try:
            logger.info("Iniciando Sistema de Trading...")
            self.start_time = datetime.now()
            self.running = True

            # Iniciar monitoramento QQQ
            self.qqq_monitor.start_monitoring()

            # Registrar início
            self.audit_logger.log_system_start("TradingSystem", {
                'mode': 'live_trading',
                'active_setups': self.config.get('trading_parameters', {}).get('active_setups', [])
            })

            logger.info("Sistema de Trading iniciado com sucesso")
            return True

        except Exception as e:
            logger.error(f"Erro ao iniciar sistema: {e}")
            self.audit_logger.log_error("MainSystem", e, {'component': 'system_start'})
            return False

    def stop(self):
        """Para sistema de trading"""
        if not self.running:
            logger.warning("Sistema já está parado")
            return True

        try:
            logger.info("Parando Sistema de Trading...")
            self.running = False

            # Parar monitoramento
            if self.qqq_monitor:
                self.qqq_monitor.stop_monitoring()

            # Fechar posições se necessário
            if self.mt5_trader:
                close_result = self.mt5_trader.close_all_positions()
                logger.info(f"Posições fechadas: {close_result}")

            # Calcular tempo de operação
            uptime = datetime.now() - self.start_time if self.start_time else timedelta()

            # Registrar parada
            self.audit_logger.log_system_stop("TradingSystem", uptime.total_seconds())

            # Flush logs finais
            self.audit_logger.force_flush()

            logger.info(f"Sistema parado. Tempo de operação: {uptime}")
            return True

        except Exception as e:
            logger.error(f"Erro ao parar sistema: {e}")
            return False

    def get_system_status(self) -> Dict:
        """Retorna status completo do sistema"""
        try:
            uptime = datetime.now() - self.start_time if self.start_time else timedelta()

            return {
                'running': self.running,
                'initialized': self.initialized,
                'uptime_seconds': uptime.total_seconds(),
                'start_time': self.start_time.isoformat() if self.start_time else None,
                'signals_detected': self.signals_detected,
                'signals_executed': self.signals_executed,
                'total_pnl': self.total_pnl,
                'components': {
                    'qqq_monitor': self.qqq_monitor.get_monitoring_status() if self.qqq_monitor else None,
                    'mt5_trader': self.mt5_trader.get_trading_status() if self.mt5_trader else None,
                    'risk_manager': self.risk_manager.get_risk_status() if self.risk_manager else None,
                    'audit_logger': self.audit_logger.get_audit_summary() if self.audit_logger else None,
                    'deduplicator': self.deduplicator.get_deduplication_stats() if self.deduplicator else None
                },
                'last_update': datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Erro ao obter status do sistema: {e}")
            return {'error': str(e)}

    def run_diagnostics(self) -> Dict:
        """Executa diagnóstico completo do sistema"""
        try:
            logger.info("Executando diagnóstico do sistema...")

            # Executar teste de integração
            tester = IntegrationTester(self.config)
            test_results = tester.run_full_integration_test()

            # Obter status atual
            system_status = self.get_system_status()

            # Gerar relatório de diagnóstico
            diagnostic_report = {
                'timestamp': datetime.now().isoformat(),
                'system_status': system_status,
                'integration_test_results': test_results,
                'recommendations': test_results.get('recommendations', []),
                'health_score': test_results.get('integration_score', 0),
                'components_health': {}
            }

            # Verificar saúde de cada componente
            components = system_status.get('components', {})

            for component_name, component_status in components.items():
                if component_status and isinstance(component_status, dict):
                    # Calcular saúde baseada em métricas específicas
                    if component_name == 'mt5_trader':
                        health = 100 if component_status.get('connected', False) else 0
                    elif component_name == 'qqq_monitor':
                        health = 100 if component_status.get('running', False) else 0
                    elif component_name == 'risk_manager':
                        health = 100 if component_status.get('trading_enabled', False) else 50
                    elif component_name == 'audit_logger':
                        health = 100 if component_status.get('total_events', 0) > 0 else 0
                    else:
                        health = 50  # Desconhecido

                    diagnostic_report['components_health'][component_name] = {
                        'health_score': health,
                        'status': 'healthy' if health > 80 else 'warning' if health > 50 else 'critical',
                        'details': component_status
                    }

            logger.info(f"Diagnóstico concluído. Health Score: {diagnostic_report['health_score']}")
            return diagnostic_report

        except Exception as e:
            logger.error(f"Erro no diagnóstico: {e}")
            return {'error': str(e)}

    def emergency_stop(self):
        """Para emergencial do sistema"""
        try:
            logger.critical("PARADA EMERGÊNCIAL DO SISTEMA")

            # Parar tudo imediatamente
            self.running = False

            # Fechar todas as posições
            if self.mt5_trader:
                self.mt5_trader.close_all_positions()

            # Registrar emergência
            self.audit_logger.log_event(
                self.audit_logger.log_event.__self__._AuditLogger__class__.SYSTEM_STOP,
                "EmergencyStop",
                "emergency_shutdown",
                {'reason': 'manual_emergency_stop'},
                severity="CRITICAL"
            )

            # Flush imediato
            self.audit_logger.force_flush()

            logger.critical("Parada emergencial concluída")

        except Exception as e:
            logger.error(f"Erro na parada emergencial: {e}")

def signal_handler(signum, frame):
    """Handler para sinais do sistema"""
    logger.info(f"Sinal recebido: {signum}")
    if 'trading_system' in globals():
        trading_system.emergency_stop()
    sys.exit(0)

def main():
    """Função principal"""
    global trading_system

    # Registrar handlers de sinal
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Inicializar sistema
    trading_system = EzOptionsTradingSystem()

    if not trading_system.initialize():
        logger.error("Falha na inicialização do sistema")
        sys.exit(1)

    # Iniciar sistema
    if not trading_system.start():
        logger.error("Falha ao iniciar sistema")
        sys.exit(1)

    # Manter sistema rodando
    try:
        logger.info("Sistema de Trading EzOptions rodando...")
        logger.info("Pressione Ctrl+C para parar")

        # Loop principal
        while trading_system.running:
            time.sleep(60)  # Verificar a cada minuto

            # Mostrar status resumido
            status = trading_system.get_system_status()
            logger.info(f"Status: Sinais detectados: {status['signals_detected']}, Executados: {status['signals_executed']}")

    except KeyboardInterrupt:
        logger.info("Interrupção recebida...")
    finally:
        # Parar sistema
        trading_system.stop()
        logger.info("Sistema encerrado")

if __name__ == "__main__":
    main()