"""
Sistema Avançado de Logs para Auditoria
Registra todas as operações do sistema de trading para auditoria completa
"""

import logging
import json
import gzip
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from pathlib import Path
import threading
from dataclasses import dataclass, asdict
from enum import Enum

logger = logging.getLogger(__name__)

class AuditEventType(Enum):
    """Tipos de eventos para auditoria"""
    SYSTEM_START = "system_start"
    SYSTEM_STOP = "system_stop"
    SIGNAL_DETECTED = "signal_detected"
    SIGNAL_CONFIRMED = "signal_confirmed"
    SIGNAL_EXECUTED = "signal_executed"
    SIGNAL_EXPIRED = "signal_expired"
    SIGNAL_CANCELLED = "signal_cancelled"
    ORDER_PLACED = "order_placed"
    ORDER_FILLED = "order_filled"
    ORDER_MODIFIED = "order_modified"
    ORDER_CANCELLED = "order_cancelled"
    POSITION_OPENED = "position_opened"
    POSITION_CLOSED = "position_closed"
    RISK_VALIDATION = "risk_validation"
    RISK_LIMIT_BREACH = "risk_limit_breach"
    ACCOUNT_UPDATE = "account_update"
    MARKET_DATA_UPDATE = "market_data_update"
    ERROR = "error"
    WARNING = "warning"

@dataclass
class AuditEvent:
    """Evento de auditoria"""
    event_type: AuditEventType
    timestamp: datetime
    session_id: str
    user_id: str
    component: str  # QQQ_Monitor, MT5_AutoTrader, Risk_Manager, etc.
    action: str
    details: Dict[str, Any]
    severity: str = "INFO"  # INFO, WARNING, ERROR, CRITICAL
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict:
        """Converte evento para dicionário"""
        data = asdict(self)
        data['event_type'] = self.event_type.value
        data['timestamp'] = self.timestamp.isoformat()
        return data

class AuditLogger:
    """
    Sistema avançado de auditoria com compressão e rotação de logs
    """

    def __init__(self, config: Dict):
        self.config = config

        # Configurações de log
        self.log_dir = Path(config.get('logging', {}).get('audit_dir', 'audit_logs'))
        self.max_file_size = config.get('logging', {}).get('max_file_size_mb', 100) * 1024 * 1024
        self.backup_count = config.get('logging', {}).get('backup_count', 30)
        self.compression_enabled = config.get('logging', {}).get('compression_enabled', True)

        # Estado do sistema
        self.session_id = self._generate_session_id()
        self.user_id = config.get('user_id', 'system_user')
        self.start_time = datetime.now()

        # Buffers para performance
        self.event_buffer: List[AuditEvent] = []
        self.buffer_size = config.get('buffer_size', 100)
        self.last_flush = datetime.now()

        # Controle de arquivo atual
        self.current_file = None
        self.current_file_size = 0

        # Lock para thread safety
        self._lock = threading.Lock()

        # Criar diretório de logs
        self.log_dir.mkdir(exist_ok=True)

        # Configurar logging padrão
        self._setup_logging()

    def _generate_session_id(self) -> str:
        """Gera ID único para sessão"""
        timestamp = int(datetime.now().timestamp())
        random_suffix = os.urandom(4).hex()
        return f"session_{timestamp}_{random_suffix}"

    def _setup_logging(self):
        """Configura sistema de logging padrão"""
        log_config = self.config.get('logging', {})

        # Configurar logger raiz
        logging.basicConfig(
            level=getattr(logging, log_config.get('level', 'INFO')),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_config.get('file', 'trading_system.log')),
                logging.StreamHandler()
            ]
        )

    def log_event(self, event_type: AuditEventType, component: str, action: str,
                  details: Dict[str, Any], severity: str = "INFO", metadata: Dict = None):
        """
        Registra evento de auditoria

        Args:
            event_type: Tipo do evento
            component: Componente que gerou o evento
            action: Ação realizada
            details: Detalhes do evento
            severity: Severidade (INFO, WARNING, ERROR, CRITICAL)
            metadata: Metadados adicionais
        """
        try:
            event = AuditEvent(
                event_type=event_type,
                timestamp=datetime.now(),
                session_id=self.session_id,
                user_id=self.user_id,
                component=component,
                action=action,
                details=details,
                severity=severity,
                metadata=metadata or {}
            )

            with self._lock:
                self.event_buffer.append(event)

                # Flush buffer se necessário
                if len(self.event_buffer) >= self.buffer_size:
                    self._flush_buffer()

        except Exception as e:
            logger.error(f"Erro ao registrar evento de auditoria: {e}")

    def _flush_buffer(self):
        """Faz flush do buffer para arquivo"""
        try:
            if not self.event_buffer:
                return

            # Escrever eventos no arquivo
            self._write_events_to_file(self.event_buffer)

            # Limpar buffer
            self.event_buffer.clear()
            self.last_flush = datetime.now()

        except Exception as e:
            logger.error(f"Erro ao fazer flush do buffer: {e}")

    def _write_events_to_file(self, events: List[AuditEvent]):
        """Escreve eventos no arquivo de log"""
        try:
            # Determinar arquivo atual
            current_file = self._get_current_log_file()

            # Preparar dados para escrita
            events_data = [event.to_dict() for event in events]

            # Adicionar informações de contexto
            log_entry = {
                'timestamp': datetime.now().isoformat(),
                'session_id': self.session_id,
                'events_count': len(events_data),
                'events': events_data
            }

            # Escrever no arquivo
            with open(current_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')

            # Atualizar tamanho do arquivo
            self.current_file_size = os.path.getsize(current_file)

            # Verificar se precisa rotacionar
            if self.current_file_size > self.max_file_size:
                self._rotate_log_file()

        except Exception as e:
            logger.error(f"Erro ao escrever eventos no arquivo: {e}")

    def _get_current_log_file(self) -> Path:
        """Obtém arquivo de log atual"""
        if self.current_file and os.path.exists(self.current_file):
            return self.current_file

        # Criar novo arquivo baseado na data
        date_str = datetime.now().strftime('%Y%m%d')
        file_name = f"audit_{date_str}.json"

        if self.compression_enabled:
            file_name += '.gz'
            self.current_file = self.log_dir / file_name

            # Se arquivo comprimido não existe, criar arquivo normal primeiro
            if not self.current_file.exists():
                normal_file = self.log_dir / f"audit_{date_str}.json"
                self.current_file = normal_file
        else:
            self.current_file = self.log_dir / file_name

        return self.current_file

    def _rotate_log_file(self):
        """Rotaciona arquivo de log quando atinge tamanho máximo"""
        try:
            if not self.current_file or not self.current_file.exists():
                return

            # Comprimir arquivo atual se necessário
            if self.compression_enabled and not str(self.current_file).endswith('.gz'):
                self._compress_file(self.current_file)

            # Limpar referência para forçar criação de novo arquivo
            self.current_file = None
            self.current_file_size = 0

            logger.info("Arquivo de log rotacionado")

        except Exception as e:
            logger.error(f"Erro ao rotacionar arquivo de log: {e}")

    def _compress_file(self, file_path: Path):
        """Comprime arquivo de log"""
        try:
            compressed_path = file_path.with_suffix(file_path.suffix + '.gz')

            with open(file_path, 'rb') as f_in:
                with gzip.open(compressed_path, 'wt', encoding='utf-8') as f_out:
                    f_out.writelines(f_in)

            # Remover arquivo original
            file_path.unlink()

            logger.info(f"Arquivo comprimido: {compressed_path}")

        except Exception as e:
            logger.error(f"Erro ao comprimir arquivo: {e}")

    def log_system_start(self, component: str, config: Dict):
        """Registra início do sistema"""
        self.log_event(
            AuditEventType.SYSTEM_START,
            component,
            "system_initialization",
            {
                'config_summary': self._sanitize_config(config),
                'start_time': self.start_time.isoformat(),
                'python_version': os.sys.version,
                'platform': os.sys.platform
            },
            severity="INFO"
        )

    def log_system_stop(self, component: str, uptime_seconds: float):
        """Registra parada do sistema"""
        self.log_event(
            AuditEventType.SYSTEM_STOP,
            component,
            "system_shutdown",
            {
                'uptime_seconds': uptime_seconds,
                'stop_time': datetime.now().isoformat(),
                'total_events_logged': len(self.event_buffer)  # Aproximado
            },
            severity="INFO"
        )

    def log_signal_detected(self, component: str, signal: Dict):
        """Registra sinal detectado"""
        self.log_event(
            AuditEventType.SIGNAL_DETECTED,
            component,
            "signal_detection",
            {
                'setup_number': signal.get('setup_number'),
                'signal_type': signal.get('signal_type'),
                'confidence': signal.get('confidence'),
                'current_price': signal.get('current_price'),
                'timestamp': signal.get('timestamp')
            },
            severity="INFO"
        )

    def log_signal_confirmed(self, component: str, signal: Dict):
        """Registra sinal confirmado"""
        self.log_event(
            AuditEventType.SIGNAL_CONFIRMED,
            component,
            "signal_confirmation",
            {
                'setup_number': signal.get('setup_number'),
                'signal_type': signal.get('signal_type'),
                'confidence': signal.get('confidence'),
                'stop_loss': signal.get('stop_loss'),
                'take_profit': signal.get('take_profit'),
                'risk_reward_ratio': signal.get('risk_reward_ratio')
            },
            severity="INFO"
        )

    def log_signal_executed(self, component: str, execution_result: Dict):
        """Registra sinal executado"""
        self.log_event(
            AuditEventType.SIGNAL_EXECUTED,
            component,
            "signal_execution",
            {
                'order_id': execution_result.get('order_id'),
                'execution_price': execution_result.get('price'),
                'volume': execution_result.get('volume'),
                'success': execution_result.get('success'),
                'error': execution_result.get('error')
            },
            severity="INFO" if execution_result.get('success') else "ERROR"
        )

    def log_order_placed(self, component: str, order: Dict):
        """Registra ordem executada"""
        self.log_event(
            AuditEventType.ORDER_PLACED,
            component,
            "order_execution",
            {
                'order_id': order.get('order_id'),
                'symbol': order.get('symbol'),
                'order_type': order.get('order_type'),
                'volume': order.get('volume'),
                'price': order.get('price'),
                'stop_loss': order.get('stop_loss'),
                'take_profit': order.get('take_profit')
            },
            severity="INFO"
        )

    def log_position_opened(self, component: str, position: Dict):
        """Registra posição aberta"""
        self.log_event(
            AuditEventType.POSITION_OPENED,
            component,
            "position_open",
            {
                'ticket': position.get('ticket'),
                'symbol': position.get('symbol'),
                'type': position.get('type'),
                'volume': position.get('volume'),
                'price_open': position.get('price_open'),
                'stop_loss': position.get('sl'),
                'take_profit': position.get('tp')
            },
            severity="INFO"
        )

    def log_position_closed(self, component: str, position: Dict, pnl: float):
        """Registra posição fechada"""
        self.log_event(
            AuditEventType.POSITION_CLOSED,
            component,
            "position_close",
            {
                'ticket': position.get('ticket'),
                'symbol': position.get('symbol'),
                'type': position.get('type'),
                'volume': position.get('volume'),
                'price_open': position.get('price_open'),
                'price_close': position.get('price_current'),
                'pnl': pnl,
                'profit': position.get('profit')
            },
            severity="INFO"
        )

    def log_risk_validation(self, component: str, validation_result: Dict):
        """Registra validação de risco"""
        self.log_event(
            AuditEventType.RISK_VALIDATION,
            component,
            "risk_check",
            {
                'valid': validation_result.get('valid'),
                'reason': validation_result.get('reason'),
                'risk_metrics': validation_result.get('risk_metrics', {}),
                'signal_details': validation_result.get('signal_details', {})
            },
            severity="WARNING" if not validation_result.get('valid') else "INFO"
        )

    def log_risk_limit_breach(self, component: str, breach_details: Dict):
        """Registra violação de limites de risco"""
        self.log_event(
            AuditEventType.RISK_LIMIT_BREACH,
            component,
            "risk_limit_exceeded",
            breach_details,
            severity="CRITICAL"
        )

    def log_account_update(self, component: str, old_balance: float, new_balance: float, reason: str):
        """Registra atualização de conta"""
        self.log_event(
            AuditEventType.ACCOUNT_UPDATE,
            component,
            "account_balance_change",
            {
                'old_balance': old_balance,
                'new_balance': new_balance,
                'change': new_balance - old_balance,
                'change_percent': ((new_balance - old_balance) / old_balance * 100) if old_balance != 0 else 0,
                'reason': reason
            },
            severity="INFO"
        )

    def log_error(self, component: str, error: Exception, context: Dict = None):
        """Registra erro"""
        self.log_event(
            AuditEventType.ERROR,
            component,
            "error_occurred",
            {
                'error_type': type(error).__name__,
                'error_message': str(error),
                'context': context or {}
            },
            severity="ERROR"
        )

    def log_warning(self, component: str, message: str, details: Dict = None):
        """Registra warning"""
        self.log_event(
            AuditEventType.WARNING,
            component,
            "warning_issued",
            {
                'message': message,
                'details': details or {}
            },
            severity="WARNING"
        )

    def _sanitize_config(self, config: Dict) -> Dict:
        """Remove informações sensíveis da configuração"""
        sanitized = config.copy()

        # Remover senhas e informações sensíveis
        sensitive_keys = ['password', 'api_key', 'secret', 'token']

        def remove_sensitive(obj):
            if isinstance(obj, dict):
                return {k: remove_sensitive(v) if k not in sensitive_keys else '***REDACTED***'
                       for k, v in obj.items()}
            elif isinstance(obj, list):
                return [remove_sensitive(item) for item in obj]
            else:
                return obj

        return remove_sensitive(sanitized)

    def get_audit_summary(self, hours: int = 24) -> Dict:
        """Gera resumo de auditoria"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours)

            # Contar eventos por tipo
            event_counts = {}
            error_count = 0
            warning_count = 0

            for event in self.event_buffer:
                if event.timestamp < cutoff_time:
                    continue

                event_type = event.event_type.value
                event_counts[event_type] = event_counts.get(event_type, 0) + 1

                if event.severity in ['ERROR', 'CRITICAL']:
                    error_count += 1
                elif event.severity == 'WARNING':
                    warning_count += 1

            return {
                'session_id': self.session_id,
                'period_hours': hours,
                'total_events': len(self.event_buffer),
                'event_counts': event_counts,
                'error_count': error_count,
                'warning_count': warning_count,
                'generated_at': datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Erro ao gerar resumo de auditoria: {e}")
            return {'error': str(e)}

    def export_audit_log(self, start_date: datetime = None, end_date: datetime = None,
                        output_file: str = None) -> str:
        """
        Exporta logs de auditoria para arquivo

        Args:
            start_date: Data inicial (opcional)
            end_date: Data final (opcional)
            output_file: Arquivo de saída (opcional)

        Returns:
            Caminho do arquivo exportado
        """
        try:
            if output_file is None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                output_file = f"audit_export_{timestamp}.json"

            output_path = Path(output_file)

            # Filtrar eventos por período
            filtered_events = self.event_buffer.copy()

            if start_date:
                filtered_events = [e for e in filtered_events if e.timestamp >= start_date]

            if end_date:
                filtered_events = [e for e in filtered_events if e.timestamp <= end_date]

            # Preparar dados para exportação
            export_data = {
                'export_info': {
                    'session_id': self.session_id,
                    'exported_at': datetime.now().isoformat(),
                    'period': {
                        'start': start_date.isoformat() if start_date else None,
                        'end': end_date.isoformat() if end_date else None
                    },
                    'total_events': len(filtered_events)
                },
                'events': [event.to_dict() for event in filtered_events]
            }

            # Escrever arquivo
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, ensure_ascii=False, indent=2)

            logger.info(f"Logs de auditoria exportados para: {output_path}")
            return str(output_path)

        except Exception as e:
            logger.error(f"Erro ao exportar logs de auditoria: {e}")
            return None

    def force_flush(self):
        """Força flush do buffer"""
        with self._lock:
            self._flush_buffer()

    def cleanup_old_logs(self, days_to_keep: int = 30):
        """Remove logs antigos"""
        try:
            cutoff_date = datetime.now() - timedelta(days=days_to_keep)

            for log_file in self.log_dir.glob("audit_*.json*"):
                try:
                    # Extrair data do nome do arquivo
                    filename = log_file.stem
                    if filename.startswith('audit_'):
                        date_str = filename.split('_')[1]

                        try:
                            file_date = datetime.strptime(date_str, '%Y%m%d')

                            if file_date < cutoff_date:
                                log_file.unlink()
                                logger.info(f"Arquivo de log antigo removido: {log_file}")

                        except ValueError:
                            # Nome do arquivo não segue padrão, pular
                            continue

                except Exception as e:
                    logger.error(f"Erro ao processar arquivo {log_file}: {e}")

        except Exception as e:
            logger.error(f"Erro ao limpar logs antigos: {e}")

# Instância global do auditor
audit_logger = None

def initialize_audit_logger(config_path: str = 'config.json') -> AuditLogger:
    """Inicializa logger de auditoria global"""
    global audit_logger

    try:
        with open(config_path, 'r') as f:
            config = json.load(f)

        audit_logger = AuditLogger(config)

        # Registrar inicialização
        audit_logger.log_system_start("AuditLogger", config)

        logger.info("Sistema de auditoria inicializado")
        return audit_logger

    except Exception as e:
        logger.error(f"Erro ao inicializar sistema de auditoria: {e}")
        return None

def get_audit_logger() -> Optional[AuditLogger]:
    """Obtém instância global do auditor"""
    return audit_logger

# Decoradores para facilitar logging
def audit_log(event_type: AuditEventType, component: str, action: str, severity: str = "INFO"):
    """Decorador para logging automático de funções"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                # Executar função
                result = func(*args, **kwargs)

                # Registrar evento
                if audit_logger:
                    audit_logger.log_event(
                        event_type,
                        component,
                        action,
                        {
                            'function': func.__name__,
                            'args_count': len(args),
                            'kwargs_keys': list(kwargs.keys()),
                            'success': True
                        },
                        severity=severity
                    )

                return result

            except Exception as e:
                # Registrar erro
                if audit_logger:
                    audit_logger.log_event(
                        AuditEventType.ERROR,
                        component,
                        f"{action}_error",
                        {
                            'function': func.__name__,
                            'error': str(e),
                            'success': False
                        },
                        severity="ERROR"
                    )

                raise

        return wrapper
    return decorator

if __name__ == "__main__":
    # Exemplo de uso
    config = {
        'logging': {
            'audit_dir': 'audit_logs',
            'max_file_size_mb': 10,
            'backup_count': 5,
            'compression_enabled': True
        },
        'user_id': 'test_user'
    }

    # Inicializar auditor
    auditor = AuditLogger(config)

    # Exemplos de logging
    auditor.log_system_start("TestComponent", config)
    auditor.log_signal_detected("QQQMonitor", {
        'setup_number': 1,
        'signal_type': 'BUY',
        'confidence': 0.85,
        'current_price': 15000.0
    })

    # Simular erro
    try:
        raise ValueError("Teste de erro")
    except Exception as e:
        auditor.log_error("TestComponent", e, {'context': 'test'})

    # Gerar resumo
    summary = auditor.get_audit_summary(hours=1)
    print(f"Resumo de auditoria: {summary}")

    # Forçar flush
    auditor.force_flush()