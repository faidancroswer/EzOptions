"""
Sistema de Verificação de Duplicatas
Evita execuções repetidas de sinais similares baseado em critérios avançados
"""

import logging
import hashlib
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Set
import threading
from dataclasses import dataclass, asdict
from enum import Enum

logger = logging.getLogger(__name__)

class DuplicateStatus(Enum):
    """Status de verificação de duplicata"""
    UNIQUE = "unique"
    DUPLICATE = "duplicate"
    SIMILAR = "similar"
    EXPIRED = "expired"

@dataclass
class SignalFingerprint:
    """Impressão digital única do sinal"""
    setup_number: int
    signal_type: str
    price_level: float  # Nível de preço arredondado
    time_window: str    # Janela de tempo (ex: "14:30-14:45")
    market_context: str # Contexto de mercado
    hash_key: str       # Hash único

    def to_dict(self) -> Dict:
        """Converte fingerprint para dicionário"""
        data = asdict(self)
        return data

@dataclass
class DuplicateCheckResult:
    """Resultado da verificação de duplicata"""
    status: DuplicateStatus
    confidence: float
    similar_signals: List[str]
    reason: str
    fingerprint: SignalFingerprint
    timestamp: datetime

    def to_dict(self) -> Dict:
        """Converte resultado para dicionário"""
        data = asdict(self)
        data['status'] = self.status.value
        data['timestamp'] = self.timestamp.isoformat()
        return data

class DuplicateChecker:
    """
    Sistema avançado de verificação de duplicatas
    """

    def __init__(self, config: Dict):
        self.config = config

        # Configurações de verificação
        self.time_window_minutes = config.get('duplicate_check', {}).get('time_window_minutes', 15)
        self.price_tolerance_percent = config.get('duplicate_check', {}).get('price_tolerance_percent', 0.1)
        self.max_similar_signals = config.get('duplicate_check', {}).get('max_similar_signals', 3)
        self.duplicate_memory_hours = config.get('duplicate_check', {}).get('memory_hours', 24)

        # Estado do sistema
        self.signal_history: List[Tuple[SignalFingerprint, datetime]] = []
        self.duplicate_cache: Dict[str, DuplicateCheckResult] = {}

        # Lock para thread safety
        self._lock = threading.Lock()

        # Cleanup automático
        self._start_cleanup_timer()

    def check_duplicate(self, signal: Dict) -> DuplicateCheckResult:
        """
        Verifica se sinal é duplicata

        Args:
            signal: Sinal de trading completo

        Returns:
            DuplicateCheckResult com status e detalhes
        """
        try:
            # Gerar fingerprint do sinal
            fingerprint = self._generate_fingerprint(signal)

            with self._lock:
                # Verificar cache primeiro
                cache_key = fingerprint.hash_key
                if cache_key in self.duplicate_cache:
                    cached_result = self.duplicate_cache[cache_key]

                    # Verificar se cache ainda é válido
                    if datetime.now() - cached_result.timestamp < timedelta(minutes=self.time_window_minutes):
                        return cached_result

                # Verificar contra histórico
                similar_signals = []
                duplicate_confidence = 0.0

                for hist_fingerprint, timestamp in self.signal_history:
                    # Verificar se ainda está dentro da janela de tempo
                    if datetime.now() - timestamp > timedelta(hours=self.duplicate_memory_hours):
                        continue

                    # Calcular similaridade
                    similarity = self._calculate_similarity(fingerprint, hist_fingerprint)

                    if similarity > 0.8:  # 80% similar
                        similar_signals.append(hist_fingerprint.hash_key)

                        if similarity > duplicate_confidence:
                            duplicate_confidence = similarity

                # Determinar status
                if duplicate_confidence > 0.95:
                    status = DuplicateStatus.DUPLICATE
                    reason = f"Sinal duplicado detectado (confiança: {duplicate_confidence:.2f})"
                elif duplicate_confidence > 0.8:
                    status = DuplicateStatus.SIMILAR
                    reason = f"Sinal similar detectado (confiança: {duplicate_confidence:.2f})"
                else:
                    status = DuplicateStatus.UNIQUE
                    reason = "Sinal único"

                # Criar resultado
                result = DuplicateCheckResult(
                    status=status,
                    confidence=duplicate_confidence,
                    similar_signals=similar_signals,
                    reason=reason,
                    fingerprint=fingerprint,
                    timestamp=datetime.now()
                )

                # Adicionar ao histórico se for único ou similar
                if status in [DuplicateStatus.UNIQUE, DuplicateStatus.SIMILAR]:
                    self.signal_history.append((fingerprint, datetime.now()))

                    # Limitar tamanho do histórico
                    if len(self.signal_history) > 1000:
                        self.signal_history = self.signal_history[-1000:]

                # Atualizar cache
                self.duplicate_cache[cache_key] = result

                return result

        except Exception as e:
            logger.error(f"Erro ao verificar duplicata: {e}")

            # Retornar resultado padrão em caso de erro
            return DuplicateCheckResult(
                status=DuplicateStatus.UNIQUE,
                confidence=0.0,
                similar_signals=[],
                reason=f"Erro na verificação: {e}",
                fingerprint=self._generate_fingerprint(signal),
                timestamp=datetime.now()
            )

    def _generate_fingerprint(self, signal: Dict) -> SignalFingerprint:
        """Gera impressão digital única do sinal"""
        try:
            # Extrair componentes principais
            setup_number = signal.get('setup_number', 0)
            signal_type = signal.get('signal_type', 'BUY')
            current_price = signal.get('current_price', 0)

            # Arredondar preço para reduzir granularidade
            price_tolerance = current_price * (self.price_tolerance_percent / 100)
            price_level = round(current_price / price_tolerance) * price_tolerance

            # Calcular janela de tempo
            current_minute = datetime.now().minute
            window_start = (current_minute // self.time_window_minutes) * self.time_window_minutes
            window_end = window_start + self.time_window_minutes
            time_window = f"{window_start:02d}:{window_end:02d}"

            # Gerar contexto de mercado baseado em indicadores
            market_context = self._generate_market_context(signal)

            # Criar chave única
            key_components = [
                str(setup_number),
                signal_type,
                f"{price_level:.2f}",
                time_window,
                market_context
            ]

            key_string = "|".join(key_components)
            hash_key = hashlib.md5(key_string.encode()).hexdigest()

            return SignalFingerprint(
                setup_number=setup_number,
                signal_type=signal_type,
                price_level=price_level,
                time_window=time_window,
                market_context=market_context,
                hash_key=hash_key
            )

        except Exception as e:
            logger.error(f"Erro ao gerar fingerprint: {e}")

            # Fingerprint mínima de emergência
            return SignalFingerprint(
                setup_number=signal.get('setup_number', 0),
                signal_type=signal.get('signal_type', 'BUY'),
                price_level=signal.get('current_price', 0),
                time_window=datetime.now().strftime('%H:%M'),
                market_context='unknown',
                hash_key=hashlib.md5(str(datetime.now().timestamp()).encode()).hexdigest()
            )

    def _generate_market_context(self, signal: Dict) -> str:
        """Gera contexto de mercado baseado em indicadores"""
        try:
            context_indicators = []

            # Contexto baseado em preço relativo à VWAP
            vwap_data = signal.get('vwap_data', {})
            current_price = signal.get('current_price', 0)
            vwap_price = vwap_data.get('current_vwap', current_price)

            if current_price > 0 and vwap_price > 0:
                price_vs_vwap = (current_price - vwap_price) / vwap_price
                if price_vs_vwap > 0.005:  # 0.5% acima
                    context_indicators.append('above_vwap')
                elif price_vs_vwap < -0.005:  # 0.5% abaixo
                    context_indicators.append('below_vwap')
                else:
                    context_indicators.append('at_vwap')

            # Contexto baseado em volatilidade (Bollinger Bands)
            bollinger_data = signal.get('bollinger_data', {})
            if bollinger_data:
                upper_band = bollinger_data.get('upper_band', 0)
                lower_band = bollinger_data.get('lower_band', 0)

                if current_price > 0:
                    if current_price > upper_band:
                        context_indicators.append('above_bb_upper')
                    elif current_price < lower_band:
                        context_indicators.append('below_bb_lower')
                    else:
                        context_indicators.append('inside_bb')

            # Contexto baseado em momentum (RSI se disponível)
            technical_data = signal.get('technical_indicators', {})
            rsi = technical_data.get('rsi', 50)

            if rsi > 70:
                context_indicators.append('overbought')
            elif rsi < 30:
                context_indicators.append('oversold')
            else:
                context_indicators.append('neutral_momentum')

            # Combinar indicadores em contexto único
            if len(context_indicators) >= 2:
                return "_".join(context_indicators[:2])  # Máximo 2 indicadores
            elif context_indicators:
                return context_indicators[0]
            else:
                return 'unknown'

        except Exception as e:
            logger.error(f"Erro ao gerar contexto de mercado: {e}")
            return 'unknown'

    def _calculate_similarity(self, fingerprint1: SignalFingerprint, fingerprint2: SignalFingerprint) -> float:
        """Calcula similaridade entre dois fingerprints"""
        try:
            similarity_score = 0.0
            total_components = 0

            # 1. Similaridade de setup (peso: 0.3)
            if fingerprint1.setup_number == fingerprint2.setup_number:
                similarity_score += 0.3
            total_components += 0.3

            # 2. Similaridade de tipo de sinal (peso: 0.2)
            if fingerprint1.signal_type == fingerprint2.signal_type:
                similarity_score += 0.2
            total_components += 0.2

            # 3. Similaridade de nível de preço (peso: 0.3)
            price_diff = abs(fingerprint1.price_level - fingerprint2.price_level)
            price_tolerance = max(fingerprint1.price_level, fingerprint2.price_level) * 0.001  # 0.1%

            if price_diff <= price_tolerance:
                similarity_score += 0.3
            elif price_diff <= price_tolerance * 2:
                similarity_score += 0.15  # Similaridade parcial
            total_components += 0.3

            # 4. Similaridade de janela de tempo (peso: 0.1)
            if fingerprint1.time_window == fingerprint2.time_window:
                similarity_score += 0.1
            total_components += 0.1

            # 5. Similaridade de contexto de mercado (peso: 0.1)
            if fingerprint1.market_context == fingerprint2.market_context:
                similarity_score += 0.1
            total_components += 0.1

            # Calcular média ponderada
            if total_components > 0:
                return similarity_score / total_components
            else:
                return 0.0

        except Exception as e:
            logger.error(f"Erro ao calcular similaridade: {e}")
            return 0.0

    def _start_cleanup_timer(self):
        """Inicia timer para limpeza automática"""
        def cleanup_loop():
            while True:
                try:
                    time.sleep(3600)  # Limpar a cada hora
                    self._cleanup_expired_signals()
                except Exception as e:
                    logger.error(f"Erro no cleanup automático: {e}")

        cleanup_thread = threading.Thread(target=cleanup_loop, daemon=True)
        cleanup_thread.start()
        logger.info("Timer de cleanup iniciado")

    def _cleanup_expired_signals(self):
        """Remove sinais expirados do histórico"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=self.duplicate_memory_hours)

            with self._lock:
                # Limpar histórico
                original_count = len(self.signal_history)
                self.signal_history = [
                    (fp, timestamp) for fp, timestamp in self.signal_history
                    if timestamp > cutoff_time
                ]

                cleaned_count = len(self.signal_history)

                # Limpar cache
                expired_cache_keys = [
                    key for key, result in self.duplicate_cache.items()
                    if datetime.now() - result.timestamp > timedelta(minutes=self.time_window_minutes * 2)
                ]

                for key in expired_cache_keys:
                    del self.duplicate_cache[key]

                if original_count != cleaned_count:
                    logger.info(f"Cleanup realizado: {original_count - cleaned_count} sinais removidos")

        except Exception as e:
            logger.error(f"Erro no cleanup: {e}")

    def get_duplicate_stats(self) -> Dict:
        """Retorna estatísticas de duplicatas"""
        try:
            with self._lock:
                # Estatísticas básicas
                total_signals = len(self.signal_history)
                unique_signals = len(set(fp.hash_key for fp, _ in self.signal_history))

                # Contar por status no cache
                status_counts = {}
                for result in self.duplicate_cache.values():
                    status = result.status.value
                    status_counts[status] = status_counts.get(status, 0) + 1

                # Sinais recentes (última hora)
                recent_cutoff = datetime.now() - timedelta(hours=1)
                recent_signals = [
                    (fp, timestamp) for fp, timestamp in self.signal_history
                    if timestamp > recent_cutoff
                ]

                return {
                    'total_signals_tracked': total_signals,
                    'unique_signals': unique_signals,
                    'duplicate_rate': (total_signals - unique_signals) / max(total_signals, 1) * 100,
                    'cache_size': len(self.duplicate_cache),
                    'status_distribution': status_counts,
                    'recent_signals': len(recent_signals),
                    'memory_hours': self.duplicate_memory_hours,
                    'last_cleanup': datetime.now().isoformat()
                }

        except Exception as e:
            logger.error(f"Erro ao obter estatísticas: {e}")
            return {'error': str(e)}

    def force_cleanup(self):
        """Força limpeza manual"""
        self._cleanup_expired_signals()

    def export_duplicate_log(self, filename: str = None) -> str:
        """Exporta log de duplicatas para análise"""
        try:
            if filename is None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"duplicate_log_{timestamp}.json"

            export_data = {
                'export_info': {
                    'timestamp': datetime.now().isoformat(),
                    'total_signals': len(self.signal_history),
                    'cache_size': len(self.duplicate_cache),
                    'config': {
                        'time_window_minutes': self.time_window_minutes,
                        'price_tolerance_percent': self.price_tolerance_percent,
                        'memory_hours': self.duplicate_memory_hours
                    }
                },
                'signal_history': [
                    {
                        'fingerprint': fp.to_dict(),
                        'detected_at': timestamp.isoformat()
                    }
                    for fp, timestamp in self.signal_history[-100:]  # Últimos 100
                ],
                'duplicate_cache': [
                    {
                        'hash_key': key,
                        'result': result.to_dict()
                    }
                    for key, result in list(self.duplicate_cache.items())[-100:]  # Últimos 100
                ]
            }

            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, ensure_ascii=False, indent=2)

            logger.info(f"Log de duplicatas exportado: {filename}")
            return filename

        except Exception as e:
            logger.error(f"Erro ao exportar log de duplicatas: {e}")
            return None

class SignalDeduplicator:
    """
    Deduplicador avançado de sinais com regras específicas por setup
    """

    def __init__(self, config: Dict):
        self.config = config
        self.duplicate_checker = DuplicateChecker(config)

        # Regras específicas por setup
        self.setup_rules = {
            1: {  # Bullish Breakout
                'min_interval_minutes': 30,
                'price_tolerance_percent': 0.2,
                'allow_similar_if_confidence_higher': True
            },
            2: {  # Bearish Breakout
                'min_interval_minutes': 30,
                'price_tolerance_percent': 0.2,
                'allow_similar_if_confidence_higher': True
            },
            3: {  # Pullback Topo
                'min_interval_minutes': 45,
                'price_tolerance_percent': 0.15,
                'allow_similar_if_confidence_higher': False
            },
            4: {  # Pullback Fundo
                'min_interval_minutes': 45,
                'price_tolerance_percent': 0.15,
                'allow_similar_if_confidence_higher': False
            },
            5: {  # Consolidação
                'min_interval_minutes': 60,
                'price_tolerance_percent': 0.1,
                'allow_similar_if_confidence_higher': False
            },
            6: {  # Proteção
                'min_interval_minutes': 20,
                'price_tolerance_percent': 0.25,
                'allow_similar_if_confidence_higher': True
            }
        }

    def should_process_signal(self, signal: Dict) -> Tuple[bool, str, Dict]:
        """
        Determina se sinal deve ser processado baseado em regras de deduplicação

        Returns:
            Tuple (processar, razão, detalhes)
        """
        try:
            # Verificação básica de duplicata
            check_result = self.duplicate_checker.check_duplicate(signal)

            # Se for duplicata exata, rejeitar
            if check_result.status == DuplicateStatus.DUPLICATE:
                return False, f"Sinal duplicado: {check_result.reason}", check_result.to_dict()

            # Verificar regras específicas do setup
            setup_number = signal.get('setup_number', 0)
            setup_rule = self.setup_rules.get(setup_number, {})

            if not setup_rule:
                # Setup sem regras específicas, usar verificação padrão
                if check_result.status == DuplicateStatus.SIMILAR:
                    return False, f"Sinal similar detectado: {check_result.reason}", check_result.to_dict()
                else:
                    return True, "Sinal aprovado", check_result.to_dict()

            # Aplicar regras específicas do setup
            min_interval = setup_rule.get('min_interval_minutes', 30)
            price_tolerance = setup_rule.get('price_tolerance_percent', 0.1) / 100

            # Verificar intervalo mínimo
            if check_result.similar_signals:
                # Verificar se sinais similares estão dentro do intervalo mínimo
                for similar_hash in check_result.similar_signals:
                    # Buscar timestamp do sinal similar
                    for fp, timestamp in self.duplicate_checker.signal_history:
                        if fp.hash_key == similar_hash:
                            time_diff = datetime.now() - timestamp

                            if time_diff.total_seconds() < (min_interval * 60):
                                return False, f"Intervalo mínimo não respeitado: {time_diff.total_seconds():.0f}s < {min_interval*60}s", {
                                    'rule': 'min_interval',
                                    'required_minutes': min_interval,
                                    'actual_seconds': time_diff.total_seconds(),
                                    'similar_signal': similar_hash
                                }

            # Verificar tolerância de preço específica
            current_price = signal.get('current_price', 0)
            price_tolerance_abs = current_price * price_tolerance

            # Verificar sinais similares dentro da tolerância de preço
            for fp, timestamp in self.duplicate_checker.signal_history:
                if fp.hash_key in check_result.similar_signals:
                    price_diff = abs(fp.price_level - current_price)

                    if price_diff <= price_tolerance_abs:
                        # Verificar se devemos permitir se confiança for maior
                        allow_if_higher_confidence = setup_rule.get('allow_similar_if_confidence_higher', False)

                        if allow_if_higher_confidence:
                            # Comparar confiança com sinais similares recentes
                            similar_confidence = self._get_similar_signal_confidence(fp.hash_key)

                            if signal.get('confidence', 0) <= similar_confidence:
                                return False, f"Sinal similar com confiança maior já existe: {similar_confidence:.3f}", {
                                    'rule': 'higher_confidence_exists',
                                    'current_confidence': signal.get('confidence', 0),
                                    'existing_confidence': similar_confidence
                                }

            # Sinal aprovado
            return True, "Sinal aprovado após verificação de duplicatas", check_result.to_dict()

        except Exception as e:
            logger.error(f"Erro na deduplicação: {e}")
            return True, f"Erro na verificação: {e}", {}

    def _get_similar_signal_confidence(self, signal_hash: str) -> float:
        """Obtém confiança de sinal similar"""
        try:
            # Esta é uma implementação simplificada
            # Em produção, seria armazenado junto com o fingerprint
            return 0.0

        except Exception as e:
            logger.error(f"Erro ao obter confiança de sinal similar: {e}")
            return 0.0

    def get_deduplication_stats(self) -> Dict:
        """Retorna estatísticas de deduplicação"""
        try:
            base_stats = self.duplicate_checker.get_duplicate_stats()

            # Adicionar estatísticas específicas
            setup_dedup_stats = {}

            for setup_number, rule in self.setup_rules.items():
                # Contar sinais por setup
                setup_signals = [
                    fp for fp, _ in self.duplicate_checker.signal_history
                    if fp.setup_number == setup_number
                ]

                unique_setup_signals = len(set(fp.hash_key for fp in setup_signals))

                setup_dedup_stats[setup_number] = {
                    'total_signals': len(setup_signals),
                    'unique_signals': unique_setup_signals,
                    'deduplication_rate': (len(setup_signals) - unique_setup_signals) / max(len(setup_signals), 1) * 100,
                    'rules': rule
                }

            return {
                'base_stats': base_stats,
                'setup_stats': setup_dedup_stats,
                'total_setups': len(self.setup_rules)
            }

        except Exception as e:
            logger.error(f"Erro ao obter estatísticas de deduplicação: {e}")
            return {'error': str(e)}

# Função para criar sistema completo de verificação de duplicatas
def create_duplicate_checking_system(config_path: str = 'config.json'):
    """Cria sistema completo de verificação de duplicatas"""
    try:
        # Carregar configuração
        with open(config_path, 'r') as f:
            config = json.load(f)

        # Inicializar componentes
        deduplicator = SignalDeduplicator(config)

        logger.info("Sistema de verificação de duplicatas inicializado")
        logger.info(f"Janela de tempo: {deduplicator.duplicate_checker.time_window_minutes} minutos")
        logger.info(f"Tolerância de preço: {deduplicator.duplicate_checker.price_tolerance_percent}%")

        return {
            'deduplicator': deduplicator,
            'duplicate_checker': deduplicator.duplicate_checker,
            'config': config
        }

    except Exception as e:
        logger.error(f"Erro ao criar sistema de verificação de duplicatas: {e}")
        return None

if __name__ == "__main__":
    # Exemplo de uso
    config = {
        'duplicate_check': {
            'time_window_minutes': 15,
            'price_tolerance_percent': 0.1,
            'max_similar_signals': 3,
            'memory_hours': 24
        }
    }

    # Inicializar sistema
    system = create_duplicate_checking_system()

    if system:
        deduplicator = system['deduplicator']

        # Exemplo de sinais para teste
        test_signals = [
            {
                'setup_number': 1,
                'signal_type': 'BUY',
                'current_price': 15000.0,
                'confidence': 0.85,
                'vwap_data': {'current_vwap': 14950.0},
                'technical_indicators': {'rsi': 65}
            },
            {
                'setup_number': 1,
                'signal_type': 'BUY',
                'current_price': 15005.0,  # Muito próximo do anterior
                'confidence': 0.82,
                'vwap_data': {'current_vwap': 14950.0},
                'technical_indicators': {'rsi': 65}
            },
            {
                'setup_number': 2,
                'signal_type': 'SELL',
                'current_price': 15000.0,
                'confidence': 0.88,
                'vwap_data': {'current_vwap': 15050.0},
                'technical_indicators': {'rsi': 35}
            }
        ]

        for i, signal in enumerate(test_signals):
            should_process, reason, details = deduplicator.should_process_signal(signal)

            print(f"Sinal {i+1}: {should_process}")
            print(f"Razão: {reason}")
            print(f"Detalhes: {details}")
            print("---")

        # Mostrar estatísticas
        stats = deduplicator.get_deduplication_stats()
        print(f"Estatísticas: {stats}")
    else:
        print("Erro ao inicializar sistema de verificação de duplicatas")