"""
Trading Setups - Definições Corretas Baseadas nas Especificações Fornecidas
Implementação dos 6 setups de trading com condições precisas
"""

import logging
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import numpy as np

logger = logging.getLogger(__name__)

class TradingSetupsCorrected:
    """
    Implementação correta dos 6 setups baseada nas especificações fornecidas
    """

    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.setup_history = []

    def analyze_all_setups(self, data: Dict) -> Dict:
        """Analisa todos os 6 setups e retorna resultados"""
        results = {}

        try:
            # Setup #1: Bullish Breakout
            results['setup_1'] = self.analyze_bullish_breakout(data)

            # Setup #2: Bearish Breakout
            results['setup_2'] = self.analyze_bearish_breakout(data)

            # Setup #3: Pullback no Topo
            results['setup_3'] = self.analyze_pullback_top(data)

            # Setup #4: Pullback no Fundo
            results['setup_4'] = self.analyze_pullback_bottom(data)

            # Setup #5: Mercado consolidado (VWAP)
            results['setup_5'] = self.analyze_consolidated_market(data)

            # Setup #6: Proteção contra Gama Negativo
            results['setup_6'] = self.analyze_negative_gamma_protection(data)

            # Registrar histórico
            self._record_setup_history(results, data)

            return results

        except Exception as e:
            logger.error(f"Erro ao analisar setups: {e}")
            return {}

    def analyze_bullish_breakout(self, data: Dict) -> Dict:
        """
        Setup #1: Alvo Acima (BULLISH BREAKOUT)
        Condições específicas das imagens:
        - Preço operando em CHARM positivo
        - CHARM crescente até o alvo
        - Barras de DELTA positivo até o alvo
        - Maior barra de GAMMA acima do preço (preferencialmente no alvo)
        - Sem barreiras de GAMMA abaixo do preço
        - Confirmação com Price Action: Rompimento do primeiro desvio da VWAP para cima
        - Stop Loss: Ativado se o preço voltar abaixo do ponto de virada da estrutura de CHARM
        """
        try:
            gamma_data = data.get('greeks_data', {}).get('gamma', {})
            delta_data = data.get('greeks_data', {}).get('delta', {})
            charm_data = data.get('greeks_data', {}).get('charm', {})
            vwap_data = data.get('vwap_data', {})
            current_price = data.get('current_price', 0)

            # 1. Preço operando em CHARM positivo
            charm_positive = charm_data.get('current', 0) > 0

            # 2. CHARM crescente até o alvo
            charm_growing = charm_data.get('growing_trend', False) and charm_data.get('direction_up', False)

            # 3. Barras de DELTA positivo até o alvo
            delta_positive_bars = delta_data.get('positive_bars_upward', False)

            # 4. Maior barra de GAMMA acima do preço (preferencialmente no alvo)
            max_gamma_price = gamma_data.get('max_level_price', 0)
            gamma_above_price = max_gamma_price > current_price

            # 5. Sem barreiras de GAMMA abaixo do preço
            no_gamma_barriers_below = not self._check_gamma_barriers_below_price(gamma_data, current_price)

            # 6. Confirmação com Price Action: Rompimento do primeiro desvio da VWAP para cima
            vwap_breakout_up = self._check_vwap_breakout_up(vwap_data, current_price)

            conditions = [charm_positive, charm_growing, delta_positive_bars,
                         gamma_above_price, no_gamma_barriers_below, vwap_breakout_up]
            confidence = sum(conditions) / len(conditions)
            confirmed = confidence >= 0.8  # Requer 80% de confiança para este setup

            return {
                'name': 'Alvo Acima (BULLISH BREAKOUT)',
                'confirmed': confirmed,
                'confidence': confidence,
                'signal': 'BUY' if confirmed else None,
                'timestamp': datetime.now(),
                'stop_loss_trigger': 'price_below_charm_structure_turn_point',
                'conditions': {
                    'charm_positive': charm_positive,
                    'charm_growing': charm_growing,
                    'delta_positive_bars': delta_positive_bars,
                    'gamma_above_price': gamma_above_price,
                    'no_gamma_barriers_below': no_gamma_barriers_below,
                    'vwap_breakout_up': vwap_breakout_up
                }
            }

        except Exception as e:
            logger.error(f"Erro no Setup #1: {e}")
            return {'confirmed': False, 'error': str(e)}

    def analyze_bearish_breakout(self, data: Dict) -> Dict:
        """
        Setup #2: Alvo Abaixo (BEARISH BREAKOUT)
        Condições específicas das imagens:
        - Preço operando em CHARM negativo
        - CHARM decrescente até o alvo
        - Barras de DELTA negativo até o alvo
        - Maior barra de GAMMA abaixo do preço (preferencialmente no alvo)
        - Sem barreiras de GAMMA acima do preço
        - Confirmação com Price Action: Rompimento do primeiro desvio da VWAP para baixo
        - Stop Loss: Ativado se o preço voltar acima do ponto de virada ou perder a estrutura de CHARM
        """
        try:
            gamma_data = data.get('greeks_data', {}).get('gamma', {})
            delta_data = data.get('greeks_data', {}).get('delta', {})
            charm_data = data.get('greeks_data', {}).get('charm', {})
            vwap_data = data.get('vwap_data', {})
            current_price = data.get('current_price', 0)

            # 1. Preço operando em CHARM negativo
            charm_negative = charm_data.get('current', 0) < 0

            # 2. CHARM decrescente até o alvo
            charm_decreasing = charm_data.get('decreasing_trend', False) and charm_data.get('direction_down', False)

            # 3. Barras de DELTA negativo até o alvo
            delta_negative_bars = delta_data.get('negative_bars_downward', False)

            # 4. Maior barra de GAMMA abaixo do preço (preferencialmente no alvo)
            max_gamma_price = gamma_data.get('max_level_price', 0)
            gamma_below_price = max_gamma_price < current_price

            # 5. Sem barreiras de GAMMA acima do preço
            no_gamma_barriers_above = not self._check_gamma_barriers_above_price(gamma_data, current_price)

            # 6. Confirmação com Price Action: Rompimento do primeiro desvio da VWAP para baixo
            vwap_breakout_down = self._check_vwap_breakout_down(vwap_data, current_price)

            conditions = [charm_negative, charm_decreasing, delta_negative_bars,
                         gamma_below_price, no_gamma_barriers_above, vwap_breakout_down]
            confidence = sum(conditions) / len(conditions)
            confirmed = confidence >= 0.8  # Requer 80% de confiança para este setup

            return {
                'name': 'Alvo Abaixo (BEARISH BREAKOUT)',
                'confirmed': confirmed,
                'confidence': confidence,
                'signal': 'SELL' if confirmed else None,
                'timestamp': datetime.now(),
                'stop_loss_trigger': 'price_above_charm_structure_turn_point',
                'conditions': {
                    'charm_negative': charm_negative,
                    'charm_decreasing': charm_decreasing,
                    'delta_negative_bars': delta_negative_bars,
                    'gamma_below_price': gamma_below_price,
                    'no_gamma_barriers_above': no_gamma_barriers_above,
                    'vwap_breakout_down': vwap_breakout_down
                }
            }

        except Exception as e:
            logger.error(f"Erro no Setup #2: {e}")
            return {'confirmed': False, 'error': str(e)}

    def analyze_pullback_top(self, data: Dict) -> Dict:
        """
        Setup #3: Reversão para Baixo (PULLBACK NO TOPO)
        Condições específicas das imagens:
        - Preço na maior barra de GAMMA positiva ou área de Gamma Flip
        - Próxima barra de GAMMA acima muito menor
        - Maior barra de CHARM positivo antes de barra menor (indica perda de força)
        - Última barra de DELTA positivo atingida (indica esgotamento de demanda)
        - Confirmação extra: Volume neutro ou fechando nos strikes acima (DDF LEVELS)
        - Stop Loss: Ativado se DELTA e CHARM continuarem crescendo acima do nível de entrada
        """
        try:
            gamma_data = data.get('greeks_data', {}).get('gamma', {})
            delta_data = data.get('greeks_data', {}).get('delta', {})
            charm_data = data.get('greeks_data', {}).get('charm', {})
            volume_data = data.get('volume_data', {})
            current_price = data.get('current_price', 0)

            # 1. Preço na maior barra de GAMMA positiva ou área de Gamma Flip
            price_at_major_gamma_positive = self._check_price_at_major_gamma_positive(gamma_data, current_price)

            # 2. Próxima barra de GAMMA acima muito menor
            next_gamma_much_smaller = self._check_next_gamma_much_smaller(gamma_data, current_price)

            # 3. Maior barra de CHARM positivo antes de barra menor (indica perda de força)
            charm_positive_before_smaller = self._check_charm_positive_before_smaller(charm_data, current_price)

            # 4. Última barra de DELTA positivo atingida (indica esgotamento de demanda)
            last_delta_positive_reached = self._check_last_delta_positive_reached(delta_data, current_price)

            # 5. Confirmação extra: Volume neutro ou fechando nos strikes acima (DDF LEVELS)
            volume_neutral_above_strikes = self._check_volume_neutral_above_strikes(volume_data, current_price)

            conditions = [price_at_major_gamma_positive, next_gamma_much_smaller,
                         charm_positive_before_smaller, last_delta_positive_reached, volume_neutral_above_strikes]
            confidence = sum(conditions) / len(conditions)
            confirmed = confidence >= 0.8  # Requer 80% de confiança para reversão

            return {
                'name': 'Reversão para Baixo (PULLBACK NO TOPO)',
                'confirmed': confirmed,
                'confidence': confidence,
                'signal': 'SELL' if confirmed else None,
                'timestamp': datetime.now(),
                'stop_loss_trigger': 'delta_charm_continue_growing_above_entry',
                'conditions': {
                    'price_at_major_gamma_positive': price_at_major_gamma_positive,
                    'next_gamma_much_smaller': next_gamma_much_smaller,
                    'charm_positive_before_smaller': charm_positive_before_smaller,
                    'last_delta_positive_reached': last_delta_positive_reached,
                    'volume_neutral_above_strikes': volume_neutral_above_strikes
                }
            }

        except Exception as e:
            logger.error(f"Erro no Setup #3: {e}")
            return {'confirmed': False, 'error': str(e)}

    def analyze_pullback_bottom(self, data: Dict) -> Dict:
        """
        Setup #4: Reversão para Cima (PULLBACK NO FUNDO)
        Condições específicas das imagens:
        - Preço na maior barra de GAMMA negativa ou área de Gamma Flip
        - Próxima barra de GAMMA abaixo muito menor
        - Maior barra de CHARM negativo antes de barra menor (indica perda de força vendedora)
        - Última barra de DELTA negativo atingida (indica exaustão de oferta)
        - Confirmação extra: Volume neutro ou ausente nos strikes abaixo (DDF LEVELS)
        - Stop Loss: Ativado se DELTA e CHARM continuarem crescendo abaixo do nível de entrada
        """
        try:
            gamma_data = data.get('greeks_data', {}).get('gamma', {})
            delta_data = data.get('greeks_data', {}).get('delta', {})
            charm_data = data.get('greeks_data', {}).get('charm', {})
            volume_data = data.get('volume_data', {})
            current_price = data.get('current_price', 0)

            # 1. Preço na maior barra de GAMMA negativa ou área de Gamma Flip
            price_at_major_gamma_negative = self._check_price_at_major_gamma_negative(gamma_data, current_price)

            # 2. Próxima barra de GAMMA abaixo muito menor
            next_gamma_below_much_smaller = self._check_next_gamma_below_much_smaller(gamma_data, current_price)

            # 3. Maior barra de CHARM negativo antes de barra menor (indica perda de força vendedora)
            charm_negative_before_smaller = self._check_charm_negative_before_smaller(charm_data, current_price)

            # 4. Última barra de DELTA negativo atingida (indica exaustão de oferta)
            last_delta_negative_reached = self._check_last_delta_negative_reached(delta_data, current_price)

            # 5. Confirmação extra: Volume neutro ou ausente nos strikes abaixo (DDF LEVELS)
            volume_neutral_below_strikes = self._check_volume_neutral_below_strikes(volume_data, current_price)

            conditions = [price_at_major_gamma_negative, next_gamma_below_much_smaller,
                         charm_negative_before_smaller, last_delta_negative_reached, volume_neutral_below_strikes]
            confidence = sum(conditions) / len(conditions)
            confirmed = confidence >= 0.8  # Requer 80% de confiança para reversão

            return {
                'name': 'Reversão para Cima (PULLBACK NO FUNDO)',
                'confirmed': confirmed,
                'confidence': confidence,
                'signal': 'BUY' if confirmed else None,
                'timestamp': datetime.now(),
                'stop_loss_trigger': 'delta_charm_continue_growing_below_entry',
                'conditions': {
                    'price_at_major_gamma_negative': price_at_major_gamma_negative,
                    'next_gamma_below_much_smaller': next_gamma_below_much_smaller,
                    'charm_negative_before_smaller': charm_negative_before_smaller,
                    'last_delta_negative_reached': last_delta_negative_reached,
                    'volume_neutral_below_strikes': volume_neutral_below_strikes
                }
            }

        except Exception as e:
            logger.error(f"Erro no Setup #4: {e}")
            return {'confirmed': False, 'error': str(e)}

    def analyze_consolidated_market(self, data: Dict) -> Dict:
        """
        Setup #5: Consolidação (MERCADO CONSOLIDADO)
        Condições específicas das imagens:
        - Maior barra de GAMMA posicionada no centro do range (normalmente na VWAP)
        - Maior barra de DELTA do range
        - CHARM neutro ou em Flip, sem direção clara
        - Maior barra de THETA DECAY no centro do range
        - VWAP e seus desvios formando linhas retas, indicando equilíbrio
        - Confirmação extra: Volume equilibrado nos strikes (DDF) e no Heat Map
        - Stop Loss: Ativado caso haja rompimento claro do range com confirmação do CHARM
        """
        try:
            gamma_data = data.get('greeks_data', {}).get('gamma', {})
            delta_data = data.get('greeks_data', {}).get('delta', {})
            charm_data = data.get('greeks_data', {}).get('charm', {})
            theta_data = data.get('greeks_data', {}).get('theta', {})
            vwap_data = data.get('vwap_data', {})
            volume_data = data.get('volume_data', {})
            current_price = data.get('current_price', 0)

            # 1. Maior barra de GAMMA posicionada no centro do range (normalmente na VWAP)
            gamma_center_positioned = self._check_gamma_center_positioned(gamma_data, vwap_data, current_price)

            # 2. Maior barra de DELTA do range
            delta_major_in_range = self._check_delta_major_in_range(delta_data, current_price)

            # 3. CHARM neutro ou em Flip, sem direção clara
            charm_neutral_or_flip = self._check_charm_neutral_or_flip(charm_data)

            # 4. Maior barra de THETA DECAY no centro do range
            theta_decay_center = self._check_theta_decay_center(theta_data, vwap_data, current_price)

            # 5. VWAP e seus desvios formando linhas retas, indicando equilíbrio
            vwap_straight_lines = self._check_vwap_straight_lines(vwap_data)

            # 6. Confirmação extra: Volume equilibrado nos strikes (DDF) e no Heat Map
            volume_balanced_strikes = self._check_volume_balanced_strikes(volume_data)

            conditions = [gamma_center_positioned, delta_major_in_range, charm_neutral_or_flip,
                         theta_decay_center, vwap_straight_lines, volume_balanced_strikes]
            confidence = sum(conditions) / len(conditions)
            confirmed = confidence >= 0.85  # Requer 85% de confiança para consolidação

            # Determine direction based on VWAP position
            signal = None
            if confirmed:
                vwap_price = vwap_data.get('current_vwap', current_price)
                if current_price > vwap_price * 1.002:  # 0.2% above VWAP
                    signal = 'SELL'
                elif current_price < vwap_price * 0.998:  # 0.2% below VWAP
                    signal = 'BUY'

            return {
                'name': 'Consolidação (MERCADO CONSOLIDADO)',
                'confirmed': confirmed,
                'confidence': confidence,
                'signal': signal,
                'timestamp': datetime.now(),
                'stop_loss_trigger': 'clear_range_breakout_with_charm_confirmation',
                'conditions': {
                    'gamma_center_positioned': gamma_center_positioned,
                    'delta_major_in_range': delta_major_in_range,
                    'charm_neutral_or_flip': charm_neutral_or_flip,
                    'theta_decay_center': theta_decay_center,
                    'vwap_straight_lines': vwap_straight_lines,
                    'volume_balanced_strikes': volume_balanced_strikes
                }
            }

        except Exception as e:
            logger.error(f"Erro no Setup #5: {e}")
            return {'confirmed': False, 'error': str(e)}

    def analyze_negative_gamma_protection(self, data: Dict) -> Dict:
        """
        Setup #6: Proteção contra Gamma Negativo (PROTEÇÃO)
        Condições específicas das imagens:
        - Preço operando em Gamma e Delta positivo
        - Maior barra de Gamma Positivo acima do preço
        - Maior barra de Delta Positivo acima do preço
        - Grande barra de Gamma Negativo e Delta Negativo abaixo do preço (perigo iminente)
        - Confirmação extra: Aumento do ratio de Puts x Calls para calls (puts perdendo força)
        - Stop Loss: Ativado caso o preço entre em Gamma Negativo sem defesa
        """
        try:
            gamma_data = data.get('greeks_data', {}).get('gamma', {})
            delta_data = data.get('greeks_data', {}).get('delta', {})
            charm_data = data.get('greeks_data', {}).get('charm', {})
            options_data = data.get('options_data', {})
            current_price = data.get('current_price', 0)

            # 1. Preço operando em Gamma e Delta positivo
            gamma_positive = gamma_data.get('current', 0) > 0
            delta_positive = delta_data.get('current', 0) > 0

            # 2. Maior barra de Gamma Positivo acima do preço
            gamma_positive_above_price = self._check_gamma_positive_above_price(gamma_data, current_price)

            # 3. Maior barra de Delta Positivo acima do preço
            delta_positive_above_price = self._check_delta_positive_above_price(delta_data, current_price)

            # 4. Grande barra de Gamma Negativo e Delta Negativo abaixo do preço (perigo iminente)
            large_negative_bars_below = self._check_large_negative_bars_below(gamma_data, delta_data, current_price)

            # 5. Confirmação extra: Aumento do ratio de Puts x Calls para calls (puts perdendo força)
            puts_calls_ratio_favoring_calls = self._check_puts_calls_ratio_favoring_calls(options_data)

            conditions = [gamma_positive, delta_positive, gamma_positive_above_price,
                         delta_positive_above_price, large_negative_bars_below, puts_calls_ratio_favoring_calls]
            confidence = sum(conditions) / len(conditions)
            confirmed = confidence >= 0.8  # Requer 80% de confiança para proteção

            return {
                'name': 'Proteção contra Gamma Negativo (PROTEÇÃO)',
                'confirmed': confirmed,
                'confidence': confidence,
                'signal': 'BUY' if confirmed else None,
                'timestamp': datetime.now(),
                'stop_loss_trigger': 'price_enters_negative_gamma_without_defense',
                'conditions': {
                    'gamma_positive': gamma_positive,
                    'delta_positive': delta_positive,
                    'gamma_positive_above_price': gamma_positive_above_price,
                    'delta_positive_above_price': delta_positive_above_price,
                    'large_negative_bars_below': large_negative_bars_below,
                    'puts_calls_ratio_favoring_calls': puts_calls_ratio_favoring_calls
                }
            }

        except Exception as e:
            logger.error(f"Erro no Setup #6: {e}")
            return {'confirmed': False, 'error': str(e)}

    # Helper methods for condition checking
    def _check_gamma_negative_below_price(self, gamma_data: Dict, current_price: float) -> bool:
        """Verifica se existe GAMMA negativo abaixo do preço"""
        try:
            levels = gamma_data.get('levels', [])
            for price, gamma_value in levels:
                if price < current_price and gamma_value < 0:
                    return True
            return False
        except:
            return False

    def _check_price_at_major_gamma(self, gamma_data: Dict, current_price: float) -> bool:
        """Verifica se o preço está na maior ou segunda maior barra de GAMMA"""
        try:
            levels = gamma_data.get('levels', [])
            if not levels:
                return False

            sorted_levels = sorted(levels, key=lambda x: abs(x[1]), reverse=True)
            tolerance = current_price * 0.005

            for i in range(min(2, len(sorted_levels))):
                level_price = sorted_levels[i][0]
                if abs(current_price - level_price) <= tolerance:
                    return True

            return False
        except:
            return False

    def _check_next_gamma_smaller(self, gamma_data: Dict, current_price: float) -> bool:
        """Verifica se a próxima barra de GAMMA é consideravelmente menor"""
        try:
            levels = gamma_data.get('levels', [])
            if not levels:
                return False

            # Find current level
            current_level = None
            next_level = None

            sorted_levels = sorted(levels, key=lambda x: x[0])

            for i, (price, gamma_value) in enumerate(sorted_levels):
                if abs(price - current_price) <= current_price * 0.005:
                    current_level = abs(gamma_value)
                    if i + 1 < len(sorted_levels):
                        next_level = abs(sorted_levels[i + 1][1])
                    break

            if current_level and next_level:
                return next_level < current_level * 0.5  # Next level is less than 50% of current

            return False
        except:
            return False

    def _check_last_relevant_delta(self, delta_data: Dict, current_price: float) -> bool:
        """Verifica última barra relevante de DELTA"""
        try:
            levels = delta_data.get('levels', [])
            if not levels:
                return False

            # Find the most recent significant delta level
            sorted_levels = sorted(levels, key=lambda x: abs(x[1]), reverse=True)
            tolerance = current_price * 0.01

            for price, delta_value in sorted_levels[:3]:  # Check top 3 levels
                if abs(current_price - price) <= tolerance:
                    return True

            return False
        except:
            return False

    def _check_gamma_center_positioned(self, gamma_data: Dict, vwap_data: Dict, current_price: float) -> bool:
        """Verifica se a maior barra de GAMMA está no centro do range (VWAP)"""
        try:
            max_gamma_price = gamma_data.get('max_level_price', 0)
            vwap_price = vwap_data.get('current_vwap', current_price)
            tolerance = current_price * 0.02
            return abs(max_gamma_price - vwap_price) <= tolerance
        except:
            return False

    def _check_delta_center_positioned(self, delta_data: Dict, vwap_data: Dict, current_price: float) -> bool:
        """Verifica se a maior barra de DELTA está no centro do range"""
        try:
            max_delta_price = delta_data.get('max_level_price', 0)
            vwap_price = vwap_data.get('current_vwap', current_price)
            tolerance = current_price * 0.02
            return abs(max_delta_price - vwap_price) <= tolerance
        except:
            return False

    def _check_gamma_below_flip(self, gamma_data: Dict, current_price: float) -> bool:
        """Verifica se a maior barra de GAMMA está bem abaixo do Flip"""
        try:
            max_gamma_price = gamma_data.get('max_level_price', 0)
            return max_gamma_price < current_price * 0.95
        except:
            return False

    def _check_delta_below_flip(self, delta_data: Dict, current_price: float) -> bool:
        """Verifica se a maior barra de DELTA está bem abaixo do Flip"""
        try:
            max_delta_price = delta_data.get('max_level_price', 0)
            return max_delta_price < current_price * 0.95
        except:
            return False

    def _check_charm_large_below_flip(self, charm_data: Dict, current_price: float) -> bool:
        """Verifica se o CHARM tem uma barra grande bem abaixo do Flip"""
        try:
            max_charm_price = charm_data.get('max_level_price', 0)
            max_charm_value = charm_data.get('max_level', 0)
            below_flip = max_charm_price < current_price * 0.95
            large_bar = abs(max_charm_value) > 0.05
            return below_flip and large_bar
        except:
            return False

    # Novos métodos auxiliares para os setups atualizados

    def _check_gamma_barriers_below_price(self, gamma_data: Dict, current_price: float) -> bool:
        """Verifica se há barreiras de GAMMA abaixo do preço"""
        try:
            levels = gamma_data.get('levels', [])
            for price, gamma_value in levels:
                if price < current_price and abs(gamma_value) > 0.1:  # Barreira significativa
                    return True
            return False
        except:
            return False

    def _check_vwap_breakout_up(self, vwap_data: Dict, current_price: float) -> bool:
        """Verifica rompimento do primeiro desvio da VWAP para cima"""
        try:
            vwap_price = vwap_data.get('current_vwap', current_price)
            first_deviation = vwap_data.get('first_deviation_up', vwap_price * 1.002)
            return current_price > first_deviation
        except:
            return False

    def _check_gamma_barriers_above_price(self, gamma_data: Dict, current_price: float) -> bool:
        """Verifica se há barreiras de GAMMA acima do preço"""
        try:
            levels = gamma_data.get('levels', [])
            for price, gamma_value in levels:
                if price > current_price and abs(gamma_value) > 0.1:  # Barreira significativa
                    return True
            return False
        except:
            return False

    def _check_vwap_breakout_down(self, vwap_data: Dict, current_price: float) -> bool:
        """Verifica rompimento do primeiro desvio da VWAP para baixo"""
        try:
            vwap_price = vwap_data.get('current_vwap', current_price)
            first_deviation = vwap_data.get('first_deviation_down', vwap_price * 0.998)
            return current_price < first_deviation
        except:
            return False

    def _check_price_at_major_gamma_positive(self, gamma_data: Dict, current_price: float) -> bool:
        """Verifica se o preço está na maior barra de GAMMA positiva"""
        try:
            levels = gamma_data.get('levels', [])
            if not levels:
                return False

            # Filtrar apenas níveis positivos
            positive_levels = [(price, gamma) for price, gamma in levels if gamma > 0]

            if not positive_levels:
                return False

            # Ordenar por magnitude do gamma
            sorted_levels = sorted(positive_levels, key=lambda x: x[1], reverse=True)
            tolerance = current_price * 0.005

            # Verificar se preço está próximo do maior nível positivo
            major_price = sorted_levels[0][0]
            return abs(current_price - major_price) <= tolerance

        except:
            return False

    def _check_next_gamma_much_smaller(self, gamma_data: Dict, current_price: float) -> bool:
        """Verifica se a próxima barra de GAMMA acima é muito menor"""
        try:
            levels = gamma_data.get('levels', [])
            if not levels:
                return False

            # Ordenar por preço
            sorted_levels = sorted(levels, key=lambda x: x[0])

            # Encontrar nível atual
            current_level = None
            for i, (price, gamma_value) in enumerate(sorted_levels):
                if abs(price - current_price) <= current_price * 0.005:
                    current_level = abs(gamma_value)
                    # Verificar próximo nível acima
                    if i + 1 < len(sorted_levels):
                        next_level = abs(sorted_levels[i + 1][1])
                        return next_level < current_level * 0.3  # Muito menor (menos de 30%)
                    break

            return False
        except:
            return False

    def _check_charm_positive_before_smaller(self, charm_data: Dict, current_price: float) -> bool:
        """Verifica se há barra de CHARM positivo antes de barra menor (perda de força)"""
        try:
            levels = charm_data.get('levels', [])
            if not levels:
                return False

            # Filtrar níveis positivos
            positive_levels = [(price, charm) for price, charm in levels if charm > 0]

            if len(positive_levels) < 2:
                return False

            # Ordenar por preço
            sorted_levels = sorted(positive_levels, key=lambda x: x[0])

            # Verificar padrão: barra maior seguida de barra menor
            for i in range(len(sorted_levels) - 1):
                current_charm = sorted_levels[i][1]
                next_charm = sorted_levels[i + 1][1]

                if next_charm < current_charm * 0.6:  # Próxima é 40% menor
                    return True

            return False
        except:
            return False

    def _check_last_delta_positive_reached(self, delta_data: Dict, current_price: float) -> bool:
        """Verifica se a última barra de DELTA positivo foi atingida (esgotamento)"""
        try:
            levels = delta_data.get('levels', [])
            if not levels:
                return False

            # Filtrar níveis positivos
            positive_levels = [(price, delta) for price, delta in levels if delta > 0]

            if not positive_levels:
                return False

            # Ordenar por preço (última seria a mais alta)
            sorted_levels = sorted(positive_levels, key=lambda x: x[0], reverse=True)

            # Verificar se preço atual está próximo do último nível positivo
            last_positive_price = sorted_levels[0][0]
            tolerance = current_price * 0.003

            return abs(current_price - last_positive_price) <= tolerance

        except:
            return False

    def _check_volume_neutral_above_strikes(self, volume_data: Dict, current_price: float) -> bool:
        """Verifica se volume está neutro ou fechando nos strikes acima (DDF LEVELS)"""
        try:
            strikes_volume = volume_data.get('strikes_volume', {})
            above_strikes = {k: v for k, v in strikes_volume.items() if k > current_price}

            if not above_strikes:
                return False

            # Verificar se volume está relativamente equilibrado
            volumes = list(above_strikes.values())
            if len(volumes) < 2:
                return False

            # Volume neutro se diferença máxima for menor que 50%
            max_vol = max(volumes)
            min_vol = min(volumes)

            return min_vol > max_vol * 0.5

        except:
            return False

    def _check_price_at_major_gamma_negative(self, gamma_data: Dict, current_price: float) -> bool:
        """Verifica se o preço está na maior barra de GAMMA negativa"""
        try:
            levels = gamma_data.get('levels', [])
            if not levels:
                return False

            # Filtrar apenas níveis negativos
            negative_levels = [(price, gamma) for price, gamma in levels if gamma < 0]

            if not negative_levels:
                return False

            # Ordenar por magnitude do gamma (mais negativo primeiro)
            sorted_levels = sorted(negative_levels, key=lambda x: x[1])  # Ordem crescente (mais negativo primeiro)
            tolerance = current_price * 0.005

            # Verificar se preço está próximo do maior nível negativo
            major_price = sorted_levels[0][0]
            return abs(current_price - major_price) <= tolerance

        except:
            return False

    def _check_next_gamma_below_much_smaller(self, gamma_data: Dict, current_price: float) -> bool:
        """Verifica se a próxima barra de GAMMA abaixo é muito menor"""
        try:
            levels = gamma_data.get('levels', [])
            if not levels:
                return False

            # Ordenar por preço (decrescente)
            sorted_levels = sorted(levels, key=lambda x: x[0], reverse=True)

            # Encontrar nível atual
            current_level = None
            for i, (price, gamma_value) in enumerate(sorted_levels):
                if abs(price - current_price) <= current_price * 0.005:
                    current_level = abs(gamma_value)
                    # Verificar próximo nível abaixo
                    if i + 1 < len(sorted_levels):
                        next_level = abs(sorted_levels[i + 1][1])
                        return next_level < current_level * 0.3  # Muito menor (menos de 30%)
                    break

            return False
        except:
            return False

    def _check_charm_negative_before_smaller(self, charm_data: Dict, current_price: float) -> bool:
        """Verifica se há barra de CHARM negativo antes de barra menor (perda de força vendedora)"""
        try:
            levels = charm_data.get('levels', [])
            if not levels:
                return False

            # Filtrar níveis negativos
            negative_levels = [(price, charm) for price, charm in levels if charm < 0]

            if len(negative_levels) < 2:
                return False

            # Ordenar por preço (decrescente para verificar padrão)
            sorted_levels = sorted(negative_levels, key=lambda x: x[0], reverse=True)

            # Verificar padrão: barra maior negativa seguida de barra menor negativa
            for i in range(len(sorted_levels) - 1):
                current_charm = abs(sorted_levels[i][1])  # Magnitude
                next_charm = abs(sorted_levels[i + 1][1])  # Magnitude

                if next_charm < current_charm * 0.6:  # Próxima é 40% menor em magnitude
                    return True

            return False
        except:
            return False

    def _check_last_delta_negative_reached(self, delta_data: Dict, current_price: float) -> bool:
        """Verifica se a última barra de DELTA negativo foi atingida (exaustão de oferta)"""
        try:
            levels = delta_data.get('levels', [])
            if not levels:
                return False

            # Filtrar níveis negativos
            negative_levels = [(price, delta) for price, delta in levels if delta < 0]

            if not negative_levels:
                return False

            # Ordenar por preço (última seria a mais baixa)
            sorted_levels = sorted(negative_levels, key=lambda x: x[0])  # Ordem crescente

            # Verificar se preço atual está próximo do último nível negativo
            last_negative_price = sorted_levels[0][0]
            tolerance = current_price * 0.003

            return abs(current_price - last_negative_price) <= tolerance

        except:
            return False

    def _check_volume_neutral_below_strikes(self, volume_data: Dict, current_price: float) -> bool:
        """Verifica se volume está neutro ou ausente nos strikes abaixo (DDF LEVELS)"""
        try:
            strikes_volume = volume_data.get('strikes_volume', {})
            below_strikes = {k: v for k, v in strikes_volume.items() if k < current_price}

            if not below_strikes:
                return True  # Ausente = neutro

            # Verificar se volume está baixo ou equilibrado
            volumes = list(below_strikes.values())

            # Volume neutro/ausente se volumes são baixos
            max_vol = max(volumes)
            return max_vol < 1000  # Threshold baixo para volume

        except:
            return False

    def _check_delta_major_in_range(self, delta_data: Dict, current_price: float) -> bool:
        """Verifica se a maior barra de DELTA está no range"""
        try:
            levels = delta_data.get('levels', [])
            if not levels:
                return False

            # Encontrar maior barra de delta (por magnitude)
            max_delta_level = max(levels, key=lambda x: abs(x[1]))

            if abs(max_delta_level[1]) < 0.1:  # Muito pequena
                return False

            # Verificar se está dentro de um range razoável do preço atual
            tolerance = current_price * 0.05  # 5% range
            return abs(max_delta_level[0] - current_price) <= tolerance

        except:
            return False

    def _check_charm_neutral_or_flip(self, charm_data: Dict) -> bool:
        """Verifica se CHARM está neutro ou em zona de flip"""
        try:
            current_charm = charm_data.get('current', 0)
            flip_zone = charm_data.get('flip_zone', False)

            # Neutro se valor absoluto muito baixo
            neutral = abs(current_charm) < 0.02

            return neutral or flip_zone

        except:
            return False

    def _check_theta_decay_center(self, theta_data: Dict, vwap_data: Dict, current_price: float) -> bool:
        """Verifica se maior barra de THETA DECAY está no centro do range"""
        try:
            max_theta_price = theta_data.get('max_level_price', 0)
            vwap_price = vwap_data.get('current_vwap', current_price)

            if max_theta_price == 0:
                return False

            tolerance = current_price * 0.02
            return abs(max_theta_price - vwap_price) <= tolerance

        except:
            return False

    def _check_vwap_straight_lines(self, vwap_data: Dict) -> bool:
        """Verifica se VWAP e desvios formam linhas retas (equilíbrio)"""
        try:
            deviations_straight = vwap_data.get('deviations_straight', False)
            equilibrium_detected = vwap_data.get('equilibrium', False)

            return deviations_straight or equilibrium_detected

        except:
            return False

    def _check_volume_balanced_strikes(self, volume_data: Dict) -> bool:
        """Verifica se volume está equilibrado nos strikes (DDF) e no Heat Map"""
        try:
            strikes_volume = volume_data.get('strikes_volume', {})
            heat_map_balance = volume_data.get('heat_map_balance', False)

            if not strikes_volume:
                return False

            # Verificar equilíbrio nos strikes
            volumes = list(strikes_volume.values())
            if len(volumes) < 3:
                return False

            # Calcular se volumes estão equilibrados (coeficiente de variação < 50%)
            mean_vol = sum(volumes) / len(volumes)
            if mean_vol == 0:
                return False

            variance = sum((v - mean_vol) ** 2 for v in volumes) / len(volumes)
            std_dev = variance ** 0.5
            coefficient_variation = std_dev / mean_vol

            strikes_balanced = coefficient_variation < 0.5

            return strikes_balanced and heat_map_balance

        except:
            return False

    def _check_gamma_positive_above_price(self, gamma_data: Dict, current_price: float) -> bool:
        """Verifica se há maior barra de Gamma Positivo acima do preço"""
        try:
            levels = gamma_data.get('levels', [])
            if not levels:
                return False

            # Filtrar níveis positivos acima do preço
            positive_above = [(price, gamma) for price, gamma in levels
                            if price > current_price and gamma > 0]

            if not positive_above:
                return False

            # Verificar se há nível positivo significativo acima
            major_positive = max(positive_above, key=lambda x: x[1])
            return major_positive[1] > 0.1  # Significativo

        except:
            return False

    def _check_delta_positive_above_price(self, delta_data: Dict, current_price: float) -> bool:
        """Verifica se há maior barra de Delta Positivo acima do preço"""
        try:
            levels = delta_data.get('levels', [])
            if not levels:
                return False

            # Filtrar níveis positivos acima do preço
            positive_above = [(price, delta) for price, delta in levels
                            if price > current_price and delta > 0]

            if not positive_above:
                return False

            # Verificar se há nível positivo significativo acima
            major_positive = max(positive_above, key=lambda x: x[1])
            return major_positive[1] > 0.1  # Significativo

        except:
            return False

    def _check_large_negative_bars_below(self, gamma_data: Dict, delta_data: Dict, current_price: float) -> bool:
        """Verifica se há grandes barras negativas de Gamma e Delta abaixo do preço"""
        try:
            # Verificar Gamma negativo abaixo
            gamma_levels = gamma_data.get('levels', [])
            gamma_negative_below = [(price, gamma) for price, gamma in gamma_levels
                                  if price < current_price and gamma < 0]

            # Verificar Delta negativo abaixo
            delta_levels = delta_data.get('levels', [])
            delta_negative_below = [(price, delta) for price, delta in delta_levels
                                  if price < current_price and delta < 0]

            # Verificar se há barras negativas significativas
            large_gamma_negative = any(abs(gamma) > 0.15 for _, gamma in gamma_negative_below)
            large_delta_negative = any(abs(delta) > 0.15 for _, delta in delta_negative_below)

            return large_gamma_negative and large_delta_negative

        except:
            return False

    def _check_puts_calls_ratio_favoring_calls(self, options_data: Dict) -> bool:
        """Verifica se ratio Puts/Calls favorece calls (puts perdendo força)"""
        try:
            puts_volume = options_data.get('puts_volume', 0)
            calls_volume = options_data.get('calls_volume', 0)

            if calls_volume == 0:
                return False

            ratio = puts_volume / calls_volume

            # Favorece calls se ratio < 0.8 (menos puts que calls)
            return ratio < 0.8

        except:
            return False

    def _record_setup_history(self, results: Dict, data: Dict):
        """Registra histórico dos setups"""
        try:
            history_entry = {
                'timestamp': datetime.now(),
                'current_price': data.get('current_price', 0),
                'results': results
            }
            self.setup_history.append(history_entry)
            if len(self.setup_history) > 100:
                self.setup_history = self.setup_history[-100:]
        except Exception as e:
            logger.error(f"Erro ao registrar histórico: {e}")

    def get_setup_summary(self) -> Dict:
        """Retorna resumo dos setups confirmados"""
        try:
            if not self.setup_history:
                return {}

            latest = self.setup_history[-1]['results']
            confirmed_setups = []

            for setup_name, setup_data in latest.items():
                if setup_data.get('confirmed', False):
                    confirmed_setups.append({
                        'name': setup_data.get('name', setup_name),
                        'signal': setup_data.get('signal'),
                        'confidence': setup_data.get('confidence', 0)
                    })

            return {
                'confirmed_setups': confirmed_setups,
                'total_confirmed': len(confirmed_setups),
                'timestamp': self.setup_history[-1]['timestamp']
            }
        except Exception as e:
            logger.error(f"Erro ao gerar resumo: {e}")
            return {}

    def get_setup(self, setup_number: int) -> Dict:
        """Retorna configuração do setup específico"""
        setups_config = {
            1: {
                'name': 'Bullish Breakout',
                'direction': 'bullish',
                'risk_multiplier': 1.0,
                'time_horizon': 'short',
                'confirmation_signals': ['gamma_increasing', 'delta_stable', 'charm_accelerating']
            },
            2: {
                'name': 'Bearish Breakout',
                'direction': 'bearish',
                'risk_multiplier': 1.2,
                'time_horizon': 'short',
                'confirmation_signals': ['gamma_decreasing', 'delta_negative', 'charm_negative']
            },
            3: {
                'name': 'Pullback no Topo',
                'direction': 'bearish',
                'risk_multiplier': 0.8,
                'time_horizon': 'medium',
                'confirmation_signals': ['gamma_peak', 'charm_peak', 'delta_reversal']
            },
            4: {
                'name': 'Pullback no Fundo',
                'direction': 'bullish',
                'risk_multiplier': 0.8,
                'time_horizon': 'medium',
                'confirmation_signals': ['gamma_bottom', 'charm_bottom', 'delta_reversal']
            },
            5: {
                'name': 'Mercado Consolidado (VWAP)',
                'direction': 'neutral',
                'risk_multiplier': 0.6,
                'time_horizon': 'long',
                'confirmation_signals': ['vwap_rejection', 'bollinger_squeeze']
            },
            6: {
                'name': 'Proteção contra Gama Negativo',
                'direction': 'bullish',
                'risk_multiplier': 1.5,
                'time_horizon': 'short',
                'confirmation_signals': ['gamma_protection', 'delta_support', 'charm_support']
            }
        }
        
        return setups_config.get(setup_number, {})

    def validate_setup_conditions(self, setup_number: int, market_data: Dict) -> Dict:
        """Valida condições do setup específico"""
        try:
            # Map setup numbers to analysis methods
            analysis_methods = {
                1: self.analyze_bullish_breakout,
                2: self.analyze_bearish_breakout,
                3: self.analyze_pullback_top,
                4: self.analyze_pullback_bottom,
                5: self.analyze_consolidated_market,
                6: self.analyze_negative_gamma_protection
            }
            
            if setup_number not in analysis_methods:
                return {'valid': False, 'confidence_score': 0.0, 'conditions_failed': ['Setup not found']}
            
            # Run analysis
            result = analysis_methods[setup_number](market_data)
            
            # Extract validation info
            valid = result.get('confirmed', False)
            confidence_score = result.get('confidence', 0.0)
            
            # Get failed conditions
            conditions = result.get('conditions', {})
            conditions_failed = [k for k, v in conditions.items() if not v]
            
            return {
                'valid': valid,
                'confidence_score': confidence_score,
                'conditions_failed': conditions_failed
            }
            
        except Exception as e:
            logger.error(f"Erro ao validar setup {setup_number}: {e}")
            return {'valid': False, 'confidence_score': 0.0, 'conditions_failed': [str(e)]}


# Instância global para uso fácil
trading_setups_corrected = TradingSetupsCorrected()

if __name__ == "__main__":
    # Exemplo de uso
    detector = TradingSetupsCorrected()

    # Dados de exemplo para teste
    mock_data = {
        'current_price': 400.0,
        'greeks_data': {
            'gamma': {
                'current': 0.1,
                'levels': [(395, 0.15), (400, 0.1), (405, 0.05)],
                'max_level': 0.15,
                'max_level_price': 395
            },
            'delta': {
                'current': 0.5,
                'levels': [(395, 0.6), (400, 0.5), (405, 0.4)],
                'max_level': 0.6,
                'max_level_price': 395,
                'quantico_positive': True,
                'max_positive': 0.6,
                'max_negative': -0.4
            },
            'charm': {
                'current': 0.02,
                'levels': [(395, 0.03), (400, 0.02), (405, 0.01)],
                'max_level': 0.03,
                'max_level_price': 395,
                'growing_trend': True,
                'peak_detected': False,
                'flip_zone': False
            }
        },
        'vwap_data': {
            'current_vwap': 399.5,
            'above_vwap': True,
            'distance_percent': 0.125
        },
        'bollinger_bands': {
            'squeeze_detected': False,
            'upper_band': 405,
            'lower_band': 395,
            'middle_band': 400
        }
    }

    results = detector.analyze_all_setups(mock_data)
    print("Análise dos Setups de Trading:")
    print("=" * 50)

    for setup_name, result in results.items():
        if result.get('confirmed', False):
            print(f"✓ {setup_name}: {result.get('name')}")
            print(f"  Sinal: {result.get('signal')}")
            print(f"  Confiança: {result.get('confidence', 0):.2f}")
            print(f"  Condições: {result.get('conditions', {})}")
            print()
# Alias para compatibilidade
trading_setups = trading_setups_corrected