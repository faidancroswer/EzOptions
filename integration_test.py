"""
Sistema de Teste e Integração Completa
Testa integração entre todos os componentes do sistema de trading
"""

import logging
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import threading
import unittest.mock as mock

logger = logging.getLogger(__name__)

class IntegrationTester:
    """
    Testa integração completa do sistema de trading
    """

    def __init__(self, config: Dict):
        self.config = config
        self.test_results = []
        self.system_components = {}

    def run_full_integration_test(self) -> Dict:
        """
        Executa teste completo de integração

        Returns:
            Dicionário com resultados dos testes
        """
        logger.info("Iniciando teste completo de integração...")

        start_time = datetime.now()
        test_results = {
            'start_time': start_time.isoformat(),
            'components_tested': [],
            'tests_passed': 0,
            'tests_failed': 0,
            'total_tests': 0,
            'component_results': {},
            'integration_score': 0.0,
            'recommendations': []
        }

        try:
            # 1. Teste de inicialização dos componentes
            init_results = self._test_component_initialization()
            test_results['component_results']['initialization'] = init_results
            test_results['components_tested'].append('initialization')

            if init_results['success']:
                test_results['tests_passed'] += 1
            else:
                test_results['tests_failed'] += 1
            test_results['total_tests'] += 1

            # 2. Teste de setups de trading
            setup_results = self._test_trading_setups()
            test_results['component_results']['trading_setups'] = setup_results
            test_results['components_tested'].append('trading_setups')

            if setup_results['success']:
                test_results['tests_passed'] += 1
            else:
                test_results['tests_failed'] += 1
            test_results['total_tests'] += 1

            # 3. Teste de gerenciamento de risco
            risk_results = self._test_risk_management()
            test_results['component_results']['risk_management'] = risk_results
            test_results['components_tested'].append('risk_management')

            if risk_results['success']:
                test_results['tests_passed'] += 1
            else:
                test_results['tests_failed'] += 1
            test_results['total_tests'] += 1

            # 4. Teste de verificação de duplicatas
            dup_results = self._test_duplicate_checking()
            test_results['component_results']['duplicate_checking'] = dup_results
            test_results['components_tested'].append('duplicate_checking')

            if dup_results['success']:
                test_results['tests_passed'] += 1
            else:
                test_results['tests_failed'] += 1
            test_results['total_tests'] += 1

            # 5. Teste de sistema de auditoria
            audit_results = self._test_audit_logging()
            test_results['component_results']['audit_logging'] = audit_results
            test_results['components_tested'].append('audit_logging')

            if audit_results['success']:
                test_results['tests_passed'] += 1
            else:
                test_results['tests_failed'] += 1
            test_results['total_tests'] += 1

            # 6. Teste de integração entre componentes
            integration_results = self._test_component_integration()
            test_results['component_results']['integration'] = integration_results
            test_results['components_tested'].append('integration')

            if integration_results['success']:
                test_results['tests_passed'] += 1
            else:
                test_results['tests_failed'] += 1
            test_results['total_tests'] += 1

            # Calcular score de integração
            if test_results['total_tests'] > 0:
                test_results['integration_score'] = (test_results['tests_passed'] / test_results['total_tests']) * 100

            # Gerar recomendações
            test_results['recommendations'] = self._generate_recommendations(test_results)

            # Tempo total
            end_time = datetime.now()
            test_results['end_time'] = end_time.isoformat()
            test_results['duration_seconds'] = (end_time - start_time).total_seconds()

            logger.info(f"Teste de integração concluído em {test_results['duration_seconds']:.2f}s")
            logger.info(f"Score de integração: {test_results['integration_score']:.1f}%")

            return test_results

        except Exception as e:
            logger.error(f"Erro no teste de integração: {e}")
            test_results['error'] = str(e)
            return test_results

    def _test_component_initialization(self) -> Dict:
        """Testa inicialização dos componentes"""
        results = {
            'success': True,
            'tests': [],
            'errors': []
        }

        try:
            # Testar importações
            try:
                from trading_setups_corrected import TradingSetupsCorrected
                from qqq_monitor import QQQMonitor
                from mt5_autotrader import MT5AutoTrader
                from advanced_risk_manager import AdvancedRiskManager
                from audit_logger import AuditLogger
                from duplicate_checker import DuplicateChecker

                results['tests'].append({
                    'test': 'component_imports',
                    'success': True,
                    'message': 'Todas as importações funcionaram'
                })

            except ImportError as e:
                results['tests'].append({
                    'test': 'component_imports',
                    'success': False,
                    'message': f'Erro de importação: {e}'
                })
                results['success'] = False
                results['errors'].append(str(e))

            # Testar inicialização básica
            try:
                # Trading Setups
                setups = TradingSetupsCorrected()
                setup_test = setups.get_setup(1)

                results['tests'].append({
                    'test': 'trading_setups_init',
                    'success': setup_test != {},
                    'message': 'TradingSetups inicializado com sucesso'
                })

                # Risk Manager
                risk_manager = AdvancedRiskManager(self.config)
                risk_status = risk_manager.get_risk_status()

                results['tests'].append({
                    'test': 'risk_manager_init',
                    'success': 'trading_enabled' in risk_status,
                    'message': 'RiskManager inicializado com sucesso'
                })

                # Duplicate Checker
                dup_checker = DuplicateChecker(self.config)
                dup_stats = dup_checker.get_duplicate_stats()

                results['tests'].append({
                    'test': 'duplicate_checker_init',
                    'success': 'total_signals_tracked' in dup_stats,
                    'message': 'DuplicateChecker inicializado com sucesso'
                })

                # Audit Logger
                audit_logger = AuditLogger(self.config)
                audit_summary = audit_logger.get_audit_summary()

                results['tests'].append({
                    'test': 'audit_logger_init',
                    'success': 'total_events' in audit_summary,
                    'message': 'AuditLogger inicializado com sucesso'
                })

            except Exception as e:
                results['tests'].append({
                    'test': 'component_initialization',
                    'success': False,
                    'message': f'Erro na inicialização: {e}'
                })
                results['success'] = False
                results['errors'].append(str(e))

        except Exception as e:
            results['success'] = False
            results['errors'].append(f"Erro geral no teste de inicialização: {e}")

        return results

    def _test_trading_setups(self) -> Dict:
        """Testa setups de trading"""
        results = {
            'success': True,
            'tests': [],
            'errors': []
        }

        try:
            from trading_setups_corrected import TradingSetupsCorrected

            # Dados de teste mock
            mock_data = {
                'current_price': 15000.0,
                'greeks_data': {
                    'gamma': {
                        'current': 0.1,
                        'levels': [(14950, 0.15), (15000, 0.1), (15050, 0.05)],
                        'max_level': 0.15,
                        'max_level_price': 14950
                    },
                    'delta': {
                        'current': 0.5,
                        'levels': [(14950, 0.6), (15000, 0.5), (15050, 0.4)],
                        'max_level': 0.6,
                        'max_level_price': 14950,
                        'quantico_positive': True,
                        'max_positive': 0.6,
                        'max_negative': -0.4
                    },
                    'charm': {
                        'current': 0.02,
                        'levels': [(14950, 0.03), (15000, 0.02), (15050, 0.01)],
                        'max_level': 0.03,
                        'max_level_price': 14950,
                        'growing_trend': True,
                        'peak_detected': False,
                        'flip_zone': False
                    }
                },
                'vwap_data': {
                    'current_vwap': 14980.0,
                    'first_deviation_up': 15020.0,
                    'first_deviation_down': 14940.0
                },
                'volume_data': {
                    'strikes_volume': {14900: 100, 15000: 150, 15100: 120}
                }
            }

            # Testar detector de setups
            detector = TradingSetupsCorrected()

            # Testar cada setup
            for setup_number in range(1, 7):
                try:
                    result = detector.analyze_all_setups(mock_data)

                    if f'setup_{setup_number}' in result:
                        setup_result = result[f'setup_{setup_number}']

                        results['tests'].append({
                            'test': f'setup_{setup_number}_analysis',
                            'success': 'confirmed' in setup_result,
                            'message': f'Setup {setup_number} analisado: {setup_result.get("name", "Unknown")}'
                        })
                    else:
                        results['tests'].append({
                            'test': f'setup_{setup_number}_analysis',
                            'success': False,
                            'message': f'Setup {setup_number} não retornou resultado'
                        })
                        results['success'] = False

                except Exception as e:
                    results['tests'].append({
                        'test': f'setup_{setup_number}_analysis',
                        'success': False,
                        'message': f'Erro no Setup {setup_number}: {e}'
                    })
                    results['success'] = False
                    results['errors'].append(str(e))

        except Exception as e:
            results['success'] = False
            results['errors'].append(f"Erro geral no teste de setups: {e}")

        return results

    def _test_risk_management(self) -> Dict:
        """Testa sistema de gerenciamento de risco"""
        results = {
            'success': True,
            'tests': [],
            'errors': []
        }

        try:
            from advanced_risk_manager import AdvancedRiskManager

            # Inicializar risk manager
            risk_manager = AdvancedRiskManager(self.config)

            # Teste 1: Validação de sinal básico
            test_signal = {
                'setup_number': 1,
                'signal_type': 'BUY',
                'entry_price': 15000.0,
                'stop_loss': 14850.0,  # 1.0% stop loss
                'take_profit': 15150.0,  # 2:1 risk-reward
                'volume': 0.01,
                'symbol': 'US100',
                'confidence': 0.85,
                'risk_reward_ratio': 2.0
            }

            is_valid, reason, details = risk_manager.validate_signal(test_signal)

            results['tests'].append({
                'test': 'risk_validation_basic',
                'success': is_valid,
                'message': f'Validação básica: {reason}'
            })

            # Teste 2: Rejeição por stop loss excessivo
            bad_signal = test_signal.copy()
            bad_signal['stop_loss'] = 14500.0  # 3.33% stop loss (excesso)

            is_valid_bad, reason_bad, _ = risk_manager.validate_signal(bad_signal)

            results['tests'].append({
                'test': 'risk_validation_rejection',
                'success': not is_valid_bad,
                'message': f'Rejeição adequada: {reason_bad}'
            })

            # Teste 3: Verificação de métricas de risco
            risk_status = risk_manager.get_risk_status()

            required_metrics = ['trading_enabled', 'account_balance', 'current_exposure', 'daily_pnl']
            metrics_success = all(metric in risk_status for metric in required_metrics)

            results['tests'].append({
                'test': 'risk_metrics',
                'success': metrics_success,
                'message': 'Métricas de risco completas' if metrics_success else 'Métricas de risco incompletas'
            })

            # Teste 4: Verificação de relatório de risco
            risk_report = risk_manager.get_risk_report()

            report_success = 'summary' in risk_report and 'statistics' in risk_report

            results['tests'].append({
                'test': 'risk_report',
                'success': report_success,
                'message': 'Relatório de risco gerado' if report_success else 'Erro no relatório de risco'
            })

        except Exception as e:
            results['success'] = False
            results['errors'].append(f"Erro geral no teste de risco: {e}")

        return results

    def _test_duplicate_checking(self) -> Dict:
        """Testa sistema de verificação de duplicatas"""
        results = {
            'success': True,
            'tests': [],
            'errors': []
        }

        try:
            from duplicate_checker import DuplicateChecker, SignalDeduplicator

            # Inicializar componentes
            dup_checker = DuplicateChecker(self.config)
            deduplicator = SignalDeduplicator(self.config)

            # Teste 1: Sinais únicos
            signal1 = {
                'setup_number': 1,
                'signal_type': 'BUY',
                'current_price': 15000.0,
                'confidence': 0.85
            }

            result1 = dup_checker.check_duplicate(signal1)

            results['tests'].append({
                'test': 'unique_signal_check',
                'success': result1.status.value == 'unique',
                'message': f'Sinal único detectado: {result1.status.value}'
            })

            # Teste 2: Sinais similares
            signal2 = signal1.copy()
            signal2['current_price'] = 15010.0  # Muito próximo

            result2 = dup_checker.check_duplicate(signal2)

            results['tests'].append({
                'test': 'similar_signal_check',
                'success': result2.status.value in ['unique', 'similar'],
                'message': f'Sinal similar tratado: {result2.status.value}'
            })

            # Teste 3: Deduplicação completa
            should_process1, reason1, _ = deduplicator.should_process_signal(signal1)

            results['tests'].append({
                'test': 'deduplication_logic',
                'success': should_process1,
                'message': f'Deduplicação lógica: {reason1}'
            })

            # Teste 4: Estatísticas
            stats = dup_checker.get_duplicate_stats()

            stats_success = 'total_signals_tracked' in stats

            results['tests'].append({
                'test': 'duplicate_statistics',
                'success': stats_success,
                'message': 'Estatísticas de duplicatas geradas' if stats_success else 'Erro nas estatísticas'
            })

        except Exception as e:
            results['success'] = False
            results['errors'].append(f"Erro geral no teste de duplicatas: {e}")

        return results

    def _test_audit_logging(self) -> Dict:
        """Testa sistema de auditoria"""
        results = {
            'success': True,
            'tests': [],
            'errors': []
        }

        try:
            from audit_logger import AuditLogger

            # Inicializar auditor
            auditor = AuditLogger(self.config)

            # Teste 1: Log de evento básico
            try:
                auditor.log_event(
                    auditor.log_event.__self__._AuditLogger__class__.SYSTEM_START,
                    "TestComponent",
                    "test_action",
                    {'test': 'data'},
                    severity="INFO"
                )

                results['tests'].append({
                    'test': 'basic_logging',
                    'success': True,
                    'message': 'Log básico registrado'
                })

            except Exception as e:
                results['tests'].append({
                    'test': 'basic_logging',
                    'success': False,
                    'message': f'Erro no log básico: {e}'
                })
                results['success'] = False

            # Teste 2: Resumo de auditoria
            summary = auditor.get_audit_summary(hours=1)

            summary_success = 'total_events' in summary

            results['tests'].append({
                'test': 'audit_summary',
                'success': summary_success,
                'message': 'Resumo de auditoria gerado' if summary_success else 'Erro no resumo'
            })

            # Teste 3: Flush do buffer
            try:
                auditor.force_flush()

                results['tests'].append({
                    'test': 'buffer_flush',
                    'success': True,
                    'message': 'Buffer feito flush com sucesso'
                })

            except Exception as e:
                results['tests'].append({
                    'test': 'buffer_flush',
                    'success': False,
                    'message': f'Erro no flush: {e}'
                })

        except Exception as e:
            results['success'] = False
            results['errors'].append(f"Erro geral no teste de auditoria: {e}")

        return results

    def _test_component_integration(self) -> Dict:
        """Testa integração entre componentes"""
        results = {
            'success': True,
            'tests': [],
            'errors': []
        }

        try:
            # Testar fluxo completo: sinal -> validação -> deduplicação -> auditoria
            from trading_setups_corrected import TradingSetupsCorrected
            from advanced_risk_manager import AdvancedRiskManager
            from duplicate_checker import SignalDeduplicator
            from audit_logger import AuditLogger

            # 1. Gerar sinal mock
            mock_signal = {
                'setup_number': 1,
                'signal_type': 'BUY',
                'current_price': 15000.0,
                'stop_loss': 14850.0,
                'take_profit': 15150.0,
                'volume': 0.01,
                'confidence': 0.85,
                'risk_reward_ratio': 2.0
            }

            # 2. Testar análise de setup
            detector = TradingSetupsCorrected()
            mock_market_data = {
                'current_price': 15000.0,
                'greeks_data': {
                    'gamma': {'current': 0.1, 'levels': [(15000, 0.1)]},
                    'delta': {'current': 0.5, 'levels': [(15000, 0.5)]},
                    'charm': {'current': 0.02, 'levels': [(15000, 0.02)]}
                }
            }

            setup_result = detector.analyze_bullish_breakout(mock_market_data)

            results['tests'].append({
                'test': 'setup_analysis_integration',
                'success': isinstance(setup_result, dict) and 'confirmed' in setup_result,
                'message': 'Análise de setup integrada funcionando'
            })

            # 3. Testar validação de risco
            risk_manager = AdvancedRiskManager(self.config)
            risk_valid, risk_reason, risk_details = risk_manager.validate_signal(mock_signal)

            results['tests'].append({
                'test': 'risk_validation_integration',
                'success': risk_valid and isinstance(risk_details, dict),
                'message': f'Validação de risco integrada: {risk_reason}'
            })

            # 4. Testar deduplicação
            deduplicator = SignalDeduplicator(self.config)
            should_process, dup_reason, dup_details = deduplicator.should_process_signal(mock_signal)

            results['tests'].append({
                'test': 'deduplication_integration',
                'success': isinstance(should_process, bool),
                'message': f'Deduplicação integrada: {dup_reason}'
            })

            # 5. Testar auditoria
            auditor = AuditLogger(self.config)
            auditor.log_signal_detected("IntegrationTest", mock_signal)

            results['tests'].append({
                'test': 'audit_integration',
                'success': True,
                'message': 'Auditoria integrada funcionando'
            })

            # 6. Testar fluxo completo
            full_flow_success = (
                setup_result.get('confirmed', False) and
                risk_valid and
                should_process
            )

            results['tests'].append({
                'test': 'full_integration_flow',
                'success': full_flow_success,
                'message': 'Fluxo completo de integração funcionando' if full_flow_success else 'Problemas no fluxo de integração'
            })

        except Exception as e:
            results['success'] = False
            results['errors'].append(f"Erro geral no teste de integração: {e}")

        return results

    def _generate_recommendations(self, test_results: Dict) -> List[str]:
        """Gera recomendações baseado nos resultados dos testes"""
        recommendations = []

        try:
            # Analisar resultados por componente
            component_results = test_results.get('component_results', {})

            # Recomendações gerais
            if test_results.get('integration_score', 0) < 80:
                recommendations.append("Score de integração abaixo de 80% - revisar configurações")

            if test_results.get('tests_failed', 0) > 0:
                recommendations.append(f"{test_results['tests_failed']} testes falharam - verificar logs de erro")

            # Recomendações específicas por componente
            for component, results in component_results.items():
                if not results.get('success', False):
                    recommendations.append(f"Componente {component} com problemas - verificar implementação")

                    # Recomendações específicas
                    if component == 'risk_management':
                        recommendations.append("Verificar configurações de risco no config.json")
                    elif component == 'trading_setups':
                        recommendations.append("Verificar dados de mercado para análise de setups")
                    elif component == 'duplicate_checking':
                        recommendations.append("Verificar configurações de janela de tempo para duplicatas")

            # Recomendações de performance
            duration = test_results.get('duration_seconds', 0)
            if duration > 30:
                recommendations.append("Testes demorando muito - otimizar performance")

            # Recomendações de segurança
            if not any('audit' in comp for comp in test_results.get('components_tested', [])):
                recommendations.append("Sistema de auditoria não testado - implementar logging")

            if not recommendations:
                recommendations.append("Todos os testes passaram - sistema pronto para produção")
                recommendations.append("Considerar testes de carga e stress")

            return recommendations

        except Exception as e:
            logger.error(f"Erro ao gerar recomendações: {e}")
            return ["Erro ao gerar recomendações"]

def run_integration_test(config_path: str = 'config.json') -> Dict:
    """
    Função principal para executar teste de integração

    Args:
        config_path: Caminho para arquivo de configuração

    Returns:
        Resultados completos dos testes
    """
    try:
        # Carregar configuração
        with open(config_path, 'r') as f:
            config = json.load(f)

        # Inicializar testador
        tester = IntegrationTester(config)

        # Executar testes
        results = tester.run_full_integration_test()

        # Salvar resultados
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = f"integration_test_results_{timestamp}.json"

        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2, default=str)

        logger.info(f"Resultados salvos em: {results_file}")

        return results

    except Exception as e:
        logger.error(f"Erro ao executar teste de integração: {e}")
        return {'error': str(e)}

if __name__ == "__main__":
    # Executar teste de integração
    print("Executando teste completo de integração do sistema de trading...")
    print("=" * 60)

    results = run_integration_test()

    if 'error' not in results:
        # Mostrar resumo
        print(f"Tempo total: {results.get('duration_seconds', 0):.2f}s")
        print(f"Score de integração: {results.get('integration_score', 0):.1f}%")
        print(f"Testes realizados: {results.get('total_tests', 0)}")
        print(f"Testes aprovados: {results.get('tests_passed', 0)}")
        print(f"Testes reprovados: {results.get('tests_failed', 0)}")

        print("\nComponentes testados:")
        for component in results.get('components_tested', []):
            status = results.get('component_results', {}).get(component, {}).get('success', False)
            print(f"  {component}: {'OK' if status else 'FAIL'}")

        print("\nRecomendações:")
        for rec in results.get('recommendations', []):
            print(f"  • {rec}")

        # Detalhes por componente
        print("\nDetalhes por componente:")
        for component, details in results.get('component_results', {}).items():
            success = details.get('success', False)
            test_count = len(details.get('tests', []))
            error_count = len(details.get('errors', []))
            print(f"  {component}: {success} ({test_count} testes, {error_count} erros)")

    else:
        print(f"Erro no teste: {results['error']}")

    print("=" * 60)