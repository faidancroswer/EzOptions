#!/usr/bin/env python3
"""
Dashboard Diagnostics Tool
Verifica erros e performance do EzOptions Dashboard
"""

import requests
import json
import time
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DashboardDiagnostics:
    def __init__(self, base_url="http://localhost:8501"):
        self.base_url = base_url
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "url": base_url,
            "checks": {}
        }

    def check_connectivity(self):
        """Verifica se o dashboard está acessível"""
        try:
            response = requests.get(self.base_url, timeout=10)
            self.results["checks"]["connectivity"] = {
                "status": "PASS" if response.status_code == 200 else "FAIL",
                "status_code": response.status_code,
                "response_time": response.elapsed.total_seconds(),
                "server": response.headers.get("Server", "Unknown")
            }
            logger.info(f"Conectividade: {response.status_code} - {response.elapsed.total_seconds():.2f}s")
        except Exception as e:
            self.results["checks"]["connectivity"] = {
                "status": "FAIL",
                "error": str(e)
            }
            logger.error(f"Erro de conectividade: {e}")

    def check_streamlit_health(self):
        """Verifica endpoints de saúde do Streamlit"""
        try:
            health_response = requests.get(f"{self.base_url}/_stcore/health", timeout=5)
            self.results["checks"]["streamlit_health"] = {
                "status": "PASS" if health_response.status_code == 200 else "❌ FAIL",
                "response": health_response.json() if health_response.headers.get("content-type", "").startswith("application/json") else "Non-JSON response"
            }
        except Exception as e:
            self.results["checks"]["streamlit_health"] = {
                "status": "FAIL",
                "error": str(e)
            }
            logger.warning(f"Streamlit health check failed: {e}")

    def check_static_resources(self):
        """Verifica se recursos estáticos estão carregando"""
        static_checks = []

        # Lista de recursos comuns para verificar
        resources = [
            "/_stcore/static/js/main.js",
            "/_stcore/static/css/main.css",
            "/static/media/logo.svg"
        ]

        for resource in resources:
            try:
                response = requests.get(f"{self.base_url}{resource}", timeout=5)
                static_checks.append({
                    "resource": resource,
                    "status": "PASS" if response.status_code == 200 else "FAIL",
                    "status_code": response.status_code,
                    "size": len(response.content)
                })
            except Exception as e:
                static_checks.append({
                    "resource": resource,
                    "status": "FAIL",
                    "error": str(e)
                })

        self.results["checks"]["static_resources"] = static_checks

    def analyze_performance(self):
        """Analisa performance de carregamento"""
        start_time = time.time()

        try:
            # Simula carregamento completo da página
            response = requests.get(self.base_url, timeout=30)
            load_time = time.time() - start_time

            # Analisa response headers
            headers = dict(response.headers)

            self.results["checks"]["performance"] = {
                "status": "PASS" if load_time < 5 else "⚠️ SLOW",
                "load_time_seconds": round(load_time, 2),
                "content_size": len(response.content),
                "headers": {
                    "content_type": headers.get("content-type", "Unknown"),
                    "cache_control": headers.get("cache-control", "Not set"),
                    "server": headers.get("server", "Unknown")
                }
            }

            logger.info(f"Performance: {load_time:.2f}s - {len(response.content)} bytes")

        except Exception as e:
            self.results["checks"]["performance"] = {
                "status": "FAIL",
                "error": str(e)
            }

    def check_api_endpoints(self):
        """Verifica endpoints de API do sistema de trading"""
        api_endpoints = [
            "/api/trading_status",
            "/api/account_info",
            "/api/active_positions"
        ]

        api_results = []

        for endpoint in api_endpoints:
            try:
                response = requests.get(f"{self.base_url}{endpoint}", timeout=5)
                api_results.append({
                    "endpoint": endpoint,
                    "status": "PASS" if response.status_code == 200 else "FAIL",
                    "status_code": response.status_code,
                    "response_time": response.elapsed.total_seconds()
                })
            except Exception as e:
                api_results.append({
                    "endpoint": endpoint,
                    "status": "FAIL",
                    "error": str(e)
                })

        self.results["checks"]["api_endpoints"] = api_results

    def generate_report(self):
        """Gera relatório completo de diagnósticos"""
        print("\n" + "="*60)
        print("EZOPTIONS DASHBOARD DIAGNOSTICS REPORT")
        print("="*60)
        print(f"Timestamp: {self.results['timestamp']}")
        print(f"URL: {self.results['url']}")
        print()

        # Sumário geral
        total_checks = len(self.results["checks"])
        passed_checks = sum(1 for check in self.results["checks"].values()
                          if isinstance(check, dict) and check.get("status", "").startswith("PASS"))

        print(f"SUMARIO: {passed_checks}/{total_checks} testes passaram")
        print()

        # Detalhes dos testes
        for check_name, check_result in self.results["checks"].items():
            print(f"{check_name.upper().replace('_', ' ')}")

            if isinstance(check_result, list):
                for item in check_result:
                    if isinstance(item, dict):
                        status = item.get("status", "Unknown")
                        name = item.get("resource", item.get("endpoint", "Unknown"))
                        print(f"   {status} {name}")
            else:
                status = check_result.get("status", "Unknown")
                print(f"   Status: {status}")

                if "error" in check_result:
                    print(f"   Erro: {check_result['error']}")
                elif "response_time" in check_result:
                    print(f"   Tempo: {check_result['response_time']:.3f}s")
                elif "load_time_seconds" in check_result:
                    print(f"   Carregamento: {check_result['load_time_seconds']}s")
                    print(f"   Tamanho: {check_result['content_size']} bytes")

            print()

        # Recomendações
        print("RECOMENDACOES:")

        if self.results["checks"].get("connectivity", {}).get("status") != "PASS":
            print("   Verifique se o dashboard está rodando em localhost:8501")

        if self.results["checks"].get("performance", {}).get("status") == "SLOW":
            print("   Dashboard lento. Considere otimizar graficos e consultas")

        failed_apis = [api for api in self.results["checks"].get("api_endpoints", [])
                      if isinstance(api, dict) and api.get("status", "").startswith("FAIL")]
        if failed_apis:
            print("   Endpoints de API com falhas. Verifique sistema de trading")

        print("="*60)

        return self.results

def main():
    """Função principal de diagnóstico"""
    diagnostics = DashboardDiagnostics()

    print("Iniciando diagnosticos do EzOptions Dashboard...")

    # Executa todos os testes
    diagnostics.check_connectivity()
    diagnostics.check_streamlit_health()
    diagnostics.check_static_resources()
    diagnostics.analyze_performance()
    diagnostics.check_api_endpoints()

    # Gera relatório
    report = diagnostics.generate_report()

    # Salva relatório em arquivo
    with open("dashboard_diagnostics_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"📄 Relatório salvo em: dashboard_diagnostics_report.json")
    print("\n🎯 Para verificar erros de JavaScript no navegador:")
    print("   1. Abra http://localhost:8501")
    print("   2. Pressione F12 (Chrome DevTools)")
    print("   3. Vá para aba 'Console'")
    print("   4. Procure por erros vermelhos")
    print("   5. Vá para aba 'Network' e verifique requisições falhando")

if __name__ == "__main__":
    main()