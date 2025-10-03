"""
MT5 Bridge Simulado - Responde na porta 8082
Fornece endpoints para integração com MT5
"""

from flask import Flask, jsonify
from datetime import datetime
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

@app.route('/api/account/info', methods=['GET'])
def get_account_info():
    """Endpoint para informações da conta MT5"""
    try:
        account_data = {
            "login": 111655745,
            "server": "FBS-Real",
            "balance": 10000.0,
            "equity": 10250.0,
            "profit": 250.0,
            "margin": 500.0,
            "margin_free": 9500.0,
            "leverage": 1000,
            "currency": "USD",
            "timestamp": datetime.now().isoformat()
        }

        logger.info("Informações da conta MT5 enviadas com sucesso")
        return jsonify(account_data)

    except Exception as e:
        logger.error(f"Erro no endpoint /api/account/info: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/positions', methods=['GET'])
def get_positions():
    """Endpoint para obter posições abertas"""
    try:
        positions = [
            {
                "ticket": 123456,
                "symbol": "US100",
                "type": "BUY",
                "volume": 0.01,
                "price_open": 4500.0,
                "price_current": 4525.0,
                "profit": 25.0,
                "time": datetime.now().isoformat()
            }
        ]

        logger.info("Posições enviadas com sucesso")
        return jsonify({"positions": positions})

    except Exception as e:
        logger.error(f"Erro no endpoint /api/positions: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/orders', methods=['GET'])
def get_orders():
    """Endpoint para obter ordens abertas"""
    try:
        orders = []  # Nenhuma ordem pendente no momento

        logger.info("Ordens enviadas com sucesso")
        return jsonify({"orders": orders})

    except Exception as e:
        logger.error(f"Erro no endpoint /api/orders: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/history', methods=['GET'])
def get_history():
    """Endpoint para obter histórico de negociações"""
    try:
        history = [
            {
                "ticket": 123455,
                "symbol": "US100",
                "type": "BUY",
                "volume": 0.01,
                "price_open": 4480.0,
                "price_close": 4500.0,
                "profit": 20.0,
                "time": datetime.now().isoformat()
            }
        ]

        logger.info("Histórico enviado com sucesso")
        return jsonify({"history": history})

    except Exception as e:
        logger.error(f"Erro no endpoint /api/history: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/status', methods=['GET'])
def get_status():
    """Endpoint para verificar status do MT5 Bridge"""
    return jsonify({
        "status": "running",
        "mt5_connected": True,
        "timestamp": datetime.now().isoformat(),
        "version": "1.0"
    })

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "healthy"}), 200

if __name__ == '__main__':
    logger.info("Iniciando MT5 Bridge simulado na porta 8082")
    app.run(host='localhost', port=8082, debug=False, threaded=True)