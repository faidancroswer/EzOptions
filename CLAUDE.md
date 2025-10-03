# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

EzOptions is a real-time options trading analysis tool that provides interactive visualizations of options data, Greeks, and market indicators. The project includes both a web dashboard and an automated trading system with MetaTrader 5 integration.

## Development Commands

### Starting the Application
```bash
# Install dependencies and start the main dashboard
python main.py

# Start the dashboard directly (after manual dependency installation)
streamlit run ezoptions.py --server.port 8501
```

### Trading System Commands
```bash
# Start the automated trading system
python main_trading_system.py

# Run integration tests
python integration_test.py

# Run system diagnostics
python trading_system.py

# Test individual components
python test_system.py
```

### Environment Setup
```bash
# Create virtual environment
python -m venv venv

# Activate environment
# Windows: .\venv\Scripts\activate
# macOS/Linux: source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Architecture Overview

### Core Components

**Main Dashboard (`ezoptions.py`)**
- Streamlit-based web application for options analysis
- Real-time data fetching from yfinance and options chains
- Interactive charts for volume, open interest, Greeks exposure
- Multi-ticker support with focus on SPX, QQQ, and major indices

**Trading System (`trading_system.py`, `main_trading_system.py`)**
- Automated trading with 6 sophisticated setup detection algorithms
- Integration with MetaTrader 5 for US100/NASDAQ-100 trading
- Real-time risk management and position sizing
- Comprehensive audit logging and duplicate signal prevention

**Setup Detection (`trading_setups_corrected.py`)**
- 6 distinct trading setups based on options Greeks analysis
- CHARM, DELTA, GAMMA, and VWAP-based signal generation
- Bullish/bearish breakout, pullback, and consolidation strategies

**Risk Management (`advanced_risk_manager.py`)**
- Dynamic position sizing with confidence-based calculations
- Maximum 1% account risk per trade
- Daily loss limits and consecutive loss controls
- Real-time exposure monitoring

**Data Integration (`ezoptions_connector.py`)**
- Fetches real-time data from ezOptions dashboard
- Caching mechanism for efficient data retrieval
- Support for multiple tickers and options data

### Configuration System

All system settings are centralized in `config.json`:
- MT5 connection credentials and server settings
- Trading parameters (symbol, lot size, risk limits)
- Risk management rules and position sizing
- Logging and audit configuration
- Active trading setups and check intervals

### Key Dependencies

**Data & Visualization**
- `streamlit`: Web dashboard framework
- `yfinance`: Market data fetching
- `plotly`: Interactive charts
- `pandas`: Data manipulation

**Trading Integration**
- `MetaTrader5`: MT5 API integration
- `requests`: HTTP client for data fetching

**Analysis Libraries**
- `scipy`: Statistical calculations
- `numpy`: Numerical computations
- `beautifulsoup4`: Web scraping for options data

## Development Workflow

1. **Local Development**: Use `python main.py` to start the dashboard with automatic dependency installation
2. **Trading System**: Configure MT5 credentials in `config.json` before running trading components
3. **Testing**: Run `integration_test.py` to verify system components and connectivity
4. **Monitoring**: Check logs in `trading_system.log` and audit logs in `audit_logs/`

## Important Notes

- The main application fetches ezoptions.py directly from GitHub on startup
- Trading system requires MetaTrader 5 installation and configured account
- All trading operations include comprehensive risk management and audit trails
- The system supports both paper and live trading modes through MT5 configuration