"""
Dashboard Streamlit para Monitoramento de Paper Trading
"""

import streamlit as st
import pandas as pd
import json
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import os
from pathlib import Path
import requests
import glob
import asyncio
from real_paper_trading import real_paper_trading  # Usar instância global

# Configurar página
st.set_page_config(
    page_title="Paper Trading Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Título principal
st.title("📊 Dashboard de Paper Trading")
st.markdown("---")

# Função helper para obter P&L em % (suporta campo pnl antigo)
def get_pnl_percent(trade):
    """Obtém P&L em % do trade, com fallback para campo pnl antigo"""
    pnl_percent = trade.get("pnl_percent")
    if pnl_percent is None and trade.get("pnl") is not None:
        # Converter pnl absoluto para % aproximado
        entry = trade.get("entry_price", 1)
        size = trade.get("position_size", 1)
        if entry > 0 and size > 0:
            pnl_percent = (trade["pnl"] / (entry * size)) * 100
        else:
            pnl_percent = 0
    return pnl_percent if pnl_percent is not None else 0

@st.cache_data(ttl=5)
def get_current_price(symbol):
    """Obtém o preço atual do símbolo via Binance API"""
    try:
        url = f"https://api.binance.com/api/v3/ticker/price?symbol={symbol}"
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            data = response.json()
            return float(data['price'])
        else:
            return None
    except Exception as e:
        st.error(f"Erro ao obter preço de {symbol}: {e}")
        return None

# Função para carregar dados do portfólio (CORRIGIDO: cache reduzido para 2s)
@st.cache_data(ttl=2)
def load_portfolio_data():
    """Carrega dados do portfólio"""
    try:
        if os.path.exists("portfolio/state.json"):
            with open("portfolio/state.json", "r", encoding='utf-8') as f:
                return json.load(f)
    except Exception as e:
        st.error(f"Erro ao carregar dados: {e}")
    return None

# Função para carregar histórico de trades
@st.cache_data(ttl=5)
def load_trade_history():
    """Carrega histórico de trades"""
    try:
        state = load_portfolio_data()
        if state and "trade_history" in state:
            return state["trade_history"]
    except Exception as e:
        st.error(f"Erro ao carregar histórico: {e}")
    return []

@st.cache_data(ttl=5)
def load_last_signals():
    """Carrega os últimos sinais gerados para cada par/fonte"""
    signals_by_pair = {}

    # Top 10 pares
    pairs = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "ADAUSDT",
             "XRPUSDT", "DOGEUSDT", "AVAXUSDT", "DOTUSDT", "LINKUSDT"]

    for pair in pairs:
        signals_by_pair[pair] = {
            "DEEPSEEK": None,
            "AGNO": None
        }

    # Buscar arquivos de sinais
    signal_files = glob.glob("signals/agno_*_*.json")

    # Filtrar apenas arquivos de sinais (não os _last_analysis)
    signal_files = [f for f in signal_files if "_last_analysis" not in f]

    # Ordenar por data (mais recente primeiro)
    signal_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)

    for filepath in signal_files:
        try:
            with open(filepath, "r", encoding='utf-8') as f:
                signal = json.load(f)

            symbol = signal.get("symbol", "")
            source = signal.get("source", "UNKNOWN")

            # Apenas processar se for um dos 10 pares
            if symbol in pairs:
                # Se ainda não temos sinal para este par/fonte, usar este
                if source in ["DEEPSEEK", "AGNO"]:
                    if signals_by_pair[symbol][source] is None:
                        signals_by_pair[symbol][source] = signal

        except Exception as e:
            continue

    return signals_by_pair


@st.cache_data(ttl=5)
def load_last_analysis_timestamps():
    """Carrega timestamps da última análise de cada par"""
    analysis_times = {}

    pairs = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "ADAUSDT",
             "XRPUSDT", "DOGEUSDT", "AVAXUSDT", "DOTUSDT", "LINKUSDT"]

    for pair in pairs:
        analysis_times[pair] = None

        # Verificar arquivo de última análise
        last_analysis_file = f"signals/agno_{pair}_last_analysis.json"
        if os.path.exists(last_analysis_file):
            try:
                with open(last_analysis_file, "r", encoding='utf-8') as f:
                    data = json.load(f)
                    analysis_times[pair] = {
                        "timestamp": data.get("timestamp"),
                        "signal": data.get("signal"),
                        "confidence": data.get("confidence", 0)
                    }
            except:
                pass

    return analysis_times

# Função para obter preços atuais dos principais pares
@st.cache_data(ttl=10)
def get_market_prices():
    """Obtém preços atuais dos principais pares de criptomoedas"""
    symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT", "DOGEUSDT", "AVAXUSDT", "DOTUSDT", "MATICUSDT"]
    prices = {}
    
    for symbol in symbols:
        try:
            response = requests.get(f"https://fapi.binance.com/fapi/v1/ticker/price", params={'symbol': symbol}, timeout=5)
            if response.status_code == 200:
                data = response.json()
                # Verificar se a chave 'price' existe e se é válida
                if isinstance(data, dict) and 'price' in data:
                    try:
                        prices[symbol] = float(data['price'])
                    except (ValueError, TypeError):
                        # Se não conseguir converter, pular este símbolo
                        continue
                else:
                    # Resposta não tem a estrutura esperada
                    continue
            else:
                # Status code não é 200, pular este símbolo
                continue
        except requests.exceptions.RequestException:
            # Erro de conexão/timeout, pular este símbolo
            continue
        except Exception as e:
            # Outro erro, pular este símbolo silenciosamente
            continue
    
    return prices

# Carregar dados
portfolio_data = load_portfolio_data()
trade_history = load_trade_history()

# Sidebar - Controles
with st.sidebar:
    st.header("⚙️ Controles")
    
    # Botão para iniciar análise contínua
    st.subheader("🚀 Sistema de Trading")
    if st.button("▶️ Iniciar Análise Contínua", type="primary", use_container_width=True):
        st.info("📡 Iniciando análise contínua...")
        st.code("python main.py --symbol BTCUSDT --mode monitor --paper", language="bash")
        st.warning("⚠️ Execute este comando no terminal para iniciar a análise contínua")
    
    if st.button("⏹️ Parar Análise", use_container_width=True):
        st.info("⏹️ Comando para parar será executado")
    
    st.markdown("---")
    
    # Auto-refresh
    auto_refresh = st.checkbox("🔄 Auto-refresh (5s)", value=False)
    
    # Botão de refresh manual
    if st.button("🔄 Atualizar Agora"):
        st.rerun()
    
    st.markdown("---")
    
    # Informações do sistema
    st.header("ℹ️ Informações")
    st.info("Dashboard atualizado em tempo real com dados do paper trading.")
    st.markdown("""
    **Recursos:**
    - 📊 Resumo do portfólio
    - 📈 Gráficos de performance
    - 💰 Posições abertas
    - 📜 Histórico de trades
    - 📉 Análise de resultados
    """)

# Layout principal
if portfolio_data:
    # KPIs principais - Foco em P&L em PORCENTAGEM
    col1, col2, col3, col4 = st.columns(4)
    
    # Calcular P&L acumulado em % (soma de todos os trades fechados)
    closed_trades = [t for t in trade_history if t.get("status") in ["CLOSED", "CLOSED_PARTIAL"]]
    realized_pnl_percent = sum([get_pnl_percent(t) for t in closed_trades])
    
    # Calcular P&L médio não realizado (posições abertas)
    open_positions = portfolio_data.get("positions", {})
    unrealized_pnl_percent = 0.0
    market_prices = get_market_prices()
    
    open_pnl_list = []
    for pos_key, position in open_positions.items():
        symbol = position.get("symbol")
        entry_price = position.get("entry_price", 0)
        signal_type = position.get("signal", "BUY")
        current_price = market_prices.get(symbol, entry_price)
        
        if entry_price > 0:
            if signal_type == "BUY":
                pnl_percent = ((current_price - entry_price) / entry_price) * 100
            else:  # SELL
                pnl_percent = ((entry_price - current_price) / entry_price) * 100
            open_pnl_list.append(pnl_percent)
    
    if open_pnl_list:
        unrealized_pnl_percent = sum(open_pnl_list) / len(open_pnl_list)
    
    # P&L total acumulado (soma de todos os trades fechados)
    total_pnl_percent = realized_pnl_percent
    
    with col1:
        st.metric(
            "💰 P&L Acumulado",
            f"{realized_pnl_percent:+.2f}%",
            delta="Trades fechados"
        )
    
    with col2:
        st.metric(
            "📈 P&L Médio Aberto",
            f"{unrealized_pnl_percent:+.2f}%",
            delta="Posições abertas"
        )
    
    with col3:
        st.metric(
            "💵 P&L Total",
            f"{total_pnl_percent:+.2f}%",
            delta=f"{'✅' if total_pnl_percent >= 0 else '❌'}"
        )
    
    with col4:
        open_count = len(open_positions)
        st.metric(
            "📊 Posições Abertas",
            open_count
        )
    
    st.markdown("---")
    
    # Tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["📈 Overview", "💰 Posições Abertas", "📜 Histórico", "📉 Análise", "💹 Preços de Mercado", "🔍 Monitor Sistema"])
    
    with tab1:
        st.header("📈 Visão Geral do Portfólio")
        
        # Calcular estatísticas (apenas %)
        closed_trades = [t for t in trade_history if t.get("status") in ["CLOSED", "CLOSED_PARTIAL"]]
        open_trades = [t for t in trade_history if t.get("status") == "OPEN"]
        winning_trades = len([t for t in closed_trades if get_pnl_percent(t) > 0])
        losing_trades = len([t for t in closed_trades if get_pnl_percent(t) < 0])
        win_rate = (winning_trades / len(closed_trades) * 100) if closed_trades else 0
        total_pnl_percent = sum([get_pnl_percent(t) for t in closed_trades])
        
        # Métricas de performance
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("🎯 Win Rate", f"{win_rate:.1f}%")
        
        with col2:
            st.metric("✅ Trades Ganhadores", winning_trades)
        
        with col3:
            st.metric("❌ Trades Perdedores", losing_trades)
        
        with col4:
            st.metric("💰 P&L Acumulado", f"{total_pnl_percent:+.2f}%")
        
        with col5:
            st.metric("📊 Trades Abertos", len(open_trades))
        
        # Mostrar detalhes dos trades fechados
        if closed_trades:
            st.subheader("📋 Últimos Trades Fechados")
            
            closed_list = []
            for trade in closed_trades[-10:]:  # Últimos 10 trades
                entry_price = trade.get('entry_price', 0)
                stop_loss = trade.get('stop_loss', 0)
                take_profit_1 = trade.get('take_profit_1', 0)
                take_profit_2 = trade.get('take_profit_2', 0)
                position_size = trade.get('position_size', 0)
                position_value = trade.get('position_value', 0)
                
                # Calcular diferenças percentuais
                sl_diff = ((stop_loss - entry_price) / entry_price * 100) if entry_price > 0 else 0
                tp1_diff = ((take_profit_1 - entry_price) / entry_price * 100) if entry_price > 0 else 0
                tp2_diff = ((take_profit_2 - entry_price) / entry_price * 100) if entry_price > 0 else 0
                
                pnl_percent = get_pnl_percent(trade)
                closed_list.append({
                    "Data": trade.get("timestamp", "N/A")[:16],
                    "Símbolo": trade.get("symbol", "N/A"),
                    "Fonte": trade.get("source", "UNKNOWN"),
                    "Tipo": trade.get("signal", "N/A"),
                    "Entrada": f"${entry_price:,.2f}",
                    "Saída": f"${trade.get('close_price', 0):,.2f}" if trade.get('close_price') else "N/A",
                    "P&L": f"{pnl_percent:+.2f}%",
                    "Motivo": trade.get('close_reason', 'N/A')
                })
            
            df_closed = pd.DataFrame(closed_list)
            st.dataframe(df_closed, use_container_width=True, hide_index=True)
        
        # Mostrar posições abertas no overview também
        if open_trades:
            st.subheader("🔄 Posições Abertas Atualmente")
            
            open_list = []
            for trade in open_trades:
                entry_price = trade.get('entry_price', 0)
                stop_loss = trade.get('stop_loss', 0)
                take_profit_1 = trade.get('take_profit_1', 0)
                take_profit_2 = trade.get('take_profit_2', 0)
                position_size = trade.get('position_size', 0)
                position_value = trade.get('position_value', 0)
                
                # Calcular diferenças percentuais
                sl_diff = ((stop_loss - entry_price) / entry_price * 100) if entry_price > 0 else 0
                tp1_diff = ((take_profit_1 - entry_price) / entry_price * 100) if entry_price > 0 else 0
                tp2_diff = ((take_profit_2 - entry_price) / entry_price * 100) if entry_price > 0 else 0
                
                open_list.append({
                    "Data": trade.get("timestamp", "N/A")[:16],
                    "Símbolo": trade.get("symbol", "N/A"),
                    "Fonte": trade.get("source", "UNKNOWN"),  # DEEPSEEK ou AGNO
                    "Tipo": trade.get("signal", "N/A"),
                    "Entrada": f"${entry_price:,.2f}",
                    "Tamanho": f"{position_size:.6f}",
                    "Valor": f"${position_value:,.2f}",
                    "Stop Loss": f"${stop_loss:,.2f} ({sl_diff:+.1f}%)",
                    "Take Profit 1": f"${take_profit_1:,.2f} ({tp1_diff:+.1f}%)",
                    "Take Profit 2": f"${take_profit_2:,.2f} ({tp2_diff:+.1f}%)",
                    "Confiança": f"{trade.get('confidence', 0)}/10"
                })
            
            df_open = pd.DataFrame(open_list)
            st.dataframe(df_open, use_container_width=True, hide_index=True)
        
        # Gráfico de performance
        if len(trade_history) > 0:
            st.subheader("📊 Performance ao Longo do Tempo")
            
            # Preparar dados para gráfico
            trades_df = pd.DataFrame(trade_history)
            trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'])
            trades_df = trades_df.sort_values('timestamp')
            
            # Verificar se coluna 'pnl_percent' existe e preencher valores nulos
            if 'pnl_percent' not in trades_df.columns:
                trades_df['pnl_percent'] = 0.0
            else:
                # Preencher valores nulos com 0 (trades abertos ainda não têm P&L)
                trades_df['pnl_percent'] = trades_df['pnl_percent'].fillna(0.0)
            
            # Calcular P&L acumulado em % apenas para trades fechados
            trades_df['cumulative_pnl_percent'] = trades_df['pnl_percent'].cumsum()
            
            # Criar gráfico
            fig = go.Figure()
            
            last_pnl = trades_df['cumulative_pnl_percent'].iloc[-1] if len(trades_df) > 0 else 0
            fig.add_trace(go.Scatter(
                x=trades_df['timestamp'],
                y=trades_df['cumulative_pnl_percent'],
                mode='lines+markers',
                name='P&L Acumulado',
                line=dict(color='green' if last_pnl >= 0 else 'red', width=2),
                marker=dict(size=8)
            ))
            
            fig.update_layout(
                title="Evolução do P&L Acumulado",
                xaxis_title="Data",
                yaxis_title="P&L Acumulado (%)",
                hovermode='x unified',
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.header("💰 Posições Abertas")

        positions = portfolio_data.get("positions", {})

        if positions:
            market_prices = get_market_prices()

            # Exibir posições em cards com botões de fechar
            for position_key, position in positions.items():
                # Extrair dados da posição
                symbol = position.get("symbol", position_key.split("_")[0])
                source = position.get("source", "UNKNOWN")
                entry_price = position.get('entry_price', 0)
                position_size = position.get('position_size', 0)
                signal_type = position.get("signal", "BUY")
                confidence = position.get('confidence', 0)
                operation_type = position.get("operation_type", "SWING_TRADE")

                # Emoji para tipo de operação
                type_emoji = {
                    "SCALP": "⚡",
                    "DAY_TRADE": "☀️",
                    "SWING_TRADE": "🌊",
                    "POSITION_TRADE": "🏔️"
                }
                type_display = f"{type_emoji.get(operation_type, '📊')} {operation_type.replace('_', ' ')}"

                # Obter preço atual e calcular P&L
                current_price = market_prices.get(symbol, entry_price)
                if signal_type == "BUY":
                    pnl_percent = ((current_price - entry_price) / entry_price * 100) if entry_price > 0 else 0
                else:  # SELL
                    pnl_percent = ((entry_price - current_price) / entry_price * 100) if entry_price > 0 else 0

                # Cor do P&L
                pnl_color = "green" if pnl_percent > 0 else "red" if pnl_percent < 0 else "gray"

                # Card da posição
                with st.container():
                    col1, col2 = st.columns([4, 1])

                    with col1:
                        st.markdown(f"""
                        **{symbol}** ({source}) - {signal_type} | {type_display}
                        - **Entrada:** ${entry_price:,.2f}
                        - **Atual:** ${current_price:,.2f}
                        - **P&L:** :{pnl_color}[{pnl_percent:+.2f}%]
                        - **Confiança:** {confidence}/10
                        """)

                    with col2:
                        # Botão para fechar posição
                        if st.button(f"❌ Fechar", key=f"close_{position_key}"):
                            # Obter preço atualizado
                            fresh_price = get_current_price(symbol)
                            if fresh_price:
                                # Usar instância global do sistema de trading
                                try:
                                    result = asyncio.run(real_paper_trading.close_position_manual(position_key, fresh_price))

                                    if result.get("success"):
                                        st.success(result.get("message"))
                                        st.rerun()
                                    else:
                                        st.error(result.get("error"))
                                except Exception as e:
                                    st.error(f"Erro ao fechar posição: {e}")
                            else:
                                st.error(f"Não foi possível obter preço atual de {symbol}")

                    st.markdown("---")
        else:
            st.info("ℹ️ Nenhuma posição aberta no momento.")
    
    with tab3:
        st.header("📜 Histórico de Trades")
        
        if trade_history:
            # Preparar dados para tabela
            history_list = []
            for trade in trade_history:
                pnl_percent = get_pnl_percent(trade)
                history_list.append({
                    "ID": trade.get("trade_id", "N/A"),
                    "Símbolo": trade.get("symbol", "N/A"),
                    "Tipo": trade.get("signal", "N/A"),
                    "Entrada": f"${trade.get('entry_price', 0):,.2f}",
                    "Status": trade.get("status", "N/A"),
                    "P&L": f"{pnl_percent:+.2f}%" if pnl_percent != 0 else "N/A",
                    "Data": trade.get("timestamp", "N/A")[:19] if trade.get("timestamp") else "N/A"
                })
            
            df_history = pd.DataFrame(history_list)
            st.dataframe(df_history, use_container_width=True, hide_index=True)
        else:
            st.info("ℹ️ Nenhum trade registrado ainda.")
    
    with tab4:
        st.header("📉 Análise Detalhada")
        
        if len(closed_trades) > 0:
            # Estatísticas dos trades fechados
            st.subheader("📊 Estatísticas dos Trades")

            # Filtrar apenas trades com P&L válido (em %)
            pnl_percent_values = [get_pnl_percent(t) for t in closed_trades if get_pnl_percent(t) != 0]
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Distribuição de P&L
                if pnl_percent_values:
                    fig_pnl = px.histogram(
                        x=pnl_percent_values,
                        nbins=20,
                        title="Distribuição de P&L",
                        labels={"x": "P&L (%)", "y": "Frequência"}
                    )
                    st.plotly_chart(fig_pnl, use_container_width=True)
                else:
                    st.info("Nenhum dado de P&L disponível")
            
            with col2:
                # Box plot de P&L
                if pnl_percent_values:
                    fig_box = go.Figure()
                    fig_box.add_trace(go.Box(
                        y=pnl_percent_values,
                        name="P&L Distribution",
                        boxmean='sd'
                    ))
                    fig_box.update_layout(
                        title="Distribuição de P&L (Box Plot)",
                        yaxis_title="P&L (%)"
                    )
                    st.plotly_chart(fig_box, use_container_width=True)
                else:
                    st.info("Nenhum dado de P&L disponível")
            
            # Estatísticas descritivas (em %)
            st.subheader("📈 Estatísticas Descritivas")
            
            if pnl_percent_values:
                stats = {
                    "Média": f"{sum(pnl_percent_values) / len(pnl_percent_values):+.2f}%",
                    "Mediana": f"{sorted(pnl_percent_values)[len(pnl_percent_values)//2]:+.2f}%",
                    "Máximo": f"{max(pnl_percent_values):+.2f}%",
                    "Mínimo": f"{min(pnl_percent_values):+.2f}%",
                    "Total Acumulado": f"{sum(pnl_percent_values):+.2f}%"
                }
            else:
                stats = {"Mensagem": "Nenhum dado disponível"}
            
            st.json(stats)
        else:
            st.info("ℹ️ Não há trades fechados para análise.")

    with tab5:
        st.header("💹 Preços de Mercado em Tempo Real")
        
        # Obter preços atuais
        market_prices = get_market_prices()
        
        if market_prices:
            # Criar DataFrame com preços
            prices_data = []
            for symbol, price in market_prices.items():
                prices_data.append({
                    "Par": symbol,
                    "Preço Atual": f"${price:,.2f}" if price >= 1 else f"${price:.6f}",
                    "Preço Numérico": price
                })
            
            df_prices = pd.DataFrame(prices_data)
            df_prices = df_prices.sort_values("Preço Numérico", ascending=False)
            
            # Mostrar tabela
            st.dataframe(
                df_prices[["Par", "Preço Atual"]], 
                use_container_width=True, 
                hide_index=True
            )
            
            # Gráfico de barras
            fig_prices = px.bar(
                df_prices,
                x="Par",
                y="Preço Numérico",
                title="Preços Atuais dos Principais Pares",
                labels={"Preço Numérico": "Preço (USDT)", "Par": "Par de Negociação"}
            )
            fig_prices.update_layout(height=500)
            st.plotly_chart(fig_prices, use_container_width=True)
        else:
            st.warning("⚠️ Não foi possível carregar preços de mercado.")

    # =====================
    # NOVA ABA: MONITOR DO SISTEMA
    # =====================
    with tab6:
        st.header("🔍 Monitor do Sistema de Sinais")
        st.markdown("Acompanhe se o sistema está gerando sinais corretamente para todos os 10 pares monitorados.")

        # Top 10 pares
        monitored_pairs = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "ADAUSDT",
                          "XRPUSDT", "DOGEUSDT", "AVAXUSDT", "DOTUSDT", "LINKUSDT"]

        # Carregar dados
        positions = portfolio_data.get("positions", {}) if portfolio_data else {}
        last_signals = load_last_signals()
        last_analysis = load_last_analysis_timestamps()

        # Construir tabela de monitoramento
        monitor_data = []

        for pair in monitored_pairs:
            row = {
                "Par": pair,
                "DEEPSEEK": "—",
                "AGNO": "—",
                "Última Análise": "—",
                "Status": "⚪"
            }

            # Verificar posição DEEPSEEK aberta
            deepseek_key = f"{pair}_DEEPSEEK"
            deepseek_short_key = f"{pair}_DEEPSEEK_SHORT"

            has_deepseek_position = False
            if deepseek_key in positions and positions[deepseek_key].get("status") == "OPEN":
                signal_type = positions[deepseek_key].get("signal", "?")
                confidence = positions[deepseek_key].get("confidence", 0)
                row["DEEPSEEK"] = f"✅ {signal_type} ({confidence}/10)"
                has_deepseek_position = True
            elif deepseek_short_key in positions and positions[deepseek_short_key].get("status") == "OPEN":
                signal_type = positions[deepseek_short_key].get("signal", "?")
                confidence = positions[deepseek_short_key].get("confidence", 0)
                row["DEEPSEEK"] = f"✅ {signal_type} ({confidence}/10)"
                has_deepseek_position = True

            # Se não tem posição DEEPSEEK, mostrar último sinal
            if not has_deepseek_position:
                if last_signals.get(pair, {}).get("DEEPSEEK"):
                    last_ds = last_signals[pair]["DEEPSEEK"]
                    signal_type = last_ds.get("signal", "N/A")
                    confidence = last_ds.get("confidence", 0)

                    if signal_type == "NO_SIGNAL":
                        row["DEEPSEEK"] = f"⏸️ NO_SIGNAL ({confidence}/10)"
                    elif confidence < 7:
                        row["DEEPSEEK"] = f"❌ {signal_type} ({confidence}/10) - Baixa"
                    else:
                        row["DEEPSEEK"] = f"⚠️ {signal_type} ({confidence}/10) - Não exec."
                else:
                    row["DEEPSEEK"] = "❓ Sem sinal"

            # Verificar posição AGNO aberta
            agno_key = f"{pair}_AGNO"
            agno_short_key = f"{pair}_AGNO_SHORT"

            has_agno_position = False
            if agno_key in positions and positions[agno_key].get("status") == "OPEN":
                signal_type = positions[agno_key].get("signal", "?")
                confidence = positions[agno_key].get("confidence", 0)
                row["AGNO"] = f"✅ {signal_type} ({confidence}/10)"
                has_agno_position = True
            elif agno_short_key in positions and positions[agno_short_key].get("status") == "OPEN":
                signal_type = positions[agno_short_key].get("signal", "?")
                confidence = positions[agno_short_key].get("confidence", 0)
                row["AGNO"] = f"✅ {signal_type} ({confidence}/10)"
                has_agno_position = True

            # Se não tem posição AGNO, mostrar último sinal
            if not has_agno_position:
                if last_signals.get(pair, {}).get("AGNO"):
                    last_ag = last_signals[pair]["AGNO"]
                    signal_type = last_ag.get("signal", "N/A")
                    confidence = last_ag.get("confidence", 0)

                    if signal_type == "NO_SIGNAL":
                        row["AGNO"] = f"⏸️ NO_SIGNAL ({confidence}/10)"
                    elif confidence < 7:
                        row["AGNO"] = f"❌ {signal_type} ({confidence}/10) - Baixa"
                    else:
                        row["AGNO"] = f"⚠️ {signal_type} ({confidence}/10) - Não exec."
                else:
                    row["AGNO"] = "❓ Sem sinal"

            # Verificar posições antigas (UNKNOWN/LEGACY)
            unknown_key = f"{pair}_SHORT"

            if unknown_key in positions and positions[unknown_key].get("status") == "OPEN":
                source = positions[unknown_key].get("source", "LEGACY")
                signal_type = positions[unknown_key].get("signal", "?")
                confidence = positions[unknown_key].get("confidence", 0)
                # Adicionar nota sobre posição legada
                if row["DEEPSEEK"] == "—" or "Sem sinal" in row["DEEPSEEK"]:
                    row["DEEPSEEK"] = f"🔄 {signal_type} ({confidence}/10) - {source}"

            # Última análise
            if last_analysis.get(pair):
                timestamp = last_analysis[pair].get("timestamp", "")
                if timestamp:
                    try:
                        dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                        if dt.tzinfo is None:
                            now = datetime.now()
                        else:
                            now = datetime.now(dt.tzinfo)
                        diff = now - dt
                        minutes = int(diff.total_seconds() / 60)

                        if minutes < 60:
                            row["Última Análise"] = f"há {minutes}min"
                        elif minutes < 1440:
                            hours = minutes // 60
                            row["Última Análise"] = f"há {hours}h"
                        else:
                            days = minutes // 1440
                            row["Última Análise"] = f"há {days}d"
                    except:
                        row["Última Análise"] = timestamp[:16]

            # Status geral
            if has_deepseek_position and has_agno_position:
                row["Status"] = "🟢 Completo"
            elif has_deepseek_position or has_agno_position:
                row["Status"] = "🟡 Parcial"
            elif "NO_SIGNAL" in row["DEEPSEEK"] or "NO_SIGNAL" in row["AGNO"]:
                row["Status"] = "⚪ Aguardando"
            elif "Baixa" in row["DEEPSEEK"] and "Baixa" in row["AGNO"]:
                row["Status"] = "🔴 Sem oportunidade"
            else:
                row["Status"] = "⚪ Aguardando"

            monitor_data.append(row)

        # Mostrar tabela
        df_monitor = pd.DataFrame(monitor_data)

        # Estilizar tabela
        st.dataframe(
            df_monitor,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Par": st.column_config.TextColumn("Par", width="small"),
                "DEEPSEEK": st.column_config.TextColumn("🤖 DEEPSEEK", width="medium"),
                "AGNO": st.column_config.TextColumn("🧠 AGNO", width="medium"),
                "Última Análise": st.column_config.TextColumn("⏰ Última Análise", width="small"),
                "Status": st.column_config.TextColumn("📊 Status", width="small"),
            }
        )

        # Legenda
        st.markdown("---")
        st.markdown('''
        **Legenda:**
        - ✅ **Posição aberta** - Trade em andamento
        - ❌ **Baixa confiança** - Sinal gerado mas não executado (confiança < 7)
        - ⏸️ **NO_SIGNAL** - Análise feita mas sem oportunidade
        - ⚠️ **Não executado** - Sinal válido mas não executado (posição existente, etc.)
        - ❓ **Sem sinal** - Nenhum sinal encontrado para este par
        - 🔄 **LEGACY** - Posição antiga do sistema anterior

        **Status:**
        - 🟢 **Completo** - Tem posição DEEPSEEK e AGNO
        - 🟡 **Parcial** - Tem posição de uma fonte apenas
        - ⚪ **Aguardando** - Sem posição, aguardando próxima análise
        - 🔴 **Sem oportunidade** - Ambas fontes com confiança baixa
        ''')

        # Seção de Fechamento Rápido
        st.markdown("---")
        st.subheader("⚡ Fechamento Rápido")
        st.markdown("Feche posições abertas com um clique (usa preço atual de mercado)")

        if positions:
            # Organizar posições em grid 3 colunas
            position_keys = list(positions.keys())
            cols_per_row = 3

            for i in range(0, len(position_keys), cols_per_row):
                cols = st.columns(cols_per_row)

                for j in range(cols_per_row):
                    idx = i + j
                    if idx < len(position_keys):
                        position_key = position_keys[idx]
                        position = positions[position_key]

                        symbol = position.get("symbol", position_key.split("_")[0])
                        source = position.get("source", "UNKNOWN")
                        signal_type = position.get("signal", "?")

                        with cols[j]:
                            button_label = f"❌ {symbol} ({source})"
                            if st.button(button_label, key=f"quick_close_{position_key}"):
                                # Obter preço atualizado
                                fresh_price = get_current_price(symbol)
                                if fresh_price:
                                    try:
                                        result = asyncio.run(real_paper_trading.close_position_manual(position_key, fresh_price))

                                        if result.get("success"):
                                            st.success(f"✅ {symbol} fechado!")
                                            st.rerun()
                                        else:
                                            st.error(result.get("error"))
                                    except Exception as e:
                                        st.error(f"Erro: {e}")
                                else:
                                    st.error(f"Erro ao obter preço de {symbol}")
        else:
            st.info("Nenhuma posição aberta para fechar.")

        # Estatísticas rápidas
        st.markdown("---")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            positions_deepseek = sum(1 for row in monitor_data if "✅" in row["DEEPSEEK"])
            st.metric("🤖 Posições DEEPSEEK", positions_deepseek)

        with col2:
            positions_agno = sum(1 for row in monitor_data if "✅" in row["AGNO"])
            st.metric("🧠 Posições AGNO", positions_agno)

        with col3:
            low_confidence = sum(1 for row in monitor_data if "Baixa" in row["DEEPSEEK"] or "Baixa" in row["AGNO"])
            st.metric("❌ Baixa Confiança", low_confidence)

        with col4:
            no_signal = sum(1 for row in monitor_data if "Sem sinal" in row["DEEPSEEK"] or "Sem sinal" in row["AGNO"])
            st.metric("❓ Sem Sinal", no_signal)

else:
    st.warning("⚠️ Nenhum dado de portfólio encontrado. Execute alguns trades primeiro!")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
        📊 Paper Trading Dashboard | Atualizado em tempo real
    </div>
    """,
    unsafe_allow_html=True
)
