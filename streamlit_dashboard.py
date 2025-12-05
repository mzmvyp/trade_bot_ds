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

# Função para obter preços atuais dos principais pares
@st.cache_data(ttl=10)
def get_market_prices():
    """Obtém preços atuais dos principais pares de criptomoedas"""
    symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT", "DOGEUSDT", "AVAXUSDT", "DOTUSDT", "MATICUSDT"]
    prices = {}
    
    try:
        for symbol in symbols:
            response = requests.get(f"https://fapi.binance.com/fapi/v1/ticker/price", params={'symbol': symbol}, timeout=5)
            if response.status_code == 200:
                data = response.json()
                prices[symbol] = float(data['price'])
    except Exception as e:
        st.warning(f"Erro ao obter preços: {e}")
    
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
    # KPIs principais
    col1, col2, col3, col4 = st.columns(4)
    
    initial_balance = portfolio_data.get("initial_balance", 10000.0)
    current_balance = portfolio_data.get("current_balance", 10000.0)
    total_return = ((current_balance - initial_balance) / initial_balance * 100) if initial_balance > 0 else 0
    
    with col1:
        st.metric(
            "💰 Saldo Inicial",
            f"${initial_balance:,.2f}"
        )
    
    with col2:
        st.metric(
            "💵 Saldo Atual",
            f"${current_balance:,.2f}",
            delta=f"{total_return:.2f}%"
        )
    
    with col3:
        open_positions = len(portfolio_data.get("positions", {}))
        st.metric(
            "📊 Posições Abertas",
            open_positions
        )
    
    with col4:
        total_trades = len(trade_history)
        st.metric(
            "📜 Total de Trades",
            total_trades
        )
    
    st.markdown("---")
    
    # Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📈 Overview", "💰 Posições Abertas", "📜 Histórico", "📉 Análise", "💹 Preços de Mercado"])
    
    with tab1:
        st.header("📈 Visão Geral do Portfólio")
        
        # Calcular estatísticas
        closed_trades = [t for t in trade_history if t.get("status") == "CLOSED"]
        open_trades = [t for t in trade_history if t.get("status") == "OPEN"]
        winning_trades = len([t for t in closed_trades if t.get("pnl", 0) > 0])
        losing_trades = len([t for t in closed_trades if t.get("pnl", 0) < 0])
        win_rate = (winning_trades / len(closed_trades) * 100) if closed_trades else 0
        total_pnl = sum([t.get("pnl", 0) for t in closed_trades])
        
        # Métricas de performance
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("🎯 Win Rate", f"{win_rate:.1f}%")
        
        with col2:
            st.metric("✅ Trades Ganhadores", winning_trades)
        
        with col3:
            st.metric("❌ Trades Perdedores", losing_trades)
        
        with col4:
            st.metric("💰 P&L Total", f"${total_pnl:,.2f}")
        
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
                
                closed_list.append({
                    "Data": trade.get("timestamp", "N/A")[:16],
                    "Símbolo": trade.get("symbol", "N/A"),
                    "Tipo": trade.get("signal", "N/A"),
                    "Entrada": f"${entry_price:,.2f}",
                    "Tamanho": f"{position_size:.6f}",
                    "Valor": f"${position_value:,.2f}",
                    "Stop Loss": f"${stop_loss:,.2f} ({sl_diff:+.1f}%)",
                    "Take Profit 1": f"${take_profit_1:,.2f} ({tp1_diff:+.1f}%)",
                    "Take Profit 2": f"${take_profit_2:,.2f} ({tp2_diff:+.1f}%)",
                    "Saída": f"${trade.get('close_price', 0):,.2f}" if trade.get('close_price') else "N/A",
                    "P&L": f"${trade.get('pnl', 0):,.2f}",
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
            
            # CORRIGIDO: Verificar se coluna 'pnl' existe e preencher valores nulos
            if 'pnl' not in trades_df.columns:
                trades_df['pnl'] = 0.0
            else:
                # Preencher valores nulos com 0 (trades abertos ainda não têm P&L)
                trades_df['pnl'] = trades_df['pnl'].fillna(0.0)
            
            # Calcular P&L acumulado apenas para trades fechados
            trades_df['cumulative_pnl'] = trades_df['pnl'].cumsum()
            
            # Criar gráfico
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=trades_df['timestamp'],
                y=trades_df['cumulative_pnl'],
                mode='lines+markers',
                name='P&L Acumulado',
                line=dict(color='green' if total_pnl >= 0 else 'red', width=2),
                marker=dict(size=8)
            ))
            
            fig.update_layout(
                title="Evolução do P&L Acumulado",
                xaxis_title="Data",
                yaxis_title="P&L Acumulado ($)",
                hovermode='x unified',
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.header("💰 Posições Abertas")
        
        positions = portfolio_data.get("positions", {})
        
        if positions:
            # Preparar dados para tabela
            positions_list = []
            for symbol, position in positions.items():
                entry_price = position.get('entry_price', 0)
                position_size = position.get('position_size', 0)
                stop_loss = position.get('stop_loss', 0)
                take_profit_1 = position.get('take_profit_1', 0)
                take_profit_2 = position.get('take_profit_2', 0)
                
                # Calcular diferenças percentuais
                sl_diff = ((stop_loss - entry_price) / entry_price * 100) if entry_price > 0 else 0
                tp1_diff = ((take_profit_1 - entry_price) / entry_price * 100) if entry_price > 0 else 0
                tp2_diff = ((take_profit_2 - entry_price) / entry_price * 100) if entry_price > 0 else 0
                
                positions_list.append({
                    "Símbolo": symbol.replace("_SHORT", ""),
                    "Tipo": position.get("signal", "N/A"),
                    "Preço Entrada": f"${entry_price:,.2f}",
                    "Tamanho": f"{position_size:.6f}",
                    "Valor Total": f"${position.get('position_value', 0):,.2f}",
                    "Stop Loss": f"${stop_loss:,.2f} ({sl_diff:+.2f}%)",
                    "Take Profit 1": f"${take_profit_1:,.2f} ({tp1_diff:+.2f}%)",
                    "Take Profit 2": f"${take_profit_2:,.2f} ({tp2_diff:+.2f}%)",
                    "Confiança": f"{position.get('confidence', 0)}/10"
                })
            
            df_positions = pd.DataFrame(positions_list)
            st.dataframe(df_positions, use_container_width=True, hide_index=True)
        else:
            st.info("ℹ️ Nenhuma posição aberta no momento.")
    
    with tab3:
        st.header("📜 Histórico de Trades")
        
        if trade_history:
            # Preparar dados para tabela
            history_list = []
            for trade in trade_history:
                history_list.append({
                    "ID": trade.get("trade_id", "N/A"),
                    "Símbolo": trade.get("symbol", "N/A"),
                    "Tipo": trade.get("signal", "N/A"),
                    "Entrada": f"${trade.get('entry_price', 0):,.2f}",
                    "Tamanho": f"{trade.get('position_size', 0):.4f}",
                    "Status": trade.get("status", "N/A"),
                    "P&L": f"${trade.get('pnl', 0):,.2f}" if trade.get('pnl') is not None else "N/A",
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
            
            # CORRIGIDO: Filtrar apenas trades com P&L válido
            pnl_values = [t.get("pnl", 0) for t in closed_trades if t.get("pnl") is not None]
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Distribuição de P&L
                fig_pnl = px.histogram(
                    x=pnl_values,
                    nbins=20,
                    title="Distribuição de P&L",
                    labels={"x": "P&L ($)", "y": "Frequência"}
                )
                st.plotly_chart(fig_pnl, use_container_width=True)
            
            with col2:
                # Box plot de P&L
                fig_box = go.Figure()
                fig_box.add_trace(go.Box(
                    y=pnl_values,
                    name="P&L Distribution",
                    boxmean='sd'
                ))
                fig_box.update_layout(
                    title="Distribuição de P&L (Box Plot)",
                    yaxis_title="P&L ($)"
                )
                st.plotly_chart(fig_box, use_container_width=True)
            
            # Estatísticas descritivas
            st.subheader("📈 Estatísticas Descritivas")
            
            stats = {
                "Média": f"${sum(pnl_values) / len(pnl_values):,.2f}",
                "Mediana": f"${sorted(pnl_values)[len(pnl_values)//2]:,.2f}",
                "Máximo": f"${max(pnl_values):,.2f}",
                "Mínimo": f"${min(pnl_values):,.2f}",
                "Total": f"${sum(pnl_values):,.2f}"
            }
            
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
