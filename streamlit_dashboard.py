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

# Função para carregar dados do portfólio
@st.cache_data(ttl=5)
def load_portfolio_data():
    """Carrega dados do portfólio"""
    try:
        if os.path.exists("portfolio/state.json"):
            with open("portfolio/state.json", "r") as f:
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

# Carregar dados
portfolio_data = load_portfolio_data()
trade_history = load_trade_history()

# Sidebar - Controles
with st.sidebar:
    st.header("⚙️ Controles")
    
    # Auto-refresh
    auto_refresh = st.checkbox("🔄 Auto-refresh (5s)", value=True)
    if auto_refresh:
        st.rerun()
    
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
    
    initial_balance = portfolio_data.get("initial_balance", 10000)
    current_balance = portfolio_data.get("current_balance", 10000)
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
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Overview", "💰 Posições Abertas", "📜 Histórico", "📉 Análise"])
    
    with tab1:
        st.header("📈 Visão Geral do Portfólio")
        
        # Calcular estatísticas
        closed_trades = [t for t in trade_history if t.get("status") == "CLOSED"]
        winning_trades = len([t for t in closed_trades if t.get("pnl", 0) > 0])
        losing_trades = len([t for t in closed_trades if t.get("pnl", 0) < 0])
        win_rate = (winning_trades / len(closed_trades) * 100) if closed_trades else 0
        total_pnl = sum([t.get("pnl", 0) for t in closed_trades])
        
        # Métricas de performance
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("🎯 Win Rate", f"{win_rate:.1f}%")
        
        with col2:
            st.metric("✅ Trades Ganhadores", winning_trades)
        
        with col3:
            st.metric("❌ Trades Perdedores", losing_trades)
        
        with col4:
            st.metric("💰 P&L Total", f"${total_pnl:,.2f}")
        
        # Gráfico de performance
        if len(trade_history) > 0:
            st.subheader("📊 Performance ao Longo do Tempo")
            
            # Preparar dados para gráfico
            trades_df = pd.DataFrame(trade_history)
            trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'])
            trades_df = trades_df.sort_values('timestamp')
            
            # Calcular P&L acumulado
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
                positions_list.append({
                    "Símbolo": symbol.replace("_SHORT", ""),
                    "Tipo": position.get("signal", "N/A"),
                    "Entrada": f"${position.get('entry_price', 0):,.2f}",
                    "Tamanho": f"{position.get('position_size', 0):.4f}",
                    "Valor": f"${position.get('position_value', 0):,.2f}",
                    "Stop Loss": f"${position.get('stop_loss', 0):,.2f}",
                    "Take Profit 1": f"${position.get('take_profit_1', 0):,.2f}",
                    "Take Profit 2": f"${position.get('take_profit_2', 0):,.2f}",
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
                    "P&L": f"${trade.get('pnl', 0):,.2f}" if trade.get('pnl') else "N/A",
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
            
            pnl_values = [t.get("pnl", 0) for t in closed_trades]
            
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
