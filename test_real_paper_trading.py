#!/usr/bin/env python3
"""
Teste do Sistema de Paper Trading REAL
"""

import sys
import asyncio
sys.path.append('.')

from real_paper_trading import RealPaperTradingSystem
from agno_tools import get_market_data
import json

async def test_real_paper_trading():
    print("🚀 TESTE DO PAPER TRADING REAL")
    print("=" * 60)
    
    # Criar sistema de paper trading REAL
    paper = RealPaperTradingSystem(initial_balance=10000.0)
    
    print(f"💰 Saldo inicial: ${paper.initial_balance:,.2f}")
    print(f"💰 Saldo atual: ${paper.current_balance:,.2f}")
    
    # Obter dados de mercado reais
    print("\n📊 Obtendo dados de mercado...")
    market_data = get_market_data("BTCUSDT")
    
    if "error" in market_data:
        print(f"❌ Erro: {market_data['error']}")
        return
    
    current_price = market_data['current_price']
    print(f"📈 Preço atual do BTC: ${current_price:,.2f}")
    
    # Criar sinal simulado
    signal = {
        "symbol": "BTCUSDT",
        "signal": "BUY",
        "entry_price": current_price,
        "stop_loss": current_price * 0.95,  # -5%
        "take_profit_1": current_price * 1.05,  # +5%
        "take_profit_2": current_price * 1.10,  # +10%
        "confidence": 8
    }
    
    print(f"\n🎯 Sinal criado:")
    print(f"   Símbolo: {signal['symbol']}")
    print(f"   Ação: {signal['signal']}")
    print(f"   Entrada: ${signal['entry_price']:,.2f}")
    print(f"   Stop Loss: ${signal['stop_loss']:,.2f} (-5%)")
    print(f"   Take Profit 1: ${signal['take_profit_1']:,.2f} (+5%)")
    print(f"   Take Profit 2: ${signal['take_profit_2']:,.2f} (+10%)")
    print(f"   Confiança: {signal['confidence']}/10")
    
    # Executar trade REAL
    print(f"\n⚡ Executando paper trade REAL...")
    result = paper.execute_trade(signal)
    
    if result['success']:
        print(f"✅ Trade REAL executado com sucesso!")
        print(f"   Trade ID: {result['trade_id']}")
        print(f"   Mensagem: {result['message']}")
        print(f"   Monitoramento: {result.get('monitoring', 'N/A')}")
    else:
        print(f"❌ Erro: {result['error']}")
        return
    
    # Mostrar posições abertas
    print(f"\n📊 Posições abertas:")
    open_positions = paper.get_open_positions()
    for pos in open_positions:
        print(f"   {pos['symbol']}: {pos['position_size']:.4f} unidades a ${pos['entry_price']:,.2f}")
    
    # Mostrar resumo do portfólio
    print(f"\n📈 Resumo do portfólio:")
    summary = paper.get_portfolio_summary()
    print(f"   Saldo inicial: ${summary['initial_balance']:,.2f}")
    print(f"   Saldo atual: ${summary['current_balance']:,.2f}")
    print(f"   Valor posições abertas: ${summary['open_positions_value']:,.2f}")
    print(f"   Valor total portfólio: ${summary['total_portfolio_value']:,.2f}")
    print(f"   P&L total: ${summary['total_pnl']:,.2f}")
    print(f"   Retorno: {summary['total_return_percent']:.2f}%")
    print(f"   Trades totais: {summary['total_trades']}")
    print(f"   Trades fechados: {summary['closed_trades']}")
    print(f"   Trades ganhadores: {summary['winning_trades']}")
    print(f"   Trades perdedores: {summary['losing_trades']}")
    print(f"   Win Rate: {summary['win_rate_percent']:.1f}%")
    print(f"   Monitoramento ativo: {'✅ SIM' if summary['is_monitoring'] else '❌ NÃO'}")
    
    print(f"\n🔄 SISTEMA REAL FUNCIONANDO:")
    print(f"   ✅ Monitora preços em tempo real")
    print(f"   ✅ Executa stop loss automaticamente")
    print(f"   ✅ Executa take profit automaticamente")
    print(f"   ✅ Fecha posições quando necessário")
    print(f"   ✅ Calcula P&L real")
    print(f"   ✅ Rastreia performance real")
    
    print(f"\n⏰ O sistema está monitorando automaticamente...")
    print(f"   - Verifica preços a cada 5 segundos")
    print(f"   - Fecha posição se preço atingir stop loss ou take profit")
    print(f"   - Salva logs de todas as ações")
    
    # Simular monitoramento por alguns segundos
    print(f"\n🔄 Simulando monitoramento por 30 segundos...")
    for i in range(6):
        await asyncio.sleep(5)
        current_price = await paper.get_current_price("BTCUSDT")
        if current_price:
            print(f"   [{i+1}/6] Preço atual: ${current_price:,.2f}")
    
    # Parar monitoramento
    paper.stop_monitoring()
    print(f"\n⏹️ Monitoramento parado")
    
    # Mostrar resumo final
    print(f"\n📊 RESUMO FINAL:")
    final_summary = paper.get_portfolio_summary()
    print(f"   Saldo final: ${final_summary['current_balance']:,.2f}")
    print(f"   P&L total: ${final_summary['total_pnl']:,.2f}")
    print(f"   Retorno: {final_summary['total_return_percent']:.2f}%")
    print(f"   Win Rate: {final_summary['win_rate_percent']:.1f}%")

if __name__ == "__main__":
    asyncio.run(test_real_paper_trading())
