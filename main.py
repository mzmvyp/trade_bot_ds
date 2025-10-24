"""
Sistema de Trading com AGNO Agent
"""
import asyncio
import argparse
import sys
from pathlib import Path
from trading_agent_agno import AgnoTradingAgent

async def main():
    parser = argparse.ArgumentParser(
        description='Sistema de Trading de Criptomoedas com AGNO Agent'
    )
    parser.add_argument(
        '--symbol', 
        default='BTCUSDT',
        help='Símbolo para trading (ex: BTCUSDT)'
    )
    parser.add_argument(
        '--mode',
        choices=['single', 'monitor', 'top5', 'top10'],
        default='single',
        help='Modo de operação'
    )
    parser.add_argument(
        '--interval',
        type=int,
        default=300,
        help='Intervalo para monitoramento em segundos'
    )
    parser.add_argument(
        '--paper',
        action='store_true',
        default=True,
        help='Usar paper trading (simulado)'
    )
    
    args = parser.parse_args()
    
    # Banner
    print("\n" + "="*60)
    print("🤖 SISTEMA DE TRADING COM AGNO AGENT")
    print("="*60)
    print(f"📊 Símbolo: {args.symbol}")
    print(f"🔄 Modo: {args.mode}")
    print(f"📝 Paper Trading: {'Sim' if args.paper else 'Não'}")
    print("="*60)
    
    # Criar agent
    agent = AgnoTradingAgent(paper_trading=args.paper)
    
    try:
        if args.mode == 'single':
            # Análise única
            signal = await agent.analyze(args.symbol)
            
            if signal.get('signal') in ['BUY', 'SELL'] and signal.get('confidence', 0) >= 7:
                print("\n⚠️  ALERTA: Sinal forte detectado!")
                print("Considere executar o trade com cautela.")
        
        elif args.mode == 'monitor':
            # Monitoramento contínuo
            await agent.monitor_continuous([args.symbol], args.interval)
        
        elif args.mode == 'top5':
            # Top 5 criptomoedas (BTC + 4 maiores)
            from config import settings
            symbols = settings.top_crypto_pairs[:5]
            
            print(f"\n🔝 Analisando Top 5 criptomoedas...")
            print("📊 BTC + 4 maiores por market cap")
            print("="*60)
            
            for i, symbol in enumerate(symbols, 1):
                print(f"\n[{i}/5] 🔍 Analisando {symbol}...")
                print("-" * 40)
                signal = await agent.analyze(symbol)
                
                # Mostrar resumo rápido
                if signal.get('signal') in ['BUY', 'SELL']:
                    print(f"⚠️  ALERTA: {signal.get('signal')} com confiança {signal.get('confidence', 0)}/10")
                else:
                    print(f"📊 {signal.get('signal')} - Confiança: {signal.get('confidence', 0)}/10")
                
                await asyncio.sleep(3)  # Pausa entre análises
        
        elif args.mode == 'top10':
            # Top 10 criptomoedas
            from config import settings
            symbols = settings.top_crypto_pairs[:10]
            
            print(f"\n🔝 Analisando Top 10 criptomoedas...")
            for i, symbol in enumerate(symbols, 1):
                print(f"\n[{i}/10] Analisando {symbol}...")
                await agent.analyze(symbol)
                await asyncio.sleep(5)  # Pausa entre análises
    
    except KeyboardInterrupt:
        print("\n\n⏹️  Sistema interrompido pelo usuário")
    except Exception as e:
        print(f"\n❌ Erro fatal: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())