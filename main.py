"""
Sistema de Trading de Criptomoedas - Versão Corrigida
"""
import asyncio
import argparse
import sys
import os
from pathlib import Path
from datetime import datetime
from trading_agent import EnhancedTradingAgent

async def main():
    """
    Função principal do sistema
    """
    parser = argparse.ArgumentParser(description='Sistema de Trading de Criptomoedas')
    parser.add_argument('--symbol', default='BTCUSDT', help='Símbolo para trading (padrão: BTCUSDT)')
    parser.add_argument('--mode', choices=['single', 'monitor', 'top10'], default='single', 
                       help='Modo de execução: single (análise única), monitor (contínuo) ou top10 (top 10 criptos)')
    parser.add_argument('--interval', type=int, default=300, 
                       help='Intervalo em segundos para modo monitor (padrão: 300)')
    
    args = parser.parse_args()
    
    print("🚀 Sistema de Trading de Criptomoedas")
    print(f"📊 Símbolo: {args.symbol}")
    print(f"🔄 Modo: {args.mode}")
    
    if args.mode == 'monitor':
        print(f"⏰ Intervalo: {args.interval} segundos")
    elif args.mode == 'top10':
        print("🔝 Analisando top 10 criptomoedas por market cap")
    
    print("-" * 50)
    
    # Criar pastas necessárias
    Path("signals").mkdir(exist_ok=True)
    Path("logs").mkdir(exist_ok=True)
    
    # Criar instância do agent
    trading_agent = EnhancedTradingAgent()
    
    try:
        if args.mode == 'single':
            # Executar análise única
            signal = await trading_agent.run_single_analysis(args.symbol)
            
            # Salvar resultado
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"signals/signal_{args.symbol}_{timestamp}.json"
            
            import json
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(signal, f, indent=2, ensure_ascii=False, default=str)
            
            print(f"💾 Sinal salvo em: {filename}")
            
        elif args.mode == 'monitor':
            # Executar monitoramento contínuo
            print(f"📊 Iniciando monitoramento do mercado para {args.symbol}")
            print(f"⏰ Intervalo: {args.interval} segundos")
            
            while True:
                try:
                    signal = await trading_agent.run_single_analysis(args.symbol)
                    
                    # Salvar sinal
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    filename = f"signals/signal_{args.symbol}_{timestamp}.json"
                    
                    import json
                    with open(filename, 'w', encoding='utf-8') as f:
                        json.dump(signal, f, indent=2, ensure_ascii=False, default=str)
                    
                    print(f"💾 Sinal salvo em: {filename}")
                    
                except Exception as e:
                    print(f"Erro no monitoramento: {e}")
                
                await asyncio.sleep(args.interval)
                
        elif args.mode == 'top10':
            # Analisar top 10 criptomoedas
            from config import settings
            
            print("🔝 Analisando top 10 criptomoedas por market cap...")
            print("=" * 60)
            
            all_signals = []
            
            for i, symbol in enumerate(settings.top_crypto_pairs, 1):
                try:
                    print(f"\n📊 [{i}/10] Analisando {symbol}...")
                    signal = await trading_agent.run_single_analysis(symbol)
                    all_signals.append(signal)
                    
                    # Salvar sinal individual
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    filename = f"signals/signal_{symbol}_{timestamp}.json"
                    
                    import json
                    with open(filename, 'w', encoding='utf-8') as f:
                        json.dump(signal, f, indent=2, ensure_ascii=False, default=str)
                    
                    print(f"💾 Sinal salvo em: {filename}")
                    
                except Exception as e:
                    print(f"❌ Erro ao analisar {symbol}: {e}")
                    continue
            
            # Salvar resumo de todos os sinais
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            summary_filename = f"signals/top10_summary_{timestamp}.json"
            
            import json
            with open(summary_filename, 'w', encoding='utf-8') as f:
                json.dump(all_signals, f, indent=2, ensure_ascii=False, default=str)
            
            print(f"\n📋 Resumo salvo em: {summary_filename}")
            print(f"✅ Análise completa: {len(all_signals)}/10 criptomoedas analisadas")
            
    except KeyboardInterrupt:
        print("\n⏹️  Monitoramento interrompido pelo usuário")
    except Exception as e:
        print(f"❌ Erro no sistema: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
