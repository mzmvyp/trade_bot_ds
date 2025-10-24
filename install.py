"""
Instalação e configuração do sistema
"""
import os
import sys
import subprocess
import shutil
from pathlib import Path

def install_dependencies():
    """
    Instala as dependências do projeto
    """
    print("📦 Instalando dependências...")
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependências instaladas com sucesso!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Erro ao instalar dependências: {e}")
        return False

def create_env_file():
    """
    Cria arquivo .env se não existir
    """
    env_file = Path(".env")
    example_file = Path("config.env.example")
    
    if not env_file.exists():
        if example_file.exists():
            shutil.copy(example_file, env_file)
            print("📝 Arquivo .env criado")
            print("   Edite o arquivo .env com suas configurações")
        else:
            print("⚠️  Arquivo de exemplo não encontrado")
    else:
        print("✅ Arquivo .env já existe")

def main():
    """
    Função principal de instalação
    """
    print("🚀 Configurando Sistema de Trading de Criptomoedas")
    print("=" * 50)
    
    # Instalar dependências
    if not install_dependencies():
        print("❌ Falha na instalação das dependências")
        sys.exit(1)
    
    # Criar arquivo .env
    create_env_file()
    
    print("\n🎉 Sistema configurado com sucesso!")
    print("\n📋 Próximos passos:")
    print("1. Edite o arquivo .env com suas configurações")
    print("2. Execute: python main.py --mode single")
    print("3. Execute: python main.py --mode monitor")

if __name__ == "__main__":
    main()
