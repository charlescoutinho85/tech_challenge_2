import sys
import importlib
from pathlib import Path

def validate_environment() -> None:
    """
    Valida se as dependências essenciais e os dados brutos estão presentes
    no ambiente antes da execução do pipeline do projeto.
    """
    print("[INFO] Iniciando validação do ambiente do projeto...")
    
    # 1. Validação de Pacotes Críticos
    required_packages = ["torch", "sklearn", "mlflow", "pandas", "dvc"]
    missing_packages = []

    for pkg in required_packages:
        try:
            importlib.import_module(pkg)
            print(f"  -> [OK] Pacote '{pkg}' carregado com sucesso.")
        except ImportError:
            missing_packages.append(pkg)

    if missing_packages:
        print(f"\n[ERRO] Faltam pacotes críticos no ambiente: {', '.join(missing_packages)}")
        print("[DICA] Execute 'poetry install' no terminal para instalar as dependências corretas.")
        sys.exit(1)

    # 2. Validação do Arquivo de Dados
    data_path = Path("data/raw/ratings.csv")
    if data_path.exists():
        print(f"  -> [OK] Dados brutos encontrados em: {data_path}")
    else:
        print(f"\n[ERRO] Dados brutos não encontrados no caminho: '{data_path}'.")
        print("[DICA] Certifique-se de executar 'dvc pull' para baixar os dados ou adicione o arquivo manualmente.")
        sys.exit(1)

    print("\n[SUCESSO] Ambiente estruturado, validado e pronto para execução!\n")


if __name__ == "__main__":
    validate_environment()