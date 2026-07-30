"""Script de validação do ambiente antes da execução do pipeline."""

import importlib
import sys
from pathlib import Path

from src.config import settings


def _check_required_packages(packages: list[str]) -> list[str]:
    """Verifica quais pacotes críticos não estão disponíveis no ambiente.

    Args:
        packages: Nomes dos pacotes a importar.

    Returns:
        Lista dos pacotes que falharam ao importar.
    """
    missing_packages = []
    for pkg in packages:
        try:
            importlib.import_module(pkg)
            print(f"  -> [OK] Pacote '{pkg}' carregado com sucesso.")
        except ImportError:
            missing_packages.append(pkg)
    return missing_packages


def _check_data_file(data_path: Path) -> bool:
    """Verifica se o arquivo de dados brutos está presente.

    Args:
        data_path: Caminho esperado do arquivo de dados.

    Returns:
        True se o arquivo existir, False caso contrário.
    """
    if data_path.exists():
        print(f"  -> [OK] Dados brutos encontrados em: {data_path}")
        return True
    print(f"\n[ERRO] Dados brutos não encontrados no caminho: '{data_path}'.")
    print(
        "[DICA] Certifique-se de executar 'dvc pull' para baixar os dados "
        "ou adicione o arquivo manualmente."
    )
    return False


def validate_environment() -> None:
    """Valida se as dependências essenciais e os dados brutos estão presentes
    no ambiente antes da execução do pipeline do projeto.
    """
    print("[INFO] Iniciando validação do ambiente do projeto...")

    required_packages = ["torch", "sklearn", "mlflow", "pandas", "dvc"]
    missing_packages = _check_required_packages(required_packages)
    if missing_packages:
        faltando = ", ".join(missing_packages)
        print(f"\n[ERRO] Faltam pacotes críticos no ambiente: {faltando}")
        print(
            "[DICA] Execute 'poetry install' no terminal para instalar as "
            "dependências corretas."
        )
        sys.exit(1)

    if not _check_data_file(Path(settings.raw_data_path)):
        sys.exit(1)

    print("\n[SUCESSO] Ambiente estruturado, validado e pronto para execução!\n")


if __name__ == "__main__":
    validate_environment()
