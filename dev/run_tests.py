"""
Script para executar testes com cobertura e gerar relatório.

Uso:
    python run_tests.py                 # Executa todos os testes
    python run_tests.py --unit          # Apenas testes unitários
    python run_tests.py --integration   # Apenas testes de integração
    python run_tests.py --coverage      # Com relatório de cobertura detalhado
"""

import subprocess
import sys
from pathlib import Path


def run_tests(test_type: str = "all", coverage: bool = True) -> int:
    """
    Executar testes com pytest.

    Args:
        test_type: 'all', 'unit', 'integration', 'data', 'model', 'api'
        coverage: Se deve gerar relatório de cobertura

    Returns:
        Código de saída do pytest
    """
    project_root = Path(__file__).parent
    tests_dir = project_root / "tests"

    # Comando base
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        str(tests_dir),
        "-v",
        "--tb=short",
        "-ra",
    ]

    # Adicionar cobertura
    if coverage:
        cmd.extend([
            "--cov=src",
            "--cov=app",
            "--cov-report=term-missing:skip-covered",
            "--cov-report=html",
            "--cov-fail-under=90",
        ])

    # Filtrar por tipo de teste
    marker_map = {
        "unit": "unit",
        "integration": "integration",
        "data": "data_loading or data_cleaning",
        "model": "model_training",
        "api": "api",
    }

    if test_type != "all" and test_type in marker_map:
        cmd.extend(["-m", marker_map[test_type]])

    print(f"\n{'='*80}")
    print(f"Executando testes: {test_type.upper()}")
    print(f"{'='*80}\n")

    result = subprocess.run(cmd, cwd=project_root)

    if coverage:
        print(f"\n{'='*80}")
        print("Relatório de Cobertura")
        print(f"{'='*80}")
        print("HTML Report gerado em: htmlcov/index.html")
        print(f"{'='*80}\n")

    return result.returncode


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Executar suite de testes")
    parser.add_argument(
        "--type",
        choices=["all", "unit", "integration", "data", "model", "api"],
        default="all",
        help="Tipo de teste a executar",
    )
    parser.add_argument(
        "--no-coverage",
        action="store_true",
        help="Desabilitar relatório de cobertura",
    )
    parser.add_argument(
        "--markers",
        action="store_true",
        help="Listar todos os marcadores de teste disponíveis",
    )

    args = parser.parse_args()

    if args.markers:
        subprocess.run([sys.executable, "-m", "pytest", "--markers"])
        sys.exit(0)

    exit_code = run_tests(args.type, coverage=not args.no_coverage)
    sys.exit(exit_code)
