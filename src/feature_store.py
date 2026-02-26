"""Camada simples de Feature Store local para o projeto.

Centraliza registro de features e versionamento de datasets derivados para
melhorar rastreabilidade de treino e consistência entre treino e inferência.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from typing import Any

import pandas as pd


def compute_dataset_version(df: pd.DataFrame) -> str:
    """Calcula versão determinística de um dataset.

    Args:
        df (pd.DataFrame): DataFrame a ser versionado.

    Returns:
        str: Hash curto representando o conteúdo do dataset.

    Raises:
        ValueError: Se o DataFrame estiver vazio.
    """
    if df.empty:
        raise ValueError("Não é possível versionar um DataFrame vazio")

    row_hashes = pd.util.hash_pandas_object(df, index=True)
    digest = hashlib.sha256(row_hashes.values.tobytes()).hexdigest()
    return digest[:12]


def persist_dataset_version(
    df: pd.DataFrame,
    dataset_name: str,
    output_dir: str = "data/versions",
) -> tuple[str, str]:
    """Persiste snapshot versionado de dataset em CSV.

    Args:
        df (pd.DataFrame): Dataset a persistir.
        dataset_name (str): Nome lógico do dataset (ex.: train_features).
        output_dir (str): Diretório de saída para snapshots.

    Returns:
        tuple[str, str]: Caminho do arquivo salvo e versão calculada.
    """
    os.makedirs(output_dir, exist_ok=True)
    version = compute_dataset_version(df)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    file_name = f"{dataset_name}_{timestamp}_{version}.csv"
    full_path = os.path.join(output_dir, file_name)
    df.to_csv(full_path, index=False)
    return full_path, version


def register_feature_view(
    feature_df: pd.DataFrame,
    target_name: str,
    output_path: str = "data/feature_store/feature_registry.json",
) -> str:
    """Registra metadados das features em um catálogo JSON local.

    Args:
        feature_df (pd.DataFrame): DataFrame com features do modelo.
        target_name (str): Nome da variável alvo.
        output_path (str): Caminho do registro JSON.

    Returns:
        str: Caminho do catálogo atualizado.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    registry: dict[str, Any]
    if os.path.exists(output_path):
        with open(output_path, encoding="utf-8") as file:
            registry = json.load(file)
    else:
        registry = {"views": []}

    feature_entries = [
        {
            "name": col,
            "dtype": str(feature_df[col].dtype),
            "nullable": bool(feature_df[col].isnull().any()),
        }
        for col in feature_df.columns
    ]

    registry["views"].append(
        {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "target": target_name,
            "n_rows": int(feature_df.shape[0]),
            "n_features": int(feature_df.shape[1]),
            "features": feature_entries,
        }
    )

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(registry, file, ensure_ascii=False, indent=2)

    return output_path
