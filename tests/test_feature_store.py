"""Testes unitários para a camada de feature store local."""

import json
from pathlib import Path

import pandas as pd

from src.feature_store import (
    compute_dataset_version,
    persist_dataset_version,
    register_feature_view,
)


def test_compute_dataset_version_is_deterministic() -> None:
    """Deve gerar a mesma versão para o mesmo conteúdo."""
    df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})

    version_1 = compute_dataset_version(df)
    version_2 = compute_dataset_version(df)

    assert version_1 == version_2
    assert len(version_1) == 12


def test_persist_dataset_version_creates_csv(tmp_path: Path) -> None:
    """Deve salvar snapshot versionado do dataset em CSV."""
    df = pd.DataFrame({"a": [1, 2, 3]})

    saved_path, version = persist_dataset_version(
        df=df,
        dataset_name="train_features",
        output_dir=str(tmp_path),
    )

    assert Path(saved_path).exists()
    assert version in Path(saved_path).name


def test_register_feature_view_creates_registry_file(tmp_path: Path) -> None:
    """Deve criar catálogo JSON com metadados das features."""
    df = pd.DataFrame({"IDADE": [12, 13], "INDE_23": [6.5, 7.2]})
    output_path = tmp_path / "feature_registry.json"

    registry_path = register_feature_view(
        feature_df=df,
        target_name="TARGET",
        output_path=str(output_path),
    )

    with open(registry_path, encoding="utf-8") as file:
        loaded = json.load(file)

    assert output_path.exists()
    assert "views" in loaded
    assert len(loaded["views"]) >= 1
