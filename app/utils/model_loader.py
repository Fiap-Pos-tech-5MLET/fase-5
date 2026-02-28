"""
Utilitários para carregamento e gerenciamento de modelos.

Centraliza a lógica de carregamento do modelo em produção
e metadados associados.
"""

import logging
import os
from datetime import datetime
from typing import Any, Optional, Tuple

import joblib

logger = logging.getLogger("model_loader")

# Global state
_loaded_model: Optional[Any] = None
_model_loaded_at: Optional[str] = None


def get_model_paths() -> Tuple[str, str, str]:
    """
    Retorna paths dos modelos e metadados.

    Returns:
        Tuple com (models_dir, model_path, candidate_path).
    """
    base_models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
    champion_model_path = os.path.join(base_models_dir, "model.pkl")
    candidate_model_path = os.path.join(base_models_dir, "model_candidate.pkl")
    return base_models_dir, champion_model_path, candidate_model_path


def load_model() -> Tuple[Optional[Any], Optional[str]]:
    """
    Carrega o modelo champion do disco.

    Returns:
        Tuple com (model, loaded_at_timestamp).

    Raises:
        FileNotFoundError: Se model.pkl não existir.
    """
    global _loaded_model, _model_loaded_at  # pylint: disable=global-statement

    _, model_path, _ = get_model_paths()

    try:
        _loaded_model = joblib.load(model_path)
        _model_loaded_at = datetime.now().isoformat()
        logger.info("Model loaded successfully from %s", model_path)
        return _loaded_model, _model_loaded_at
    except FileNotFoundError:
        _loaded_model = None
        _model_loaded_at = None
        logger.warning("Model not found at %s", model_path)
        return None, None
    except (OSError, EOFError, ValueError, TypeError) as exc:
        _loaded_model = None
        _model_loaded_at = None
        logger.exception("Failed to load model at %s: %s", model_path, exc)
        return None, None


def reload_model() -> Tuple[Optional[Any], Optional[str]]:
    """
    Recarrega o modelo (útil após promoção).

    Returns:
        Tuple com (model, loaded_at_timestamp).
    """
    return load_model()


def get_current_model() -> Optional[Any]:
    """
    Retorna o modelo atualmente carregado.

    Returns:
        Modelo carregado ou None.
    """
    return _loaded_model


def get_model_info() -> dict:
    """
    Retorna informações sobre o modelo carregado.

    Returns:
        Dicionário com metadados do modelo.
    """
    _, champion_path, _ = get_model_paths()

    return {
        "model_loaded": _loaded_model is not None,
        "model_path": champion_path,
        "loaded_at": _model_loaded_at,
    }
