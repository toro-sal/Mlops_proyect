from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Tuple

import pytest
from fastapi.testclient import TestClient

from src.full_pipeline import run_full_pipeline


@pytest.fixture
def api_client(tmp_project, monkeypatch) -> Tuple[TestClient, Path]:
    outputs = run_full_pipeline(
        config_path=tmp_project["config_path"],
        project_root=tmp_project["project_root"],
        generate_clean_report=False,
    )
    metadata_path = Path(outputs["artifacts"]["model_registry_latest"])
    monkeypatch.setenv("MODEL_REGISTRY_METADATA_PATH", str(metadata_path))

    app_module = importlib.import_module("src.api.app")
    app_module = importlib.reload(app_module)
    app_module.state.configure(metadata_path)
    app_module.state.ensure_loaded()
    client = TestClient(app_module.app)
    return client, metadata_path


def test_predict_endpoint_returns_predictions(api_client) -> None:
    client, metadata_path = api_client
    metadata = json.loads(metadata_path.read_text())
    feature_names = metadata.get("feature_names", [])
    assert feature_names, "El registry debe listar feature_names para servir el modelo."

    payload = {
        "records": [
            {name: 0 for name in feature_names},
            {name: 1 for name in feature_names},
        ],
        "return_probabilities": True,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["algorithm"] == metadata["algorithm"]
    assert len(body["predictions"]) == len(payload["records"])
    assert body["probabilities"] is None or len(body["probabilities"]) == len(payload["records"])
