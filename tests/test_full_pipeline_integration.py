from __future__ import annotations

import json
from pathlib import Path

from src.full_pipeline import run_full_pipeline


def test_full_pipeline_runs_end_to_end(tmp_project) -> None:
    outputs = run_full_pipeline(
        config_path=tmp_project["config_path"],
        project_root=tmp_project["project_root"],
        generate_clean_report=False,
    )

    clean_path = Path(outputs["clean_path"])
    assert clean_path.exists()

    metrics = outputs["model_metrics"]
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert outputs["algorithm"] == "logreg"

    registry_latest = outputs["artifacts"].get("model_registry_latest")
    assert registry_latest is not None
    assert Path(registry_latest).exists()

    metadata = json.loads(Path(registry_latest).read_text())
    model_artifact = metadata.get("model_artifact")
    assert model_artifact is not None
    assert Path(model_artifact).exists()
    assert metadata.get("feature_names")
