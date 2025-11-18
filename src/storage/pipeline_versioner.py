from __future__ import annotations

import logging
import os
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional

import boto3
from botocore.exceptions import BotoCoreError, ClientError, NoCredentialsError

logger = logging.getLogger(__name__)


@dataclass
class S3Settings:
    enabled: bool = False
    bucket: Optional[str] = None
    region: Optional[str] = None
    profile: Optional[str] = None
    prefix: str = ""
    endpoint_url: Optional[str] = None
    artifacts: Dict[str, str] = field(default_factory=dict)


@dataclass
class DVCSettings:
    auto_push: bool = False
    remote: Optional[str] = None
    stage_name: Optional[str] = None


class PipelineVersioner:
    """
    Coordinates secure uploads/downloads against S3 and DVC for a pipeline stage.

    The versioner is intentionally conservative: if a backend is disabled or misconfigured,
    the pipeline will continue without raising, but a warning is logged.
    """

    def __init__(self, project_root: Path, s3: S3Settings, dvc: DVCSettings) -> None:
        self.project_root = project_root
        self.s3 = s3
        self.dvc = dvc
        self._s3_client = None

    # ------------------------------------------------------------------ #
    # Factory helpers
    # ------------------------------------------------------------------ #
    @classmethod
    def from_params(
        cls,
        params: Mapping[str, object],
        *,
        stage_name: str,
        project_root: Path,
    ) -> Optional["PipelineVersioner"]:
        storage_cfg = params.get("storage") or {}
        if not isinstance(storage_cfg, Mapping):
            storage_cfg = {}

        s3_cfg = storage_cfg.get("s3") or {}
        dvc_cfg = storage_cfg.get("dvc") or {}

        if not isinstance(s3_cfg, Mapping):
            s3_cfg = {}
        if not isinstance(dvc_cfg, Mapping):
            dvc_cfg = {}

        s3_settings = S3Settings(
            enabled=bool(s3_cfg.get("enabled")),
            bucket=_coerce_str(s3_cfg.get("bucket")),
            region=_coerce_str(s3_cfg.get("region")),
            profile=_coerce_str(s3_cfg.get("profile") or os.environ.get("AWS_PROFILE")),
            prefix=_coerce_str(s3_cfg.get("prefix") or ""),
            endpoint_url=_coerce_str(s3_cfg.get("endpoint_url")),
            artifacts=_coerce_mapping(s3_cfg.get("artifacts")),
        )

        stage_map_cfg = dvc_cfg.get("stage_map") or {}
        mapped_stage = stage_name
        if isinstance(stage_map_cfg, Mapping):
            mapped_stage = _coerce_str(stage_map_cfg.get(stage_name) or stage_name) or stage_name

        dvc_settings = DVCSettings(
            auto_push=bool(dvc_cfg.get("auto_push")),
            remote=_coerce_str(dvc_cfg.get("remote")),
            stage_name=mapped_stage,
        )

        if not s3_settings.enabled and not dvc_settings.auto_push:
            return None
        return cls(project_root=project_root, s3=s3_settings, dvc=dvc_settings)

    # Maintained for backward compatibility with the name used elsewhere
    @classmethod
    def build(
        cls,
        params: Mapping[str, object],
        *,
        stage_name: str,
        project_root: Path,
    ) -> Optional["PipelineVersioner"]:
        return cls.from_params(params, stage_name=stage_name, project_root=project_root)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def ensure_local_artifact(self, label: str, path: Path) -> Path:
        """
        Downloads an artifact from S3 only if `path` does not exist locally.
        """
        if not self._s3_enabled():
            return path
        path = path.resolve()
        if path.exists():
            return path

        key = self._artifact_key(label, path)
        try:
            client = self._get_s3_client()
        except RuntimeError as exc:
            logger.warning("S3 client no disponible: %s", exc)
            return path

        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            client.download_file(self.s3.bucket, key, str(path))
            logger.info("Descargado '%s' desde s3://%s/%s", label, self.s3.bucket, key)
        except (ClientError, BotoCoreError, NoCredentialsError) as exc:
            logger.warning("No se pudo descargar '%s' de S3: %s", label, exc)
        return path

    def finalize_stage(self, artifacts: Mapping[str, Path]) -> None:
        """
        Once a pipeline stage finishes, call this to push artifacts to S3 and DVC.
        """
        if self._s3_enabled():
            for label, path in artifacts.items():
                self._upload_artifact(label, Path(path))
        self._push_dvc_stage()

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _s3_enabled(self) -> bool:
        return bool(self.s3.enabled and self.s3.bucket)

    def _artifact_key(self, label: str, path: Path) -> str:
        manual = self.s3.artifacts.get(label)
        if manual:
            return manual.strip("/")

        try:
            rel = path.relative_to(self.project_root)
        except ValueError:
            rel = Path(path.name)

        prefix = self.s3.prefix.strip("/") if self.s3.prefix else ""
        rel_key = rel.as_posix().lstrip("/")
        return "/".join(part for part in (prefix, rel_key) if part)

    def _get_s3_client(self):
        if not self._s3_enabled():
            raise RuntimeError("S3 deshabilitado o sin bucket configurado.")
        if self._s3_client is None:
            session_kwargs = {}
            if self.s3.profile:
                session_kwargs["profile_name"] = self.s3.profile
            session = boto3.session.Session(**session_kwargs)
            self._s3_client = session.client(
                "s3",
                region_name=self.s3.region,
                endpoint_url=self.s3.endpoint_url,
            )
        return self._s3_client

    def _upload_artifact(self, label: str, path: Path) -> None:
        if not path.exists():
            logger.debug("El artefacto '%s' no existe, se omite.", label)
            return

        key = self._artifact_key(label, path)
        try:
            client = self._get_s3_client()
        except RuntimeError as exc:
            logger.warning("No se puede subir '%s' a S3: %s", label, exc)
            return

        if path.is_dir():
            for file_path in path.rglob("*"):
                if not file_path.is_file():
                    continue
                rel_key = "/".join(
                    [
                        key.rstrip("/"),
                        file_path.relative_to(path).as_posix(),
                    ]
                )
                self._upload_file(client, file_path, rel_key)
        else:
            self._upload_file(client, path, key)

    def _upload_file(self, client, file_path: Path, key: str) -> None:
        try:
            client.upload_file(str(file_path), self.s3.bucket, key)
            logger.info("Subido '%s' → s3://%s/%s", file_path, self.s3.bucket, key)
        except (ClientError, BotoCoreError, NoCredentialsError) as exc:
            logger.warning("Fallo al subir '%s' a S3: %s", file_path, exc)

    def _push_dvc_stage(self) -> None:
        if not self.dvc.auto_push or not self.dvc.stage_name:
            return
        if shutil.which("dvc") is None:
            logger.warning("DVC no está instalado en el entorno, se omite push automático.")
            return

        cmd = ["dvc", "push", self.dvc.stage_name]
        if self.dvc.remote:
            cmd.extend(["-r", self.dvc.remote])
        try:
            subprocess.run(cmd, check=True, cwd=self.project_root, capture_output=True)
            logger.info("Stage '%s' versionado en DVC (remote=%s).", self.dvc.stage_name, self.dvc.remote or "default")
        except subprocess.CalledProcessError as exc:
            logger.warning(
                "Fallo al ejecutar '%s' (code=%s): %s",
                " ".join(cmd),
                exc.returncode,
                exc.stderr.decode("utf-8", errors="ignore") if exc.stderr else exc,
            )


def build_pipeline_versioner(
    params: Mapping[str, object],
    *,
    stage_name: str,
    project_root: Path,
) -> Optional[PipelineVersioner]:
    """
    Convenience wrapper to keep backwards compatibility with earlier imports.
    """
    return PipelineVersioner.from_params(params, stage_name=stage_name, project_root=project_root)


def _coerce_str(value: object) -> Optional[str]:
    if value is None:
        return None
    return str(value).strip()


def _coerce_mapping(value: object) -> Dict[str, str]:
    if not isinstance(value, Mapping):
        return {}
    return {str(k): str(v) for k, v in value.items() if v is not None}
