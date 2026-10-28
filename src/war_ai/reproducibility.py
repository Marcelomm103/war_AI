from __future__ import annotations

import hashlib
import io
import json
import os
import platform as platform_info
import re
import tempfile
from datetime import datetime, timezone
from importlib.metadata import distributions
from pathlib import Path
from typing import Annotated, BinaryIO, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    field_validator,
    model_validator,
)

NonEmptyString = Annotated[str, Field(min_length=1)]
NonNegativeInt = Annotated[int, Field(ge=0)]
PositiveInt = Annotated[int, Field(gt=0)]
Sha256Hash = Annotated[str, Field(pattern=r"^sha256:[0-9a-f]{64}$")]
CommitSha = Annotated[str, Field(pattern=r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")]
RunId = Annotated[str, Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")]
RUN_ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$"


class ReproducibilityModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class ArtifactReference(ReproducibilityModel):
    sha256: Sha256Hash
    size_bytes: NonNegativeInt
    media_type: NonEmptyString | None = None


class SourceRevision(ReproducibilityModel):
    repository: NonEmptyString
    commit_sha: CommitSha
    dirty: bool
    working_tree_patch: ArtifactReference | None = None

    @model_validator(mode="after")
    def require_patch_for_dirty_tree(self) -> SourceRevision:
        if self.dirty != (self.working_tree_patch is not None):
            raise ValueError(
                "dirty revisions require a stored working-tree patch; "
                "clean revisions must not reference one"
            )
        return self


class PlatformSnapshot(ReproducibilityModel):
    python_version: NonEmptyString
    python_implementation: NonEmptyString
    operating_system: NonEmptyString
    os_release: NonEmptyString
    architecture: NonEmptyString


class AcceleratorSnapshot(ReproducibilityModel):
    kind: Literal["cuda", "rocm", "metal", "other"]
    model: NonEmptyString
    driver_version: NonEmptyString | None = None
    runtime_version: NonEmptyString | None = None

    @model_validator(mode="after")
    def require_driver_for_cuda_or_rocm(self) -> AcceleratorSnapshot:
        if self.kind in ("cuda", "rocm") and self.driver_version is None:
            raise ValueError(f"{self.kind} accelerators require a driver version")
        return self


class HardwareSnapshot(ReproducibilityModel):
    cpu_model: NonEmptyString
    logical_cores: PositiveInt
    accelerators: tuple[AcceleratorSnapshot, ...] = ()


class RunManifest(ReproducibilityModel):
    schema_version: Literal["1.0"] = "1.0"
    run_id: RunId
    created_at: datetime
    source_revisions: tuple[SourceRevision, ...] = Field(min_length=1)
    dependencies: dict[str, str] = Field(min_length=1)
    dependency_lock_sha256: Sha256Hash | None = None
    platform: PlatformSnapshot
    hardware: HardwareSnapshot
    configuration: dict[str, JsonValue] = Field(min_length=1)
    seed_streams: dict[NonEmptyString, NonNegativeInt] = Field(min_length=1)
    dataset_artifacts: dict[NonEmptyString, ArtifactReference] = Field(min_length=1)
    weights_artifact: ArtifactReference
    normalizer_artifacts: dict[NonEmptyString, ArtifactReference] = Field(
        default_factory=dict
    )
    optimizer_configuration: dict[str, JsonValue] = Field(min_length=1)
    optimizer_state_artifact: ArtifactReference
    training_state_artifact: ArtifactReference
    benchmark_configuration: dict[str, JsonValue] = Field(min_length=1)
    inference_budget_ms: PositiveInt

    @field_validator("created_at")
    @classmethod
    def require_timezone_aware_timestamp(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("created_at must include a timezone")
        return value.astimezone(timezone.utc)

    @field_validator("source_revisions")
    @classmethod
    def sort_source_revisions(
        cls, revisions: tuple[SourceRevision, ...]
    ) -> tuple[SourceRevision, ...]:
        return tuple(sorted(revisions, key=lambda revision: revision.repository))

    @field_validator("dependencies")
    @classmethod
    def require_resolved_dependency_versions(
        cls, dependencies: dict[str, str]
    ) -> dict[str, str]:
        name_pattern = r"^[a-z0-9](?:[a-z0-9-]*[a-z0-9])?$"
        version_pattern = r"^[A-Za-z0-9][A-Za-z0-9.!+_-]*$"
        for name, version in dependencies.items():
            if (
                re.fullmatch(name_pattern, name) is None
                or _canonicalize_distribution_name(name) != name
            ):
                raise ValueError(
                    "dependency names must use canonical lowercase spelling"
                )
            if (
                version.lower() in {"latest", "unknown"}
                or re.fullmatch(version_pattern, version) is None
            ):
                raise ValueError(
                    "dependencies must record resolved versions, not specifiers"
                )
        return dependencies

    @model_validator(mode="after")
    def require_unique_repositories(self) -> RunManifest:
        repositories = [revision.repository for revision in self.source_revisions]
        if len(repositories) != len(set(repositories)):
            raise ValueError("source_revisions must have unique repository names")
        return self

    def referenced_artifacts(self) -> tuple[ArtifactReference, ...]:
        references = [
            *self.dataset_artifacts.values(),
            self.weights_artifact,
            *self.normalizer_artifacts.values(),
            self.optimizer_state_artifact,
            self.training_state_artifact,
        ]
        references.extend(
            revision.working_tree_patch
            for revision in self.source_revisions
            if revision.working_tree_patch is not None
        )
        return tuple(references)

    def canonical_json_bytes(self) -> bytes:
        return json.dumps(
            self.model_dump(mode="json"),
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")

    @property
    def manifest_sha256(self) -> str:
        digest = hashlib.sha256(self.canonical_json_bytes()).hexdigest()
        return f"sha256:{digest}"


class ArtifactIntegrityError(ValueError):
    pass


class ArtifactStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def put_bytes(
        self, content: bytes, media_type: str | None = None
    ) -> ArtifactReference:
        return self._put_stream(io.BytesIO(content), media_type)

    def put_file(
        self, source: str | Path, media_type: str | None = None
    ) -> ArtifactReference:
        with Path(source).open("rb") as stream:
            return self._put_stream(stream, media_type)

    def path_for(self, reference: ArtifactReference) -> Path:
        digest = reference.sha256.removeprefix("sha256:")
        return self.root / "sha256" / digest[:2] / digest[2:]

    def verify(self, reference: ArtifactReference) -> Path:
        path = self.path_for(reference)
        digest = hashlib.sha256()
        size_bytes = 0
        with path.open("rb") as artifact:
            while chunk := artifact.read(1024 * 1024):
                digest.update(chunk)
                size_bytes += len(chunk)

        actual_sha256 = f"sha256:{digest.hexdigest()}"
        if actual_sha256 != reference.sha256 or size_bytes != reference.size_bytes:
            raise ArtifactIntegrityError(
                f"Artifact {reference.sha256} failed its hash or size check"
            )
        return path

    def _put_stream(
        self, source: BinaryIO, media_type: str | None
    ) -> ArtifactReference:
        self.root.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256()
        size_bytes = 0
        temporary_path: Path | None = None

        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=self.root,
                prefix=".artifact-",
                delete=False,
            ) as temporary:
                temporary_path = Path(temporary.name)
                while chunk := source.read(1024 * 1024):
                    digest.update(chunk)
                    size_bytes += len(chunk)
                    temporary.write(chunk)
                temporary.flush()
                os.fsync(temporary.fileno())

            reference = ArtifactReference(
                sha256=f"sha256:{digest.hexdigest()}",
                size_bytes=size_bytes,
                media_type=media_type,
            )
            destination = self.path_for(reference)
            destination.parent.mkdir(parents=True, exist_ok=True)

            if destination.exists():
                self.verify(reference)
            else:
                if temporary_path is None:
                    raise RuntimeError("artifact temporary file was not created")
                os.replace(temporary_path, destination)
                temporary_path = None
            return reference
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)


class ReproducibilityStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.artifacts = ArtifactStore(self.root / "artifacts")
        self.runs_directory = self.root / "runs"

    def save_run(self, manifest: RunManifest) -> Path:
        for reference in manifest.referenced_artifacts():
            self.artifacts.verify(reference)

        destination = self._run_path(manifest.run_id)
        destination.parent.mkdir(parents=True, exist_ok=True)
        content = manifest.canonical_json_bytes()
        temporary_path: Path | None = None

        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=destination.parent,
                prefix=".run-",
                delete=False,
            ) as temporary:
                temporary_path = Path(temporary.name)
                temporary.write(content)
                temporary.flush()
                os.fsync(temporary.fileno())

            if temporary_path is None:
                raise RuntimeError("run temporary file was not created")
            try:
                os.link(temporary_path, destination)
            except FileExistsError:
                if destination.read_bytes() != content:
                    raise FileExistsError(
                        f"Run {manifest.run_id!r} already exists with different data"
                    )
            return destination
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

    def load_run(self, run_id: str) -> RunManifest:
        path = self._run_path(run_id)
        return RunManifest.model_validate_json(path.read_bytes())

    def verify_run(self, run_id: str) -> RunManifest:
        manifest = self.load_run(run_id)
        for reference in manifest.referenced_artifacts():
            self.artifacts.verify(reference)
        return manifest

    def _run_path(self, run_id: str) -> Path:
        if re.fullmatch(RUN_ID_PATTERN, run_id) is None:
            raise ValueError("run_id contains unsupported path characters")
        return self.runs_directory / f"{run_id}.json"


def capture_installed_dependencies() -> dict[str, str]:
    versions: dict[str, str] = {}
    for distribution in distributions():
        try:
            raw_name = distribution.metadata["Name"]
        except KeyError as error:
            raise ValueError("installed distribution is missing a name") from error
        version = distribution.version
        if not raw_name or raw_name.strip() != raw_name or not version:
            raise ValueError("installed distribution is missing name or version")

        name = _canonicalize_distribution_name(raw_name)
        if name in versions and versions[name] != version:
            raise ValueError(f"duplicate installed distribution name: {name}")
        versions[name] = version
    return dict(sorted(versions.items()))


def capture_platform_snapshot() -> PlatformSnapshot:
    return PlatformSnapshot(
        python_version=platform_info.python_version(),
        python_implementation=platform_info.python_implementation(),
        operating_system=platform_info.system(),
        os_release=platform_info.release(),
        architecture=platform_info.machine(),
    )


def _canonicalize_distribution_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()
