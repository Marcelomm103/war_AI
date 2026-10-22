from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path

import pytest
from pydantic import JsonValue, ValidationError

from war_ai.reproducibility import (
    AcceleratorSnapshot,
    ArtifactIntegrityError,
    ArtifactStore,
    HardwareSnapshot,
    ReproducibilityStore,
    RunManifest,
    SourceRevision,
    capture_installed_dependencies,
    capture_platform_snapshot,
)

CREATED_AT = datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc)


def _manifest(
    store: ReproducibilityStore,
    run_id: str = "training-run-1",
    configuration: Mapping[str, JsonValue] | None = None,
) -> RunManifest:
    artifact_store = store.artifacts
    return RunManifest(
        run_id=run_id,
        created_at=CREATED_AT,
        source_revisions=(
            SourceRevision(
                repository="Marcelomm103/war_AI",
                commit_sha="a" * 40,
                dirty=False,
            ),
        ),
        dependencies={"pydantic": "2.14.1", "war-ai": "0.1.0"},
        dependency_lock_sha256=None,
        platform=capture_platform_snapshot(),
        hardware=HardwareSnapshot(cpu_model="Test CPU", logical_cores=8),
        configuration=dict(configuration or {"algorithm": "ppo"}),
        seed_streams={"training": 42, "evaluation": 123},
        dataset_artifacts={
            "training": artifact_store.put_bytes(b"training-data"),
            "validation": artifact_store.put_bytes(b"validation-data"),
        },
        weights_artifact=artifact_store.put_bytes(b"model-weights"),
        normalizer_artifacts={
            "observation": artifact_store.put_bytes(b"normalizer-state")
        },
        optimizer_configuration={"name": "adam", "learning_rate": 0.001},
        optimizer_state_artifact=artifact_store.put_bytes(b"optimizer-state"),
        training_state_artifact=artifact_store.put_bytes(b"training-checkpoint"),
        benchmark_configuration={"games": 20, "players": 4},
        inference_budget_ms=500,
    )


def test_artifact_store_deduplicates_and_verifies_content(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "artifacts")

    first = store.put_bytes(b"same content", "application/octet-stream")
    second = store.put_bytes(b"same content", "application/octet-stream")

    assert first == second
    assert store.verify(first).read_bytes() == b"same content"


def test_run_manifest_is_canonical_and_stored_with_verified_artifacts(
    tmp_path: Path,
) -> None:
    store = ReproducibilityStore(tmp_path)
    first = _manifest(store, configuration={"z": "last", "a": "first"})
    second = _manifest(store, configuration={"a": "first", "z": "last"})

    assert first.manifest_sha256 == second.manifest_sha256
    run_path = store.save_run(first)

    assert run_path.read_bytes() == first.canonical_json_bytes()
    assert store.load_run(first.run_id) == first
    assert store.verify_run(first.run_id).manifest_sha256 == first.manifest_sha256


def test_run_store_does_not_overwrite_a_different_manifest(tmp_path: Path) -> None:
    store = ReproducibilityStore(tmp_path)
    store.save_run(_manifest(store, configuration={"algorithm": "ppo"}))

    changed = _manifest(store, configuration={"algorithm": "other"})
    with pytest.raises(FileExistsError, match="already exists with different data"):
        store.save_run(changed)


def test_run_store_requires_artifacts_to_exist_locally(tmp_path: Path) -> None:
    source = ReproducibilityStore(tmp_path / "source")
    manifest = _manifest(source)
    destination = ReproducibilityStore(tmp_path / "destination")

    with pytest.raises(FileNotFoundError):
        destination.save_run(manifest)


def test_dirty_source_patch_is_verified_with_the_run(tmp_path: Path) -> None:
    store = ReproducibilityStore(tmp_path)
    clean_manifest = _manifest(store)
    patch = store.artifacts.put_bytes(b"complete working-tree patch")
    changed_data = clean_manifest.model_dump()
    changed_data["source_revisions"] = (
        SourceRevision(
            repository="Marcelomm103/war_AI",
            commit_sha="a" * 40,
            dirty=True,
            working_tree_patch=patch,
        ),
    )
    dirty_manifest = RunManifest.model_validate(changed_data)

    assert patch in dirty_manifest.referenced_artifacts()
    store.save_run(dirty_manifest)
    assert store.verify_run(dirty_manifest.run_id) == dirty_manifest


def test_artifact_tampering_is_reported(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "artifacts")
    reference = store.put_bytes(b"original")
    store.path_for(reference).write_bytes(b"tampered")

    with pytest.raises(ArtifactIntegrityError, match="failed its hash or size check"):
        store.verify(reference)


def test_dirty_revisions_and_cuda_accelerators_require_provenance() -> None:
    with pytest.raises(ValidationError, match="require a stored working-tree patch"):
        SourceRevision(
            repository="Marcelomm103/war_AI",
            commit_sha="a" * 40,
            dirty=True,
        )

    with pytest.raises(ValidationError, match="require a driver version"):
        AcceleratorSnapshot(kind="cuda", model="NVIDIA GPU")


def test_run_manifest_rejects_unresolved_dependency_specifiers(tmp_path: Path) -> None:
    store = ReproducibilityStore(tmp_path)
    data = _manifest(store).model_dump()
    data["dependencies"] = {"pydantic": ">=2.0"}

    with pytest.raises(ValidationError, match="resolved versions"):
        RunManifest.model_validate(data)


def test_dependency_capture_returns_canonical_installed_versions() -> None:
    dependencies = capture_installed_dependencies()

    assert "pydantic" in dependencies
    assert list(dependencies) == sorted(dependencies)


def test_run_ids_cannot_escape_the_runs_directory(tmp_path: Path) -> None:
    store = ReproducibilityStore(tmp_path)

    with pytest.raises(ValueError, match="unsupported path characters"):
        store.load_run("../outside")
