# war_AI
Repository containing an AI that can play war (risk) boardgame

## Package and optional dependencies

The distribution is named `war-ai`; its Python import name is `war_ai`, while
the repository name remains `war_AI`.

This package requires Python 3.11 or newer. CI and the aggregate development
container use Python 3.11.

Install only the features needed:

```bash
python -m pip install -e ".[test]"
python -m pip install -e ".[quality]"
python -m pip install -e ".[training]"
python -m pip install -e ".[service]"
```

The optional `training` extra pins PyTorch 2.14.1, PettingZoo, Gymnasium, and NumPy.
The `service` extra pins FastAPI and Uvicorn. Direct dependency versions are
recorded in `pyproject.toml`.

## GPU system requirements

CPU training does not require a GPU driver. GPU support is system-specific and
the host GPU driver is not installed or pinned by this Python package. The
default PyPI wheel may add platform-specific CUDA runtime packages. For a
CPU-only install or a different GPU backend, use the official
[PyTorch installation selector](https://pytorch.org/get-started/locally/) and
its corresponding package index. For NVIDIA hardware, select a build
compatible with the installed driver, check the driver with `nvidia-smi`, and
verify the PyTorch build with `torch.cuda.is_available()`.

The manual GPU smoke workflow is
`.github/workflows/gpu.yml`; it requires a self-hosted Linux runner labeled
`gpu` with a driver compatible with the selected PyTorch wheel. Ordinary CI
uses CPU-only imports and does not install the training extra.

## Experiment tracking

Experiments use a self-contained local store; no hosted tracking service is
required. `war_ai.reproducibility.RunManifest` records source commits and dirty
patches, resolved package versions and an optional lockfile hash, Python/OS,
CPU/accelerator and driver details, complete configuration, per-stream seeds,
dataset/weight/normalizer hashes, optimizer and training-state hashes, and
benchmark settings with the inference budget.

`ReproducibilityStore` writes canonical run JSON under `runs/<run_id>.json` and
content-addressed artifacts under `artifacts/sha256/<prefix>/<digest>`.
`ArtifactStore` streams files while hashing them; saving or verifying a run
checks every referenced artifact's SHA-256 and size. Reusing a run ID with
different metadata is rejected. `capture_installed_dependencies()` records
exact installed distribution versions and `capture_platform_snapshot()`
captures Python and OS details; hardware and accelerator driver details are
provided explicitly by the caller.

These manifests are local provenance, not gameplay contracts. Seeds and
training artifacts must never be added to a player's observation or exposed to
an agent at inference time.

Ruff, mypy, and mutmut are pinned in the `quality` extra. Mutation campaigns
will run once the package contains behavior and corresponding tests.
