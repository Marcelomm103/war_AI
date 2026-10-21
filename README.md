# war_AI
Repository containing an AI that can play war (risk) boardgame

## Package and optional dependencies

The distribution is named `war-ai`; its Python import name is `war_ai`, while
the repository name remains `war_AI`.

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

Experiments will use a self-contained local tracker rather than a hosted
tracking service. Artifact files will be stored by SHA-256 content hash, with
run metadata stored alongside their references. The complete reproducibility
record is defined in roadmap step 1.6 before training runs are introduced.

Ruff, mypy, and mutmut are pinned in the `quality` extra. Mutation campaigns
will run once the package contains behavior and corresponding tests.
