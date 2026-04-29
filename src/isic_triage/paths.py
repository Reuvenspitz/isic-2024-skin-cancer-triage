"""Path handling for the public ISIC 2024 notebook.

The notebook is expected to live in ``repo_root/notebooks``. Data and
artifacts are intentionally kept outside version control, with optional
environment-variable overrides for local machines or Colab-style runs.
"""

from __future__ import annotations

import os
from pathlib import Path


def find_project_root(start: str | Path | None = None) -> Path:
    """Return the repository root from a notebook or repo working directory."""
    current = Path(start or Path.cwd()).expanduser().resolve()

    for candidate in (current, *current.parents):
        if (candidate / '.git').exists():
            return candidate
        if (candidate / 'notebooks').exists() and (candidate / 'README.md').exists():
            return candidate

    return current.parent if current.name == 'notebooks' else current


def _env_path(name: str, default: Path) -> Path:
    return Path(os.environ.get(name, default)).expanduser().resolve()


def _first_existing(candidates: list[Path], fallback: Path) -> Path:
    for path in candidates:
        if path.exists():
            return path.resolve()
    return fallback.resolve()


def build_project_paths(project_root: str | Path | None = None) -> dict[str, Path]:
    """Build all paths used by the notebook.

    Expected local data layout is flexible. The function supports the Kaggle
    layout (``data/train-image/image`` and ``data/train-image.hdf5``) and the
    working layout used while developing the project (``data/image/...``).
    """
    root = find_project_root(project_root)
    data_dir = _env_path('ISIC_DATA_DIR', root / 'data')
    artifacts_dir = _env_path('ISIC_ARTIFACT_DIR', root / 'artifacts')
    split_dir = artifacts_dir / 'dataset_splits'

    images_dir = _first_existing(
        [
            data_dir / 'train-image' / 'image',
            data_dir / 'image' / 'train-image',
            data_dir / 'image' / 'image',
            data_dir / 'images',
        ],
        data_dir / 'train-image' / 'image',
    )

    image_hdf5 = _first_existing(
        [
            data_dir / 'train-image.hdf5',
            data_dir / 'image' / 'train-image.hdf5',
            data_dir / 'train-image' / 'train-image.hdf5',
        ],
        data_dir / 'train-image.hdf5',
    )

    return {
        'project_root': root,
        'data_dir': data_dir,
        'image_dir': images_dir.parent if images_dir.name == 'image' else images_dir.parent,
        'images_dir': images_dir,
        'image_hdf5': image_hdf5,
        'raw_meta': data_dir / 'train-metadata.csv',
        'artifacts_dir': artifacts_dir,
        'output_dir': split_dir,
        'train_csv': split_dir / 'train_set.csv',
        'test_csv': split_dir / 'test_set.csv',
    }


def ensure_output_dirs(paths: dict[str, Path]) -> None:
    """Create local output directories used by the notebook."""
    paths['artifacts_dir'].mkdir(parents=True, exist_ok=True)
    paths['output_dir'].mkdir(parents=True, exist_ok=True)
