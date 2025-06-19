from functools import lru_cache
from pathlib import Path

@lru_cache(maxsize=None)
def project_root(io_utility_path: Path = Path(__file__).resolve(), configuration_at_root: str = "pyproject.toml") -> Path:
    for parent in [io_utility_path] + list(io_utility_path.parents):
        if (parent / configuration_at_root).exists():
            return parent.resolve()
    raise FileNotFoundError("Couldn't locate " + configuration_at_root + " in IO utility parents")