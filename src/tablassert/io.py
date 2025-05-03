__author__ = "Skye Lane Goetz"


from importlib.metadata import files, PackageNotFoundError
from tablassert.cfg import load_yaml
from rich.console import Console
from functools import lru_cache
from rich.panel import Panel
from pathlib import Path


def load_sections(dirs: list[str]) -> list[dict[str, object]]:
    console = Console()
    sections = []
    for d in dirs:
        for path in Path(d).rglob("*"):
            if path.suffix.lower() == ".yaml":
                try:
                    table_cfg = load_yaml(path.resolve(), "TableConfig")
                    sections.extend(table_cfg["sections"])
                    console.print(f"[green]✓[/green] Loaded [dim]{path.name}[/dim]")
                except Exception as e:
                    console.print(
                        Panel.fit(
                            f"[red]Cannot load {path.name}:[/red]\n{str(e)}",
                            border_style="red",
                        )
                    )
    console.print(
        f"\n[bold green]Loaded [cyan]{len(sections)}[/cyan] sections[/bold green]"
    )
    return sections


@lru_cache(maxsize=None)
def get_root(package: str = "tablassert") -> str:
    try:
        package_contents = files(package)
        first_file = Path(str(package_contents[0].locate()))
        candidate = first_file.parent.parent
        if "site-packages" in str(candidate):
            return get_true_root(candidate).resolve().as_posix()
        return candidate.resolve().as_posix()
    except (PackageNotFoundError, StopIteration):
        return get_true_root(Path(__file__).parent).resolve().as_posix()


@lru_cache(maxsize=None)
def get_true_root(candidate: Path, identifier: str = "pyproject.toml") -> Path:
    for parent in [candidate] + list(candidate.parents):
        if (parent / identifier).exists():
            return parent
    raise FileNotFoundError(
        f"Could not find project root, no {identifier} in {candidate}"
    )
