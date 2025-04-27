# 2025 Skye Lane Goetz

from tablassert.cfg import load_yaml
from rich.console import Console
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
                    if "sections" in table_cfg:
                        sections.append(table_cfg["sections"])
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
