from tablassert.src.tablassert.utils.io import project_root
from textual.widgets import Markdown, Button, Static
from textual.app import App, ComposeResult
from textual.containers import Container
from textual.screen import Screen
from textwrap import dedent
from pathlib import Path
from typing import Self
from typing import Any

ROOT: Path = project_root()

class BuildInstructions(Container):

    def compose(self: Self) -> ComposeResult:

        md_text = dedent("""
        ## Select a graph config to build a knowledge graph

        some other instructions here        
        """.upper())

        yield Markdown(md_text, id="build-markdown")

class Sidebar(Container):

    def compose(self: Self) -> ComposeResult:
        yield Button("BUILD", classes="sidebar buttons", id="build")
        yield Button("QUIT", classes="sidebar buttons", id="quit")

class HomePage(Screen[None]):

    def compose(self: Self) -> ComposeResult:

        yield Sidebar(id="sidebar")
        yield BuildInstructions(id="build-instructions")
        yield Static("placeholder", id="build-file-explorer")

    def on_button_pressed(self: Self, event: Button.Pressed) -> None:
        match event.button.id:
            case "quit":
                self.app.exit()
            case _:
                pass

class MainMenu(Screen[None]):

    def compose(self: Self) -> ComposeResult:

        md_text = dedent("""
        ## Tablassert

        ### Version 4.2.0

        #### By Skye Goetz & Gwênlyn Glusman

        Tablassert is a versatile tool that creates knowledge assertions from tabular data,
        enhances knowledge with configurable options, and exports KGX-compliant TSVs.  
      

        **(Press any key to continue)**
        """.upper())

        yield Container(
            Markdown(md_text, id="intro-markdown"), id="intro-container"
        )

    def on_key(self: Self) -> None:
        self.app.push_screen(HomePage(classes="home-page"))

TUI: Path = ROOT / "tablassert/src/tablassert/tui"

class Tablassert(App[None]):

    CSS_PATH: Path = TUI / "styles.tcss"

    def on_mount(self: Self) -> None:
        self.theme = "tokyo-night"
        self.push_screen(MainMenu(classes="main-menu"))

def app() -> None: 
    Tablassert().run()