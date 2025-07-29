# Tablassert CLI

## Version 2.0.0

### By Skye Lane Goetz

The Tablassert CLI is the command line interface for Tablassert and it's methods. Currently, it's the only way to use Tablassert, but I see a GUI implementation or something like it in the future.

## Commands and Usage

| command | arguments | notes |
|-|-|-|
| `--help` | NA | the help screen for Tablassert's typer CLI |
| `train` |  `trainingdata:str`-path to a ndjson (jsonlines) file with the schema defined in `--help` `saveto:str`-path to the file you want to save weights to `epochs:str`-the number of epochs to train model | produces PyTorch weights for the edge scoring model associated with a build |
| `build` | `graphconfig:str`-path to a valid graph config | NA |
