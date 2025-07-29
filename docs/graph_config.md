# GraphConfigs

## Version 2.0.0

### By Skye Lane Goetz

Configuration files used to specifiy a new knowledge graph, which table configs it uses, and general graph wide parameters. You must specifiy one GraphConfig to create a knowledge graph with Tablassert.

## Base Schema

```yaml
metadata:
  knowledge_graph_name:
  graph_version:
  graph_description:

location:
  table_config_containing_directories:
    - /dir/
  local_pmc_download:
  edge_scoring_model_weights:
  sqlite_databases:
    babel:
    kg2:
    pubmed:
    pmc:

hyperparameters:
  number_of_parallel_processes_to_run:
  maximum_p_value_in_graph:
  sql_progress_handler_timeout:
```

## Field Usage

| Field | Required with Parent | Options and Comments | Default |
|-|-|-|-|
| **metadata** | Y | NA | NA |
| knowledge_graph_name | Y | the name of the KG, used for storage, caching, and exports | NA |
| graph_version | N | the version of the KG, used for storage, caching, and exports | "0.0.0" |
| graph_description | N | an optional description of the KG (this literally does nothing, I just put it here, so I remember what each GraphConfig is all about) | None |
| **location** | Y | NA | NA |
| table_config_containing_directories | Y | a list of directories that contain table configs you wish to incorporate into the graph | NA |
| local_pmc_download | N | a path to local PMC tar file installs (I expect nobody to use this, this is more of an internal field we have to leverage an existing local download) | "NA" |
| edge_scoring_model_weights | Y | path to the PyTorch edge scoring model weights you created with the CLI's train method | NA |
| sqlite_databases | Y | path to the several SQLite databases required for the build method | NA |
| babel | Y | path to babel.db | NA |
| kg2 | Y | path to kg2.db | NA |
| pubmed | Y | path to pubmed.db | NA |
| pmc | Y | path to pmc.db | NA |
| **hyperparameters** | Y | NA | NA |
| number_of_parallel_processes_to_run | N | the number of multiprocessing.Pool processes to leverage while building your KG | 1 |
| maximum_p_value_in_graph | N | the p_value cutoff for the significant field in the graph, anything over this cutoff is labeled as not-significant | 1.0 |
| sql_progress_handler_timeout | N | the maximum amount of time to let any SQL query to run for, these errors are caught and logged without stopping the build process | 1.0 |
