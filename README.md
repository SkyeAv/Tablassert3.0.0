# Tablassert(3.0.0)

## By Skye Goetz (ISB) & Gwênlyn Glusman (ISB)

Tablassert is a multipurpose tool that crafts knowledge assertions from tabular data, augments knowledge with configuration, and exports knowledge as Knowledge Graph Exchange (KGX) consistent TSVs.

### Dependencies (Python)

```python
pip install scikit-learn
pip install pyyaml
pip install pandas
pip install numpy
pip install nltk
pip install xlrd
```

### Usage (Unix)

```bash
pip install -e <path_to_tablassert>
main <kg_config>
```

### KG Config

KG Configs are YAML configuration for any new Tablassert-generated knowledge graph. They contain basic information about the graph, what databases you want to connect, the table_configs you wish to include, which vectorizers/models to use, and some knowledge graph-wide hyperparameters.

```yaml
knowledge_graph_name :
max_workers : # Max Parallel Processes
p_value_cutoff : # Max P Value
progress_handler_timeout : # For all SQL Databases and Queries

config_directories:
 - # List of Directories That Contain Table Configs
 

override_sqlite : # Paths
supplement_sqlite : 
babel_sqlite : 
kg2_sqlite :
predicates_sqlite : 

confidence_model : # Pretrained Sklearn Linear Regression Model
tfidf_vectorizer : # Pretrained Sklearn TFIDF Vectorizer Model
```

### Table Configs 

Table Configs are YAML configuration for tabular data incorporated in a knowledge graph. They contain information about what to mine for knowledge, how to mine it for knowledge, adjustments to that knowledge, and hyperparameters dictating how Tablassert should behave. Typically, there are multiple Table Configs in each Tablassert-generated knowledge graph. 

```yaml
# USE "~" FOR "None"

column_style: # alphabetic (A-ZZ), numeric (1-100), else normal

method_notes : # Addtional details describing the methodology of the tabular data

data_location :
 path_to_file : # <Path to the Supplemental Table, Absolute or Relative to Where You Execute the Script>
 delimiter : # ONLY if CSV, TSV, TXT File
 sheet_to_use : # ONLY if XLS or XLSX File
 first_line : # First Line to Use / ONLY if XLS or XLSX File
 last_line : # Last Line to Use / ONLY if XLS or XLSX File

provenance : 
 publication : # PMC/PMID/doi Identifier
 publication_name : # Paper Title
 authors : 
 year_published : 
 journal :
 table_url : # Valid URL Telling Tablassert Where to Download the Desired Table
 yaml_curator :
 curator_organization : 

subject :
 curie : # <A CURIE for the Entire Table>
 # value : <A Value for the Entire Table>
 # curie_column_name : <A Name of a Column Containing CURIEs>
 # value_column_name : <A Name of a Column Containing Values>
 expected_classes : # List
   - # biolink:Class
 Taxons : # Only For biolink:Gene Filtering / List
   - # NCBITaxon:Taxon
 regex_replacements : # List
   - pattern :
     replacement : 

predicate : # biolink:Predicate

object :
 value_column_name : # Name of Column with Values
 prefix : # List
   - prefix : # Prefix
 suffix :
   - suffix : # Suffix
 explode : # Delimeter to Split Values by Before Exploding to Separate Rows
 fill_values : # How to Fill Empty Values in Column (ffil or bfill)

reindex : # List 
  - mode : # Mode (greater_than_or_equal_to, less_than_or_equal_to, not_equal_to)
    column : #nGoes By Final Column Names ONLY if Column is Included in the Final KG
    value :

attributes : 
 p : # P-Value
   value : # <Attribute Value for Entire Table>
   # column_name : <Name of Column to Containing Attribute>
   math : # List
     - operation : # <Python math Module Attribute>
       parameter : # Optional: <Second Parameter for Attribute>
       order_last : # Optional: <yes/no About Whether to Order parameter Last>
       # order_last is Required when parameter is Specified (Vice-Versa)
 n : # Sample Size
 relationship_strength : # Field Describing the Strength of an Edge
 relationship_type : # Method for Strength
 p_correction_method : # Field Describing If/How P-Value was Corrected
 knowledge_level :
 agent_type :

sections : # Can List Multiple
 - # <Copy of Section Formatted Like the Rest of the Config Nested in A Sections Section> 
   # For example...
   # attributes :
     # p :
       # value :
   # object :
     # curie :
     # prefix :
       # - prefix :
```