# Tablassert(4.0.0)

By Skye Goetz & Gwênlyn Glusman (ISB)

## GraphConfig

```yaml
# Every Parameter Here is Mandatory
name: str
version: str
# Directories with TableConfigs
dirs:
 - list[str]

# Number of Parallel Processes to Run
workers: int
# Max Time Per SQL Query
progress_handler: float
# For Identiying Tables in Images
identification_accuracy: float
# For Identiying Tables in Text Based PDFs
extraction_accuracy: float
# For Maximum P-Value Allowed in KG
cutoff: float

# Paths To Required SQLite3 Databases
override: str
babel: str
kg2: str
supplement: str
pubmed: str
names: str
preds: str

# TSV File with Training Data For Edge Scoring
training_data: str
```

## TableConfig

```yaml
# Templates are Optional but Encouraged
# Sections Must Have Every Mandatory Parameter (Denoted with a "##") in This Template 
# Use null for none
template: &template
 location: ##
   # Where to download_from File You're Mining
   download_from: str ##
   # Extension Specific Parameters
   params: ##
     ext: str ## Required To Differentiate All Below
     # Text Based PDF
     pages: str
     flavor: str
     # Delimited File
     delimiter: str
     sheet: str ##
     # Row To Start At
     start: int
     # Row To End At
     end: int
     # List of Specific Rows (Alt to Start + End)
     rows:
       - list[str]
     # Excel SpreadSheet
     sheet: str ##
     # Row To Start At
     start: int
     # Row To End At
     end: int
     # List of Specific Rows (Alt to Start + End)
     rows:
       - list[str]
 provenance: ##
   # PubMed Curie (or doi) for the Paper You're download_froming From
   publication_id: str ##
   curator: str ##
   # Your Organization
   org: str ##
 attributes: ##
   sample_size: 
     mode: str ##
     # "column" or "predefined"
     value: str ##
     math:
         # list[str, dict[str, object]]
       - attr: str ##
         args: ##
           - list[float | None]
           # Use Null In Place of What You Want To Transform
           - null
   p_value: 
   fdr:
   strength:
   statistics:
   notes:
 triple: ##
   subj: ##
     mode: str ## value, cvalue, scvalue, curie, ccurie, sccurie
     value: str ## representing column
     # NCBITaxon or null for unspecified
     in_organism: str
     prioritize: 
       - list[str] # biolink:Classes
     avoid:
       - list[str] # biolink:Classes
     prefix: str
     suffix: str
     # substrings to remove
     cfill: str # polars fill_null strategy
     remove:
       # substrings to remove
       - list[str] 
     regex:
       # list[dict[str, str]]
       - pattern: str ##
         replacement: str ##
      dexplode: str # delimiter to split strings by before exploding
   obj: ##
   pred: str ## biolink:preds
 reindexing:
   # list[dict[str, object]]
   - when: str ## "before" or "after"
     mode: str ## ge, le, gt, lt, eq, ne
     column: str ##
     # mode = et/ne can use strings for value
     value: float | str ##

# You Must Have At Least One Section Per TableConfig
sections: ##
 - <<: *template ##
   changes_to_template: list[dict[str, object]]
    # Inherits ALL base_template properties automatically
    # Only adds/overrides what's unique
```
