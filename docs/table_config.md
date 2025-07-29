# TableConfigs

Configuration files used to specify how Tablassert should encode the knowledge in a unit of Tabular content. These are highly flexible and are the backbone of a Tablassert KG.

## Base Schema

```yaml
template:
  location:
    where_to_download_data_from:
    download_hyperparameters:
      # delimited files
      file_extension: csv | tsv | txt
      file_delimiter:
      start_at_line_number:
      end_at_line_number:
      use_row_numbers:
      # excel speadsheets
      file_extension: xls | xlsx
      which_excel_sheet_to_use:
      end_at_line_number:
      use_row_numbers:
  provenance:
    publication:
    config_curator_name:
    config_curator_organization:
  attributes:
    sample_size:
      encoding_method:
      value_for_encoding:
      math_module_transformations:
        - attribute:
          arguments:
            - ~
    p_value:
      encoding_method:
      value_for_encoding:
      math_module_transformations:
        - attribute:
          arguments:
            - ~
    multiple_testing_correction_method:
      encoding_method:
      value_for_encoding:
      math_module_transformations:
        - attribute:
          arguments:
            - ~
    relationship_strength:
      encoding_method:
      value_for_encoding:
      math_module_transformations:
        - attribute:
          arguments:
            - ~
    assertion_method:
      encoding_method:
      value_for_encoding:
      math_module_transformations:
        - attribute:
          arguments:
            - ~
    notes:
  triple:
    triple_subject:
      encoding_method:
      value_for_encoding:
      mapping_hyperparameters:
        in_this_organism:
        classes_to_prioritize:
          - biolink:category
        classes_to_avoid:
          - biolink:category
        prefix:
        suffix:
        how_to_fill_column:
        substrings_to_remove:
          - substring
        regular_expressions:
          - pattern:
            replacement:
        explode_by_delimiter:
    triple_object:
      encoding_method:
      value_for_encoding:
      mapping_hyperparameters:
        in_this_organism:
        classes_to_prioritize:
          - biolink:category
        classes_to_avoid:
          - biolink:category
        prefix:
        suffix:
        how_to_fill_column:
        substrings_to_remove:
          - substring
        regular_expressions:
          - pattern:
            replacement:
        explode_by_delimiter:
    triple_predicate:
  reindexing:
    - when:
      column:
      comparison:
      value_for_comparison:

sections:
  - ~
```

## Field Usage

| Field | Required with Parent | Options and Comments | Default |
|-|-|-|-|
| **posix_filepath** | N | User specified FilePath to data, used to bypass programmatic downloads | NA |
| **location** | Y | NA | NA |
| where_to_download_data_from | Y | A link specifying where the data was downloaded from, tries to download the data if posix_filepath isn't specified | NA |
| download_hyperparameters | Y | NA | NA |
| file_extension | Y | the file extension of the file you're working with, it dictates how Tablassert reads the data and what downstream parameters are required | NA |
| file_delimiter | Y$^*$ | a delimiter for a delimited file, only required for csv, tsv, and txt files | "," |
| which_excel_sheet_to_use | Y$^*$ | the excel spreadsheet sheet name of the sheet where the tabular data you're working with lies | "Sheet1" |
| start_at_line_number | N | if specified, start transforming the tabular data at this line number (1 based indexing) | None |
| end_at_line_number | N | if specified, stop transforming the tabular data at this line number (1 based indexing) | None |
| use_row_numbers | N | if specified, only transform the tabular data with these line numbers (1 based indexing) | None |
| **provenance** | Y | NA | NA |
| publication | Y | either a PMC:, PMID:, or doi: curie defining the publication that the tabular data originates from | NA |
| config_curator_name | Y | the name of the person curating a config, for edge provenance | NA |
| config_curator_organization | Y | the name of the organization from which the person curating a config works, for edge provenance | NA |
| **attributes** | N | NA | NA |
| sample_size | N | the sample size used to make an assertion | NA |
| p_value | N | the p_value for an assertion | NA |
| multiple_testing_correction_method | N | the multiple_testing_correction_method used for the p_value of an assertion | NA |
| relationship_strength | N | the relationship_strength of an assertion, can be something like a beta value or regression coefficient | NA |
| assertion_method | N | the method used for an assertion | NA |
| notes | N | any miscellaneous notes you have regarding an assertion that cannot fit into any other fields | None |
| encoding_method | N | either value (a single value to use for an attribute) or column_of_values (the column name where the values for an attribute are found) | "value" |
| value_for_encoding | N | the value corresponding to encoding method that you'd like to describe | "NA" |
| math_module_transformations | N | any math module transformations you want to apply to an attribute | NA |
| attribute | Y | a valid math module attribute that you want to use | NA |
| arguments | Y | a list of arguments you wish to pass to the math module attribute, use None (or ~) in place of the existing attribute (these are ordered) | NA |
| **triple** | Y | NA | NA |
| triple_subject | Y | the subject of a subject predicate object knowledge triple | NA |
| triple_object | Y | the predicate of a subject predicate object knowledge triple | NA |
| triple_predicate | Y | the object of a subject predicate object knowledge triple | NA |
| encoding_method | N | either value (a single value to use for an attribute) or column_of_values (the column name where the values for an attribute are found) | "value" |
| value_for_encoding | Y | the value corresponding to encoding method that you'd like to describe | NA |
| mapping_hyperparameters | N | NA | NA |
| in_this_organism | N | a specified NCBITaxon:ID organism from which the knowledge assertions specified are derived | None |
| classes_to_prioritize | N | a list of valid biolink:categories that you want to prefer when mapping raw strings to CURIES | None |
| classes_to_avoid | N | a list of valid biolink:categories that you want to exclude when mapping raw strings to CURIES | None |
| prefix | N | a prefix to add to all values | None |
| suffix | N | a suffix to add to all values | None |
| how_to_fill_column | a valid polars fill_null method to fill nulls | test | None |
| substrings_to_remove | a list of substrings to remove from all values | test | None |
| regular_expressions | N | a list of regular expression substitutions to apply to all values, they are computed with polars.Expr.str.replace | None |
| pattern | Y | a regular expression pattern to apply | NA |
| replacement | Y | a regular expression replacement to apply | NA |
| explode_by_delimiter | N | a delimiter to split all values by into a list before exploding each of these lists into thier own separate values for further transformation | None |
| **reindexing** | N | test | NA |
| when | Y | before (before column names are renamed, they still follow excel style conventions) (these transformations are applied before all other transformations are applied) or after (after column names are renames, they follow final KG naming conventions) (these transformations are applied after all other transformations are applied) | "after" |
| column | Y | the name of the column you wish to reindex values by | NA |
| comparison | Y | ge (greater than or equal to), gt (greater than), le (less than or equal to), lt (less than), ne (does not equal, can be a string), eq (does not equal, can be a string) | NA |
| value_for_comparison | Y | the value to give compare the values you're specifying in a column by | NA |
| **sections** | N | an optional list of changes to a specified template, each section inherits all unspecified characteristics from the template (there can be an infinite number of sections) | [None] |
