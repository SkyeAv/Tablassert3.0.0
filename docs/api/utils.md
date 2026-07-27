# Utilities (utils)

The `tablassert.utils` module provides the shared working-directory constants and a compact hashing helper. Deterministic UUID generation for KGX edge identifiers lives in the Rust extension (`tablassert.rs`) and is documented below as well.

## Constants

**`BASE: Path`** — `Path("./.tablassert")`

The single parent working directory. All runtime artifacts live beneath it.

**`STORE: Path`** — `BASE / "store"`

Intermediate parquet storage for compiled subgraphs (`.tablassert/store/`). Created on import.

The loguru sink (`.tablassert/log/tablassert.log`, `log.LOGASSERT`) and the cached BioBERT model (`.tablassert/biobert/`, `qc.MODEL`) are likewise derived from `BASE`.

## mkhash()

Generates a compact **xxh32** digest (hex string, 8 characters) for arbitrary input, computed by the Rust extension (`tablassert.rs.xxh32`).

```python
def mkhash(x: Any) -> str
```

The input is converted to a string and UTF-8 encoded before hashing. Used for compact identifiers in the CLI — e.g., the 8-character section hash shown in the progress display and the per-section parquet store filename (`{hash}.parquet`).

```python
from tablassert.utils import mkhash

mkhash({"source": "data.csv", "statement": {...}})  # e.g. "abc123de"
```

**Deterministic:** the same input always produces the same digest.

---

## namespace_uuid()  *(Rust extension: `tablassert.rs`)*

Generates deterministic UUIDs for KGX edge identifiers using UUID v3 (MD5-based namespacing). Provided by the Rust extension, not `tablassert.utils`.

### Function Signature

```python
def namespace_uuid(domain: str, values: list[str]) -> str
```

### Parameters

**`domain: str`**

Domain string used to create the namespace UUID. The default domain used internally for KGX edge IDs is `"TABLASSERT"`.

**`values: list[str]`**

The values to incorporate into the UUID. Empty/None entries are filtered out, the rest are joined with tabs (`"\t"`) and hashed within the domain namespace.

### Return Value

Returns a string representation of a UUID v3: `"xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx"`.

### How It Works

**Step 1:** Create the domain namespace from the nil UUID:

```python
domain_uuid = uuid3(UUID("00000000-0000-0000-0000-000000000000"), domain)
```

**Step 2:** Join the (filtered) values with tabs and hash within the namespace:

```python
return str(uuid3(domain_uuid, "\t".join(values)))
```

### Deterministic Behavior

Same inputs always produce the same UUID; different inputs (or different domains) produce different UUIDs.

### Use Case: KGX Edge IDs

Tablassert uses this function to generate reproducible edge identifiers from the edge's subject, predicate, object, qualifiers, and publication:

```python
from tablassert.rs import namespace_uuid

edge_id = namespace_uuid(
  "TABLASSERT",
  ["HGNC:11998", "biolink:associated_with", "MONDO:0005148", "PMC11708054"],
)
# e.g. "2cfea591-0f8f-33af-a7df-03da531d3359"
```

**Benefits:**
- **Reproducible:** the same edge always gets the same ID across runs
- **Collision-resistant:** MD5 hashing makes collisions extremely unlikely
- **Traceable:** the ID incorporates the edge components (subject, predicate, object, provenance)

### KGX Compliance

NCATS Translator KGX requires edge IDs to be globally unique and, where possible, deterministic. `namespace_uuid()` satisfies both: UUID v3 with domain namespacing yields unique, reproducible identifiers.

## Next Steps

- **[Entity Resolution](fullmap.md)** - Core entity mapping
- **[Tutorial](../tutorial.md)** - See UUIDs in action
