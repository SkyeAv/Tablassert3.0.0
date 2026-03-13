# Utilities (utils)

The `utils` module provides utility functions for deterministic UUID generation and other helper functions.

## namespace_uuid()

Generates deterministic UUIDs for KGX edge identifiers using UUID v3 (MD5-based namespacing).

### Function Signature

```python
def namespace_uuid(domain: Any, *values: list[Any]) -> str
```

### Parameters

**`domain: Any`**

Domain string used to create the namespace UUID.

Converted to string internally. Common domains:
- `"TABLASSERT"` - Default domain used for knowledge graph edge IDs
- `"edges"` - Optional custom domain for edge IDs
- `"nodes"` - For custom node IDs
- `"tablassert"` - For application-specific IDs

**`*values: list[Any]`**

Variable number of values to include in the UUID.

Values are:
1. Converted to strings
2. Filtered (None values removed)
3. Joined with tabs (`"\t"`)
4. Hashed within the domain namespace

### Return Value

Returns a string representation of UUID v3.

Format: `"xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx"`

### How It Works

**Step 1:** Create domain namespace

```python
domain_uuid = uuid3(UUID("00000000-0000-0000-0000-000000000000"), domain)
```

Each domain gets a unique namespace UUID.

**Step 2:** Join values

```python
joined = "\t".join([str(v) for v in values if v])
```

Example: `("HGNC:1234", "biolink:treats", "MONDO:0005148")` → `"HGNC:1234\tbiolink:treats\tMONDO:0005148"`

**Step 3:** Generate UUID within namespace

```python
return str(uuid3(domain_uuid, joined))
```

### Deterministic Behavior

**Same inputs always produce same UUID:**

```python
uuid1 = namespace_uuid("edges", "HGNC:1", "treats", "MONDO:1")
uuid2 = namespace_uuid("edges", "HGNC:1", "treats", "MONDO:1")

assert uuid1 == uuid2  # Always True
```

**Different inputs produce different UUIDs:**

```python
uuid1 = namespace_uuid("edges", "HGNC:1", "treats", "MONDO:1")
uuid2 = namespace_uuid("edges", "HGNC:1", "treats", "MONDO:2")

assert uuid1 != uuid2  # Always True
```

**Different domains produce different UUIDs:**

```python
uuid1 = namespace_uuid("edges", "A", "B")
uuid2 = namespace_uuid("nodes", "A", "B")

assert uuid1 != uuid2  # Different namespaces
```

### Use Case: KGX Edge IDs

Tablassert uses this function to generate edge identifiers:

```python
edge_id = namespace_uuid(
  "TABLASSERT",
  subject_curie,
  predicate,
  object_curie,
  *qualifiers,
  publication_id
)
```

**Example:**

```python
edge_id = namespace_uuid(
  "TABLASSERT",
  "HGNC:11998",  # TP53
  "biolink:associated_with",
  "MONDO:0005148",  # Type 2 diabetes
  "PMC11708054"  # Publication
)
# Returns: "uuid:a1b2c3d4-e5f6-7890-abcd-ef1234567890"
```

**Benefits:**
- **Reproducible:** Same edge always gets same ID across runs
- **Collision-resistant:** MD5 hash makes collisions extremely unlikely
- **Traceable:** ID incorporates edge components (subject, predicate, object, provenance)

### KGX Compliance

NCATS Translator KGX specification requires:
- Edge IDs must be globally unique
- Edge IDs should be deterministic when possible

`namespace_uuid()` satisfies both requirements:
- **Unique:** UUID v3 with domain namespacing
- **Deterministic:** Same inputs → same output

### Example Usage

**Simple case:**

```python
from tablassert.utils import namespace_uuid

# Generate edge ID
edge_id = namespace_uuid("edges", "subject", "predicate", "object")
print(edge_id)  # "uuid:12345678-1234-1234-1234-123456789abc"
```

**With qualifiers:**

```python
# Include anatomical context qualifier
edge_id = namespace_uuid(
  "edges",
  "HGNC:1",
  "biolink:expressed_in",
  "UBERON:0002048",  # Lung
  "anatomical_context",
  "UBERON:0002048"
)
```

**Filtering None values:**

```python
# Automatically removes None
edge_id = namespace_uuid("edges", "A", None, "B", None)
# Equivalent to: namespace_uuid("edges", "A", "B")
```

### Performance

- **O(1)** time complexity
- **No external dependencies** (uses Python stdlib `uuid`)
- **Minimal memory** (temporary string allocation only)

Suitable for millions of ID generations per second.

### Related Functions

**`basespace(domain)`** - Creates namespace UUID from domain

```python
def basespace(domain: str) -> UUID:
  namespace = UUID("00000000-0000-0000-0000-000000000000")
  return uuid3(namespace, domain)
```

Used internally by `namespace_uuid()`.

## Next Steps

- **[Entity Resolution](fullmap.md)** - Core entity mapping
- **[Tutorial](../tutorial.md)** - See UUIDs in action
