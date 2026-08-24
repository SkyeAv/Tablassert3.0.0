# Real-world config fixtures

Production configurations vendored from sibling repositories. In every file each
`local:` value was rewritten to `local: payload.placeholder` (validation never reads
payloads). The three `dakp/` configs additionally have their `avoid:` entry
`AffinityMeasurement` rewritten to `ProteinLigandAssayResult`, the name biolink-model
4.4.4 renamed that class to; the original spelling no longer validates. Everything else
is byte-identical to the source.

## positive/

| File | Source repo | Original path |
|---|---|---|
| `mokg-v12/AYOGLU1.v12.yaml` | MultiomicsNext | `.tablassert/mokg-v12/AYOGLU1.v12.yaml` |
| `mokg-v12/CORREIA3.v12.yaml` | MultiomicsNext | `.tablassert/mokg-v12/CORREIA3.v12.yaml` |
| `mokg-v12/HOYER1.v12.yaml` | MultiomicsNext | `.tablassert/mokg-v12/HOYER1.v12.yaml` |
| `refconfigs/AYOGLU1.yaml` | MultiomicsNext | `.tablassert/refconfigs/AYOGLU1.yaml` |
| `dakp/approved_treats.yaml` | DAKP | `tables/approved_treats.yaml` |
| `dakp/contraindications.yaml` | DAKP | `tables/contraindications.yaml` |
| `dakp/faers_applied_to_treat.yaml` | DAKP | `tables/faers_applied_to_treat.yaml` |
| `dakp/graph.yaml` | DAKP | `tables/graph.yaml` |
| `tableconfigs/AAMER1.yaml` | TableConfigs | `TABLE/FLAKASSIST/AAMER1.yaml` |
| `tableconfigs/ALAM1.yaml` | TableConfigs | `TABLE/MBKG/ALAM1.yaml` |
| `tableconfigs/SILVARODRGUEZ3.yaml` | TableConfigs | `TABLE/FLAKASSIST/SILVARODRGUEZ3.yaml` |
| `tableconfigs/BRUNDAGE5.yaml` | TableConfigs | `TABLE/FLAKASSIST/BRUNDAGE5.yaml` |

## negative/ (deliberate legacy configs; 12.x must reject them)

| File | Source repo | Original path | Why rejected |
|---|---|---|---|
| `DRUGIBD.yaml` | TableConfigs | `TABLE/QI/DRUGIBD.yaml` | legacy keys `syntax: TC3`, `status: alpha`, `provenance.contributors` |
| `MOKG.yaml` | TableConfigs | `GRAPH/MOKG.yaml` | legacy graph layout: no `rig:`, top-level `description:` |
