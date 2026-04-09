# Classification Methodology

## Overview

This document describes the regex-based classification system used to categorize microplastics research projects in the NIH grants database. Classifications are pre-computed and stored in the CSV file to avoid runtime regex matching.

---

## Model Organism Classifications

Last updated: April 2026

### Summary Table

| Category | Column Name | Count | Description |
|----------|-------------|-------|-------------|
| In Vitro | MODEL_INVITRO | 31 | Cell-based experiments |
| Rodent | MODEL_RODENT | 44 | Mouse/rat animal models (active use) |
| Zebrafish | MODEL_ZEBRAFISH | 5 | Zebrafish/Danio rerio |
| Human | MODEL_HUMAN | 69 | Human subjects/epidemiology |
| Environmental | MODEL_ENVIRONMENTAL | 8 | Environmental sampling |
| Other Animal | MODEL_OTHER_ANIMAL | 3 | C. elegans, Drosophila, etc. |

### Pattern Details

#### MODEL_INVITRO (In Vitro / Cell-based)
```regex
in\s+vitro|cell\s+line|cell\s+culture|primary\s+cell|
organoid|3D\s+culture|spheroid|tissue\s+culture|
cultured\s+cell|HepG2|Caco-2|HEK293|A549|MCF|HeLa
```
**Rationale**: Matches explicit cell culture terminology and common cell line names.

#### MODEL_RODENT (Mouse/Rat) - Medium Approach
**Tightened from 51 → 44 projects**

Uses a "medium" approach to avoid classifying projects that merely cite rodent studies:

**Criteria (match ANY):**
1. Rodent term in TITLE (strong signal)
2. 2+ mentions of rodent terms in abstract (suggests actual use)
3. Active use language patterns:
```regex
(?:we|will|to)\s+(?:use|used|using)\s+(?:mouse|mice|rat|rats|rodent)
(?:mice|rats|mouse)\s+(?:were|was|are|will be)\s+(?:exposed|treated|fed|given|injected|dosed)
(?:pregnant|female|male|adult|neonatal)\s+(?:mice|rats|mouse)
c57bl|balb|sprague|wistar  # strain names
(?:apoe|ldlr).{0,5}(?:mice|mouse)  # knockout mice
```
**Rationale**: A single mention of "mice" in an abstract often means citing other studies, not using rodent models. Requiring active use language or multiple mentions ensures the project actually uses rodents.

#### MODEL_ZEBRAFISH
```regex
zebrafish|danio\s+rerio|\bzebrafish\s+model\b|\bzebrafish\s+larva
```
**Rationale**: Explicit zebrafish/Danio terms only. Tightened from original pattern that matched 11 projects to 5.

#### MODEL_HUMAN
```regex
human\s+(?:subject|participant|volunteer|sample|tissue|blood|urine|placenta|stool)|
(?:cohort|epidemiol|population.based|cross.sectional)\s+stud|
clinical\s+(?:trial|study|sample)|patient\s+(?:sample|cohort)|
NHANES|biomonitoring|human\s+exposure|
(?:serum|plasma|blood)\s+sample.*(?:from|of)\s+(?:human|patient|participant)
```
**Rationale**: Expanded to include human tissue/sample studies, biomonitoring, and specific databases like NHANES.

#### MODEL_ENVIRONMENTAL
```regex
environmental\s+(?:sample|sampling|monitoring|survey)|
(?:ocean|marine|river|lake|coastal|aquatic)\s+(?:sample|sampling|monitoring)|
field\s+(?:sample|sampling|survey|collection)|
water\s+(?:sample|sampling).*(?:collect|analyz)|
sediment\s+(?:sample|sampling)|ambient\s+air\s+sample
```
**Rationale**: **Significantly tightened** (42→8). Now requires explicit sampling/monitoring language rather than just environmental location words like "ocean" or "river" which appeared in many non-environmental studies.

#### MODEL_OTHER_ANIMAL
```regex
drosophila|c\.\s*elegans|caenorhabditis|xenopus|
\b(?:rabbit|pig|primate|chicken|frog)\s+model
```
**Rationale**: Matches alternative model organisms used in toxicology.

---

## Organ System Classifications

Last updated: April 2026

### Summary Table

| Organ System | Column Name | Count |
|--------------|-------------|-------|
| Brain/Nervous | ORGAN_BRAIN_NERVOUS | 38 |
| Cardiovascular | ORGAN_CARDIOVASCULAR | 23 |
| GI/Gut | ORGAN_GI_GUT | 24 |
| Respiratory | ORGAN_RESPIRATORY | 34 |
| Reproductive | ORGAN_REPRODUCTIVE | 30 |
| Liver | ORGAN_LIVER | 8 |
| Kidney | ORGAN_KIDNEY | 6 |
| Immune | ORGAN_IMMUNE | 22 |
| Endocrine | ORGAN_ENDOCRINE | 5 |

### Key Fixes Applied

#### ORGAN_REPRODUCTIVE
- **Problem**: "reproduce" and "reproducibility" triggered false positives
- **Solution**: Pattern uses `reproduct(?:ive|ion)` to exclude "reproduce/reproducibility"
- **Additional**: Excludes "embryonic day" (neuroscience term) and "pregnane" (chemical name)
- **Result**: 47 → 30 projects

#### ORGAN_IMMUNE
- **Problem**: "inflammation" alone triggered too many false positives
- **Solution**: Requires core immune terms (immune system, lymphocyte, T/B cells, macrophage activation)
- **Note**: Inflammasome/NLRP3 still counts as immune-focused
- **Result**: 40 → 22 projects

#### ORGAN_RESPIRATORY
- **Problem**: "inhalation" as exposure route triggered non-respiratory studies
- **Solution**: Only counts inhalation if in title (suggesting respiratory focus)
- **Requires**: Core respiratory terms (lung, pulmonary, airway, alveolar)
- **Result**: 37 → 34 projects

#### All Organ Systems
- **General Rule**: Require organ term in TITLE or multiple mentions in abstract
- **Rationale**: Single incidental mentions in abstracts often don't indicate organ-focused research

---

## How to Re-run Classifications

### Model Organisms
```bash
source venv/bin/activate
python scripts/update_model_organisms.py
```

### Organ Systems
```bash
source venv/bin/activate
python scripts/update_organ_systems.py
```

### After Updates
1. Bump the cache version in `app.py` (search for `_cache_version`)
2. Commit and push changes
3. Streamlit Cloud will auto-deploy

---

## Appendix: Classification Philosophy

### Precision vs. Recall Trade-off
We favor **precision** (avoiding false positives) over recall (capturing all possible matches). This is because:
1. Users exploring the database expect filters to return relevant results
2. False positives erode trust in the classification system
3. Missing a few edge cases is less harmful than including many irrelevant ones

### Pattern Design Principles
1. **Compound patterns**: Require context words (e.g., "liver toxicity" not just "liver")
2. **Word boundaries**: Use `\b` or character class negation to avoid partial matches
3. **Title priority**: Terms in titles are stronger signals than abstract mentions
4. **Explicit exclusions**: Remove known false positive triggers before matching
