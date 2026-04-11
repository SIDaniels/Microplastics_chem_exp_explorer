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

---

## LLM-Based Classification (Papers)

Last updated: April 2026

### Overview

For bioRxiv, medRxiv, and PMC papers (not NIH grants), we use Claude Sonnet for classification. This provides more nuanced categorization than regex matching, especially for determining human health relevance and identifying specific toxicity mechanisms.

### Pre-Processing Pipeline

Before running the LLM classifier, papers go through several pre-processing steps:

#### Step 1: Data Collection
- **Source**: Papers extracted from bioRxiv, medRxiv, and PMC using Paperclip CLI
- **Search terms**: "microplastic*", "nanoplastic*", "plastic particle*"
- **Date range**: 2025-2026
- **Initial count**: ~300 papers

#### Step 2: Deduplication
- Remove duplicate papers (same title appearing in multiple sources)
- Prefer PMC (published) over preprint versions
- Result: Reduced to ~250 unique papers

#### Step 3: Initial Relevance Filtering
Manual/semi-automated filtering to remove clearly non-relevant papers:
- Plant/soil contamination studies (no animal toxicity)
- Plastic biodegradation by microbes/enzymes
- Pure methodology papers without biological samples
- Environmental sampling without health endpoints
- Result: ~200 papers for classification

#### Step 4: Conference Abstract Integration
- Added 162 conference abstracts from STOMP 2025 symposium
- Extracted via PDF parsing (full-digital-program.pdf)
- Manually corrected 77 entries with garbled PI names/organizations

### LLM Classification Process

#### Model & Cost
- **Model**: Claude Sonnet (claude-sonnet-4-20250514)
- **Cost**: ~$1.35 for 198 papers (~$0.007/paper)
- **Script**: `scripts/llm_classify_all.py`

#### Classification Categories

Each paper is classified across 4 dimensions:

**1. Human Health Relevance**
```json
{
  "human_health_relevant": true/false,
  "confidence": "high/medium/low"
}
```

**2. Study Types**
- `detection`: Identifying/measuring plastics in biological samples
- `exposure_assessment`: Estimating human exposure levels/routes

**3. Mechanisms (10 categories)**
| Key | Description |
|-----|-------------|
| oxidative_stress | ROS, mitochondrial dysfunction, antioxidant depletion |
| inflammation | Cytokines, NLRP3, NF-κB signaling |
| barrier_disruption | Gut/BBB/placental barrier integrity |
| microbiome | Dysbiosis, bacterial diversity changes |
| endocrine | Hormone disruption, estrogenic/androgenic effects |
| neurodegeneration | Cognitive impairment, Alzheimer's/Parkinson's links |
| immune_dysfunction | Immunotoxicity, altered immune cell function |
| dna_damage | Genotoxicity, chromosomal aberrations |
| receptor_signaling | AhR, TLRs, MAPK pathways |
| cell_death | Apoptosis, senescence, autophagy |

**4. Organ Systems (9 categories)**
brain_nervous, cardiovascular, gi_gut, respiratory, reproductive, liver, kidney, immune, endocrine

**5. Model Organisms (6 categories)**
invitro, rodent, zebrafish, human, environmental, other_animal

#### Prompt Structure

The LLM receives a structured prompt with:
1. Clear inclusion/exclusion criteria for health relevance
2. Detailed definitions for each category with examples
3. Request for JSON-only output (no markdown)

Example prompt excerpt:
```
STEP 1: Is this study relevant to HEALTH/TOXICOLOGY?
Answer TRUE if the study investigates BIOLOGICAL EFFECTS or TOXICITY MECHANISMS...

Answer FALSE only if:
- Plant or soil contamination studies (no animal toxicity)
- Plastic biodegradation by microbes/enzymes
- Pure methodology/analytical development without biological samples

STEP 2: Classify across all categories...
```

#### Output Format

LLM returns structured JSON:
```json
{
  "human_health_relevant": true,
  "confidence": "high",
  "study_types": {"detection": false, "exposure_assessment": true},
  "mechanisms": {"oxidative_stress": true, "inflammation": true, ...},
  "organs": {"gi_gut": true, "liver": true, ...},
  "models": {"rodent": true, "invitro": false, ...}
}
```

Results are flattened to columns with `LLM_` prefix:
- `LLM_HUMAN_HEALTH_RELEVANT`
- `LLM_CONFIDENCE`
- `LLM_MECH_INFLAMMATION`, `LLM_MECH_BARRIER_DISRUPTION`, etc.
- `LLM_ORGAN_BRAIN_NERVOUS`, `LLM_ORGAN_GI_GUT`, etc.
- `LLM_MODEL_RODENT`, `LLM_MODEL_HUMAN`, etc.

### Post-Classification Filtering

After LLM classification:

#### Step 5: Health Relevance Filter
- Papers with `LLM_HUMAN_HEALTH_RELEVANT = 0` were reviewed
- 104 papers filtered out as non-health-relevant (saved to `filtered_non_health_papers.csv`)
- Categories filtered: plant studies, environmental sampling only, pure detection methods

#### Step 6: Manual Validation
- Spot-checked ~20% of classifications
- Corrected obvious errors (e.g., zebrafish developmental toxicity miscategorized)
- Adjusted confidence thresholds based on abstract quality

### Final Data Integration

The final dataset combines:
1. **NIH Grants** (204 entries) - regex-classified using MODEL_*, ORGAN_*, MECH_* columns
2. **Conference Abstracts** (152 entries) - LLM-classified
3. **Papers** (152 entries from bioRxiv/medRxiv/PMC) - LLM-classified

**Total**: 356 unique entries in `microplastic_grants_with_papers.csv`

### Running the LLM Classifier

```bash
# Set API key
export ANTHROPIC_API_KEY="your-key-here"

# Run classifier
source venv/bin/activate
python scripts/llm_classify_all.py

# Output files:
# - data/llm_full_classifications.csv (classifications only)
# - data/papers_for_classification.csv (updated with LLM columns)
```

### Limitations

1. **Confidence varies**: Papers with brief abstracts get lower confidence scores
2. **Category overlap**: Some mechanisms co-occur (e.g., oxidative stress → inflammation)
3. **Model vs. regex**: LLM columns (LLM_MODEL_*) don't always match regex columns (MODEL_*) for the same paper
4. **Cost**: Re-running on large datasets requires API budget

### Merging LLM + Regex Classifications

The app uses:
- **Mechanisms tab**: `LLM_MECH_*` columns (from LLM classification)
- **Model Organisms tab**: `MODEL_*` columns (from regex classification)
- **Organ Systems tab**: Mix of `ORGAN_*` (regex) and `LLM_ORGAN_*` (LLM)

This hybrid approach leverages:
- Regex precision for well-defined terms (cell lines, species names)
- LLM nuance for complex concepts (mechanisms, health relevance)
