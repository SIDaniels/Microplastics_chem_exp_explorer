# Paperclip Paper Extraction Process

## Overview

This document describes how microplastics/nanoplastics papers (2025-2026) were extracted from Paperclip and classified.

## Search Terms Used

The SQL query searched for papers containing these terms in **title OR abstract**:

```sql
WHERE (title ILIKE '%microplastic%' OR title ILIKE '%nanoplastic%'
       OR abstract_text ILIKE '%microplastic%' OR abstract_text ILIKE '%nanoplastic%')
  AND pub_date >= '2025-01-01'
```

### Terms Covered
- **microplastic** (includes: microplastics, microplastic-induced, etc.)
- **nanoplastic** (includes: nanoplastics, nanoplastic-induced, etc.)

### Terms NOT Currently Searched
Consider adding these for more comprehensive coverage:
- `plastic particle` / `plastic particles`
- `polymer particle` / `polymer particles`
- `polystyrene` (PS) - common microplastic type
- `polyethylene` (PE) - common microplastic type
- `PET` / `polyethylene terephthalate`
- `HDPE` / `LDPE`
- `polypropylene` (PP)
- `PVC` / `polyvinyl chloride`
- `micro-sized plastic`
- `nano-sized plastic`
- `plastic debris`
- `plastic pollution` (broader, may get irrelevant results)

## Extraction Process

### Step 1: Get Paper IDs via SQL

```bash
paperclip sql "SELECT id FROM documents
    WHERE (title ILIKE '%microplastic%' OR title ILIKE '%nanoplastic%'
           OR abstract_text ILIKE '%microplastic%' OR abstract_text ILIKE '%nanoplastic%')
      AND pub_date >= '2025-01-01'
    LIMIT 200"
```

Output saved to `/tmp/sql_ids.txt` - returned **198 paper IDs**

### Step 2: Fetch Full Metadata

For each paper ID, fetch complete metadata:

```bash
paperclip cat /papers/<doc_id>/meta.json
```

This returns JSON with:
- `document_id` or `id`
- `title`
- `authors`
- `doi`
- `pub_date`
- `abstract` or `abstract_text`
- `journal`
- `pmc_id` (for PMC papers)

### Step 3: Apply Regex Classifications

Using patterns from existing scripts in `/scripts/`:

#### Model Organisms (from `update_model_organisms.py`)
| Category | Description |
|----------|-------------|
| MODEL_INVITRO | Cell lines, organoids, 3D culture |
| MODEL_RODENT | Mouse, rat, murine models |
| MODEL_ZEBRAFISH | Danio rerio |
| MODEL_HUMAN | Human subjects, clinical samples |
| MODEL_ENVIRONMENTAL | Field sampling, water/sediment |
| MODEL_OTHER_ANIMAL | Drosophila, C. elegans, etc. |

#### Organ Systems (from `update_organ_systems.py`)
| Category | Title Pattern | Abstract Pattern |
|----------|--------------|------------------|
| ORGAN_BRAIN_NERVOUS | brain, neuro, CNS | neurotoxic, blood-brain barrier |
| ORGAN_CARDIOVASCULAR | cardiac, heart, vascular | cardiotoxic, atherosclerosis |
| ORGAN_GI_GUT | gut, intestin, digest | microbiome, IBD, IBS |
| ORGAN_RESPIRATORY | lung, pulmonary, airway | COPD, asthma, fibrosis |
| ORGAN_REPRODUCTIVE | ovari, testes, placent | fertility, sperm, oocyte |
| ORGAN_LIVER | liver, hepat | hepatotoxic, NAFLD |
| ORGAN_KIDNEY | kidney, renal, nephro | nephrotoxic, CKD |
| ORGAN_IMMUNE | immune, lymph, macrophage | immunotoxic, T-cell, B-cell |
| ORGAN_ENDOCRINE | endocrine, thyroid, hormone | endocrine disrupt |

#### Mechanisms (from `llm_classify_mechanisms.py` regex fallback)
| Category | Key Terms |
|----------|-----------|
| MECH_OXIDATIVE | oxidative stress, ROS, mitochondria |
| MECH_INFLAMMATION | IL-1, IL-6, TNF, NF-kB, NLRP3 |
| MECH_BARRIER | tight junction, ZO-1, occludin |
| MECH_MICROBIOME | microbiota, dysbiosis, 16S rRNA |
| MECH_ENDOCRINE | estrogen, BPA, phthalate |
| MECH_NEURODEGENERATION | Alzheimer, Parkinson, amyloid |
| MECH_IMMUNE | T-cell, B-cell, macrophage activation |
| MECH_DNA_DAMAGE | genotoxic, comet assay, 8-OHdG |
| MECH_RECEPTOR | AhR, TLR, PPAR |
| MECH_CELL_DEATH | apoptosis, caspase, ferroptosis |

## Output Files

- `data/microplastics_papers_2025_2026.xlsx` - Excel with all columns
- `data/microplastics_papers_2025_2026.csv` - CSV format

## Classification Results Summary (198 papers)

### Model Organisms
- MODEL_INVITRO: 27 papers
- MODEL_RODENT: 20 papers
- MODEL_ZEBRAFISH: 13 papers
- MODEL_HUMAN: 14 papers
- MODEL_ENVIRONMENTAL: 8 papers
- MODEL_OTHER_ANIMAL: 5 papers

### Organ Systems
- ORGAN_GI_GUT: 22 papers
- ORGAN_LIVER: 14 papers
- ORGAN_CARDIOVASCULAR: 12 papers
- ORGAN_REPRODUCTIVE: 11 papers
- ORGAN_BRAIN_NERVOUS: 10 papers
- ORGAN_IMMUNE: 9 papers
- ORGAN_RESPIRATORY: 6 papers
- ORGAN_KIDNEY: 0 papers (likely undercounted - see validation)
- ORGAN_ENDOCRINE: 0 papers (likely undercounted - see validation)

### Mechanisms
- MECH_OXIDATIVE: 106 papers
- MECH_INFLAMMATION: 19 papers
- MECH_CELL_DEATH: 18 papers
- MECH_ENDOCRINE: 17 papers
- MECH_MICROBIOME: 14 papers
- MECH_BARRIER: 4 papers
- MECH_IMMUNE: 6 papers
- MECH_DNA_DAMAGE: 6 papers
- MECH_NEURODEGENERATION: 5 papers
- MECH_RECEPTOR: 5 papers

## Validation Issues Found

Regex classification has significant **false negatives**:

| Category | Regex Count | Estimated Actual | Issue |
|----------|-------------|-----------------|-------|
| ORGAN_KIDNEY | 0 | 5+ | Regex too strict |
| ORGAN_ENDOCRINE | 0 | 5+ | Regex too strict |
| MECH_BARRIER | 4 | 8+ | Missing simple "barrier" mentions |
| MECH_IMMUNE | 6 | 11+ | Missing "immune" without qualifiers |
| MODEL_HUMAN | 14 | 19+ | Missing "human health" general mentions |

## Recommended Improvements

1. **Expand search terms**: Add specific polymer types (PS, PE, PET, PP, PVC)
2. **Use LLM classification**: Run `llm_classify_mechanisms.py` for better mechanism detection (~$0.70 for 198 papers)
3. **Loosen regex patterns**: Reduce requirement for compound phrases
4. **Manual validation sample**: Spot-check 20 papers for accuracy

## Script Location

Main extraction script:
```
/Users/sarahdaniels/Documents/microplastics_chem_exp_explorer/scripts/paperclip_extract_papers.py
```

## Date Extracted

April 2026

## Data Sources

Paperclip indexes:
- PubMed Central (PMC)
- bioRxiv
- medRxiv
- ~8M+ papers total
