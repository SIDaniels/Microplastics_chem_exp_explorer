# Paper Integration Plan: Adding PMC Papers to Dataset

## Current State

| Dataset | File | Count | Source |
|---------|------|-------|--------|
| Preprints | `microplastics_papers_2025_2026.csv` | 198 | bioRxiv (187) + medRxiv (11) |
| PMC | `pmc_papers_classified.csv` | 58 | PubMed Central (published) |
| **Combined** | - | **256** | All sources |

## Column Structure

### Core Columns (both datasets have these)
- `doc_id` - Unique identifier (PMC12345678 or bio_xxx)
- `title`, `authors`, `abstract`, `doi`, `pub_date`, `year`
- `source_type` - "bioRxiv (preprint)", "medRxiv (preprint)", or "PMC (published)"

### LLM Classification Columns (both datasets have these)
- `LLM_MECH_*` - 10 mechanism columns
- `LLM_ORGAN_*` - 9 organ system columns
- `LLM_MODEL_*` - 6 model organism columns

### Legacy Regex Columns (preprints only)
- `MECH_*` - Old regex-based mechanism classifications
- `ORGAN_*` - Old regex-based organ classifications
- `MODEL_*` - Old regex-based model classifications
- `STUDY_DETECTION_HUMAN` - Human detection study flag

## Integration Steps

### Step 1: Merge Datasets
```python
import pandas as pd

preprints = pd.read_csv('data/microplastics_papers_2025_2026.csv')
pmc = pd.read_csv('data/pmc_papers_classified.csv')

# Add missing columns to PMC (fill with 0 or empty)
for col in preprints.columns:
    if col not in pmc.columns:
        if col.startswith(('MECH_', 'ORGAN_', 'MODEL_', 'STUDY_')):
            pmc[col] = 0  # Binary columns default to 0
        else:
            pmc[col] = ''

# Concatenate
combined = pd.concat([preprints, pmc], ignore_index=True)

# Verify no duplicates
assert combined['doc_id'].nunique() == len(combined), "Duplicate doc_ids!"

combined.to_csv('data/microplastics_papers_2025_2026.csv', index=False)
```

### Step 2: Add STUDY_DETECTION_HUMAN to PMC Papers
```python
# Apply same regex pattern used for preprints
def is_human_detection(row):
    text = (str(row.get('title', '')) + ' ' + str(row.get('abstract', ''))).lower()
    has_human = any(x in text for x in ['human', 'placenta', 'blood sample', 'serum sample'])
    has_detection = any(x in text for x in ['detect', 'quantif', 'identif', 'characteriz'])
    return 1 if (has_human and has_detection) else 0

pmc['STUDY_DETECTION_HUMAN'] = pmc.apply(is_human_detection, axis=1)
```

### Step 3: Verify Data Integrity
```python
# Check counts
print(f"Total papers: {len(combined)}")
print(f"bioRxiv: {len(combined[combined['source_type'].str.contains('bioRxiv')])}")
print(f"medRxiv: {len(combined[combined['source_type'].str.contains('medRxiv')])}")
print(f"PMC: {len(combined[combined['source_type'].str.contains('PMC')])}")

# Check no NaN in critical columns
assert combined['doc_id'].notna().all()
assert combined['title'].notna().all()
```

## App Changes Required

### 1. Add Source Filter to Left Panel

In `app.py`, add a new filter section:

```python
# Source type filter
st.sidebar.markdown("### Source")
source_options = ["All", "Published (PMC)", "Preprints (bioRxiv/medRxiv)"]
selected_source = st.sidebar.radio("Source type:", source_options)

# Apply filter
if selected_source == "Published (PMC)":
    df = df[df['source_type'].str.contains('PMC')]
elif selected_source == "Preprints (bioRxiv/medRxiv)":
    df = df[df['source_type'].str.contains('Rxiv')]
```

### 2. Add Source Badge to Paper Display

```python
def get_source_badge(source_type):
    if 'PMC' in source_type:
        return "🔬 Published"
    elif 'bioRxiv' in source_type:
        return "📝 bioRxiv"
    elif 'medRxiv' in source_type:
        return "📝 medRxiv"
    return "❓ Unknown"

# In paper display
st.markdown(f"**{row['title']}** {get_source_badge(row['source_type'])}")
```

### 3. Summary Statistics Update

Add source breakdown to summary stats:

```python
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Papers", len(df))
col2.metric("Published", len(df[df['source_type'].str.contains('PMC')]))
col3.metric("Preprints", len(df[df['source_type'].str.contains('Rxiv')]))
```

## Potential Issues & Solutions

### Issue 1: Legacy Regex Columns Missing for PMC
**Problem:** PMC papers won't have `MECH_*`, `ORGAN_*`, `MODEL_*` (legacy regex) columns filled.
**Solution:** Either:
- a) Fill with 0 (safe, but inconsistent)
- b) Run regex classification on PMC papers too (consistent)
- c) Deprecate legacy columns, use only `LLM_*` columns

**Recommendation:** Option (c) - use only LLM columns going forward. They're more accurate anyway.

### Issue 2: Filtering by Source May Reduce Results
**Problem:** Users filtering to "Published only" will only see 58 papers (23% of data).
**Solution:** Show count in filter label: "Published (PMC) - 58 papers"

### Issue 3: Preprint-to-Publication Tracking
**Problem:** Some preprints may later be published to PMC. Over time, duplicates could appear.
**Solution:**
- Add `preprint_id` column to PMC papers if they originated from bioRxiv/medRxiv
- Run periodic deduplication checks
- For now: the 0% overlap we found suggests this isn't urgent

### Issue 4: Date Range Confusion
**Problem:** PMC `pub_date` is publication date; preprint `pub_date` is first posted date.
**Solution:** Add `date_type` column or document this in the app.

## Biggest Problems & Why

### 1. Column Inconsistency (HIGH PRIORITY)
The legacy regex columns (`MECH_*`, `ORGAN_*`, `MODEL_*`) exist only for preprints. If the app filters on these, PMC papers will always show as 0.

**Fix:** Update app to use only `LLM_*` columns for all filtering.

### 2. No Journal/Venue Information (MEDIUM PRIORITY)
PMC papers come from various journals, but we don't capture journal name. This could be useful for filtering by journal quality/impact.

**Fix:** Fetch journal name from Paperclip metadata and add `journal` column.

### 3. Preprint Status Unknown (LOW PRIORITY)
We don't know if bioRxiv papers have been peer-reviewed/published elsewhere.

**Fix:** Could cross-reference with PMC/PubMed, but complex and not critical.

## Final Checklist

- [ ] Run merge script to combine datasets
- [ ] Add `source_type` column to all rows
- [ ] Apply `STUDY_DETECTION_HUMAN` to PMC papers
- [ ] Update `app.py` to add source filter
- [ ] Update `app.py` to use `LLM_*` columns only (deprecate legacy)
- [ ] Add source badges to paper display
- [ ] Test all filters work correctly
- [ ] Verify total count = 256 papers
- [ ] Update documentation

## Files to Modify

| File | Changes |
|------|---------|
| `data/microplastics_papers_2025_2026.csv` | Add 58 PMC papers |
| `data/pmc_papers_classified.csv` | Temporary file, can delete after merge |
| `app.py` | Add source filter, update to use LLM columns |
| `docs/CLASSIFICATION_DEFINITIONS.md` | Note that legacy columns are deprecated |

---

*Created: April 2026*
