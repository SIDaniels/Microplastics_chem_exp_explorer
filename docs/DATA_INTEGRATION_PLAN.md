# Data Integration Plan

## Current Data Sources

| Source | File | Count | ID Field | Has DOI? |
|--------|------|-------|----------|----------|
| NIH Grants | `microplastic_grants_cleaned.csv` | 204 | `CORE_PROJECT_NUM` | No |
| Conference Abstracts | `conference_abstracts.csv` | 192 | `#` (index) | No |
| Published Papers | `microplastics_papers_2025_2026.csv` | 198 | `doc_id` | Yes |

## Overlap Analysis

### Grants vs Papers: NO OVERLAP
- Grants are NIH funding records (proposals/awards)
- Papers are published research articles
- These are fundamentally different data types - no deduplication needed

### Conference Abstracts vs Papers: POSSIBLE OVERLAP
- Conference was January 2026 in New Mexico
- Papers are from 2025-2026
- Some conference presentations may have become published papers
- **Deduplication needed**

## Deduplication Strategy

### Step 1: Title Similarity Matching
Since conference abstracts don't have DOIs, we must match by title similarity.

```python
# Fuzzy matching approach
from rapidfuzz import fuzz

def find_duplicates(abstracts_df, papers_df, threshold=85):
    duplicates = []
    for _, abstract in abstracts_df.iterrows():
        abstract_title = str(abstract['Title']).lower().strip()
        for _, paper in papers_df.iterrows():
            paper_title = str(paper['title']).lower().strip()
            score = fuzz.ratio(abstract_title, paper_title)
            if score >= threshold:
                duplicates.append({
                    'abstract_idx': abstract['#'],
                    'paper_id': paper['doc_id'],
                    'abstract_title': abstract['Title'],
                    'paper_title': paper['title'],
                    'similarity': score
                })
    return duplicates
```

### Step 2: Manual Review
- Review matches with similarity 85-95% (potential duplicates)
- Confirm matches with similarity >95% (likely duplicates)
- Decide whether to:
  a) Remove from conference abstracts (paper is more complete)
  b) Remove from papers (keep conference version)
  c) Link them (add `related_paper_id` column to abstracts)

### Step 3: Schema Harmonization
The three datasets have different column structures:

**Option A: Keep Separate (Current)**
- Pro: Simple, no schema changes
- Con: Can't search across all data

**Option B: Create Unified Table**
- Merge into single `all_research.csv` with:
  - `source_type`: grant / conference / paper
  - `source_id`: original ID
  - `title`, `abstract`, `authors`
  - `MODEL_*`, `ORGAN_*`, `MECH_*` columns
- Pro: Unified search, consistent classifications
- Con: Need to apply LLM classification to grants/conference abstracts

**Option C: Keep Separate + Cross-Reference**
- Keep three files separate
- Add `related_ids` columns to link duplicates
- Pro: Preserves original data, shows relationships
- Con: More complex queries

## Recommended Approach: Option C

1. **Run duplicate detection** between conference abstracts and papers
2. **Add `published_paper_id` column** to `conference_abstracts.csv` for any matches
3. **Keep datasets separate** in app (different tabs already exist)
4. **Apply consistent LLM classification** to all three datasets for filtering

## Implementation Steps

### Phase 1: Deduplication (Required)
1. Install rapidfuzz: `pip install rapidfuzz`
2. Run title similarity matching script
3. Review matches, confirm duplicates
4. Add cross-reference column to conference abstracts

### Phase 2: Classification Alignment (Recommended)
1. Apply `llm_classify_all.py` to conference abstracts (~$1 for 192 items)
2. Verify grants already have `LLM_MECH_*` columns (they do)
3. Ensure all datasets use same column naming convention

### Phase 3: App Updates (Optional)
1. Add "Papers" tab to Streamlit app
2. Update filtering to work across all datasets
3. Add cross-reference links in UI

## Testing Checklist

- [ ] Run duplicate detection between conference abstracts and papers
- [ ] Verify no abstracts appear twice in papers dataset
- [ ] Confirm classification columns are consistent across datasets
- [ ] Test app filters work correctly after data update
- [ ] Verify row counts match expected totals

## Files to Modify

| File | Change |
|------|--------|
| `data/conference_abstracts.csv` | Add `published_paper_id` column |
| `data/microplastics_papers_2025_2026.csv` | Already complete |
| `app.py` | Add Papers tab (optional) |
| `scripts/deduplicate_data.py` | New script for matching |

---

*Created: April 2026*
