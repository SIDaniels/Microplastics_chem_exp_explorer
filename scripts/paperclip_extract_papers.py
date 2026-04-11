#!/usr/bin/env python3
"""
Paperclip Paper Extraction Script
==================================
Extracts microplastics papers from Paperclip (2025-2026) and classifies them
using the same patterns as the grant classification scripts.

Output columns:
- doc_id, title, authors, doi, pub_date, year, abstract
- MODEL_* columns (INVITRO, RODENT, ZEBRAFISH, HUMAN, ENVIRONMENTAL, OTHER_ANIMAL)
- ORGAN_* columns (BRAIN_NERVOUS, CARDIOVASCULAR, GI_GUT, RESPIRATORY, etc.)
- MECH_* columns (from LLM classification via Paperclip map)

Usage:
    python scripts/paperclip_extract_papers.py

Requires:
    - Paperclip CLI installed and authenticated
    - pandas, openpyxl
"""

import subprocess
import json
import re
import pandas as pd
from pathlib import Path
from typing import Optional

# ============================================================================
# MODEL ORGANISM PATTERNS (from update_model_organisms.py)
# ============================================================================

RODENT_TERMS = r'\bmouse\b|\bmice\b|\bmurine\b|\brat\b|\brats\b|\brodent\b'
RODENT_ACTIVE_USE = [
    r'(?:we|will|to)\s+(?:use|used|using)\s+(?:mouse|mice|rat|rats|rodent)',
    r'(?:mouse|mice|rat|rats|rodent)\s+(?:model|studies|experiment)',
    r'(?:in|using|with)\s+(?:mice|rats|mouse)\b',
    r'(?:mice|rats|mouse)\s+(?:were|was|are|will be)\s+(?:exposed|treated|fed|given|injected|dosed)',
    r'(?:pregnant|female|male|adult|neonatal)\s+(?:mice|rats|mouse)',
    r'c57bl|balb|sprague|wistar',
    r'(?:mice|rats)\s+(?:fed|received|consumed)',
    r'murine\s+(?:model|tissue|cell|lung|brain|liver)',
    r'(?:apoe|ldlr).{0,5}(?:mice|mouse)',
]

MODEL_PATTERNS = {
    'MODEL_INVITRO': r'in\s+vitro|cell\s+line|cell\s+culture|primary\s+cell|'
                     r'organoid|3D\s+culture|spheroid|tissue\s+culture|'
                     r'cultured\s+cell|HepG2|Caco-2|HEK293|A549|MCF|HeLa',
    'MODEL_ZEBRAFISH': r'zebrafish|danio\s+rerio|\bzebrafish\s+model\b|\bzebrafish\s+larva',
    'MODEL_HUMAN': r'human\s+(?:subject|participant|volunteer|sample|tissue|blood|urine|placenta|stool)|'
                   r'(?:cohort|epidemiol|population.based|cross.sectional)\s+stud|'
                   r'clinical\s+(?:trial|study|sample)|patient\s+(?:sample|cohort)|'
                   r'NHANES|biomonitoring|human\s+exposure|'
                   r'(?:serum|plasma|blood)\s+sample.*(?:from|of)\s+(?:human|patient|participant)',
    'MODEL_ENVIRONMENTAL': r'environmental\s+(?:sample|sampling|monitoring|survey)|'
                           r'(?:ocean|marine|river|lake|coastal|aquatic)\s+(?:sample|sampling|monitoring)|'
                           r'field\s+(?:sample|sampling|survey|collection)|'
                           r'water\s+(?:sample|sampling).*(?:collect|analyz)|'
                           r'sediment\s+(?:sample|sampling)|ambient\s+air\s+sample',
    'MODEL_OTHER_ANIMAL': r'drosophila|c\.\s*elegans|caenorhabditis|xenopus|'
                          r'\b(?:rabbit|pig|primate|chicken|frog)\s+model',
}

# ============================================================================
# ORGAN SYSTEM PATTERNS (from update_organ_systems.py)
# ============================================================================

ORGAN_PATTERNS = {
    'ORGAN_BRAIN_NERVOUS': {
        'title': r'brain|neuro|nervous|CNS|cerebr|cognitive',
        'abstract': r'neurotoxic|neurodegenerat|brain\s+(?:damage|injury|toxicity)|'
                   r'blood.brain\s+barrier|cognitive\s+(?:impair|dysfunction)|'
                   r'nervous\s+system|neuroinflam|dopamin|serotonin',
    },
    'ORGAN_CARDIOVASCULAR': {
        'title': r'cardiovasc|cardiac|heart|vascular|coronary',
        'abstract': r'cardiotoxic|cardiovascular|heart\s+(?:disease|failure|damage)|'
                   r'cardiac\s+(?:function|toxicity)|atheroscler|coronary|'
                   r'vascular\s+(?:damage|dysfunction)|endothelial|hypertension',
    },
    'ORGAN_GI_GUT': {
        'title': r'gut|intestin|gastro|GI\s+tract|colon|bowel|digest',
        'abstract': r'gut\s+(?:barrier|health|microbiome)|intestin|'
                   r'GI\s+(?:toxicity|tract)|colitis|bowel|microbiome|'
                   r'IBD|IBS|enteric|digest',
    },
    'ORGAN_RESPIRATORY': {
        'title': r'lung|pulmonary|respiratory|airway|alveolar|inhal',
        'abstract': r'pulmonary\s+(?:toxicity|fibrosis|disease)|'
                   r'lung\s+(?:damage|disease|injury|function)|'
                   r'respiratory|airway|alveolar|bronchitis|COPD|asthma',
    },
    'ORGAN_REPRODUCTIVE': {
        'title': r'reproduct(?:ive|ion)|fertil|ovari|testes|placent|pregnan|fetal|uterine',
        'abstract': r'reproduct(?:ive|ion)|ovari|testes|fertil|infertil|'
                   r'sperm|oocyte|uter|placent|pregnan|fetal|endometri',
    },
    'ORGAN_LIVER': {
        'title': r'liver|hepat',
        'abstract': r'hepatotoxic|liver\s+(?:damage|disease|injury|fibrosis|function)|'
                   r'hepat|cirrhosis|fatty\s+liver|NAFLD|NASH',
    },
    'ORGAN_KIDNEY': {
        'title': r'kidney|renal|nephro',
        'abstract': r'nephrotoxic|kidney\s+(?:damage|disease|injury|failure)|'
                   r'renal\s+(?:toxicity|dysfunction)|glomerul|CKD|proteinuria',
    },
    'ORGAN_IMMUNE': {
        'title': r'immune|immuno|lymph|macrophage|T\s*cell|B\s*cell',
        'abstract': r'immune\s+system|immunotoxic|immunodeficien|immunosuppress|'
                   r'lymphocyte|macrophage\s+activation|T\s*cell|B\s*cell|'
                   r'inflammasome|NLRP3|autoimmun',
    },
    'ORGAN_ENDOCRINE': {
        'title': r'endocrine|thyroid|hormone|adrenal',
        'abstract': r'endocrine\s+disrupt|thyroid|hormone\s+(?:disrupt|dysfunction)|'
                   r'adrenal|pituitary|insulin\s+resist',
    },
}

# ============================================================================
# MECHANISM PATTERNS (from llm_classify_mechanisms.py - regex fallback)
# ============================================================================

MECH_PATTERNS = {
    'MECH_OXIDATIVE': r'oxidative\s+stress|ROS|reactive\s+oxygen|mitochondr|'
                      r'antioxidant|glutathione|SOD|catalase|lipid\s+peroxid',
    'MECH_INFLAMMATION': r'inflammat|cytokine|IL-1|IL-6|IL-8|TNF|NF.kB|NLRP3|'
                         r'pro.inflammatory|inflammasome',
    'MECH_BARRIER': r'barrier\s+(?:disrupt|dysfunction|integrity)|tight\s+junction|'
                    r'ZO-1|occludin|claudin|TEER|leaky\s+gut|permeability',
    'MECH_MICROBIOME': r'microbiome|microbiota|gut\s+flora|dysbiosis|'
                       r'firmicutes|bacteroidetes|16S\s+rRNA',
    'MECH_ENDOCRINE': r'endocrine\s+disrupt|estrogen|androgen|thyroid|'
                      r'phthalate|BPA|bisphenol|hormone',
    'MECH_NEURODEGENERATION': r'neurodegenerat|alzheimer|parkinson|amyloid|'
                              r'tau\s+protein|alpha.synuclein|cognitive\s+decline',
    'MECH_IMMUNE': r'immunotoxic|immunosuppress|T.cell|B.cell|lymphocyte|'
                   r'macrophage\s+(?:activation|polarization)|NK\s+cell',
    'MECH_DNA_DAMAGE': r'DNA\s+damage|genotoxic|comet\s+assay|micronucleus|'
                       r'chromosom|mutagen|8-OHdG',
    'MECH_RECEPTOR': r'AhR|aryl\s+hydrocarbon|PXR|pregnane|PPAR|TLR|'
                     r'toll.like\s+receptor|receptor\s+(?:activation|binding)',
    'MECH_CELL_DEATH': r'apoptosis|caspase|Bcl-2|necrosis|pyroptosis|'
                       r'ferroptosis|autophagy|senescence|cell\s+death',
}


def run_paperclip_command(cmd: list) -> str:
    """Run a paperclip command and return output."""
    full_cmd = ['/Users/sarahdaniels/.local/bin/paperclip'] + cmd
    result = subprocess.run(full_cmd, capture_output=True, text=True, timeout=120)
    return result.stdout + result.stderr


def parse_paperclip_sql_output(output: str) -> list[dict]:
    """Parse the pipe-delimited SQL output from paperclip."""
    lines = output.strip().split('\n')
    if len(lines) < 2:
        return []

    # First line is headers
    headers = [h.strip() for h in lines[0].split('|')]

    # Skip the separator line (dashes)
    data_lines = [l for l in lines[1:] if not l.startswith('-') and l.strip()]

    records = []
    for line in data_lines:
        values = [v.strip() for v in line.split('|')]
        if len(values) == len(headers):
            records.append(dict(zip(headers, values)))

    return records


def classify_rodent(title: str, abstract: str) -> int:
    """Classify rodent using medium approach from update_model_organisms.py."""
    title_lower = title.lower() if title else ''
    abstract_lower = abstract.lower() if abstract else ''

    # Title match is strong signal
    if re.search(RODENT_TERMS, title_lower):
        return 1

    # Multiple mentions in abstract suggests actual use
    mentions = len(re.findall(RODENT_TERMS, abstract_lower))
    if mentions >= 2:
        return 1

    # Single mention with active use language
    for pattern in RODENT_ACTIVE_USE:
        if re.search(pattern, abstract_lower):
            return 1

    return 0


def classify_model_organisms(title: str, abstract: str) -> dict:
    """Classify model organisms for a paper."""
    combined = f"{title} {abstract}".lower() if title or abstract else ''

    result = {'MODEL_RODENT': classify_rodent(title or '', abstract or '')}

    for col, pattern in MODEL_PATTERNS.items():
        result[col] = 1 if re.search(pattern, combined, re.IGNORECASE) else 0

    return result


def classify_organ_systems(title: str, abstract: str) -> dict:
    """Classify organ systems for a paper."""
    title_lower = title.lower() if title else ''
    abstract_lower = abstract.lower() if abstract else ''

    result = {}
    for col, patterns in ORGAN_PATTERNS.items():
        # Title match OR 2+ abstract matches
        title_match = bool(re.search(patterns['title'], title_lower, re.IGNORECASE))
        abstract_matches = len(re.findall(patterns['abstract'], abstract_lower, re.IGNORECASE))
        result[col] = 1 if (title_match or abstract_matches >= 2) else 0

    return result


def classify_mechanisms(title: str, abstract: str) -> dict:
    """Classify mechanisms using regex patterns (fallback from LLM)."""
    combined = f"{title} {abstract}".lower() if title or abstract else ''

    result = {}
    for col, pattern in MECH_PATTERNS.items():
        result[col] = 1 if re.search(pattern, combined, re.IGNORECASE) else 0

    return result


def extract_year(pub_date: str) -> Optional[int]:
    """Extract year from pub_date string."""
    if not pub_date:
        return None
    # Try to find 4-digit year
    match = re.search(r'(20\d{2})', str(pub_date))
    if match:
        return int(match.group(1))
    return None


def fetch_paper_metadata(doc_id: str) -> Optional[dict]:
    """Fetch full metadata for a paper from Paperclip."""
    output = run_paperclip_command(['cat', f'/papers/{doc_id}/meta.json'])
    try:
        # Find JSON in output
        start = output.find('{')
        end = output.rfind('}') + 1
        if start >= 0 and end > start:
            return json.loads(output[start:end])
    except json.JSONDecodeError:
        pass
    return None


def main():
    data_dir = Path(__file__).parent.parent / 'data'

    print("=" * 60)
    print("Paperclip Paper Extraction Script")
    print("=" * 60)

    # Use SQL to get paper IDs (more reliable than parsing search output)
    print("\nQuerying Paperclip for microplastics paper IDs (2025-2026)...")

    sql_query = """SELECT id FROM documents
    WHERE (title ILIKE '%microplastic%' OR title ILIKE '%nanoplastic%'
           OR abstract_text ILIKE '%microplastic%' OR abstract_text ILIKE '%nanoplastic%')
      AND pub_date >= '2025-01-01'
    LIMIT 200"""

    output = run_paperclip_command(['sql', sql_query])

    # Parse SQL output - each line after header is an ID
    doc_ids = []
    lines = output.strip().split('\n')
    for line in lines:
        line = line.strip()
        # Skip header and separator lines
        if line.startswith('id') or line.startswith('-') or not line:
            continue
        if line.startswith('(') and 'rows' in line:  # Skip footer
            continue
        # UUID format or PMC format
        if re.match(r'^[a-f0-9-]{36}$', line) or line.startswith('PMC'):
            doc_ids.append(line)

    print(f"Found {len(doc_ids)} paper IDs")

    if not doc_ids:
        print("No papers found. Check Paperclip authentication.")
        return

    # Fetch full metadata for each paper
    print("\nFetching full metadata for each paper...")
    records = []
    for i, doc_id in enumerate(doc_ids):
        meta = fetch_paper_metadata(doc_id)
        if meta:
            records.append(meta)
        if (i + 1) % 20 == 0:
            print(f"  Fetched {i + 1}/{len(doc_ids)} papers...")

    print(f"Successfully fetched {len(records)} paper metadata")

    if not records:
        print("No papers found. Check Paperclip authentication.")
        return

    # Process each paper
    print("\nClassifying papers...")
    results = []

    for i, record in enumerate(records):
        doc_id = record.get('document_id', record.get('id', record.get('pmc_id', '')))
        title = record.get('title', '')
        authors = record.get('authors', '')
        doi = record.get('doi', '')
        pub_date = record.get('pub_date', '')
        abstract = record.get('abstract', record.get('abstract_text', ''))

        year = extract_year(pub_date)

        # Classify
        model_org = classify_model_organisms(title, abstract)
        organs = classify_organ_systems(title, abstract)
        mechs = classify_mechanisms(title, abstract)

        # Combine into single record
        row = {
            'doc_id': doc_id,
            'title': title,
            'authors': authors,
            'doi': doi,
            'pub_date': pub_date,
            'year': year,
            'abstract': abstract[:2000] if abstract else '',  # Truncate for Excel
            **model_org,
            **organs,
            **mechs,
        }
        results.append(row)

        if (i + 1) % 20 == 0:
            print(f"  Processed {i + 1}/{len(records)} papers...")

    # Create DataFrame
    df = pd.DataFrame(results)

    # Print summary
    print("\n" + "=" * 60)
    print("CLASSIFICATION SUMMARY")
    print("=" * 60)

    print("\nModel Organisms:")
    for col in [c for c in df.columns if c.startswith('MODEL_')]:
        print(f"  {col}: {df[col].sum()}")

    print("\nOrgan Systems:")
    for col in [c for c in df.columns if c.startswith('ORGAN_')]:
        print(f"  {col}: {df[col].sum()}")

    print("\nMechanisms:")
    for col in [c for c in df.columns if c.startswith('MECH_')]:
        print(f"  {col}: {df[col].sum()}")

    # Save to Excel
    output_path = data_dir / 'microplastics_papers_2025_2026.xlsx'
    df.to_excel(output_path, index=False, engine='openpyxl')
    print(f"\nSaved to {output_path}")

    # Also save as CSV
    csv_path = data_dir / 'microplastics_papers_2025_2026.csv'
    df.to_csv(csv_path, index=False)
    print(f"Saved to {csv_path}")

    print("\nDone!")


if __name__ == '__main__':
    main()
