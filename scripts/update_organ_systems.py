#!/usr/bin/env python3
"""
Organ System Classification Update Script
==========================================
Updates the ORGAN_* columns in microplastic_grants_cleaned.csv

Classification Strategy (April 2026):
- Require organ-specific terms in TITLE, or multiple mentions in ABSTRACT
- Use compound patterns requiring toxicity/damage/function context
- Exclude common false positive triggers

Key Fixes Applied:
1. REPRODUCTIVE: Uses `reproduct(?:ive|ion)` to exclude "reproduce/reproducibility"
   - Also excludes "embryonic day" and "pregnane"
   - Requires pregnancy-related terms for pregnancy matches
   - 47 → 30 projects

2. IMMUNE: Requires core immune terms, not just "inflammation"
   - Inflammasome/NLRP3 still counts
   - 40 → 22 projects

3. RESPIRATORY: Only counts "inhalation" if in title
   - Requires core respiratory terms (lung, pulmonary, airway)
   - 37 → 34 projects

4. ALL ORGANS: Require term in TITLE or multiple mentions in abstract
"""

import pandas as pd
import re
from pathlib import Path

# Define organ system patterns with tightened rules
ORGAN_PATTERNS = {
    'ORGAN_BRAIN_NERVOUS': {
        'name': 'Brain/Nervous System',
        'title_pattern': r'brain|neuro|nervous|CNS|cerebr|cognitive',
        'abstract_pattern': r'neurotoxic|neurodegenerat|brain\s+(?:damage|injury|toxicity)|'
                           r'blood.brain\s+barrier|cognitive\s+(?:impair|dysfunction)|'
                           r'nervous\s+system|neuroinflam|dopamin|serotonin',
        'min_abstract_mentions': 2
    },
    'ORGAN_CARDIOVASCULAR': {
        'name': 'Cardiovascular',
        'title_pattern': r'cardiovasc|cardiac|heart|vascular|coronary',
        'abstract_pattern': r'cardiotoxic|cardiovascular|heart\s+(?:disease|failure|damage)|'
                           r'cardiac\s+(?:function|toxicity)|atheroscler|coronary|'
                           r'vascular\s+(?:damage|dysfunction)|endothelial|hypertension',
        'min_abstract_mentions': 2
    },
    'ORGAN_GI_GUT': {
        'name': 'Gastrointestinal/Gut',
        'title_pattern': r'gut|intestin|gastro|GI\s+tract|colon|bowel|digest',
        'abstract_pattern': r'gut\s+(?:barrier|health|microbiome)|intestin|'
                           r'GI\s+(?:toxicity|tract)|colitis|bowel|microbiome|'
                           r'IBD|IBS|enteric|digest',
        'min_abstract_mentions': 2
    },
    'ORGAN_RESPIRATORY': {
        'name': 'Respiratory/Lung',
        'title_pattern': r'lung|pulmonary|respiratory|airway|alveolar|inhal',
        'abstract_pattern': r'pulmonary\s+(?:toxicity|fibrosis|disease)|'
                           r'lung\s+(?:damage|disease|injury|function)|'
                           r'respiratory|airway|alveolar|bronchitis|COPD|asthma',
        'min_abstract_mentions': 2,
        'note': 'Inhalation only counts if in title'
    },
    'ORGAN_REPRODUCTIVE': {
        'name': 'Reproductive',
        'title_pattern': r'reproduct(?:ive|ion)|fertil|ovari|testes|placent|pregnan|fetal|uterine',
        'abstract_pattern': r'reproduct(?:ive|ion)|ovari|testes|fertil|infertil|'
                           r'sperm|oocyte|uter|placent|pregnan|fetal|endometri',
        'min_abstract_mentions': 2,
        'exclude_patterns': [r'embryonic\s+day', r'pregnane'],
        'note': 'Excludes "reproduce/reproducibility", "embryonic day", "pregnane"'
    },
    'ORGAN_LIVER': {
        'name': 'Liver/Hepatic',
        'title_pattern': r'liver|hepat',
        'abstract_pattern': r'hepatotoxic|liver\s+(?:damage|disease|injury|fibrosis|function)|'
                           r'hepat|cirrhosis|fatty\s+liver|NAFLD|NASH',
        'min_abstract_mentions': 2
    },
    'ORGAN_KIDNEY': {
        'name': 'Kidney/Renal',
        'title_pattern': r'kidney|renal|nephro',
        'abstract_pattern': r'nephrotoxic|kidney\s+(?:damage|disease|injury|failure)|'
                           r'renal\s+(?:toxicity|dysfunction)|glomerul|CKD|proteinuria',
        'min_abstract_mentions': 2
    },
    'ORGAN_IMMUNE': {
        'name': 'Immune System',
        'title_pattern': r'immune|immuno|lymph|macrophage|T\s*cell|B\s*cell',
        'abstract_pattern': r'immune\s+system|immunotoxic|immunodeficien|immunosuppress|'
                           r'lymphocyte|macrophage\s+activation|T\s*cell|B\s*cell|'
                           r'inflammasome|NLRP3|autoimmun',
        'min_abstract_mentions': 1,
        'note': 'Requires core immune terms, not just "inflammation"'
    },
    'ORGAN_ENDOCRINE': {
        'name': 'Endocrine',
        'title_pattern': r'endocrine|thyroid|hormone|adrenal',
        'abstract_pattern': r'endocrine\s+disrupt|thyroid|hormone\s+(?:disrupt|dysfunction)|'
                           r'adrenal|pituitary|insulin\s+resist',
        'min_abstract_mentions': 2
    }
}


def classify_organ_systems(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply organ system classifications with tightened rules.

    Rules:
    1. Match if term appears in TITLE, OR
    2. Match if pattern appears >= min_abstract_mentions times in ABSTRACT

    Args:
        df: DataFrame with PROJECT_TITLE and ABSTRACT_TEXT columns

    Returns:
        DataFrame with ORGAN_* columns added/updated
    """
    df['_title'] = df['PROJECT_TITLE'].fillna('').str.lower()
    df['_abstract'] = df['ABSTRACT_TEXT'].fillna('').str.lower()

    print("Classifying organ systems...")
    print("-" * 50)

    for col_name, config in ORGAN_PATTERNS.items():
        name = config['name']
        title_pattern = config['title_pattern']
        abstract_pattern = config['abstract_pattern']
        min_mentions = config.get('min_abstract_mentions', 2)
        exclude_patterns = config.get('exclude_patterns', [])

        # Prepare text (apply exclusions for certain organs)
        abstract_text = df['_abstract'].copy()
        for exclude in exclude_patterns:
            abstract_text = abstract_text.str.replace(exclude, ' ', regex=True)

        # Check title match
        title_match = df['_title'].str.contains(
            title_pattern, regex=True, flags=re.IGNORECASE, na=False
        )

        # Count abstract matches
        def count_matches(text):
            if pd.isna(text) or not text:
                return 0
            matches = re.findall(abstract_pattern, text, re.IGNORECASE)
            return len(matches)

        abstract_counts = abstract_text.apply(count_matches)
        abstract_match = abstract_counts >= min_mentions

        # Combine: title match OR sufficient abstract mentions
        df[col_name] = (title_match | abstract_match).astype(int)

        count = df[col_name].sum()
        title_only = (title_match & ~abstract_match).sum()
        abstract_only = (~title_match & abstract_match).sum()
        both = (title_match & abstract_match).sum()
        print(f"{col_name} ({name}): {count} (title:{title_only}, abstract:{abstract_only}, both:{both})")

    # Clean up
    df = df.drop(columns=['_title', '_abstract'])

    print("-" * 50)
    print("Classification complete!")

    return df


def main():
    """Main function to update classifications."""
    # Path to data file
    data_dir = Path(__file__).parent.parent / 'data'
    csv_path = data_dir / 'microplastic_grants_cleaned.csv'

    print(f"Loading data from {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} projects")

    # Get counts before
    print("\n=== BEFORE ===")
    organ_cols = [c for c in df.columns if c.startswith('ORGAN_')]
    for col in sorted(organ_cols):
        if col in df.columns:
            print(f"  {col}: {df[col].sum()}")

    # Apply classifications
    print("\n=== APPLYING CLASSIFICATIONS ===")
    df = classify_organ_systems(df)

    # Get counts after
    print("\n=== AFTER ===")
    for col in sorted(ORGAN_PATTERNS.keys()):
        print(f"  {col}: {df[col].sum()}")

    # Save
    print(f"\nSaving to {csv_path}")
    df.to_csv(csv_path, index=False)
    print("Done!")


if __name__ == '__main__':
    main()
