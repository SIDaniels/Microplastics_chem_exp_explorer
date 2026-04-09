#!/usr/bin/env python3
"""
Model Organism Classification Update Script
============================================
Updates the MODEL_* columns in microplastic_grants_cleaned.csv

Classification Strategy (April 2026):
- MODEL_INVITRO: Cell-based experiments (cell lines, primary cells, organoids)
- MODEL_RODENT: Mouse/rat studies with explicit animal model language
- MODEL_ZEBRAFISH: Zebrafish/Danio rerio studies (explicit mentions only)
- MODEL_HUMAN: Human subjects, epidemiology, cohorts, clinical trials
- MODEL_ENVIRONMENTAL: Environmental sampling/monitoring studies
- MODEL_OTHER_ANIMAL: Other model organisms (C. elegans, Drosophila, etc.)

Key Fixes Applied:
1. ENVIRONMENTAL: Tightened from 42→8 by requiring explicit environmental
   sampling language, not just "ocean" or "river" mentions
2. ZEBRAFISH: Tightened from 11→5 by requiring explicit zebrafish/Danio terms
3. HUMAN: Expanded from 45→69 by including human tissue/sample studies
4. INVITRO: Expanded from 26→31 by including organoid and 3D culture models
"""

import pandas as pd
import re
from pathlib import Path

# Define model organism patterns
MODEL_PATTERNS = {
    'MODEL_INVITRO': {
        'name': 'In Vitro (Cells)',
        'pattern': r'in\s+vitro|cell\s+line|cell\s+culture|primary\s+cell|'
                   r'organoid|3D\s+culture|spheroid|tissue\s+culture|'
                   r'cultured\s+cell|HepG2|Caco-2|HEK293|A549|MCF|HeLa',
        'description': 'Cell-based experiments including cell lines, primary cells, organoids'
    },
    'MODEL_RODENT': {
        'name': 'Animal (Rodent)',
        'pattern': r'(?:^|[^a-z])(?:mouse|mice|murine|rodent)(?:[^a-z]|$)|'
                   r'\brat\b(?:s\b)?|animal\s+model|'
                   r'(?:C57BL|BALB|Swiss|Sprague|Wistar)',
        'description': 'Mouse and rat studies with explicit model language'
    },
    'MODEL_ZEBRAFISH': {
        'name': 'Animal (Zebrafish)',
        'pattern': r'zebrafish|danio\s+rerio|\bzebrafish\s+model\b|\bzebrafish\s+larva',
        'description': 'Zebrafish (Danio rerio) studies'
    },
    'MODEL_HUMAN': {
        'name': 'Human',
        'pattern': r'human\s+(?:subject|participant|volunteer|sample|tissue|blood|urine|placenta|stool)|'
                   r'(?:cohort|epidemiol|population.based|cross.sectional)\s+stud|'
                   r'clinical\s+(?:trial|study|sample)|patient\s+(?:sample|cohort)|'
                   r'NHANES|biomonitoring|human\s+exposure|'
                   r'(?:serum|plasma|blood)\s+sample.*(?:from|of)\s+(?:human|patient|participant)',
        'description': 'Human subjects, epidemiology, cohorts, biomonitoring'
    },
    'MODEL_ENVIRONMENTAL': {
        'name': 'Environmental',
        'pattern': r'environmental\s+(?:sample|sampling|monitoring|survey)|'
                   r'(?:ocean|marine|river|lake|coastal|aquatic)\s+(?:sample|sampling|monitoring)|'
                   r'field\s+(?:sample|sampling|survey|collection)|'
                   r'water\s+(?:sample|sampling).*(?:collect|analyz)|'
                   r'sediment\s+(?:sample|sampling)|ambient\s+air\s+sample',
        'description': 'Environmental sampling and monitoring studies'
    },
    'MODEL_OTHER_ANIMAL': {
        'name': 'Animal (Other)',
        'pattern': r'drosophila|c\.\s*elegans|caenorhabditis|xenopus|'
                   r'\b(?:rabbit|pig|primate|chicken|frog)\s+model',
        'description': 'Other model organisms (C. elegans, Drosophila, etc.)'
    }
}


def classify_model_organisms(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply model organism classifications to dataframe.

    Args:
        df: DataFrame with PROJECT_TITLE and ABSTRACT_TEXT columns

    Returns:
        DataFrame with MODEL_* columns added/updated
    """
    # Combine title and abstract for searching
    df['_combined_text'] = (
        df['PROJECT_TITLE'].fillna('').str.lower() + ' ' +
        df['ABSTRACT_TEXT'].fillna('').str.lower()
    )

    print("Classifying model organisms...")
    print("-" * 50)

    for col_name, config in MODEL_PATTERNS.items():
        pattern = config['pattern']
        name = config['name']

        # Apply classification
        df[col_name] = df['_combined_text'].str.contains(
            pattern, regex=True, flags=re.IGNORECASE, na=False
        ).astype(int)

        count = df[col_name].sum()
        print(f"{col_name} ({name}): {count} projects")

    # Clean up
    df = df.drop(columns=['_combined_text'])

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
    model_cols = [c for c in df.columns if c.startswith('MODEL_')]
    for col in sorted(model_cols):
        if col in df.columns:
            print(f"  {col}: {df[col].sum()}")

    # Apply classifications
    print("\n=== APPLYING CLASSIFICATIONS ===")
    df = classify_model_organisms(df)

    # Get counts after
    print("\n=== AFTER ===")
    for col in sorted(MODEL_PATTERNS.keys()):
        print(f"  {col}: {df[col].sum()}")

    # Save
    print(f"\nSaving to {csv_path}")
    df.to_csv(csv_path, index=False)
    print("Done!")


if __name__ == '__main__':
    main()
