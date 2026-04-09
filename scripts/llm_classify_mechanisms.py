#!/usr/bin/env python3
"""
LLM-based Mechanism Classification Script
==========================================
Uses Claude Sonnet to classify microplastics grants by mechanism of toxicity.

Cost estimate: ~$0.70 for 204 grants using Sonnet

Usage:
    export ANTHROPIC_API_KEY="your-key-here"
    python scripts/llm_classify_mechanisms.py

Output:
    - data/llm_mechanism_classifications.csv (new classifications)
    - Prints comparison with existing regex-based classifications
"""

import os
import json
import time
import pandas as pd
from pathlib import Path
from anthropic import Anthropic

# Mechanism definitions - aligned with app.py MECHANISM_SUMMARIES
# These describe what each mechanism means in microplastics/nanoplastics research
MECHANISMS = {
    "oxidative_stress": """Oxidative stress and mitochondrial dysfunction research examines how
micro/nanoplastics generate reactive oxygen species (ROS), deplete cellular antioxidants
(glutathione, SOD, catalase), damage mitochondrial membranes, disrupt electron transport chains,
and cause lipid peroxidation. Key markers include MDA (malondialdehyde), 8-OHdG, protein carbonyls,
and changes in Nrf2/ARE pathway activation. Studies may measure mitochondrial membrane potential,
ATP production, or use fluorescent ROS probes.""",

    "inflammation": """Inflammatory response research studies how plastic particles trigger
pro-inflammatory cytokine release (IL-1β, IL-6, IL-8, TNF-α), activate the NLRP3 inflammasome,
induce NF-κB signaling, or cause tissue inflammation. Includes studying macrophage/monocyte
activation, neutrophil infiltration, and inflammatory gene expression. Must be investigating
inflammation as a DIRECT OUTCOME of plastic exposure, not just background mention. Chronic
low-grade inflammation and sterile inflammation from particle exposure count.""",

    "barrier_disruption": """Barrier disruption research investigates how microplastics compromise
biological barriers: intestinal/gut epithelial barrier (increased permeability, "leaky gut"),
blood-brain barrier (BBB) integrity, pulmonary/alveolar barrier, or placental barrier. Key
markers include tight junction proteins (ZO-1, occludin, claudins), transepithelial electrical
resistance (TEER), paracellular permeability assays (FITC-dextran), and particle translocation
studies. Research on how particles cross from gut to bloodstream or other compartments.""",

    "microbiome": """Gut microbiome research examines how ingested micro/nanoplastics alter
intestinal microbial communities: dysbiosis, changes in bacterial diversity (alpha/beta diversity),
shifts in Firmicutes/Bacteroidetes ratios, effects on beneficial bacteria (Lactobacillus,
Bifidobacterium), or pathobiont overgrowth. Includes effects on microbial metabolites
(short-chain fatty acids, bile acids), gut-brain axis signaling, and microbiome-mediated
immune effects. May use 16S rRNA sequencing, metagenomics, or metabolomics approaches.""",

    "endocrine": """Endocrine disruption research studies how plastics or their chemical additives
interfere with hormonal systems: estrogenic/anti-estrogenic activity (estrogen receptor binding),
androgenic effects, thyroid hormone disruption (T3/T4, TSH), or effects on reproductive hormones
(testosterone, progesterone, FSH, LH). Includes research on plastic additives like phthalates,
BPA/BPS/BPF, and other plasticizers leaching from particles. May study effects on hormone-sensitive
tissues, puberty timing, or metabolic hormones (insulin, leptin).""",

    "neurodegeneration": """Neurotoxicity and neurodegeneration research examines how micro/nanoplastics
affect the brain and nervous system: cognitive impairment, memory deficits, behavioral changes,
links to Alzheimer's disease (amyloid-β, tau), Parkinson's disease (α-synuclein, dopaminergic
neurons), neuroinflammation (microglial activation), or general neurotoxicity. Includes studies
on particle accumulation in brain tissue, effects on neurons/glia, neurotransmitter changes,
and blood-brain barrier crossing. The grant must be studying BRAIN/NERVOUS SYSTEM as a primary
target organ, not just mentioning neurological endpoints.""",

    "immune_dysfunction": """Immune system dysfunction research studies how plastics impair immune
function beyond acute inflammation: immunotoxicity, immunosuppression, altered lymphocyte
(T-cell, B-cell) function, changes in antibody production, NK cell activity, dendritic cell
maturation, or macrophage polarization (M1/M2). May examine effects on immune organs (thymus,
spleen, lymph nodes), autoimmune responses, or vaccine response impairment. Distinct from
acute inflammatory response - focuses on immune system function and competence.""",

    "dna_damage": """DNA damage and genotoxicity research examines whether micro/nanoplastics cause
genetic damage: DNA strand breaks (comet assay), chromosomal aberrations, micronucleus formation,
mutations, or genomic instability. May study oxidative DNA damage (8-OHdG), DNA adduct formation,
effects on DNA repair pathways, or potential carcinogenic/mutagenic effects. Includes research
on particle interactions with DNA-damaging gut bacteria (e.g., pks+ E. coli producing colibactin)
and transgenerational effects on genomic integrity.""",

    "receptor_signaling": """Receptor and signaling pathway research studies how plastic particles
or their leachates interact with cellular receptors and signaling cascades: aryl hydrocarbon
receptor (AhR) activation, pregnane X receptor (PXR), peroxisome proliferator-activated receptors
(PPARs), Toll-like receptors (TLRs), or other pattern recognition receptors. Includes effects
on downstream signaling pathways (MAPK, PI3K/Akt, Wnt), ion channel function (calcium signaling,
mechanosensitive channels), and receptor-mediated cellular responses.""",

    "cell_death": """Cell death and senescence research examines how plastic exposure triggers
programmed cell death or cellular aging: apoptosis (caspase activation, Bcl-2 family, TUNEL),
necrosis/necroptosis, pyroptosis, ferroptosis, autophagy dysregulation (LC3, p62, Beclin-1),
or cellular senescence (p16, p21, SA-β-gal, SASP). Studies may examine dose-dependent
cytotoxicity, viability assays, or specific cell death pathways activated by particle exposure.""",
}

def build_prompt(title: str, abstract: str) -> str:
    """Build the classification prompt with mechanism definitions."""
    mechanism_list = "\n".join([f"- {k}: {v}" for k, v in MECHANISMS.items()])

    return f"""You are classifying NIH research grants about microplastics/nanoplastics toxicity.

For each grant, determine which mechanisms of toxicity the research DIRECTLY INVESTIGATES (not just mentions in background).

IMPORTANT: Only mark a mechanism as TRUE if:
1. It appears in the project TITLE, OR
2. The abstract describes INVESTIGATING/STUDYING that mechanism (not just mentioning it exists)

Known mechanisms to classify:
{mechanism_list}

GRANT TO CLASSIFY:
Title: {title}
Abstract: {abstract}

Respond with ONLY a JSON object (no markdown, no explanation).
Include all known mechanisms plus an "other_mechanisms" field listing any additional toxicity mechanisms investigated that don't fit the categories above (as a comma-separated string, or empty string if none):

{{"oxidative_stress": true/false, "inflammation": true/false, "barrier_disruption": true/false, "microbiome": true/false, "endocrine": true/false, "neurodegeneration": true/false, "immune_dysfunction": true/false, "dna_damage": true/false, "receptor_signaling": true/false, "cell_death": true/false, "other_mechanisms": ""}}
"""


def classify_grant(client: Anthropic, title: str, abstract: str, retries: int = 3) -> dict:
    """Classify a single grant using Claude Sonnet."""
    prompt = build_prompt(title, abstract or "No abstract available")

    for attempt in range(retries):
        try:
            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=200,
                messages=[{"role": "user", "content": prompt}]
            )

            # Parse JSON response
            text = response.content[0].text.strip()
            # Handle potential markdown code blocks
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]

            return json.loads(text)

        except json.JSONDecodeError as e:
            print(f"  JSON parse error (attempt {attempt+1}): {e}")
            if attempt == retries - 1:
                return {k: False for k in MECHANISMS.keys()}
        except Exception as e:
            print(f"  API error (attempt {attempt+1}): {e}")
            if attempt < retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                return {k: False for k in MECHANISMS.keys()}

    return {k: False for k in MECHANISMS.keys()}


def main():
    # Check for API key
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: Set ANTHROPIC_API_KEY environment variable")
        print("  export ANTHROPIC_API_KEY='your-key-here'")
        return

    client = Anthropic(api_key=api_key)

    # Load data
    data_dir = Path(__file__).parent.parent / "data"
    csv_path = data_dir / "microplastic_grants_cleaned.csv"

    print(f"Loading {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} grants")

    # Classify each grant
    results = []
    total = len(df)

    print(f"\nClassifying {total} grants with Claude Sonnet...")
    print("=" * 60)

    for idx, row in df.iterrows():
        project_num = row.get("PROJECT_NUM", row.get("CORE_PROJECT_NUM", f"row_{idx}"))
        title = row.get("PROJECT_TITLE", "")
        abstract = row.get("ABSTRACT_TEXT", "")

        print(f"[{idx+1}/{total}] {project_num}: {title[:50]}...")

        classifications = classify_grant(client, title, abstract)
        classifications["PROJECT_NUM"] = project_num
        results.append(classifications)

        # Rate limiting - be nice to the API
        if (idx + 1) % 10 == 0:
            time.sleep(1)

    # Create results dataframe
    results_df = pd.DataFrame(results)

    # Rename columns to match existing schema
    col_map = {
        "oxidative_stress": "LLM_MECH_OXIDATIVE",
        "inflammation": "LLM_MECH_INFLAMMATION",
        "barrier_disruption": "LLM_MECH_BARRIER",
        "microbiome": "LLM_MECH_MICROBIOME",
        "endocrine": "LLM_MECH_ENDOCRINE",
        "neurodegeneration": "LLM_MECH_NEURODEGENERATION",
        "immune_dysfunction": "LLM_MECH_IMMUNE",
        "dna_damage": "LLM_MECH_DNA_DAMAGE",
        "receptor_signaling": "LLM_MECH_RECEPTOR",
        "cell_death": "LLM_MECH_CELL_DEATH",
    }
    results_df = results_df.rename(columns=col_map)

    # Keep other_mechanisms as string, rename it
    if "other_mechanisms" in results_df.columns:
        results_df = results_df.rename(columns={"other_mechanisms": "LLM_OTHER_MECHANISMS"})

    # Convert booleans to integers
    for col in col_map.values():
        if col in results_df.columns:
            results_df[col] = results_df[col].astype(int)

    # Save results
    output_path = data_dir / "llm_mechanism_classifications.csv"
    results_df.to_csv(output_path, index=False)
    print(f"\nSaved LLM classifications to {output_path}")

    # Compare with existing regex classifications
    print("\n" + "=" * 60)
    print("COMPARISON: LLM vs Regex Classifications")
    print("=" * 60)

    regex_to_llm = {
        "MECH_OXIDATIVE_MITOCHONDRIAL": "LLM_MECH_OXIDATIVE",
        "MECH_INFLAMMATION": "LLM_MECH_INFLAMMATION",
        "MECH_BARRIER_DISRUPTION": "LLM_MECH_BARRIER",
        "MECH_MICROBIOME": "LLM_MECH_MICROBIOME",
        "MECH_ENDOCRINE": "LLM_MECH_ENDOCRINE",
        "MECH_NEURODEGENERATION": "LLM_MECH_NEURODEGENERATION",
        "MECH_IMMUNE_DYSFUNCTION": "LLM_MECH_IMMUNE",
        "MECH_DNA_DAMAGE": "LLM_MECH_DNA_DAMAGE",
        "MECH_RECEPTOR_SIGNALING": "LLM_MECH_RECEPTOR",
        "MECH_SENESCENCE_CELL_DEATH": "LLM_MECH_CELL_DEATH",
    }

    for regex_col, llm_col in regex_to_llm.items():
        if regex_col in df.columns and llm_col in results_df.columns:
            regex_count = df[regex_col].sum()
            llm_count = results_df[llm_col].sum()
            diff = llm_count - regex_count
            sign = "+" if diff > 0 else ""
            print(f"{regex_col.replace('MECH_', ''):25} Regex: {regex_count:3}  LLM: {llm_count:3}  ({sign}{diff})")

    # Show other mechanisms discovered
    if "LLM_OTHER_MECHANISMS" in results_df.columns:
        other_mechs = results_df[results_df["LLM_OTHER_MECHANISMS"].notna() & (results_df["LLM_OTHER_MECHANISMS"] != "")]
        if len(other_mechs) > 0:
            print("\n" + "=" * 60)
            print("OTHER MECHANISMS DISCOVERED")
            print("=" * 60)
            all_others = []
            for _, row in other_mechs.iterrows():
                others = str(row["LLM_OTHER_MECHANISMS"])
                if others and others != "nan":
                    print(f"  {row['PROJECT_NUM']}: {others}")
                    all_others.extend([m.strip() for m in others.split(",")])

            # Count frequency of other mechanisms
            if all_others:
                from collections import Counter
                counts = Counter(all_others)
                print("\nMost common 'other' mechanisms:")
                for mech, count in counts.most_common(10):
                    if mech:
                        print(f"  {mech}: {count}")

    print("\nDone! Review llm_mechanism_classifications.csv and merge if satisfied.")


if __name__ == "__main__":
    main()
