#!/usr/bin/env python3
"""
LLM-based Full Classification Script
=====================================
Uses Claude Sonnet to classify microplastics papers across ALL categories:
- 10 Mechanisms
- 9 Organ Systems
- 6 Model Organisms

Cost estimate: ~$1.35 for 198 papers using Sonnet (single combined call per paper)

Usage:
    export ANTHROPIC_API_KEY="your-key-here"
    python scripts/llm_classify_all.py

Output:
    - Updates data/microplastics_papers_2025_2026.csv with LLM_* columns
    - Also saves data/llm_full_classifications.csv (classifications only)
"""

import os
import sys
import json
import time
import pandas as pd
from pathlib import Path
from anthropic import Anthropic

# =============================================================================
# CATEGORY DEFINITIONS
# =============================================================================

MECHANISMS = {
    "oxidative_stress": "Oxidative stress and mitochondrial dysfunction: ROS generation, antioxidant depletion (glutathione, SOD, catalase), mitochondrial damage, lipid peroxidation. Markers: MDA, 8-OHdG, Nrf2/ARE pathway.",

    "inflammation": "Inflammatory response: pro-inflammatory cytokines (IL-1β, IL-6, TNF-α), NLRP3 inflammasome, NF-κB signaling, tissue inflammation. Must be DIRECTLY INVESTIGATED as a mechanism.",

    "barrier_disruption": "Barrier disruption: gut epithelial barrier, blood-brain barrier, pulmonary barrier, placental barrier, epithelial integrity. Markers: tight junctions (ZO-1, occludin, claudins), TEER, permeability assays.",

    "microbiome": "Gut microbiome alterations: dysbiosis, bacterial diversity changes, Firmicutes/Bacteroidetes shifts, altered microbial metabolites (SCFAs, bile acids). Methods: 16S rRNA, metagenomics.",

    "endocrine": "Endocrine disruption: interference with hormones via plastics or leached additives (phthalates, BPA). Includes estrogenic/androgenic effects, thyroid disruption, reproductive hormone changes.",

    "neurodegeneration": "Neurotoxicity/neurodegeneration: effects on nervous system, cognitive impairment, links to Alzheimer's (amyloid-β, tau) or Parkinson's (α-synuclein), neuroinflammation, microglial activation, neural development defects.",

    "immune_dysfunction": "Immune dysfunction: immunotoxicity, immunosuppression, altered immune cell function, macrophage activation, autoimmune responses.",

    "dna_damage": "Genotoxicity/DNA damage: DNA strand breaks, chromosomal aberrations, micronucleus formation, mutations, genomic instability. Assays: comet assay, 8-OHdG, micronucleus test.",

    "receptor_signaling": "Receptor/signaling pathway effects: AhR (aryl hydrocarbon receptor), PXR, PPARs, Toll-like receptors (TLRs), downstream pathways (MAPK, PI3K/Akt, Wnt).",

    "cell_death": "Cell death/senescence: apoptosis (caspases, Bcl-2), necrosis, pyroptosis, ferroptosis, autophagy dysregulation, cellular senescence (p16, p21, SA-β-gal).",
}

STUDY_TYPES = {
    "detection": "Detection/quantification study: identifying, measuring, or characterizing microplastics/nanoplastics in biological samples (blood, tissue, placenta, urine, stool) or products (food, water, consumer goods). Methods: pyrolysis-GC/MS, FTIR, Raman spectroscopy, fluorescence microscopy.",

    "exposure_assessment": "Exposure assessment: estimating human exposure levels, exposure routes (ingestion, inhalation, dermal), exposure sources (food packaging, bottled water, air, textiles), biomonitoring, dose estimation, risk characterization.",
}

ORGANS = {
    "brain_nervous": "Brain/nervous system: neurotoxicity, neuroinflammation, blood-brain barrier, cognitive impairment, neurodegeneration, neural development. Includes: mammalian brain, zebrafish neurotoxicity, C. elegans neuronal effects.",

    "cardiovascular": "Cardiovascular: heart, blood vessels, cardiotoxicity, atherosclerosis, endothelial dysfunction, vascular effects. Includes: mammalian heart, zebrafish cardiac effects, endothelial cells.",

    "gi_gut": "Gastrointestinal/gut: intestinal effects, gut barrier integrity, microbiome alterations, intestinal inflammation, gut permeability. Includes: mammalian gut, zebrafish intestine, Daphnia gut, C. elegans intestine.",

    "respiratory": "Respiratory/lung: pulmonary toxicity, lung inflammation, fibrosis, airway effects, inhalation toxicity. Includes: mammalian lung, fish gills (if studying toxicity mechanisms).",

    "reproductive": "Reproductive: effects on gonads (ovary/testis), fertility, gamete quality (sperm/oocytes), placental effects, fetal/embryonic development. Includes: mammalian reproduction, zebrafish reproduction, Daphnia reproduction, C. elegans reproduction - if studying TOXICITY MECHANISMS.",

    "liver": "Liver/hepatic: hepatotoxicity, liver damage, steatosis, fibrosis, metabolic effects. Includes: mammalian liver, zebrafish liver, fish hepatotoxicity.",

    "kidney": "Kidney/renal: nephrotoxicity, kidney damage, renal function effects. Includes: mammalian kidney, zebrafish kidney/pronephros.",

    "immune": "Immune system: immunotoxicity, immune cell effects, macrophage activation, inflammatory responses. Includes: mammalian immune cells, fish immune response, invertebrate immune/hemocyte effects.",

    "endocrine": "Endocrine: hormone disruption, thyroid effects, estrogen/androgen effects, endocrine disruptors. Includes: mammalian hormones, fish vitellogenin/hormone disruption, invertebrate endocrine effects.",
}

MODELS = {
    "invitro": "In vitro: cell lines, primary cells, organoids, spheroids, 3D cultures. Examples: HepG2, Caco-2, HEK293, A549, HeLa.",

    "rodent": "Rodent: mice or rats used as experimental subjects. Any strain (C57BL/6, BALB/c, Sprague-Dawley, Wistar).",

    "zebrafish": "Zebrafish: Danio rerio at any life stage (embryos, larvae, adults).",

    "human": "Human: human subjects, human biospecimens (blood, urine, placenta, tissue), epidemiological studies, clinical trials, biomonitoring.",

    "environmental": "Environmental: field sampling of water, sediment, soil, or air to characterize plastic contamination (not controlled experiments).",

    "other_animal": "Other animals: C. elegans, Drosophila, Xenopus, marine invertebrates, fish (non-zebrafish), rabbits, pigs, primates.",
}


def build_prompt(title: str, abstract: str) -> str:
    """Build the combined classification prompt."""

    mech_list = "\n".join([f"  - {k}: {v}" for k, v in MECHANISMS.items()])
    organ_list = "\n".join([f"  - {k}: {v}" for k, v in ORGANS.items()])
    model_list = "\n".join([f"  - {k}: {v}" for k, v in MODELS.items()])
    study_type_list = "\n".join([f"  - {k}: {v}" for k, v in STUDY_TYPES.items()])

    return f"""You are classifying microplastics/nanoplastics research papers for a TOXICOLOGY database.

STEP 1: Is this study relevant to HEALTH/TOXICOLOGY?
Answer TRUE if the study investigates BIOLOGICAL EFFECTS or TOXICITY MECHANISMS in ANY organism:
- Humans, human samples, or human cell lines
- Mammalian models (mice, rats, rabbits, pigs, primates)
- Model organisms (zebrafish, C. elegans, Drosophila, Xenopus)
- Fish, marine invertebrates, Daphnia - if studying toxicity/biological effects
- Any study measuring oxidative stress, inflammation, organ damage, developmental effects, etc.
- Detection/quantification of microplastics in biological samples (blood, tissue, placenta)
- Exposure assessment studies

Answer FALSE only if:
- Plant or soil contamination studies (no animal toxicity)
- Plastic biodegradation by microbes/enzymes
- Pure methodology/analytical development without biological samples

STEP 2: Classify across all categories
For each category, mark TRUE only if the paper DIRECTLY INVESTIGATES that topic.

STUDY TYPES (can be true regardless of other categories):
{study_type_list}

MECHANISMS (toxicity pathways studied):
{mech_list}

ORGANS (target organs/systems):
{organ_list}

MODELS (experimental systems used):
{model_list}

PAPER:
Title: {title}
Abstract: {abstract}

Respond with ONLY valid JSON (no markdown):
{{
  "human_health_relevant": true/false,
  "confidence": "high/medium/low",
  "study_types": {{"detection": true/false, "exposure_assessment": true/false}},
  "mechanisms": {{"oxidative_stress": true/false, "inflammation": true/false, "barrier_disruption": true/false, "microbiome": true/false, "endocrine": true/false, "neurodegeneration": true/false, "immune_dysfunction": true/false, "dna_damage": true/false, "receptor_signaling": true/false, "cell_death": true/false}},
  "organs": {{"brain_nervous": true/false, "cardiovascular": true/false, "gi_gut": true/false, "respiratory": true/false, "reproductive": true/false, "liver": true/false, "kidney": true/false, "immune": true/false, "endocrine": true/false}},
  "models": {{"invitro": true/false, "rodent": true/false, "zebrafish": true/false, "human": true/false, "environmental": true/false, "other_animal": true/false}}
}}"""


def classify_paper(client: Anthropic, title: str, abstract: str, retries: int = 3) -> dict:
    """Classify a single paper using Claude Sonnet."""
    prompt = build_prompt(title, abstract or "No abstract available")

    for attempt in range(retries):
        try:
            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=300,
                messages=[{"role": "user", "content": prompt}]
            )

            text = response.content[0].text.strip()

            # Handle markdown code blocks
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]

            return json.loads(text)

        except json.JSONDecodeError as e:
            print(f"  JSON parse error (attempt {attempt+1}): {e}")
            if attempt == retries - 1:
                return get_empty_result()
        except Exception as e:
            print(f"  API error (attempt {attempt+1}): {e}")
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                return get_empty_result()

    return get_empty_result()


def get_empty_result() -> dict:
    """Return empty classification result."""
    return {
        "human_health_relevant": False,
        "confidence": "low",
        "study_types": {k: False for k in STUDY_TYPES.keys()},
        "mechanisms": {k: False for k in MECHANISMS.keys()},
        "organs": {k: False for k in ORGANS.keys()},
        "models": {k: False for k in MODELS.keys()},
    }


def flatten_result(result: dict, doc_id: str) -> dict:
    """Flatten nested result into single row with LLM_ prefixes."""
    row = {"doc_id": doc_id}

    # Add human health relevance flag and confidence
    is_relevant = result.get("human_health_relevant", False)
    row["LLM_HUMAN_HEALTH_RELEVANT"] = 1 if is_relevant else 0
    row["LLM_CONFIDENCE"] = result.get("confidence", "low")

    # Study types (detection, exposure_assessment) - always include
    for study_type, val in result.get("study_types", {}).items():
        col = f"LLM_STUDY_{study_type.upper()}"
        row[col] = 1 if val else 0

    # Mechanisms
    for mech, val in result.get("mechanisms", {}).items():
        col = f"LLM_MECH_{mech.upper()}"
        row[col] = 1 if val else 0

    # Organs
    for organ, val in result.get("organs", {}).items():
        col = f"LLM_ORGAN_{organ.upper()}"
        row[col] = 1 if val else 0

    # Models
    for model, val in result.get("models", {}).items():
        col = f"LLM_MODEL_{model.upper()}"
        row[col] = 1 if val else 0

    return row


def main():
    # Check for API key
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: Set ANTHROPIC_API_KEY environment variable")
        print("  export ANTHROPIC_API_KEY='your-key-here'")
        return

    client = Anthropic(api_key=api_key)

    # Load data - accept command line argument or use default
    data_dir = Path(__file__).parent.parent / "data"
    if len(sys.argv) > 1:
        csv_path = Path(sys.argv[1])
        if not csv_path.is_absolute():
            csv_path = data_dir / csv_path
    else:
        csv_path = data_dir / "papers_for_classification.csv"

    print(f"Loading {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} papers")

    # Classify each paper
    results = []
    total = len(df)

    print(f"\nClassifying {total} papers with Claude Sonnet...")
    print("Estimated cost: ~$1.35")
    print("=" * 60)

    # Progress checkpoint file
    checkpoint_path = data_dir / "llm_checkpoint.csv"

    for idx, row in df.iterrows():
        # Support both column naming conventions
        doc_id = row.get("CORE_PROJECT_NUM", row.get("doc_id", f"row_{idx}"))
        title = row.get("PROJECT_TITLE", row.get("title", ""))
        abstract = row.get("ABSTRACT_TEXT", row.get("abstract", ""))

        # Truncate abstract if too long
        if len(str(abstract)) > 3000:
            abstract = str(abstract)[:3000] + "..."

        print(f"[{idx+1}/{total}] {str(title)[:50]}...", flush=True)

        classification = classify_paper(client, title, abstract)
        flat_row = flatten_result(classification, doc_id)
        results.append(flat_row)

        # Rate limiting
        if (idx + 1) % 10 == 0:
            time.sleep(0.5)

        # Save checkpoint every 30 entries
        if (idx + 1) % 30 == 0:
            checkpoint_df = pd.DataFrame(results)
            checkpoint_df.to_csv(checkpoint_path, index=False)
            print(f"\n>>> CHECKPOINT: Saved {idx+1}/{total} entries to {checkpoint_path}", flush=True)
            print(f">>> Health relevant so far: {checkpoint_df['LLM_HUMAN_HEALTH_RELEVANT'].sum()}/{len(checkpoint_df)}", flush=True)
            print("=" * 60, flush=True)

    # Create results dataframe
    results_df = pd.DataFrame(results)

    # Save standalone classifications
    llm_path = data_dir / "llm_full_classifications.csv"
    results_df.to_csv(llm_path, index=False)
    print(f"\nSaved LLM classifications to {llm_path}")

    # Merge with original data
    df_merged = df.merge(results_df, on="doc_id", how="left")
    df_merged.to_csv(csv_path, index=False)
    print(f"Updated {csv_path} with LLM columns")

    # Print summary
    print("\n" + "=" * 60)
    print("CLASSIFICATION SUMMARY")
    print("=" * 60)

    print("\nMechanisms:")
    for col in sorted([c for c in results_df.columns if c.startswith("LLM_MECH_")]):
        print(f"  {col}: {results_df[col].sum()}")

    print("\nOrgans:")
    for col in sorted([c for c in results_df.columns if c.startswith("LLM_ORGAN_")]):
        print(f"  {col}: {results_df[col].sum()}")

    print("\nModels:")
    for col in sorted([c for c in results_df.columns if c.startswith("LLM_MODEL_")]):
        print(f"  {col}: {results_df[col].sum()}")

    print("\nDone!")


if __name__ == "__main__":
    main()
