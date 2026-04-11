# Microplastics Research Trendspotter

An interactive tool for exploring research trends in emerging scientific fields, demonstrated here with microplastics/nanoplastics research. The app helps researchers discover what's being studied, who's studying it, and—critically—find experts in adjacent fields whose methods may be transferable.

**Live demo**: [microplasticschemexpexplorer.streamlit.app](https://microplasticschemexpexplorer.streamlit.app)

---

## Purpose

Microplastics research is a rapidly evolving field where researchers often work in silos. This tool addresses two problems:

1. **Trend Discovery**: What mechanisms, organ systems, and model organisms are currently being studied? Where are the gaps?

2. **Cross-Field Insights**: Researchers studying other pollutants (heavy metals, PFAS, pesticides, air pollution) have decades of validated methods, model organisms, and mechanistic insights. This tool helps microplastics researchers find those experts—potentially saving years of method development.

---

## Features

| Tab | Description |
|-----|-------------|
| **Projects** | Browse and search 350+ microplastics studies (NIH grants, conference abstracts, papers) with filters for mechanisms, organs, models |
| **Organ Systems** | Visual breakdown of which organ systems are being studied |
| **Model Organisms** | Distribution of in vitro, rodent, zebrafish, human, and environmental models |
| **Mechanisms** | Toxicity mechanisms under investigation (inflammation, neurodegeneration, barrier disruption, etc.) |
| **Cross-Field Insights** | Search 3,000+ NIH-funded researchers studying similar problems with other pollutants |

---

## Data Pipeline

This section documents how the data was collected and processed, so others can replicate this approach for different research domains.

### Step 1: Collect Raw Grant Data from NIH Reporter

**Source**: [NIH Reporter](https://reporter.nih.gov/)

For microplastics grants:
- Search terms: `microplastic*`, `nanoplastic*`, `plastic particle*`, `polystyrene particle*`
- Export all matching grants as CSV

For cross-field comparison (chemical exposures):
- Broader search covering: heavy metals, PFAS, pesticides, phthalates/BPA, air pollution, PAHs/dioxins/PCBs, flame retardants, solvents, nitrates
- This yielded ~2,400 grants for comparison

### Step 1b: Collect Recent Papers from Paperclip

**Source**: [Paperclip](https://paperclip.ai/) (indexes bioRxiv, medRxiv, PMC)

For recent preprints and published papers (2025-2026):
- Search terms: `microplastic*`, `nanoplastic*`
- Used SQL query to extract ~200 papers with abstracts
- See `scripts/paperclip_extract_papers.py` for extraction process
- See `docs/PAPERCLIP_EXTRACTION_PROCESS.md` for detailed methodology

### Step 2: Filter and Clean Data

Raw NIH Reporter exports contain many false positives. We developed iterative regex-based filtering to identify grants that genuinely study the target pollutant (vs. those that merely mention it in passing).

Key filtering logic:
- Title must contain target terms, OR
- Abstract must contain target terms in research context (not just background mentions)
- Exclude grants where the pollutant is only mentioned as a comparator

### Step 3: Classify by Research Dimensions

Each grant was classified across multiple dimensions using regex pattern matching:

**Exposure Types** (`EXP_*` columns):
- Heavy metals, air pollution, PFAS, pesticides, phthalates/BPA, solvents, PAHs/dioxins/PCBs, flame retardants, microplastics, nitrates

**Mechanisms of Toxicity** (`MECH_*` columns):
- Neurodegeneration, inflammation, oxidative stress/mitochondrial dysfunction, endocrine disruption, microbiome effects, immune dysfunction, DNA damage, epigenetic changes, receptor signaling, cell death/senescence, barrier disruption

**Organ Systems** (`ORGAN_*` columns):
- Brain/nervous, GI/gut, respiratory, cardiovascular, reproductive, liver, kidney, immune, skin, endocrine

**Model Systems** (`MODEL_*` columns):
- In vitro, rodent, zebrafish, other animal, human, environmental

### Step 4: Enhance with LLM Classification

Initial regex patterns had limitations—they couldn't distinguish between grants that *study* a mechanism vs. those that merely *mention* it. We used Claude (Anthropic's LLM) to re-classify mechanisms with better accuracy.

See `scripts/llm_classify_mechanisms.py` for the implementation:
- Cost: ~$0.70 for 204 grants using Claude Sonnet
- Each grant's title and abstract are analyzed
- LLM determines if the grant *directly investigates* each mechanism

The LLM classifications are stored in `LLM_MECH_*` columns and used by the app.

### Step 5: Build the Streamlit App

The app (`app.py`) provides:
- Interactive filtering and search
- Visualizations (bar charts, pie charts) of research distribution
- Paginated tables with CSV export
- Cross-field expert discovery based on shared research categories

---

## Data Files

| File | Description |
|------|-------------|
| `data/microplastic_grants_cleaned.csv` | Main dataset: NIH microplastics grants with all classifications |
| `data/chemical_exposure_grants_filtered.csv` | Cross-field dataset: 2,400+ grants on other chemical exposures |
| `data/conference_abstracts.csv` | Abstracts from 1st Micro/Nanoplastics & Human Health Conference (Jan 2026, NM) |
| `data/llm_mechanism_classifications.csv` | LLM-generated mechanism classifications |

---

## Scripts

| Script | Purpose |
|--------|---------|
| `scripts/llm_classify_mechanisms.py` | Classify grants by mechanism using Claude LLM |
| `scripts/llm_classify_all.py` | Full LLM classification pipeline for papers (health relevance, mechanisms, organs, models) |
| `scripts/paperclip_extract_papers.py` | Extract microplastics papers from Paperclip (bioRxiv, medRxiv, PMC) |
| `scripts/update_organ_systems.py` | Update organ system classifications |
| `scripts/update_model_organisms.py` | Update model organism classifications |

---

## Reproducing for Another Field

This approach can be adapted for any emerging research area. Here's how:

### 1. Define Your Domain
- What search terms capture your field? (e.g., for PFAS: `PFAS`, `perfluor*`, `PFOA`, `PFOS`)
- What adjacent fields have transferable expertise?

### 2. Collect Data from NIH Reporter
- Use the [NIH Reporter API](https://api.reporter.nih.gov/) or web export
- Cast a wide net initially; you'll filter later

### 3. Develop Classification Taxonomy
- What mechanisms are relevant to your field?
- What organ systems or endpoints matter?
- What model systems are used?

### 4. Iterate on Filtering
- Start with simple regex patterns
- Review false positives/negatives
- Refine patterns or use LLM classification for nuanced distinctions

### 5. Build Cross-Field Comparisons
- Identify related research areas
- Collect grants from those areas using similar methods
- Map shared classification dimensions

### 6. Deploy
- Fork this repo
- Replace data files with your domain's data
- Update classification definitions in `app.py`
- Deploy on Streamlit Cloud

---

## Local Development

```bash
# Clone the repo
git clone https://github.com/SIDaniels/Microplastics_chem_exp_explorer.git
cd Microplastics_chem_exp_explorer

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

---

## Deploy on Streamlit Cloud

1. Fork this repo
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub account
4. Select this repo, branch `main`, file `app.py`
5. Deploy

---

## Technical Notes

### Cross-Field Insights: Relevance Scoring Algorithm

The Cross-Field Insights tab helps researchers find experts studying similar problems with other pollutants. When a user selects a research category (e.g., "Gut Microbiome" or "Oxidative Stress"), the app scores and ranks 2,400+ grants from other pollutant fields based on research similarity.

**How it works:**

1. **Analyze the source field**: The algorithm first examines all microplastics grants in the selected category to identify:
   - Which mechanisms are commonly studied (>10% prevalence)
   - Which model systems are used (>15% prevalence)
   - Which organ systems are targeted (>10% prevalence)
   - Which molecular pathways/subthemes appear (>8% prevalence)

2. **Score each target grant**: Every grant from the cross-field dataset (heavy metals, PFAS, pesticides, etc.) is scored based on overlap with the source field:

   | Feature | Points | Rationale |
   |---------|--------|-----------|
   | Keyword match (user search) | +10 | Explicit user intent |
   | Selected category match | +5 | Direct research alignment |
   | Mechanism match | +3 each (max 2) | Mechanisms often translate across chemicals |
   | Model system match | +2 | Methods/protocols are transferable |
   | Subtheme/pathway match | +2 each (max 2) | Deep mechanistic alignment |
   | Organ system overlap | +1 each | Secondary relevance |

3. **Display results**: Grants are sorted by relevance score, with matching features shown (e.g., "★Gut Microbiome, Model: Rodent, Inflammation").

**Example**: If a microplastics researcher is studying gut microbiome effects using rodent models, they'd find high-scoring matches among PFAS or pesticide researchers who also study microbiome disruption in rodents—even though the pollutant is different.

### Classification Approach Evolution

We initially attempted pure regex-based classification for mechanisms of toxicity. This worked reasonably well for binary detection (does the grant mention inflammation?) but struggled with nuance (does the grant *study* inflammation as a primary outcome, or just mention it as background?).

**Key lesson: Abstracts are challenging to query with regex.** Scientific abstracts use varied terminology, abbreviations, and sentence structures. A grant studying "ROS-mediated mitochondrial dysfunction" might not match a simple "oxidative stress" regex. Manual validation revealed many false negatives.

The LLM-based approach (`scripts/llm_classify_mechanisms.py`) resolved this by:
1. Providing detailed definitions of each mechanism
2. Asking the LLM to determine if the grant *directly investigates* each mechanism
3. Allowing for "other mechanisms" discovery

### LLM-Generated Category Summaries

The expandable "What are microplastics researchers studying?" summaries in Cross-Field Insights were generated using Claude. For each research category, the LLM analyzed the microplastics grants in that category and wrote a human-readable summary of current research themes, common approaches, and knowledge gaps. These summaries help users understand the landscape before exploring cross-field experts.

### Manual Validation

After automated classification, we performed manual spot-checks to verify accuracy:
- Reviewed a sample of grants in each category to confirm appropriate classification
- Identified edge cases where regex patterns failed (informing the move to LLM classification)
- Validated that cross-field matches were genuinely relevant (not just keyword coincidences)

### Experimental: Pollutant Analog Predictions

We experimented with using Claude to predict which pollutants might serve as useful analogs for specific types of microplastics (e.g., would heavy metals research inform nanoplastic neurotoxicity studies?). This work is preserved in `data/mnp_analog_predictions_v3.xlsx` but was not incorporated into the final app.

---

## Acknowledgments

- Data sourced from [NIH Reporter](https://reporter.nih.gov/)
- Recent papers extracted from [Paperclip](https://paperclip.ai/) (bioRxiv, medRxiv, PMC)
- Conference abstracts from the [1st Micro/Nanoplastics & Human Health Conference](https://hsc.unm.edu/pharmacy/research/areas/cmbm/mnp-conf/) (January 2026, New Mexico)
- Built with [Streamlit](https://streamlit.io/)
- LLM classification powered by [Anthropic Claude](https://www.anthropic.com/)
- Developed as part of [Engineered Resilience](https://www.engineeredresilience.org/)

---

## License

MIT License - see LICENSE file for details.

---

## Contact

Questions or feedback? [Contact us](https://www.engineeredresilience.org/contact)
