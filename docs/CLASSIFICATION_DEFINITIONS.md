# Classification Definitions

This document consolidates all category definitions used for classifying microplastics/nanoplastics research.

---

## Model Organisms

Categories for the experimental system or study population used.

| Category | Definition | Key Patterns |
|----------|------------|--------------|
| **MODEL_INVITRO** | Cell-based experiments including immortalized cell lines, primary cells, organoids, spheroids, and 3D tissue cultures. Studies conducted outside a living organism. | `in vitro`, `cell line`, `cell culture`, `organoid`, `HepG2`, `Caco-2`, `HEK293`, `A549`, `HeLa` |
| **MODEL_RODENT** | Mouse or rat studies with active experimental use (not just citations). Requires title match, 2+ mentions in abstract, OR active use language indicating the grant actually uses rodents. | `mouse`, `mice`, `murine`, `rat`, `rodent`, `C57BL`, `BALB`, `Sprague-Dawley`, `Wistar` |
| **MODEL_ZEBRAFISH** | Zebrafish (Danio rerio) studies at any life stage. Common for developmental toxicity and high-throughput screening. | `zebrafish`, `Danio rerio` |
| **MODEL_HUMAN** | Studies involving human subjects, human biospecimens (blood, urine, stool, placenta, tissue), epidemiological analyses, or clinical trials. Includes biomonitoring and population-based studies. | `human subject`, `cohort study`, `epidemiol`, `clinical trial`, `NHANES`, `biomonitoring`, `patient sample` |
| **MODEL_ENVIRONMENTAL** | Field sampling and environmental monitoring studies. Collecting/analyzing water, sediment, soil, or air samples to characterize plastic contamination. | `environmental sample`, `field sampling`, `water sample`, `sediment sample`, `marine sampling` |
| **MODEL_OTHER_ANIMAL** | Non-rodent, non-zebrafish animal models including invertebrates (C. elegans, Drosophila) and other vertebrates (Xenopus, rabbits, pigs, primates). | `C. elegans`, `Drosophila`, `Xenopus`, `rabbit model`, `primate` |

**Source**: `scripts/update_model_organisms.py`

---

## Organ Systems

Categories for the target organ or physiological system studied.

| Category | Definition | Title Patterns | Abstract Patterns |
|----------|------------|----------------|-------------------|
| **ORGAN_BRAIN_NERVOUS** | Effects on the central or peripheral nervous system. Includes neurotoxicity, neuroinflammation, blood-brain barrier disruption, cognitive impairment, neurodegeneration. | `brain`, `neuro`, `nervous`, `CNS`, `cerebr`, `cognitive` | `neurotoxic`, `neurodegenerat`, `blood-brain barrier`, `cognitive impair` |
| **ORGAN_CARDIOVASCULAR** | Effects on heart and blood vessels. Includes cardiotoxicity, atherosclerosis, endothelial dysfunction, hypertension, thrombosis. | `cardiovasc`, `cardiac`, `heart`, `vascular`, `coronary` | `cardiotoxic`, `atheroscler`, `endothelial`, `hypertension` |
| **ORGAN_GI_GUT** | Effects on the gastrointestinal tract. Includes gut barrier integrity, intestinal inflammation, microbiome disruption, colitis. | `gut`, `intestin`, `gastro`, `colon`, `bowel`, `digest` | `gut barrier`, `microbiome`, `IBD`, `IBS`, `colitis` |
| **ORGAN_RESPIRATORY** | Effects on lungs and airways. Includes pulmonary toxicity, fibrosis, airway inflammation, COPD, asthma. Note: "inhalation" only counts if in title. | `lung`, `pulmonary`, `respiratory`, `airway`, `alveolar`, `inhal` | `pulmonary toxicity`, `COPD`, `asthma`, `bronchitis` |
| **ORGAN_REPRODUCTIVE** | Effects on male or female reproductive systems. Includes ovarian/testicular toxicity, fertility, sperm quality, placental effects, fetal development. Excludes "reproduce/reproducibility", "embryonic day", "pregnane". | `reproduct(ive\|ion)`, `fertil`, `ovari`, `testes`, `placent`, `pregnan`, `fetal` | `sperm`, `oocyte`, `infertil`, `endometri` |
| **ORGAN_LIVER** | Effects on hepatic function. Includes hepatotoxicity, steatosis (fatty liver), fibrosis, NAFLD/NASH. | `liver`, `hepat` | `hepatotoxic`, `NAFLD`, `NASH`, `cirrhosis`, `fatty liver` |
| **ORGAN_KIDNEY** | Effects on renal function. Includes nephrotoxicity, glomerular damage, proteinuria, chronic kidney disease. | `kidney`, `renal`, `nephro` | `nephrotoxic`, `CKD`, `glomerul`, `proteinuria` |
| **ORGAN_IMMUNE** | Effects on immune system function. Includes immunotoxicity, altered immune cell populations, immunosuppression, autoimmune responses. Requires core immune terms, not just "inflammation". | `immune`, `immuno`, `lymph`, `macrophage`, `T cell`, `B cell` | `immunotoxic`, `NLRP3`, `inflammasome`, `autoimmun` |
| **ORGAN_ENDOCRINE** | Effects on hormone-producing glands. Includes thyroid disruption, adrenal effects, insulin resistance, hormone level alterations. | `endocrine`, `thyroid`, `hormone`, `adrenal` | `endocrine disrupt`, `pituitary`, `insulin resist` |

**Classification Logic**: Match if term appears in TITLE, OR if pattern appears >=2 times in ABSTRACT (>=1 for ORGAN_IMMUNE).

**Source**: `scripts/update_organ_systems.py`

---

## Mechanisms of Toxicity

Categories for the biological mechanism by which plastic particles cause harm. These are the LLM-compatible definitions used in `scripts/llm_classify_mechanisms.py`.

### MECH_OXIDATIVE (Oxidative Stress)
Oxidative stress and mitochondrial dysfunction research examines how micro/nanoplastics generate reactive oxygen species (ROS), deplete cellular antioxidants (glutathione, SOD, catalase), damage mitochondrial membranes, disrupt electron transport chains, and cause lipid peroxidation.

**Key markers**: MDA (malondialdehyde), 8-OHdG, protein carbonyls, Nrf2/ARE pathway, mitochondrial membrane potential, ATP production, fluorescent ROS probes.

### MECH_INFLAMMATION (Inflammatory Response)
Inflammatory response research studies how plastic particles trigger pro-inflammatory cytokine release (IL-1β, IL-6, IL-8, TNF-α), activate the NLRP3 inflammasome, induce NF-κB signaling, or cause tissue inflammation.

**Key markers**: IL-1β, IL-6, TNF-α, NF-κB, NLRP3, macrophage/monocyte activation, neutrophil infiltration. Must be investigating inflammation as a DIRECT OUTCOME, not just background mention.

### MECH_BARRIER (Barrier Disruption)
Barrier disruption research investigates how microplastics compromise biological barriers: intestinal/gut epithelial barrier (increased permeability, "leaky gut"), blood-brain barrier (BBB), pulmonary/alveolar barrier, or placental barrier.

**Key markers**: Tight junction proteins (ZO-1, occludin, claudins), TEER (transepithelial electrical resistance), paracellular permeability assays (FITC-dextran), particle translocation studies.

### MECH_MICROBIOME (Gut Microbiome)
Gut microbiome research examines how ingested micro/nanoplastics alter intestinal microbial communities: dysbiosis, changes in bacterial diversity (alpha/beta diversity), shifts in Firmicutes/Bacteroidetes ratios, effects on beneficial bacteria.

**Key markers**: 16S rRNA sequencing, metagenomics, short-chain fatty acids, bile acids, gut-brain axis, Lactobacillus, Bifidobacterium populations.

### MECH_ENDOCRINE (Endocrine Disruption)
Endocrine disruption research studies how plastics or their chemical additives interfere with hormonal systems: estrogenic/anti-estrogenic activity, androgenic effects, thyroid hormone disruption (T3/T4, TSH), reproductive hormone effects.

**Key markers**: Estrogen receptor binding, phthalates, BPA/BPS/BPF, plasticizers, testosterone, progesterone, FSH, LH, thyroid function tests.

### MECH_NEURODEGENERATION (Neurotoxicity/Neurodegeneration)
Neurotoxicity and neurodegeneration research examines how micro/nanoplastics affect the brain and nervous system: cognitive impairment, memory deficits, behavioral changes, links to Alzheimer's or Parkinson's disease.

**Key markers**: Amyloid-β, tau protein, α-synuclein, microglial activation, dopaminergic neurons, neurotransmitter changes. Must study BRAIN/NERVOUS SYSTEM as primary target.

### MECH_IMMUNE (Immune Dysfunction)
Immune system dysfunction research studies how plastics impair immune function beyond acute inflammation: immunotoxicity, immunosuppression, altered lymphocyte function, changes in antibody production.

**Key markers**: T-cell/B-cell function, NK cell activity, dendritic cell maturation, macrophage polarization (M1/M2), thymus/spleen effects, autoimmune responses.

### MECH_DNA_DAMAGE (Genotoxicity)
DNA damage and genotoxicity research examines whether micro/nanoplastics cause genetic damage: DNA strand breaks, chromosomal aberrations, micronucleus formation, mutations, or genomic instability.

**Key markers**: Comet assay, 8-OHdG (oxidative DNA damage), DNA adducts, micronucleus assay, chromosomal aberrations, mutagenicity testing.

### MECH_RECEPTOR (Receptor/Signaling Pathways)
Receptor and signaling pathway research studies how plastic particles or their leachates interact with cellular receptors and signaling cascades.

**Key markers**: AhR (aryl hydrocarbon receptor), PXR (pregnane X receptor), PPARs, Toll-like receptors (TLRs), MAPK, PI3K/Akt, Wnt pathways, calcium signaling.

### MECH_CELL_DEATH (Cell Death/Senescence)
Cell death and senescence research examines how plastic exposure triggers programmed cell death or cellular aging.

**Key markers**: Apoptosis (caspase activation, Bcl-2 family, TUNEL), necrosis/necroptosis, pyroptosis, ferroptosis, autophagy (LC3, p62, Beclin-1), senescence (p16, p21, SA-β-gal, SASP).

---

## Classification Methods

### Regex-Based (Fast, Less Accurate)
- Used in `paperclip_extract_papers.py` for initial classification
- Searches for keyword patterns in title + abstract
- Prone to false negatives (misses synonyms, paraphrases)
- Good for high-prevalence mechanisms (oxidative stress)

### LLM-Based (Slower, More Accurate)
- Used in `llm_classify_mechanisms.py`
- Provides full definitions to Claude Sonnet
- Asks if the paper DIRECTLY INVESTIGATES each mechanism
- ~$0.70 per 200 papers
- Better for nuanced distinctions

---

## Validation Notes (April 2026)

From Paperclip paper extraction (198 papers, 2025-2026):

### Known False Negative Issues (Regex)
| Category | Regex Count | Likely Actual | Issue |
|----------|-------------|---------------|-------|
| ORGAN_KIDNEY | 0 | 5+ | Patterns too strict |
| ORGAN_ENDOCRINE | 0 | 5+ | Patterns too strict |
| MECH_BARRIER | 4 | 8+ | Missing simple "barrier" |
| MECH_IMMUNE | 6 | 11+ | Missing "immune" alone |
| MODEL_HUMAN | 14 | 19+ | Missing "human health" |

**Recommendation**: Use LLM classification for mechanisms when accuracy matters.

---

## Source Files

| Script | Purpose |
|--------|---------|
| `scripts/llm_classify_mechanisms.py` | LLM-based mechanism classification with full definitions |
| `scripts/update_organ_systems.py` | Regex-based organ system classification |
| `scripts/update_model_organisms.py` | Regex-based model organism classification |
| `scripts/paperclip_extract_papers.py` | Combined extraction and classification for Paperclip papers |

---

*Last updated: April 2026*
