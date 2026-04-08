#!/usr/bin/env python3
"""
NIH Chemical Exposure Grant Explorer
Streamlit app for exploring environmental toxicology grants.

Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import altair as alt
import re
from pathlib import Path
import anthropic
from datetime import datetime, timedelta

# Engineered Resilience color palette
ER_COLORS = {
    'dark_teal': '#0D3B3C',
    'soft_teal': '#46B3A9',
    'gold': '#D4A84B',
    'green': '#5FA872',
    'orange': '#D97706',
    'bg': '#FAFAF8',
}

# Gradient palette for charts
ER_GRADIENT = ['#0D3B3C', '#1a5455', '#2d7170', '#46B3A9', '#5FA872', '#D4A84B']

# ============== CHATBOT FUNCTIONS ==============

# Rate limiting settings
MAX_QUESTIONS_PER_SESSION = 20
RATE_LIMIT_WINDOW_HOURS = 1

def init_chat_state():
    """Initialize chat session state."""
    if 'chat_messages' not in st.session_state:
        st.session_state.chat_messages = []
    if 'question_count' not in st.session_state:
        st.session_state.question_count = 0
    if 'rate_limit_reset' not in st.session_state:
        st.session_state.rate_limit_reset = datetime.now()

def check_rate_limit() -> tuple[bool, str]:
    """Check if user is within rate limits. Returns (allowed, message)."""
    now = datetime.now()

    # Reset counter if window has passed
    if now > st.session_state.rate_limit_reset + timedelta(hours=RATE_LIMIT_WINDOW_HOURS):
        st.session_state.question_count = 0
        st.session_state.rate_limit_reset = now

    if st.session_state.question_count >= MAX_QUESTIONS_PER_SESSION:
        mins_left = int((st.session_state.rate_limit_reset + timedelta(hours=RATE_LIMIT_WINDOW_HOURS) - now).seconds / 60)
        return False, f"Rate limit reached ({MAX_QUESTIONS_PER_SESSION} questions/hour). Try again in {mins_left} minutes."

    return True, ""

def search_grants_for_chat(df: pd.DataFrame, query: str, max_results: int = 20) -> pd.DataFrame:
    """Search grants relevant to a chat query using keyword extraction."""
    # Extract meaningful keywords (skip common words)
    stop_words = {'who', 'what', 'where', 'when', 'why', 'how', 'is', 'are', 'the', 'a', 'an',
                  'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from',
                  'about', 'into', 'through', 'during', 'before', 'after', 'above', 'below',
                  'between', 'research', 'researching', 'studying', 'studies', 'study',
                  'working', 'work', 'doing', 'does', 'did', 'have', 'has', 'been', 'being',
                  'there', 'their', 'they', 'them', 'this', 'that', 'these', 'those',
                  'can', 'could', 'would', 'should', 'may', 'might', 'must', 'will',
                  'tell', 'me', 'show', 'find', 'look', 'looking', 'any', 'some'}

    # Extract keywords from query
    words = re.findall(r'\b[a-zA-Z]{3,}\b', query.lower())
    keywords = [w for w in words if w not in stop_words]

    if not keywords:
        # Fall back to full query if no keywords extracted
        keywords = [query.lower()]

    # Create combined text for searching
    text = (df['PROJECT_TITLE'].fillna('') + ' ' +
            df['ABSTRACT_TEXT'].fillna('') + ' ' +
            df['PI_NAMEs'].fillna('') + ' ' +
            df['ORG_NAME'].fillna('')).str.lower()

    # Score each grant by how many keywords it matches
    df = df.copy()
    df['_match_score'] = 0
    for keyword in keywords:
        df['_match_score'] += text.str.contains(keyword, regex=False).astype(int)

    # Filter to grants that match at least one keyword, sort by score
    results = df[df['_match_score'] > 0].sort_values('_match_score', ascending=False).head(max_results)
    results = results.drop(columns=['_match_score'])

    return results

def format_grants_for_context(grants_df: pd.DataFrame) -> str:
    """Format grants data as context for the AI."""
    if len(grants_df) == 0:
        return "No relevant grants found in the database."

    context_parts = []
    for _, grant in grants_df.iterrows():
        title = grant.get('PROJECT_TITLE', 'Unknown')
        pi = grant.get('PI_NAMEs', 'Unknown')
        org = grant.get('ORG_NAME', 'Unknown')
        fy = grant.get('FISCAL_YEAR', 'N/A')
        abstract = str(grant.get('ABSTRACT_TEXT', ''))[:800]  # More context

        context_parts.append(f"**{title}** (FY{fy})\nPI: {pi} | Institution: {org}\nAbstract: {abstract}")

    return f"Found {len(grants_df)} relevant grants:\n\n" + "\n\n---\n\n".join(context_parts)

def get_chat_response(query: str, df: pd.DataFrame) -> str:
    """Get AI response for a chat query using Claude."""
    try:
        # Check for API key
        api_key = st.secrets.get("ANTHROPIC_API_KEY", None)
        if not api_key:
            return "⚠️ Chat is not configured. Please add ANTHROPIC_API_KEY to Streamlit secrets."

        client = anthropic.Anthropic(api_key=api_key)

        # Search for relevant grants
        relevant_grants = search_grants_for_chat(df, query)
        context = format_grants_for_context(relevant_grants)

        # Build the prompt
        system_prompt = """You are a research assistant for the Microplastics & Chemical Exposure Grant Explorer database.
You help users discover NIH-funded research grants and conference abstracts.

IMPORTANT INSTRUCTIONS:
- You will be given a list of grants from the database that match the user's query
- ALWAYS reference specific grants, PIs, and institutions from the provided data
- List the most relevant researchers and their work with specific details
- If grants are provided, summarize what research is being done and by whom
- Only say "no relevant research found" if the grants list is truly empty
- Be specific: mention PI names, institutions, and grant titles"""

        user_prompt = f"""User question: {query}

DATABASE RESULTS:
{context}

Based on these grants from our database, answer the user's question. Cite specific PIs, institutions, and grant titles."""

        response = client.messages.create(
            model="claude-3-5-haiku-latest",
            max_tokens=500,
            system=system_prompt,
            messages=[
                {"role": "user", "content": user_prompt}
            ]
        )

        return response.content[0].text

    except Exception as e:
        return f"⚠️ Error: {str(e)}"


def create_horizontal_bar_chart(data: dict, title: str = "", value_label: str = "Grants") -> alt.Chart:
    """Create a styled horizontal bar chart with ER colors, sorted by count descending."""
    # Sort by value descending before creating DataFrame
    sorted_data = sorted(data.items(), key=lambda x: x[1], reverse=True)
    df = pd.DataFrame([
        {'category': k, 'value': v} for k, v in sorted_data
    ])

    chart = alt.Chart(df).mark_bar(
        cornerRadiusTopRight=4,
        cornerRadiusBottomRight=4,
    ).encode(
        x=alt.X('value:Q',
                title=value_label,
                axis=alt.Axis(grid=True, gridColor='#e0e0e0')),
        y=alt.Y('category:N',
                sort='-x',
                title=None,
                axis=alt.Axis(labelLimit=200)),
        color=alt.Color('value:Q',
                       scale=alt.Scale(range=[ER_COLORS['soft_teal'], ER_COLORS['dark_teal']]),
                       legend=None),
        tooltip=[
            alt.Tooltip('category:N', title='Category'),
            alt.Tooltip('value:Q', title=value_label, format=',')
        ]
    ).properties(
        height=max(len(data) * 28, 150),
    ).configure_axis(
        labelFontSize=12,
        titleFontSize=13,
        labelColor=ER_COLORS['dark_teal'],
        titleColor=ER_COLORS['dark_teal'],
    ).configure_view(
        strokeWidth=0
    )

    return chart


def create_donut_chart(data: dict, title: str = "") -> alt.Chart:
    """Create a styled donut chart with ER colors."""
    df = pd.DataFrame([
        {'category': k, 'value': v} for k, v in data.items()
    ])

    chart = alt.Chart(df).mark_arc(innerRadius=50, outerRadius=90).encode(
        theta=alt.Theta('value:Q'),
        color=alt.Color('category:N',
                       scale=alt.Scale(range=ER_GRADIENT),
                       legend=alt.Legend(title=None, orient='right')),
        tooltip=[
            alt.Tooltip('category:N', title='Category'),
            alt.Tooltip('value:Q', title='Grants', format=',')
        ]
    ).properties(
        height=220,
        width=300,
    )

    return chart


def clean_pi_names(pi_str):
    """Remove '(contact)' annotations from PI names."""
    if pd.isna(pi_str) or not isinstance(pi_str, str):
        return pi_str
    return re.sub(r'\s*\(contact\)', '', pi_str, flags=re.IGNORECASE)


# Page config
st.set_page_config(
    page_title="Chemical Exposure Grant Explorer | Engineered Resilience",
    page_icon="🔬",
    layout="wide"
)

# Custom CSS for Engineered Resilience branding
st.markdown("""
<style>
    /* Import fonts */
    @import url('https://fonts.googleapis.com/css2?family=Source+Sans+Pro:wght@400;600;700&family=Spectral:wght@500;600;700&display=swap');

    /* Color palette from engineeredresilience.org */
    :root {
        --er-dark-teal: #0D3B3C;
        --er-soft-teal: #46B3A9;
        --er-gold: #D4A84B;
        --er-bg: #FAFAF8;
        --er-green: #5FA872;
        --er-orange: #D97706;
    }

    /* Main background */
    .stApp {
        background-color: #FAFAF8;
    }

    /* Headers - serif font like site */
    h1, h2, h3 {
        font-family: 'Spectral', Georgia, serif !important;
        color: #0D3B3C !important;
    }

    /* Body text */
    .stMarkdown, .stText, p, span, label {
        font-family: 'Source Sans Pro', -apple-system, sans-serif !important;
    }

    /* Primary buttons */
    .stButton > button {
        background-color: #46B3A9 !important;
        color: white !important;
        border: none !important;
        border-radius: 6px !important;
        font-family: 'Source Sans Pro', sans-serif !important;
        font-weight: 600 !important;
    }

    .stButton > button:hover {
        background-color: #0D3B3C !important;
    }

    /* STOMP pill buttons - unselected state */
    [data-testid="stHorizontalBlock"] .stButton > button[kind="secondary"] {
        background-color: #f0f2f1 !important;
        color: #0D3B3C !important;
        border: none !important;
        border-radius: 20px !important;
        padding: 10px 16px !important;
        font-weight: 600 !important;
        transition: all 0.2s ease !important;
    }

    [data-testid="stHorizontalBlock"] .stButton > button[kind="secondary"]:hover {
        background-color: rgba(70, 179, 169, 0.2) !important;
        color: #0D3B3C !important;
    }

    /* Download button */
    .stDownloadButton > button {
        background-color: #D4A84B !important;
        color: #0D3B3C !important;
        border: none !important;
        font-weight: 600 !important;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #0D3B3C !important;
    }

    [data-testid="stSidebar"] .stMarkdown,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] .stText,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] div {
        color: #FAFAF8 !important;
    }

    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #D4A84B !important;
    }

    /* Sidebar selectbox/multiselect styling for better contrast */
    [data-testid="stSidebar"] [data-baseweb="select"] {
        background-color: #FAFAF8 !important;
    }

    [data-testid="stSidebar"] [data-baseweb="select"] span,
    [data-testid="stSidebar"] [data-baseweb="select"] div {
        color: #0D3B3C !important;
    }

    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stMultiSelect label {
        color: #FAFAF8 !important;
        font-weight: 600 !important;
    }

    /* Slider labels in sidebar */
    [data-testid="stSidebar"] .stSlider label {
        color: #FAFAF8 !important;
        font-weight: 600 !important;
    }

    [data-testid="stSidebar"] .stSlider [data-testid="stTickBarMin"],
    [data-testid="stSidebar"] .stSlider [data-testid="stTickBarMax"] {
        color: #FAFAF8 !important;
    }

    /* Metrics in sidebar */
    [data-testid="stSidebar"] [data-testid="stMetricValue"] {
        color: #46B3A9 !important;
        font-size: 1.8rem !important;
    }

    /* Tabs - refined styling matching engineeredresilience.org */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background-color: #f0f2f1;
        padding: 6px;
        border-radius: 10px;
        border: none;
    }

    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        color: #0D3B3C;
        border-radius: 8px;
        font-family: 'Source Sans Pro', sans-serif;
        font-weight: 600;
        font-size: 0.95rem;
        padding: 10px 20px;
        border: none;
        transition: all 0.2s ease;
    }

    .stTabs [data-baseweb="tab"]:hover {
        background-color: rgba(70, 179, 169, 0.15);
    }

    .stTabs [aria-selected="true"] {
        background-color: #0D3B3C !important;
        color: white !important;
        box-shadow: 0 2px 8px rgba(13, 59, 60, 0.25);
    }

    /* Remove the default bottom border/highlight */
    .stTabs [data-baseweb="tab-highlight"] {
        display: none;
    }

    .stTabs [data-baseweb="tab-border"] {
        display: none;
    }

    /* DataFrames */
    .stDataFrame {
        border-radius: 8px;
        overflow: hidden;
    }

    /* Multiselect tags */
    [data-baseweb="tag"] {
        background-color: #46B3A9 !important;
    }

    /* Info boxes */
    .stAlert {
        background-color: rgba(70, 179, 169, 0.1) !important;
        border-left-color: #46B3A9 !important;
    }

    /* Charts - teal color scheme */
    .stBarChart, .stLineChart {
        border-radius: 8px;
    }

    /* Card-like sections */
    .stExpander {
        background-color: white;
        border-radius: 8px;
        border: 1px solid rgba(13, 59, 60, 0.1);
    }

    /* Footer styling */
    .footer-text {
        color: #0D3B3C;
        font-size: 0.85rem;
        opacity: 0.7;
    }
</style>
""", unsafe_allow_html=True)

# Data path - use pre-filtered file (6,500 chemical exposure grants)
DATA_PATH = Path(__file__).parent / 'data' / 'chemical_exposure_grants_filtered.csv'

# Exposure categories
EXPOSURES = {
    'EXP_HEAVY_METALS': 'Heavy Metals (Lead, Mercury, etc.)',
    'EXP_AIR_POLLUTION': 'Air Pollution & Particulate Matter',
    'EXP_PFAS': 'PFAS (Forever Chemicals)',
    'EXP_PESTICIDES': 'Pesticides & Herbicides',
    'EXP_PHTHALATES_BPA': 'Phthalates & BPA',
    'EXP_SOLVENTS': 'Industrial Solvents',
    'EXP_PAHS_DIOXINS_PCBS': 'PAHs, Dioxins & PCBs',
    'EXP_FLAME_RETARDANTS': 'Flame Retardants',
    'EXP_MICROPLASTICS': 'Microplastics & Nanoplastics',
    'EXP_NITRATES': 'Nitrates & Nitrites',
}

# Mechanism categories (excluding Inflammation - poor classification accuracy)
MECHANISMS = {
    'MECH_NEURODEGENERATION': 'Neurodegeneration',
    'MECH_OXIDATIVE_MITOCHONDRIAL': 'Oxidative Stress / Mitochondrial',
    'MECH_ENDOCRINE': 'Endocrine Disruption',
    'MECH_MICROBIOME': 'Microbiome / Gut-Brain',
    'MECH_IMMUNE_DYSFUNCTION': 'Immune Dysfunction',
    'MECH_DNA_DAMAGE': 'DNA Damage / Genotoxicity',
    'MECH_RECEPTOR_SIGNALING': 'Receptor Signaling',
    'MECH_SENESCENCE_CELL_DEATH': 'Senescence / Cell Death',
    'MECH_BARRIER_DISRUPTION': 'Barrier Disruption (BBB, Gut)',
}

# Non-mechanism TYPE categories (for conference abstracts without mechanism focus)
# Note: These are separate from RESEARCH_TYPES which are regex-based
CONF_TYPE_CATEGORIES = {
    'TYPE_METHODS': 'Methods / Detection',
    'TYPE_ACCUMULATION': 'Tissue Accumulation',
    'TYPE_ENVIRONMENTAL': 'Environmental Studies',
    'TYPE_EXPOSURE': 'Exposure Assessment',
}

# Regex patterns for TYPE_ categories (used for dynamic classification in Cross-Field Insights)
TYPE_PATTERNS = {
    'TYPE_METHODS': (
        r'method\s+(development|validation)|assay\s+(development|optimization)|'
        r'detection\s+(method|limit|technique)|quantif\w+\s+(method|assay)|'
        r'analytical\s+(method|technique)|sensor\s+(development|design)|'
        r'biomarker\s+(discovery|development|validation)|spectro\w+\s+analys|'
        r'mass\s+spectro|HPLC|chromatograph|immunoassay'
    ),
    'TYPE_ACCUMULATION': (
        r'tissue\s+(accumul|distribution|concentration|level)|'
        r'biodistrib|organ\s+distribution|cellular\s+uptake|'
        r'bioaccumul|body\s+burden|tissue\s+burden|'
        r'uptake\s+(in|by)|accumul\w+\s+in\s+(tissue|organ)'
    ),
    'TYPE_ENVIRONMENTAL': (
        r'environmental\s+(monitoring|sampling|fate|transport|contamination)|'
        r'water\s+quality|soil\s+contamination|air\s+quality|sediment|'
        r'ecosystem|ecological\s+(risk|assessment)|wildlife|'
        r'environ\w+\s+occurrence|ambient\s+(concentration|level)'
    ),
    'TYPE_EXPOSURE': (
        r'exposure\s+(assessment|pathway|route|scenario)|'
        r'biomonitor|human\s+biomonitoring|exposure\s+biomarker|'
        r'dietary\s+exposure|occupational\s+exposure|residential\s+exposure|'
        r'inhalation\s+exposure|dermal\s+exposure|aggregate\s+exposure'
    ),
}

# Combined mechanisms + types for Cross-Field Insights dropdown
MECHANISMS_AND_TYPES = {**MECHANISMS, **CONF_TYPE_CATEGORIES}


# Common research themes to detect in abstracts
THEMES = {
    'gut_microbiome': r'gut|intestin|microbiome|gastrointestin|colon',
    'reproductive': r'reproduct|pregnan|fetal|maternal|placent|utero',
    'cardiovascular': r'cardiovasc|heart|vascul|atheroscler|blood\s+vessel',
    'neurotoxicity': r'neuro|brain|cognitive|nervous\s+system|BBB',
    'cancer': r'cancer|tumor|carcinogen|oncogen|malignant',
    'immune': r'immun|inflamm|cytokine|macrophage|T\s+cell',
    'developmental': r'develop|child|pediatric|prenatal|postnatal',
    'epigenetic': r'epigenet|methylat|histone|chromatin|transgener',
    'oxidative': r'oxidative|ROS|antioxidant|free\s+radical|mitochondr',
    'exposure_assessment': r'exposure\s+assess|biomonitor|biomarker|quantif',
}

# Granular sub-themes for similarity matching (more specific than THEMES)
# Model systems - separate category for filtering
MODEL_SYSTEMS = {
    'In Vitro (Cells)': r'in\s+vitro|cell\s+line|cell\s+culture|primary\s+cell',
    'Animal (Rodent)': r'mouse|mice|rodent|murine|rat\b|animal\s+model',
    'Animal (Zebrafish)': r'zebrafish|danio',
    'Animal (Other)': r'drosophila|c\.\s*elegans|xenopus|rabbit|pig|primate',
    'Human (Cohort/Epi)': r'cohort|epidemiol|NHANES|human\s+subject|population.based|cross.sectional',
    'Human (Clinical)': r'clinical\s+trial|patient|clinical\s+study|human\s+volunteer',
}

SUBTHEMES = {
    # Target tissues/organs
    'liver_hepatic': r'liver|hepat|hepatocyte',
    'gut_intestinal': r'gut|intestin|GI\s+tract|colon|bowel',
    'adipose_metabolic': r'adipos|metabolic|obesity|lipid|fatty',
    'lung_respiratory': r'lung|pulmonary|airway|respiratory',
    'kidney_renal': r'kidney|renal|nephro',
    'brain_neuro': r'brain|neuro|cognitive|CNS',
    'cardiovascular': r'cardiovasc|cardiac|heart|vascular',
    'reproductive': r'reproduct|ovary|testes|fertility|placent',
    # Pathways/mechanisms
    'nfkb_pathway': r'NF.?κB|NFkB|NF-kB',
    'nlrp3_inflammasome': r'NLRP3|inflammasome',
    'cytokines': r'IL-\d|TNF|interleukin|cytokine',
    'oxidative_ros': r'oxidative|ROS|reactive\s+oxygen',
    'mitochondrial': r'mitochondr',
    'apoptosis': r'apoptos|cell\s+death|caspase',
    'fibrosis': r'fibros|fibrotic|collagen\s+deposit',
    # Cell types
    'macrophage': r'macrophage',
    't_cell_lymphocyte': r'T.?cell|lymphocyte|CD4|CD8',
    'epithelial': r'epithelial|epithelium',
    'stem_cell': r'stem\s+cell|progenitor',
    # Life stages
    'developmental': r'develop|fetal|prenatal|postnatal|child|pediatric',
    'aging': r'aging|aged|elderly|senescen',
    # Conditions
    'diabetes_metabolic': r'diabet|insulin|glucose|metabolic\s+syndrome',
    'autoimmune': r'autoimmun|lupus|rheumatoid',
    'cancer': r'cancer|tumor|carcinogen|oncogen',
}

# ========== STOMP-INSPIRED CATEGORIZATIONS ==========

# 1. Target Organ Systems
# --------------------------------------------------------------------------
# ACCURACY LOG (April 6, 2026):
# Problem: Original broad patterns matched ~65% of grants (e.g., "brain" alone
#   catches incidental mentions, "blood" catches sample collection contexts)
# Solution: Compound patterns requiring toxicity/damage/function/disease context
# Expected: Reduce false positives by ~40-50% while retaining true organ targets
# --------------------------------------------------------------------------
ORGAN_SYSTEMS = {
    'ORGAN_BRAIN': ('Brain/Neurological',
        r'neurotoxic|neurodegenerat\w*\s+(disease|disorder|effect)|'
        r'brain\s+(damage|injury|toxicity|lesion)|CNS\s+(toxicity|damage)|'
        r'cerebr\w+\s+(damage|toxicity|lesion)|blood.brain\s+barrier\s+(disrupt|damage|permeab)|'
        r'BBB\s+(disrupt|damage|permeab)|cognitive\s+(impair|deficit|decline)\s+\w*\s*(expos|toxic|pollut)?|'
        r'neuronal\s+(death|damage|loss|toxicity)|neurobehavior\w*\s+(effect|deficit|toxicity)|'
        r'encephalopathy|developmental\s+neurotoxic|'
        r'synap\w+\s+(dysfunction|damage|loss|toxicity)|'
        r'neurotransmit\w*\s+(disrupt|imbalance|dysfunction)|'
        r'dopamin\w+\s+(neuron\s+damage|system\s+dysfunction|depletion)|'
        r'neuro\w*\s+(inflammation|damage|impairment)|'
        r'nervous\s+system\s+(toxicity|damage|effect)'),
    'ORGAN_CARDIOVASCULAR': ('Cardiovascular',
        r'cardiotoxic|cardiovascular\s+(disease|risk|effect|health)|heart\s+(disease|failure|damage)|'
        r'cardiac\s+(function|toxicity|dysfunction|arrest)|atheroscler|coronary\s+(artery|disease)|'
        r'vascular\s+(damage|dysfunction|disease)|endothelial\s+(dysfunction|damage)|'
        r'hypertension|arrhythmia|myocardial|stroke\s+(risk|outcome)|'
        r'blood\s+pressure|cardiomyopathy|thrombosis|aneurysm'),
    'ORGAN_GI': ('Gastrointestinal',
        r'gut\s+(barrier|health|toxicity|microbiome|dysbiosis)|intestin\w*\s+(damage|inflammation|barrier|permeab)|'
        r'GI\s+(toxicity|tract)|gastro\w+\s+(disease|toxicity)|'
        r'colitis|bowel\s+disease|digest\w+\s+(disorder|dysfunction)|'
        r'microbiome\s+(disruption|composition|health)|fecal\s+microb|'
        r'IBD\b|IBS\b|crohn|celiac|leaky\s+gut|enteric'),
    'ORGAN_LIVER': ('Liver/Hepatic',
        r'hepatotoxic|liver\s+(damage|disease|injury|fibrosis|cancer|function)|hepat\w+\s+(dysfunction|injury)|'
        r'cirrhosis|fatty\s+liver|hepatocellular|biliary\s+(damage|disease)|'
        r'NAFLD|NASH|liver\s+enzyme|ALT\s+level|AST\s+level|bilirubin'),
    'ORGAN_KIDNEY': ('Kidney/Renal',
        r'nephrotoxic|kidney\s+(damage|disease|injury|failure|function)|renal\s+(toxicity|dysfunction|failure)|'
        r'glomerul\w+\s+(disease|damage|filtration)|chronic\s+kidney|CKD\b|'
        r'proteinuria|albuminuria|tubular\s+(damage|injury)|dialysis'),
    'ORGAN_LUNG': ('Lung/Respiratory',
        r'pulmonary\s+(toxicity|fibrosis|disease|function)|lung\s+(damage|disease|injury|cancer|function)|'
        r'respiratory\s+(toxicity|disease|dysfunction|health)|airway\s+(inflammation|disease|remodel)|'
        r'alveolar\s+(damage|macrophage)|bronchitis|COPD|asthma|'
        r'inhala\w+\s+(toxicity|exposure)|pneumon|emphysema|'
        r'particulate\s+matter|PM2\.5|air\s+pollution\s+(exposure|health)'),
    'ORGAN_REPRODUCTIVE': ('Reproductive',
        r'reproduct\w+\s+(toxicity|dysfunction|health|outcome)|ovarian\s+(toxicity|function|reserve)|'
        r'test\w+\s+(damage|toxicity|function)|placent\w+\s*(toxicity|dysfunction|barrier|transfer)|'
        r'sperm\s+(damage|quality|count|motility)|fertil\w+\s+(impair|disorder)|'
        r'endometriosis|pregnancy\s+(outcome|complication|loss)|'
        r'fetal\s+(development|growth|exposure)|preterm\s+birth|'
        r'menstrual|uterine|endocrine\s+disrupt'),
    'ORGAN_IMMUNE': ('Immune System',
        r'immunotoxic|immune\s+(dysfunction|suppression|disorder|response)|'
        r'lymph\w+\s+(toxicity|dysfunction)|spleen\s+toxicity|'
        r'bone\s+marrow\s+(suppression|toxicity)|autoimmun|'
        r'inflamm\w+\s+(response|marker|cytokine)|cytokine\s+(storm|release)|'
        r'T\s*cell|B\s*cell\s+(function|response)|macrophage\s+(activation|function)|'
        r'allerg\w+\s+(response|sensitization)|hypersensitiv'),
    'ORGAN_SKIN': ('Skin/Dermal',
        r'dermal\s+toxicity|skin\s+toxicity|cutaneous\s+toxicity|'
        r'skin\s+(damage|sensitization|irritation|barrier\s+dysfunction|disease|disorder)|'
        r'epidermal\s+(damage|barrier|toxicity)|dermatitis|contact\s+dermatitis|'
        r'keratinocyte\s+(toxicity|damage)|melanocyte\s+(toxicity|damage)|'
        r'skin\s+carcinoma|skin\s+melanoma|skin\s+cancer|'
        r'atopic\s+dermatitis|psoriasis|eczema|'
        r'skin\s+sensitiz|skin\s+allerg|skin\s+inflammat'),
}

# 1b. Mechanism Pattern Classification
# --------------------------------------------------------------------------
# ACCURACY LOG (April 7, 2026):
# Problem: Pre-classified MECH_* columns use loose patterns (e.g., standalone
#   "alzheimer" matches any mention, not toxicology-focused research)
# Solution: Tightened patterns that require mechanism to be a FOCUS of research
#   or require toxicology/exposure context for disease-related mechanisms
# Expected: Reduce false positives significantly while retaining true mechanism studies
# --------------------------------------------------------------------------
MECHANISM_SYSTEMS = {
    'MECH_NEURODEGENERATION': ('Neurodegeneration',
        # TIGHTENED: Require toxicology/exposure context for disease names
        r'(?:expos\w*|toxic\w*|pollut\w*|contamin\w*|chemical).{0,50}(?:alzheimer|parkinson|dementia|neurodegenerat)|'
        r'(?:alzheimer|parkinson|dementia|neurodegenerat).{0,50}(?:expos\w*|toxic\w*|pollut\w*|contamin\w*|chemical)|'
        r'neurodegenerat.{0,30}(?:mechanism|pathway|toxicity)|'
        r'(?:study|investigat|examin).{0,30}neurodegenerat|'
        r'amyloid.{0,30}(?:toxicity|aggregat|pathology)|'
        r'alpha.?synuclein.{0,30}(?:aggregat|toxicity)|'
        r'dopamin\w+.{0,20}(?:neuron.{0,10}(?:loss|death|damage)|degenerat)|'
        r'tauopathy|lewy\s+bod'),
    'MECH_INFLAMMATION': ('Inflammation',
        # Require inflammation to be a research focus, not just mentioned
        r'(?:aim|goal|objective).{0,50}inflamm|'
        r'inflamm.{0,50}(?:aim|goal|objective|mechanism)|'
        r'(?:study|investigat|examin|characteriz).{0,30}inflammat|'
        r'inflammat.{0,30}(?:pathway|mechanism|response|signaling)|'
        r'(?:role|effect|impact).{0,20}(?:of|on).{0,20}inflammat|'
        r'inflammat.{0,20}(?:induc|mediat|driven|caused)|'
        r'pro.?inflammat|anti.?inflammat|inflammasome|nlrp3'),
    'MECH_OXIDATIVE_MITOCHONDRIAL': ('Oxidative Stress / Mitochondrial',
        r'(?:aim|goal|objective).{0,50}oxidative|'
        r'(?:study|investigat|examin).{0,30}oxidative\s+stress|'
        r'oxidative\s+stress.{0,30}(?:pathway|mechanism|role)|'
        r'(?:role|effect|impact).{0,20}(?:of|on).{0,20}oxidative|'
        r'mitochondri.{0,20}(?:dysfunction|damage|toxicity|impair)|'
        r'(?:study|investigat).{0,30}mitochondri|'
        r'\bros\b.{0,30}(?:product|generat|level|measur)|'
        r'reactive\s+oxygen\s+species|lipid\s+peroxidation'),
    'MECH_EPIGENETIC': ('Epigenetic',
        r'(?:aim|goal|objective).{0,50}epigenetic|'
        r'(?:study|investigat|examin).{0,30}epigenetic|'
        r'epigenetic.{0,30}(?:mechanism|change|effect|modific)|'
        r'dna\s+methylation.{0,30}(?:pattern|change|level|analys)|'
        r'histone.{0,20}(?:modific|acetyl|methyl)|'
        r'chromatin.{0,20}(?:remodel|modific|structure)|'
        r'(?:mirna|microrna).{0,30}(?:expression|regulat|target)|'
        r'transgenerational.{0,20}(?:effect|inherit|transmis)|'
        r'epigenetic.{0,20}inherit'),
    'MECH_ENDOCRINE': ('Endocrine Disruption',
        r'endocrine\s+disrupt|'
        r'(?:study|investigat|examin).{0,30}hormone|'
        r'hormone.{0,30}(?:disrupt|dysfunction|imbalance)|'
        r'estrogen\s+receptor.{0,30}(?:activ|bind|signal)|'
        r'androgen\s+receptor.{0,30}(?:activ|bind|signal)|'
        r'(?:anti.?estrogen|anti.?androgen)|'
        r'thyroid.{0,30}(?:disrupt|dysfunction|hormone)|'
        r'steroidogenesis|hormone.{0,20}(?:level|production|synthesis)'),
    'MECH_MICROBIOME': ('Microbiome / Gut-Brain',
        r'(?:aim|goal|objective).{0,50}microbiome|'
        r'(?:study|investigat|examin).{0,30}(?:microbiome|microbiota)|'
        r'(?:microbiome|microbiota).{0,30}(?:composition|change|alter|disrupt)|'
        r'dysbiosis|gut.?brain.{0,20}(?:axis|connect|commun)|'
        r'(?:gut|intestinal).{0,20}(?:bacteria|microb).{0,20}(?:composition|change)|'
        r'fecal\s+microb'),
    'MECH_IMMUNE_DYSFUNCTION': ('Immune Dysfunction',
        r'(?:aim|goal|objective).{0,50}immune|'
        r'(?:study|investigat|examin).{0,30}immune.{0,20}(?:function|response|system)|'
        r'immune.{0,20}(?:dysfunction|suppress|deficien|impair)|'
        r'immunotoxic|autoimmun|'
        r't.?cell.{0,20}(?:function|response|activ|exhaust)|'
        r'(?:antibody|immunoglobulin).{0,20}(?:production|response|level)'),
    'MECH_DNA_DAMAGE': ('DNA Damage / Genotoxicity',
        r'(?:aim|goal|objective).{0,50}dna\s+damage|'
        r'(?:study|investigat|examin).{0,30}dna\s+(?:damage|repair)|'
        r'dna\s+damage.{0,30}(?:mechanism|pathway|response)|'
        r'genotoxic|mutagenic|mutagenesis|'
        r'double.?strand\s+break|dna\s+adduct|'
        r'chromosom.{0,15}(?:aberration|damage|instabil)|genome\s+instability'),
    'MECH_SENESCENCE_CELL_DEATH': ('Senescence / Cell Death',
        r'(?:aim|goal|objective).{0,50}senescen|'
        r'(?:study|investigat|examin).{0,30}(?:senescen|apoptos)|'
        r'cellular?\s+senescen|'
        r'(?:apoptosis|apoptotic).{0,30}(?:pathway|mechanism|signal)|'
        r'programmed\s+cell\s+death|(?:pyroptosis|ferroptosis|necroptosis)|'
        r'caspase.{0,20}(?:activ|express|mediat)|sasp\b|'
        r'p16.{0,15}(?:express|induc|positive)'),
    'MECH_BARRIER_DISRUPTION': ('Barrier Disruption (BBB, Gut)',
        r'(?:aim|goal|objective).{0,50}barrier|'
        r'(?:study|investigat|examin).{0,30}barrier|'
        r'barrier.{0,30}(?:function|integrity|permeab|disrupt)|'
        r'blood.?brain\s+barrier.{0,30}(?:permeab|disrupt|integrity|function)|'
        r'bbb\s+(?:permeab|disrupt|integrity)|'
        r'(?:gut|intestinal)\s+barrier.{0,30}(?:function|permeab|integrity)|'
        r'tight\s+junction.{0,20}(?:disrupt|function|integrity)|leaky\s+gut'),
    'MECH_RECEPTOR_SIGNALING': ('Receptor Signaling',
        r'(?:aim|goal|objective).{0,50}receptor|'
        r'(?:study|investigat|examin).{0,30}receptor.{0,20}(?:signal|activ)|'
        r'aryl\s+hydrocarbon\s+receptor|ahr\s+(?:activ|pathway|signal)|'
        r'(?:estrogen|androgen)\s+receptor.{0,30}(?:activ|signal|mediat)|'
        r'ppar.{0,20}(?:activ|pathway|signal|agonist)|'
        r'signal\s+transduction.{0,20}(?:pathway|mechanism)|'
        r'receptor.{0,20}(?:agonist|antagonist|activ|inhibit)'),
}

# 2. Research Type Classification (simple 4-category)
# IMPROVED: Tightened patterns to reduce false positives
# - Removed overly broad terms: mouse, rat, signal, molecular, therapy, treatment
# - Added compound patterns requiring context
RESEARCH_TYPES = {
    'TYPE_EPIDEMIOLOGY': ('Epidemiology/Population',
        r'epidemiol|cohort\s+study|population.based|cross.sectional\s+study|'
        r'longitudinal\s+study|case.control|NHANES|birth\s+cohort|prospective\s+study|'
        r'odds\s+ratio|relative\s+risk|hazard\s+ratio|incidence|prevalence\s+study'),
    'TYPE_MECHANISTIC': ('Mechanistic/Basic',
        r'mechanis\w+\s+(of|underlying|study|pathway)|pathway\s+(analysis|activation)|'
        r'in\s+vitro\s+(study|model|assay)|cell\s+line\s+(model|study)|'
        r'animal\s+model\s+of|murine\s+model|rodent\s+model|'
        r'signaling\s+(pathway|cascade|mechanism)|molecular\s+(mechanism|basis|pathway)'),
    'TYPE_CLINICAL': ('Clinical/Human',
        r'clinical\s+trial|phase\s+[I123]\s+trial|human\s+subject|human\s+participant|'
        r'randomized\s+(control|trial)|RCT\b|placebo.controlled|patient\s+population|'
        r'therapeutic\s+intervention|clinical\s+outcome'),
    'TYPE_METHODS': ('Methods/Detection',
        r'method\s+(development|validation)|assay\s+(development|optimization)|'
        r'detection\s+(method|limit|technique)|quantif\w+\s+(method|assay)|'
        r'analytical\s+(method|technique)|sensor\s+(development|design)|'
        r'biomarker\s+(discovery|development|validation)'),
}

# 3. Research Phase Classification (original STOMP-inspired 5-category)
# --------------------------------------------------------------------------
# ACCURACY LOG (April 6, 2026):
# Note: These patterns are intentionally broader than Research Types above
# to capture the research PHASE (what stage of investigation) vs TYPE (methodology)
# Some overlap is expected - a grant can have multiple phases
# --------------------------------------------------------------------------
RESEARCH_PHASES = {
    'PHASE_DETECTION': ('Detection/Methods',
        r'detect\w+\s+(method|limit|technique)|method\s+(development|validation)|'
        r'assay\s+(development|optimization)|quantif\w+\s+(method|technique)|'
        r'analytic\w+\s+(method|technique)|sensor\s+(development|design)|'
        r'spectro\w+\s+analys|measure\w+\s+(method|technique)|biomonitor'),
    'PHASE_BIODISTRIBUTION': ('Biodistribution',
        r'biodistrib|tissue\s+accumul|tissue\s+distribution|organ\s+distribution|'
        r'cellular\s+uptake|tissue\s+uptake|intracellular\s+traffick'),
    'PHASE_MECHANISM': ('Mechanism Studies',
        r'mechanis\w+\s+(of|underlying|study)|pathway\s+analys|'
        r'molecular\s+(mechanism|basis)|signaling\s+pathway|receptor\s+mediat'),
    'PHASE_HEALTH_OUTCOME': ('Health Outcomes',
        r'disease\s+(outcome|association|risk)|health\s+outcome|'
        r'mortality\s+(rate|risk)|morbidity|risk\s+factor|'
        r'epidemiol\w+\s+(study|association)|cohort\s+study'),
    'PHASE_INTERVENTION': ('Intervention/Treatment',
        r'intervent\w+\s+(study|strategy)|treatment\s+(strategy|efficacy)|'
        r'therap\w+\s+(approach|intervent)|remediat|removal\s+strateg|'
        r'mitigation\s+strateg|prevent\w+\s+(strategy|measure)'),
}



def classify_stomp_categories(df: pd.DataFrame, deduplicate: bool = True) -> dict:
    """Classify grants by STOMP-inspired categories.

    Uses pre-classified ORGAN_*/TYPE_* columns if available (for conference abstracts),
    otherwise falls back to keyword pattern matching on abstract text.

    Args:
        df: DataFrame with grant data
        deduplicate: If True, count unique project titles only (default True)
    """
    if len(df) == 0:
        return {'organs': {}, 'phases': {}, 'research_types': {}}

    # Deduplicate by PROJECT_TITLE if requested
    if deduplicate:
        df_unique = df.drop_duplicates(subset=['PROJECT_TITLE'], keep='first')
    else:
        df_unique = df

    text = df_unique['PROJECT_TITLE'].fillna('') + ' ' + df_unique['ABSTRACT_TEXT'].fillna('')
    n_projects = len(df_unique)

    results = {'organs': {}, 'phases': {}, 'research_types': {}}

    # Mapping from ORGAN_SYSTEMS keys to pre-classified column names
    organ_col_map = {
        'ORGAN_BRAIN': 'ORGAN_BRAIN_NERVOUS',
        'ORGAN_CARDIOVASCULAR': 'ORGAN_CARDIOVASCULAR',
        'ORGAN_GI': 'ORGAN_GI_GUT',
        'ORGAN_LIVER': 'ORGAN_LIVER',
        'ORGAN_KIDNEY': 'ORGAN_KIDNEY',
        'ORGAN_LUNG': 'ORGAN_RESPIRATORY',
        'ORGAN_REPRODUCTIVE': 'ORGAN_REPRODUCTIVE',
        'ORGAN_IMMUNE': 'ORGAN_IMMUNE',
        'ORGAN_SKIN': 'ORGAN_SKIN',
    }

    # Organ systems - use pre-classified columns OR keyword matching
    for key, (label, pattern) in ORGAN_SYSTEMS.items():
        col_name = organ_col_map.get(key)
        if col_name and col_name in df_unique.columns:
            # Use pre-classified column OR keyword match
            preclassified = df_unique[col_name] == 1
            keyword_match = text.str.contains(pattern, regex=True, flags=re.IGNORECASE, na=False)
            matches = preclassified | keyword_match
        else:
            matches = text.str.contains(pattern, regex=True, flags=re.IGNORECASE, na=False)
        count = matches.sum()
        pct = 100 * count / n_projects if n_projects > 0 else 0
        results['organs'][label] = {'count': int(count), 'pct': round(pct, 1)}

    # Research phases (5-category, STOMP-inspired) - keyword only
    for key, (label, pattern) in RESEARCH_PHASES.items():
        matches = text.str.contains(pattern, regex=True, flags=re.IGNORECASE, na=False)
        count = matches.sum()
        pct = 100 * count / n_projects if n_projects > 0 else 0
        results['phases'][label] = {'count': int(count), 'pct': round(pct, 1)}

    # Mapping for research types
    type_col_map = {
        'TYPE_EPIDEMIOLOGY': None,  # No direct match
        'TYPE_MECHANISTIC': 'TYPE_MECHANISTIC',
        'TYPE_CLINICAL': None,
        'TYPE_METHODS': 'TYPE_METHODS',
    }

    # Research types - use pre-classified columns OR keyword matching
    for key, (label, pattern) in RESEARCH_TYPES.items():
        col_name = type_col_map.get(key)
        if col_name and col_name in df_unique.columns:
            preclassified = df_unique[col_name] == 1
            keyword_match = text.str.contains(pattern, regex=True, flags=re.IGNORECASE, na=False)
            matches = preclassified | keyword_match
        else:
            matches = text.str.contains(pattern, regex=True, flags=re.IGNORECASE, na=False)
        count = matches.sum()
        pct = 100 * count / n_projects if n_projects > 0 else 0
        results['research_types'][label] = {'count': int(count), 'pct': round(pct, 1)}

    return results


def extract_themes_from_abstracts(df: pd.DataFrame, n_grants: int = 50) -> dict:
    """Extract common themes from abstracts using keyword patterns."""
    if len(df) == 0:
        return {}

    # Sample grants for efficiency
    sample = df.head(min(n_grants, len(df)))
    text = sample['PROJECT_TITLE'].fillna('') + ' ' + sample['ABSTRACT_TEXT'].fillna('')

    theme_counts = {}
    for theme_name, pattern in THEMES.items():
        matches = text.str.contains(pattern, regex=True, flags=re.IGNORECASE, na=False)
        count = matches.sum()
        if count > 0:
            pct = 100 * count / len(sample)
            theme_counts[theme_name] = {'count': count, 'pct': pct}

    # Sort by count
    return dict(sorted(theme_counts.items(), key=lambda x: x[1]['count'], reverse=True))


def detect_model_system(text: str) -> str:
    """Detect the primary model system used in a grant."""
    for model_name, pattern in MODEL_SYSTEMS.items():
        if re.search(pattern, text, re.IGNORECASE):
            return model_name
    return 'Unknown'


def compute_grant_similarity(source_grants: pd.DataFrame, target_grants: pd.DataFrame,
                             selected_category: str = None, keyword_filter: str = None) -> pd.DataFrame:
    """
    Compute similarity scores between source grants (your field) and target grants (other fields).

    Weighting system (researcher-informed):
    - Selected category match: +5 (highest priority - what user explicitly asked for)
    - Mechanism match: +3 (mechanisms often translate across chemicals)
    - Model system match: +2 (methods/approaches are transferable)
    - Other organ overlap: +1 (secondary relevance)
    - Subtheme/pathway match: +2 (deep mechanistic alignment)
    - Keyword match: +10 (explicit user search)
    """
    if len(source_grants) == 0 or len(target_grants) == 0:
        return target_grants.assign(similarity_score=0, matching_features='', model_system='Unknown')

    # Get text for both sets
    source_text = source_grants['PROJECT_TITLE'].fillna('') + ' ' + source_grants['ABSTRACT_TEXT'].fillna('')
    target_text = target_grants['PROJECT_TITLE'].fillna('') + ' ' + target_grants['ABSTRACT_TEXT'].fillna('')

    # Find dominant model systems in source grants (>15% prevalence)
    source_models = []
    for model_name, pattern in MODEL_SYSTEMS.items():
        matches = source_text.str.contains(pattern, regex=True, flags=re.IGNORECASE, na=False)
        pct = 100 * matches.sum() / len(source_text)
        if pct >= 15:
            source_models.append(model_name)

    # Find which organ systems are common in source grants (>10% prevalence)
    source_organs = []
    for key, (label, pattern) in ORGAN_SYSTEMS.items():
        matches = source_text.str.contains(pattern, regex=True, flags=re.IGNORECASE, na=False)
        pct = 100 * matches.sum() / len(source_text)
        if pct >= 10:
            source_organs.append((key, label, pattern))

    # Find which mechanisms are common in source grants (>10% prevalence)
    source_mechanisms = []
    for key, (label, pattern) in MECHANISM_SYSTEMS.items():
        matches = source_text.str.contains(pattern, regex=True, flags=re.IGNORECASE, na=False)
        pct = 100 * matches.sum() / len(source_text)
        if pct >= 10:
            source_mechanisms.append((key, label, pattern))

    # Find which subthemes are common in source grants (>8% prevalence)
    source_subthemes = []
    for key, pattern in SUBTHEMES.items():
        matches = source_text.str.contains(pattern, regex=True, flags=re.IGNORECASE, na=False)
        pct = 100 * matches.sum() / len(source_text)
        if pct >= 8:
            source_subthemes.append((key, pattern))

    # Get the pattern for the selected category if provided
    selected_pattern = None
    selected_label = None
    if selected_category:
        # Check if it's an organ system
        if selected_category in ORGAN_SYSTEMS:
            selected_label, selected_pattern = ORGAN_SYSTEMS[selected_category]
        # Check if it's a mechanism
        elif selected_category in MECHANISM_SYSTEMS:
            selected_label, selected_pattern = MECHANISM_SYSTEMS[selected_category]
        # Check mechanisms dict (legacy)
        elif selected_category in MECHANISMS:
            selected_label = MECHANISMS[selected_category]

    # Score each target grant
    scores = []
    matching_features_list = []
    model_systems = []

    for idx, text in target_text.items():
        # Detect model system
        model = detect_model_system(text)
        model_systems.append(model)

        matches = []
        score = 0

        # HIGHEST PRIORITY: Keyword filter match (+10)
        if keyword_filter and keyword_filter.strip():
            keywords = [k.strip() for k in keyword_filter.split(',')]
            for kw in keywords:
                if kw and re.search(re.escape(kw), text, re.IGNORECASE):
                    score += 10
                    matches.insert(0, f"Keyword: {kw}")
                    break  # Only count once

        # HIGH PRIORITY: Selected category match (+5)
        if selected_pattern and re.search(selected_pattern, text, re.IGNORECASE):
            score += 5
            if selected_label and selected_label not in matches:
                matches.append(f"★{selected_label}")

        # MEDIUM PRIORITY: Mechanism matches (+3 each, max 2)
        mech_matches = 0
        for key, label, pattern in source_mechanisms:
            if mech_matches >= 2:
                break
            if re.search(pattern, text, re.IGNORECASE):
                score += 3
                mech_matches += 1
                if label not in matches and f"★{label}" not in matches:
                    matches.append(label)

        # MEDIUM PRIORITY: Model system match (+2)
        if model in source_models:
            score += 2
            matches.append(f"Model: {model}")

        # MEDIUM PRIORITY: Subtheme/pathway matches (+2 each, max 2)
        subtheme_matches = 0
        for key, pattern in source_subthemes:
            if subtheme_matches >= 2:
                break
            if re.search(pattern, text, re.IGNORECASE):
                score += 2
                subtheme_matches += 1
                # Clean up subtheme key for display
                nice_key = key.replace('_', ' ').title()
                if nice_key not in matches:
                    matches.append(nice_key)

        # LOWER PRIORITY: Other organ system matches (+1 each)
        for key, organ_label, organ_pattern in source_organs:
            # Skip if this is the selected category (already counted)
            if selected_category and key == selected_category:
                continue
            if re.search(organ_pattern, text, re.IGNORECASE):
                score += 1
                if organ_label not in matches and f"★{organ_label}" not in matches:
                    matches.append(organ_label)

        scores.append(score)
        matching_features_list.append(', '.join(matches[:5]))

    result = target_grants.copy()
    result['similarity_score'] = scores
    result['matching_features'] = matching_features_list
    result['model_system'] = model_systems

    return result


def generate_dynamic_summary(df: pd.DataFrame, exposures: list, mechanisms: list) -> str:
    """Generate a dynamic summary based on filtered grants."""
    if len(df) == 0:
        return "No grants match current filters."

    lines = []

    # Basic stats
    lines.append(f"**{len(df):,} grants** in current selection")

    # Year distribution
    if 'FISCAL_YEAR' in df.columns:
        year_counts = df['FISCAL_YEAR'].value_counts().sort_index()
        year_str = ", ".join([f"FY{int(y)}: {c}" for y, c in year_counts.items()])
        lines.append(f"Years: {year_str}")

    # Extract themes
    themes = extract_themes_from_abstracts(df)
    if themes:
        top_themes = list(themes.items())[:5]
        theme_str = ", ".join([f"{t.replace('_', ' ').title()} ({d['pct']:.0f}%)"
                               for t, d in top_themes])
        lines.append(f"**Top research themes:** {theme_str}")

    # Mechanism summary for filtered data
    mech_cols = [c for c in MECHANISMS.keys() if c in df.columns]
    if mech_cols:
        mech_counts = {MECHANISMS[m][:20]: int(df[m].sum()) for m in mech_cols}
        mech_counts = {k: v for k, v in mech_counts.items() if v > 0}
        if mech_counts:
            top_mechs = sorted(mech_counts.items(), key=lambda x: x[1], reverse=True)[:3]
            mech_str = ", ".join([f"{m} ({c})" for m, c in top_mechs])
            lines.append(f"**Top mechanisms:** {mech_str}")

    # Exposure summary
    exp_cols = [c for c in EXPOSURES.keys() if c in df.columns]
    if exp_cols:
        exp_counts = {EXPOSURES[e][:20]: int(df[e].sum()) for e in exp_cols}
        exp_counts = {k: v for k, v in exp_counts.items() if v > 0}
        if exp_counts:
            top_exps = sorted(exp_counts.items(), key=lambda x: x[1], reverse=True)[:3]
            exp_str = ", ".join([f"{e} ({c})" for e, c in top_exps])
            lines.append(f"**Top exposures:** {exp_str}")

    return "\n\n".join(lines)


@st.cache_data
def compute_cooccurrence(df: pd.DataFrame) -> dict:
    """Compute co-occurrence statistics between exposures and mechanisms.

    Deduplicates by PROJECT_TITLE to count unique projects only.
    """
    stats = {
        'exp_to_mech': {},  # For each exposure, top mechanisms
        'mech_to_exp': {},  # For each mechanism, top exposures
        'cross_field': {},  # Compare mechanism rates across exposures
    }

    # Deduplicate by PROJECT_TITLE
    df_unique = df.drop_duplicates(subset=['PROJECT_TITLE'], keep='first')

    exp_cols = [c for c in EXPOSURES.keys() if c in df_unique.columns]
    mech_cols = [c for c in MECHANISMS.keys() if c in df_unique.columns]

    # For each exposure, calculate mechanism percentages
    for exp in exp_cols:
        exp_grants = df_unique[df_unique[exp] == 1]
        if len(exp_grants) == 0:
            continue
        mech_stats = []
        for mech in mech_cols:
            count = int(exp_grants[mech].sum())
            pct = 100 * count / len(exp_grants)
            mech_stats.append((mech, count, pct))
        # Sort by percentage descending
        mech_stats.sort(key=lambda x: x[2], reverse=True)
        stats['exp_to_mech'][exp] = mech_stats

    # For each mechanism, calculate exposure percentages
    for mech in mech_cols:
        mech_grants = df_unique[df_unique[mech] == 1]
        if len(mech_grants) == 0:
            continue
        exp_stats = []
        for exp in exp_cols:
            count = int(mech_grants[exp].sum())
            pct = 100 * count / len(mech_grants)
            exp_stats.append((exp, count, pct))
        exp_stats.sort(key=lambda x: x[2], reverse=True)
        stats['mech_to_exp'][mech] = exp_stats

    # Cross-field comparison: for each mechanism, which exposures study it most?
    for mech in mech_cols:
        exp_rates = []
        for exp in exp_cols:
            exp_grants = df_unique[df_unique[exp] == 1]
            if len(exp_grants) > 0:
                rate = 100 * exp_grants[mech].sum() / len(exp_grants)
                exp_rates.append((exp, rate, len(exp_grants)))
        exp_rates.sort(key=lambda x: x[1], reverse=True)
        stats['cross_field'][mech] = exp_rates

    return stats


@st.cache_data
def load_data(_cache_version: str = "v7_conference_2026") -> pd.DataFrame:
    """Load pre-filtered grant data (6,500 chemical exposure grants + conference abstracts)."""
    if not DATA_PATH.exists():
        return pd.DataFrame()

    df = pd.read_csv(DATA_PATH, low_memory=False)

    # Set conference abstracts to fiscal year 2026
    conf_mask = df['CORE_PROJECT_NUM'].astype(str).str.startswith('CONF_')
    df.loc[conf_mask, 'FISCAL_YEAR'] = 2026

    # Create combined text field for searching
    df['_text'] = df['PROJECT_TITLE'].fillna('') + ' ' + df['ABSTRACT_TEXT'].fillna('')

    # Reset index to avoid alignment issues
    df = df.reset_index(drop=True)

    return df


def filter_grants(df: pd.DataFrame, exposures: list, mechanisms: list,
                  keyword: str, years: list, source: str = "All Sources") -> pd.DataFrame:
    """Filter grants by exposure, mechanism, keyword, year, and source."""
    mask = pd.Series([True] * len(df), index=df.index)

    # Filter by source (NIH Grants vs Conference Abstracts)
    if source == "NIH Grants Only":
        # NIH grants have project numbers that don't start with CONF_
        mask &= ~df['CORE_PROJECT_NUM'].astype(str).str.startswith('CONF_')
    elif source == "Conference Abstracts Only":
        # Conference abstracts have project numbers starting with CONF_
        mask &= df['CORE_PROJECT_NUM'].astype(str).str.startswith('CONF_')

    # Filter by year (include conference abstracts with NaN fiscal year)
    if years and 'FISCAL_YEAR' in df.columns:
        mask &= (df['FISCAL_YEAR'].isin(years) | df['FISCAL_YEAR'].isna())

    # Filter by exposures (OR logic - any selected exposure)
    if exposures:
        existing = [c for c in exposures if c in df.columns]
        if existing:
            mask &= df[existing].max(axis=1) > 0

    # Filter by mechanisms (OR logic - any selected mechanism)
    if mechanisms:
        existing = [c for c in mechanisms if c in df.columns]
        if existing:
            mask &= df[existing].max(axis=1) > 0

    # Filter by keyword (regex search in title + abstract)
    if keyword:
        try:
            pattern = re.compile(keyword, re.IGNORECASE)
            mask &= df['_text'].str.contains(pattern, regex=True, na=False)
        except re.error:
            st.error(f"Invalid regex pattern: {keyword}")

    return df[mask]


# ============== MAIN APP ==============

# Load data first to get counts for header
df = load_data()

# Initialize chat state
init_chat_state()

# ============== CHAT INTERFACE ==============
st.markdown("""
<div style="background: linear-gradient(135deg, #0D3B3C 0%, #1a5455 100%); padding: 1.5rem; border-radius: 12px; margin-bottom: 1.5rem;">
    <h3 style="color: #D4A84B !important; margin: 0 0 0.5rem 0; font-family: 'Spectral', serif;">💬 Ask about the research</h3>
    <p style="color: #FAFAF8; opacity: 0.9; margin: 0; font-size: 0.9rem;">
        Ask questions like "Who is studying microplastics and gut health?" or "What research focuses on reproductive effects?"
    </p>
</div>
""", unsafe_allow_html=True)

# Chat input
chat_col1, chat_col2 = st.columns([5, 1])
with chat_col1:
    user_question = st.text_input(
        "Your question",
        placeholder="e.g., Who is researching microplastics in drinking water?",
        label_visibility="collapsed",
        key="chat_input"
    )
with chat_col2:
    ask_button = st.button("Ask", type="primary", use_container_width=True)

# Handle chat submission
if ask_button and user_question:
    allowed, limit_msg = check_rate_limit()
    if not allowed:
        st.warning(limit_msg)
    else:
        st.session_state.question_count += 1
        with st.spinner("Searching grants and thinking..."):
            response = get_chat_response(user_question, df)
            st.session_state.chat_messages.append({"role": "user", "content": user_question})
            st.session_state.chat_messages.append({"role": "assistant", "content": response})

# Display chat history (most recent first, limit to last 3 exchanges)
if st.session_state.chat_messages:
    with st.expander(f"💬 Chat history ({len(st.session_state.chat_messages)//2} questions)", expanded=True):
        # Show messages in reverse order (most recent first)
        messages = st.session_state.chat_messages[-6:]  # Last 3 Q&A pairs
        for i in range(len(messages) - 1, -1, -2):
            if i >= 1:
                # Assistant response
                st.markdown(f"**🤖 Assistant:** {messages[i]['content']}")
                # User question
                st.markdown(f"**You:** {messages[i-1]['content']}")
                st.markdown("---")

        remaining = MAX_QUESTIONS_PER_SESSION - st.session_state.question_count
        st.caption(f"💡 {remaining} questions remaining this session")

st.markdown("<div style='height: 1rem'></div>", unsafe_allow_html=True)

# Count totals for header display
total_entries = len(df)
nih_grants_total = (~df['CORE_PROJECT_NUM'].astype(str).str.startswith('CONF_')).sum() if len(df) > 0 else 0
conf_abstracts_total = (df['CORE_PROJECT_NUM'].astype(str).str.startswith('CONF_')).sum() if len(df) > 0 else 0



if len(df) == 0:
    st.error("No data found. Make sure chemical_exposure_grants.csv exists in the data/ folder.")
    st.stop()

# Sidebar filters with branding
st.sidebar.markdown("""
<div style="text-align: center; padding: 0.5rem 0 1rem 0; border-bottom: 1px solid rgba(250,250,248,0.2); margin-bottom: 1rem;">
    <span style="font-family: 'Spectral', serif; font-size: 1.1rem; color: #D4A84B;">🔬 Grant Explorer</span>
</div>
""", unsafe_allow_html=True)
st.sidebar.header("Filters")

# Source filter (NIH Grants vs Conference Abstracts)
source_options = ["All Sources", "NIH Grants Only", "Conference Abstracts Only"]
selected_source = st.sidebar.radio(
    "Data Source",
    source_options,
    index=0
)

# Year selection - include 2026 for conference abstracts
available_years = sorted(df['FISCAL_YEAR'].dropna().unique().astype(int).tolist())
# Ensure 2026 is included for conference abstracts
if 2026 not in available_years:
    available_years = sorted(available_years + [2026])
selected_years = st.sidebar.multiselect(
    "Fiscal Years",
    available_years,
    default=available_years
)

# Exposure filter - FROZEN to Microplastics only
exp_cols = [c for c in EXPOSURES.keys() if c in df.columns]
selected_exposures = ['EXP_MICROPLASTICS'] if 'EXP_MICROPLASTICS' in exp_cols else []
st.sidebar.markdown("**Chemical Exposure:** Microplastics")

# No mechanism filter - removed per user request
selected_mechanisms = []

# Apply filters
filtered = filter_grants(df, selected_exposures, selected_mechanisms, "", selected_years, selected_source)

# Sidebar stats
st.sidebar.markdown("---")

# Global "Group by project" toggle
group_by_project = st.sidebar.checkbox(
    "Group by project",
    value=True,
    help="Count unique projects instead of individual fiscal year records"
)

# Count NIH grants and conference abstracts
nih_count = (~df['CORE_PROJECT_NUM'].astype(str).str.startswith('CONF_')).sum()
conf_count = (df['CORE_PROJECT_NUM'].astype(str).str.startswith('CONF_')).sum()

# Show counts based on grouping preference
if group_by_project:
    filtered_display = filtered.drop_duplicates(subset=['PROJECT_TITLE'], keep='first')
    st.sidebar.metric("Unique Projects", f"{len(filtered_display):,}")
else:
    st.sidebar.metric("Total Records", f"{len(filtered):,}")

st.sidebar.caption(f"NIH: {nih_count:,} | Conf: {conf_count:,}")

# Compute co-occurrence stats (full dataset for suggestions)
cooccur = compute_cooccurrence(df)

# Compute dynamic stats on filtered data
cooccur_filtered = compute_cooccurrence(filtered) if len(filtered) > 0 else {}

# Related filters section removed - app is focused on Microplastics only

# ============== SUMMARY CARD & ACTIVE FILTERS ==============
# Get deduplicated data for summary
filtered_unique = filtered.drop_duplicates(subset=['PROJECT_TITLE'], keep='first') if group_by_project else filtered

# Active filters display
active_filters = []
if selected_source != "All Sources":
    active_filters.append(selected_source.replace(" Only", ""))
if selected_years and len(selected_years) < len(available_years):
    if len(selected_years) <= 2:
        active_filters.append(f"FY {', '.join(map(str, selected_years))}")
    else:
        active_filters.append(f"FY {min(selected_years)}-{max(selected_years)}")
active_filters.append("Microplastics")  # Always active

# Summary card
st.markdown(f"""
<div style="background: linear-gradient(135deg, #0D3B3C 0%, #1a5455 100%);
            padding: 1.25rem 1.5rem; border-radius: 10px; margin-bottom: 1.5rem;
            display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 1rem;">
    <div>
        <div style="color: #D4A84B; font-family: 'Spectral', serif; font-size: 1.8rem; font-weight: 600;">
            {len(filtered_unique):,}
        </div>
        <div style="color: #FAFAF8; font-size: 0.9rem; opacity: 0.9;">
            {"unique projects" if group_by_project else "total records"}
        </div>
    </div>
    <div style="display: flex; gap: 0.5rem; flex-wrap: wrap;">
        {"".join([f'<span style="background: rgba(70,179,169,0.3); color: #FAFAF8; padding: 0.35rem 0.75rem; border-radius: 20px; font-size: 0.8rem; font-family: Source Sans Pro, sans-serif;">{f}</span>' for f in active_filters])}
    </div>
</div>
""", unsafe_allow_html=True)

# Main content - tabs
tab1, tab4, tab5, tab6 = st.tabs(["Results", "Cross-Field Insights", "STOMP Analysis", "Microplastics Focus"])

with tab1:
    # Use global toggle from sidebar
    show_unique = group_by_project

    if len(filtered) > 0:
        # Calculate tag count for each grant (exposures + mechanisms)
        exp_cols = [c for c in EXPOSURES.keys() if c in filtered.columns]
        mech_cols = [c for c in MECHANISMS.keys() if c in filtered.columns]
        tag_cols = exp_cols + mech_cols

        # Sort by total tag count (most prevalent first)
        filtered_sorted = filtered.copy()
        if tag_cols:
            filtered_sorted['_tag_count'] = filtered_sorted[tag_cols].sum(axis=1)
            filtered_sorted = filtered_sorted.sort_values('_tag_count', ascending=False)

        # Add Source column to identify NIH vs Conference entries
        filtered_sorted['Source'] = filtered_sorted['CORE_PROJECT_NUM'].apply(
            lambda x: 'Conference' if str(x).startswith('CONF_') else 'NIH Grant'
        )

        # Group by project if requested
        if show_unique and 'CORE_PROJECT_NUM' in filtered_sorted.columns:
            # Aggregate by CORE_PROJECT_NUM - show year range instead of single year
            agg_dict = {
                'PROJECT_TITLE': 'first',
                'ORG_NAME': 'first',
                'PI_NAMEs': 'first',
                'FISCAL_YEAR': lambda x: f"{int(x.min())}-{int(x.max())}" if x.min() != x.max() else str(int(x.min())),
                'Source': 'first',
                '_tag_count': 'max',
                'ABSTRACT_TEXT': lambda x: max(x.fillna(''), key=len) if len(x) > 0 else ''
            }
            # Also include tag columns for display
            for col in tag_cols:
                if col in filtered_sorted.columns:
                    agg_dict[col] = 'max'
            grouped = filtered_sorted.groupby('CORE_PROJECT_NUM').agg(agg_dict).reset_index()
            grouped = grouped.sort_values('_tag_count', ascending=False)
            grouped = grouped.rename(columns={'FISCAL_YEAR': 'Years'})

            display_df = grouped
            display_cols = ['Source', 'Years', 'PROJECT_TITLE', 'ORG_NAME', 'PI_NAMEs']
            unique_projects = len(grouped)
            total_records = len(filtered_sorted)
            st.subheader(f"Unique Projects: {unique_projects:,} ({total_records:,} total records)")
        else:
            display_df = filtered_sorted
            display_cols = ['Source', 'FISCAL_YEAR', 'PROJECT_TITLE', 'ORG_NAME', 'PI_NAMEs']
            st.subheader(f"Matching Grants: {len(filtered):,}")

        display_cols = [c for c in display_cols if c in display_df.columns]

        # Prepare display dataframe with nice column names
        table_df = display_df[display_cols].head(100).copy()
        if 'PI_NAMEs' in table_df.columns:
            table_df['PI_NAMEs'] = table_df['PI_NAMEs'].apply(clean_pi_names)
        col_names = {'PROJECT_TITLE': 'Title', 'PI_NAMEs': 'PI(s)', 'ORG_NAME': 'Organization', 'FISCAL_YEAR': 'FY', 'Source': 'Source', 'Years': 'Years'}
        table_df.columns = [col_names.get(c, c) for c in display_cols]

        # Show grants with row selection enabled
        st.caption("Select a row to view abstract below")
        selection = st.dataframe(
            table_df,
            use_container_width=True,
            hide_index=True,
            on_select="rerun",
            selection_mode="single-row"
        )

        if len(display_df) > 100:
            st.info(f"Showing first 100 of {len(display_df):,} results. Use filters to narrow results.")

        # Download button
        csv = filtered_sorted.drop(columns=['_tag_count'], errors='ignore').to_csv(index=False)
        st.download_button(
            "Download Results (CSV)",
            csv,
            "grants_export.csv",
            "text/csv"
        )

        # Show abstract for selected row
        if selection and selection.selection and selection.selection.rows:
            selected_idx = selection.selection.rows[0]
            grant = display_df.iloc[selected_idx]

            st.markdown("---")
            st.markdown("### Grant Details")

            # Get fiscal year info - handle both grouped (Years) and ungrouped (FISCAL_YEAR) cases
            if 'Years' in grant.index and pd.notna(grant.get('Years')):
                fy_display = f"FY{grant.get('Years')}"
            elif 'FISCAL_YEAR' in grant.index and pd.notna(grant.get('FISCAL_YEAR')):
                fy_display = f"FY{int(grant.get('FISCAL_YEAR'))}"
            else:
                fy_display = "FY N/A"

            # Grant details in a styled container
            st.markdown(f"""
            <div style="background: white; padding: 1.5rem; border-radius: 8px; border: 1px solid rgba(13,59,60,0.15); margin-top: 1rem;">
                <h4 style="color: #0D3B3C; margin: 0 0 0.5rem 0; font-family: 'Spectral', serif;">{grant.get('PROJECT_TITLE', 'Untitled')}</h4>
                <p style="color: #46B3A9; margin: 0 0 0.5rem 0; font-size: 0.95rem;">
                    <strong>{grant.get('ORG_NAME', 'Unknown')}</strong> | {fy_display}
                </p>
                <p style="color: #666; margin: 0; font-size: 0.9rem;">PI: {clean_pi_names(grant.get('PI_NAMEs', 'Unknown'))}</p>
            </div>
            """, unsafe_allow_html=True)

            # Show tags
            tags = []
            for exp, label in EXPOSURES.items():
                if exp in grant and grant[exp] == 1:
                    tags.append(f"🧪 {label}")
            for mech, label in MECHANISMS.items():
                if mech in grant and grant[mech] == 1:
                    tags.append(f"⚙️ {label}")

            if tags:
                st.markdown("**Tags:** " + " | ".join(tags))

            # Show full abstract - no truncation
            abstract = grant.get('ABSTRACT_TEXT', 'No abstract available')
            if pd.isna(abstract):
                abstract = 'No abstract available'
            st.markdown("**Abstract:**")
            st.markdown(f"""
            <div style="padding: 1rem; background: #f8f9fa; border-radius: 8px; line-height: 1.6; white-space: pre-wrap;">
                {abstract}
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No grants match your filters. Try broadening your search.")

with tab4:
    st.subheader("Cross-Field Insights")

    st.markdown("""
    <div style="background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); padding: 1rem; border-radius: 8px; margin-bottom: 1rem; border-left: 4px solid #D4A84B;">
        <p style="margin: 0; font-size: 0.95rem; color: #333;">
            <strong>Find established experts</strong> who already study your mechanism of interest with other chemicals.
            These researchers have existing expertise that could translate to microplastics research.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Fixed chemical exposure to Microplastics
    my_exposure = 'EXP_MICROPLASTICS'

    # Category selection with pill buttons (like STOMP)
    st.markdown("#### Select a Research Category")
    category_options_list = list(MECHANISMS_AND_TYPES.keys())
    category_labels = [MECHANISMS_AND_TYPES[k] for k in category_options_list]

    # Initialize session state for crossfield category
    if 'crossfield_category' not in st.session_state:
        st.session_state.crossfield_category = category_options_list[0] if category_options_list else None

    # Show pills in rows of 4
    for row_start in range(0, len(category_options_list), 4):
        row_items = category_options_list[row_start:row_start + 4]
        cols = st.columns(len(row_items))
        for i, (col, cat_key) in enumerate(zip(cols, row_items)):
            with col:
                cat_label = MECHANISMS_AND_TYPES.get(cat_key, cat_key)
                is_selected = st.session_state.crossfield_category == cat_key
                if is_selected:
                    st.markdown(f"""
                    <div style="background-color: #0D3B3C; color: white; padding: 8px 12px;
                                border-radius: 20px; text-align: center; font-weight: 600;
                                font-family: 'Source Sans Pro', sans-serif; font-size: 0.85rem;
                                box-shadow: 0 2px 8px rgba(13, 59, 60, 0.3); margin-bottom: 8px;">
                        {cat_label}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    if st.button(cat_label, key=f"cf_pill_{row_start}_{i}", use_container_width=True):
                        st.session_state.crossfield_category = cat_key
                        st.rerun()

    my_mechanism = st.session_state.crossfield_category

    # Secondary filters row
    st.markdown("<div style='height: 12px'></div>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        model_options = ["All Models", "In Vitro (Cells)", "Animal (Rodent)", "Animal (Zebrafish)", "Human (Cohort/Epi)", "Human (Clinical)"]
        model_filter = st.selectbox(
            "Filter by model system:",
            model_options,
            key='crossfield_model'
        )
    with col2:
        # Keyword search field
        keyword_search = st.text_input(
            "Search by keyword(s):",
            placeholder="e.g., gut barrier, inflammasome, zebrafish",
            help="Enter keywords to find grants containing these terms.",
            key='crossfield_keyword'
        )

    # Always run - exposure is fixed to Microplastics
    exp_label = EXPOSURES.get(my_exposure, my_exposure)
    mech_label = MECHANISMS_AND_TYPES.get(my_mechanism, my_mechanism) if my_mechanism else "All Categories"

    # Check if this is a TYPE_ category that needs regex-based filtering
    # TYPE_ categories always use regex since they don't have pre-classified columns for other chemicals
    use_regex_filter = my_mechanism and my_mechanism.startswith('TYPE_') and my_mechanism in TYPE_PATTERNS

    # Build combined text column for regex matching (used for TYPE_ categories)
    text_combined = df['PROJECT_TITLE'].fillna('') + ' ' + df['ABSTRACT_TEXT'].fillna('')

    # Get grants from MY field (with mechanism filter - now always applied)
    my_field_mask = (df[my_exposure] == 1)
    if my_mechanism:
        if use_regex_filter:
            # TYPE_ categories: Use regex pattern matching
            type_pattern = TYPE_PATTERNS[my_mechanism]
            my_field_mask = my_field_mask & text_combined.str.contains(type_pattern, regex=True, flags=re.IGNORECASE, na=False)
        elif my_mechanism in df.columns:
            # MECH_ categories: Use pre-classified column
            my_field_mask = my_field_mask & (df[my_mechanism] == 1)
    my_grants = df[my_field_mask]

    # Get grants from OTHER fields (with mechanism filter)
    other_exp_cols = [e for e in EXPOSURES.keys() if e != my_exposure and e in df.columns]
    other_field_mask = (df[other_exp_cols].max(axis=1) > 0) & (df[my_exposure] == 0)
    if my_mechanism:
        if use_regex_filter:
            # TYPE_ categories: Use regex pattern matching
            type_pattern = TYPE_PATTERNS[my_mechanism]
            other_field_mask = other_field_mask & text_combined.str.contains(type_pattern, regex=True, flags=re.IGNORECASE, na=False)
        elif my_mechanism in df.columns:
            # MECH_ categories: Use pre-classified column
            other_field_mask = other_field_mask & (df[my_mechanism] == 1)
    other_grants = df[other_field_mask]

    # Count by chemical field for the gap ratio display
    chemical_counts = {}
    for exp_col in other_exp_cols:
        if my_mechanism:
            if use_regex_filter:
                # TYPE_ categories: Use regex pattern matching
                type_pattern = TYPE_PATTERNS[my_mechanism]
                type_mask = text_combined.str.contains(type_pattern, regex=True, flags=re.IGNORECASE, na=False)
                count = ((df[exp_col] == 1) & type_mask & (df[my_exposure] == 0)).sum()
            elif my_mechanism in df.columns:
                # MECH_ categories: Use pre-classified column
                count = ((df[exp_col] == 1) & (df[my_mechanism] == 1) & (df[my_exposure] == 0)).sum()
            else:
                count = ((df[exp_col] == 1) & (df[my_exposure] == 0)).sum()
        else:
            count = ((df[exp_col] == 1) & (df[my_exposure] == 0)).sum()
        if count > 0:
            chemical_counts[EXPOSURES.get(exp_col, exp_col)] = count

    # Sort by count
    sorted_chemicals = sorted(chemical_counts.items(), key=lambda x: x[1], reverse=True)

    # Gap ratio stats bar
    st.markdown("---")
    gap_ratio = len(other_grants) / len(my_grants) if len(my_grants) > 0 else 0

    # Styled stats bar
    st.markdown(f"""
    <div style="background: linear-gradient(90deg, #0D3B3C 0%, #1a5a5c 100%); padding: 1rem 1.5rem; border-radius: 10px; margin-bottom: 1rem;">
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 1rem;">
            <div style="text-align: center;">
                <div style="color: #46B3A9; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 1px;">Microplastics</div>
                <div style="color: white; font-size: 1.8rem; font-weight: 700;">{len(my_grants):,}</div>
                <div style="color: rgba(255,255,255,0.7); font-size: 0.85rem;">grants</div>
            </div>
            <div style="text-align: center;">
                <div style="color: #D4A84B; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 1px;">Other Chemicals</div>
                <div style="color: white; font-size: 1.8rem; font-weight: 700;">{len(other_grants):,}</div>
                <div style="color: rgba(255,255,255,0.7); font-size: 0.85rem;">grants</div>
            </div>
            <div style="text-align: center;">
                <div style="color: #D4A84B; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 1px;">Gap Ratio</div>
                <div style="color: #D4A84B; font-size: 1.8rem; font-weight: 700;">{gap_ratio:.1f}x</div>
                <div style="color: rgba(255,255,255,0.7); font-size: 0.85rem;">more established</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Show top chemical fields for this category
    if sorted_chemicals:
        top_chems = sorted_chemicals[:5]
        chem_badges = " ".join([
            f'<span style="background-color: rgba(70,179,169,0.15); color: #0D3B3C; padding: 4px 10px; border-radius: 12px; font-size: 0.85rem; margin-right: 4px;">{name} ({count})</span>'
            for name, count in top_chems
        ])
        st.markdown(f"""
        <div style="margin-bottom: 1rem;">
            <span style="color: #666; font-size: 0.85rem; margin-right: 8px;">Top fields studying {mech_label}:</span>
            {chem_badges}
        </div>
        """, unsafe_allow_html=True)

    if len(my_grants) > 0:
        my_text = my_grants['PROJECT_TITLE'].fillna('') + ' ' + my_grants['ABSTRACT_TEXT'].fillna('')

        # Find which organ systems are most studied in source field
        my_organs = []
        for key, (label, pattern) in ORGAN_SYSTEMS.items():
            matches = my_text.str.contains(pattern, regex=True, flags=re.IGNORECASE, na=False)
            pct = 100 * matches.sum() / len(my_text)
            if pct >= 10:
                my_organs.append((label, round(pct, 0)))
        my_organs.sort(key=lambda x: x[1], reverse=True)

        # Find which model systems are used in source field
        my_models = []
        for model_name, pattern in MODEL_SYSTEMS.items():
            matches = my_text.str.contains(pattern, regex=True, flags=re.IGNORECASE, na=False)
            pct = 100 * matches.sum() / len(my_text)
            if pct >= 10:
                my_models.append((model_name, round(pct, 0)))
        my_models.sort(key=lambda x: x[1], reverse=True)

        # Results section header
        st.markdown(f"### Experts Studying {mech_label} with Other Chemicals")
        st.markdown(f"*These researchers already have expertise in **{mech_label}** using established chemical models.*")

        if len(other_grants) > 0:
            # Compute similarity scores with enhanced weighting
            scored_grants = compute_grant_similarity(
                my_grants, other_grants,
                selected_category=my_mechanism,
                keyword_filter=keyword_search
            )

            # Group by project and keep highest similarity score
            if 'CORE_PROJECT_NUM' in scored_grants.columns:
                # For grouping, we need to get the best matching themes per project
                def best_match(group):
                    best_idx = group['similarity_score'].idxmax()
                    return group.loc[best_idx]

                inspiring = scored_grants.groupby('CORE_PROJECT_NUM').apply(best_match, include_groups=False).reset_index(drop=True)
                inspiring = inspiring.sort_values('similarity_score', ascending=False)

                # Apply model system filter
                if model_filter != "All Models":
                    inspiring = inspiring[inspiring['model_system'] == model_filter]

                # Add chemical field info
                def get_exposures(row):
                    exps = []
                    for exp in other_exp_cols:
                        if exp in row and row[exp] == 1:
                            exps.append(EXPOSURES.get(exp, exp)[:15])
                    return ', '.join(exps[:2]) + ('...' if len(exps) > 2 else '')

                inspiring['Chemical(s)'] = inspiring.apply(get_exposures, axis=1)

                if len(inspiring) == 0:
                    st.info(f"No grants found with model system: {model_filter}")
                else:
                    # Format for display - PI first, then chemical badge, title, model
                    display_cols = ['PI_NAMEs', 'Chemical(s)', 'PROJECT_TITLE', 'model_system', 'ORG_NAME']
                    display_cols = [c for c in display_cols if c in inspiring.columns]
                    display_df = inspiring[display_cols].head(25).copy()
                    if 'PI_NAMEs' in display_df.columns:
                        display_df['PI_NAMEs'] = display_df['PI_NAMEs'].apply(clean_pi_names)
                    col_rename = {
                        'PI_NAMEs': 'Expert / PI',
                        'Chemical(s)': 'Chemical Field',
                        'PROJECT_TITLE': 'Project Title',
                        'model_system': 'Model',
                        'ORG_NAME': 'Institution'
                    }
                    display_df.columns = [col_rename.get(c, c) for c in display_cols]

                    st.caption(f"Found **{len(inspiring):,}** experts - click a row to view details")
                    inspiring_selection = st.dataframe(
                        display_df,
                        hide_index=True,
                        use_container_width=True,
                        on_select="rerun",
                        selection_mode="single-row"
                    )

                    if len(inspiring) > 25:
                        st.caption(f"Showing top 25 of {len(inspiring):,} unique projects, ranked by thematic similarity")

                # Show grant details for selected row
                if inspiring_selection and inspiring_selection.selection and inspiring_selection.selection.rows:
                    selected_idx = inspiring_selection.selection.rows[0]
                    orig = inspiring.iloc[selected_idx]

                    st.markdown("---")
                    st.markdown("### Expert Details")

                    # Get chemical field badge
                    chem_field = orig.get('Chemical(s)', 'Unknown')

                    st.markdown(f"""
                    <div style="background: white; padding: 1.5rem; border-radius: 8px; border: 1px solid rgba(13,59,60,0.15); margin-top: 1rem;">
                        <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 0.75rem;">
                            <div>
                                <p style="color: #0D3B3C; font-size: 1.2rem; font-weight: 700; margin: 0;">{clean_pi_names(orig.get('PI_NAMEs', 'Unknown'))}</p>
                                <p style="color: #666; margin: 0.25rem 0 0 0; font-size: 0.9rem;">{orig.get('ORG_NAME', 'Unknown')}</p>
                            </div>
                            <div style="background: linear-gradient(135deg, #46B3A9 0%, #0D3B3C 100%); color: white; padding: 6px 14px; border-radius: 16px; font-size: 0.85rem; font-weight: 600;">
                                {chem_field}
                            </div>
                        </div>
                        <p style="color: #333; margin: 0.75rem 0 0 0; font-size: 0.95rem;"><strong>Project:</strong> {orig['PROJECT_TITLE']}</p>
                    </div>
                    """, unsafe_allow_html=True)

                    # Show why this is relevant
                    if orig.get('matching_features'):
                        st.markdown(f"""
                        <div style="background: rgba(70,179,169,0.1); padding: 0.75rem 1rem; border-radius: 6px; margin-top: 0.75rem; border-left: 3px solid #46B3A9;">
                            <strong style="color: #0D3B3C;">Thematic overlap:</strong> {orig['matching_features']}
                        </div>
                        """, unsafe_allow_html=True)

                    # Show what exposures/mechanisms this grant has
                    col1, col2 = st.columns(2)
                    with col1:
                        grant_exps = [EXPOSURES[e] for e in EXPOSURES.keys() if e in orig.index and orig[e] == 1]
                        if grant_exps:
                            st.markdown(f"**Chemical exposures:** {', '.join(grant_exps)}")
                    with col2:
                        grant_mechs = [MECHANISMS[m] for m in MECHANISMS.keys() if m in orig.index and orig[m] == 1]
                        if grant_mechs:
                            st.markdown(f"**Mechanisms:** {', '.join(grant_mechs)}")

                    abstract = orig.get('ABSTRACT_TEXT', 'No abstract available')
                    if pd.isna(abstract):
                        abstract = 'No abstract available'
                    st.markdown("**Abstract:**")
                    st.write(abstract)
            else:
                st.info(f"No grants found studying {mech_label} in other chemical fields.")
        else:
            st.warning(f"Not enough data to build a research profile for {exp_label} + {mech_label}. Try a different combination.")
    else:
        st.info(f"No grants found for {exp_label} + {mech_label}. Try a different combination.")

with tab5:
    st.subheader("STOMP-Style Analysis")

    # Horizontal pill buttons instead of dropdown
    stomp_options = ["🫀 Organ Systems", "🔬 Research Phase", "📊 Research Type", "⚙️ Mechanisms"]

    # Initialize session state for STOMP category
    if 'stomp_category' not in st.session_state:
        st.session_state.stomp_category = stomp_options[0]

    # Create pill button row
    pill_cols = st.columns(len(stomp_options))
    for i, (col, option) in enumerate(zip(pill_cols, stomp_options)):
        with col:
            is_selected = st.session_state.stomp_category == option
            if is_selected:
                st.markdown(f"""
                <div style="background-color: #0D3B3C; color: white; padding: 10px 16px;
                            border-radius: 20px; text-align: center; font-weight: 600;
                            font-family: 'Source Sans Pro', sans-serif; cursor: pointer;
                            box-shadow: 0 2px 8px rgba(13, 59, 60, 0.3);">
                    {option}
                </div>
                """, unsafe_allow_html=True)
            else:
                if st.button(option, key=f"stomp_pill_{i}", use_container_width=True):
                    st.session_state.stomp_category = option
                    st.rerun()

    stomp_category = st.session_state.stomp_category

    st.markdown("<div style='height: 16px'></div>", unsafe_allow_html=True)  # Spacer

    if len(filtered) > 0:
        # Run STOMP classification
        stomp_results = classify_stomp_categories(filtered)

        # Deduplicate for all STOMP views
        filtered_stomp = filtered.drop_duplicates(subset=['PROJECT_TITLE'], keep='first')

        # ---- ORGAN SYSTEMS ----
        if stomp_category == "🫀 Organ Systems":
            st.markdown("#### Which body systems are being studied?")

            organ_data = stomp_results['organs']
            if organ_data:
                # Sort by count
                sorted_organs = sorted(organ_data.items(), key=lambda x: x[1]['count'], reverse=True)

                col1, col2 = st.columns([2, 1])
                with col1:
                    chart_data = {k: v['count'] for k, v in sorted_organs if v['count'] > 0}
                    chart = create_horizontal_bar_chart(chart_data, value_label="Projects")
                    st.altair_chart(chart, use_container_width=True)

                with col2:
                    organ_table = [{'Organ System': k, 'Projects': v['count'], '%': f"{v['pct']}%"}
                                  for k, v in sorted_organs[:8]]
                    st.dataframe(pd.DataFrame(organ_table), hide_index=True, use_container_width=True)

            # Drill-down with selectbox
            st.markdown("---")
            organ_options = [f"{name} ({info['count']})" for name, info in sorted_organs if info['count'] > 0]
            if organ_options:
                selected_organ_option = st.selectbox(
                    "Explore projects by organ system",
                    ["Select an organ system..."] + organ_options,
                    key="organ_select"
                )
                selected_organ = selected_organ_option.rsplit(" (", 1)[0] if selected_organ_option != "Select an organ system..." else None
            else:
                selected_organ = None

            if selected_organ:
                # Find the pattern and column for this organ
                organ_pattern = None
                organ_col = None
                organ_col_map = {
                    'ORGAN_BRAIN': 'ORGAN_BRAIN_NERVOUS',
                    'ORGAN_CARDIOVASCULAR': 'ORGAN_CARDIOVASCULAR',
                    'ORGAN_GI': 'ORGAN_GI_GUT',
                    'ORGAN_LIVER': 'ORGAN_LIVER',
                    'ORGAN_KIDNEY': 'ORGAN_KIDNEY',
                    'ORGAN_LUNG': 'ORGAN_RESPIRATORY',
                    'ORGAN_REPRODUCTIVE': 'ORGAN_REPRODUCTIVE',
                    'ORGAN_IMMUNE': 'ORGAN_IMMUNE',
                    'ORGAN_SKIN': 'ORGAN_SKIN',
                }
                for key, (label, pattern) in ORGAN_SYSTEMS.items():
                    if label == selected_organ:
                        organ_pattern = pattern
                        organ_col = organ_col_map.get(key)
                        break

                if organ_pattern:
                    text = filtered_stomp['PROJECT_TITLE'].fillna('') + ' ' + filtered_stomp['ABSTRACT_TEXT'].fillna('')
                    keyword_matches = text.str.contains(organ_pattern, regex=True, flags=re.IGNORECASE, na=False)
                    # Also check pre-classified column
                    if organ_col and organ_col in filtered_stomp.columns:
                        preclassified = filtered_stomp[organ_col] == 1
                        organ_matches = keyword_matches | preclassified
                    else:
                        organ_matches = keyword_matches
                    organ_grants = filtered_stomp[organ_matches].copy()

                    st.markdown(f"### {selected_organ}")
                    st.markdown(f"**{len(organ_grants):,} projects** studying this system")

                    display_cols = ['PROJECT_TITLE', 'PI_NAMEs', 'ORG_NAME', 'FISCAL_YEAR']
                    display_cols = [c for c in display_cols if c in organ_grants.columns]
                    if display_cols:
                        grants_display = organ_grants[display_cols].head(50).copy()
                        if 'PI_NAMEs' in grants_display.columns:
                            grants_display['PI_NAMEs'] = grants_display['PI_NAMEs'].apply(clean_pi_names)
                        col_names = {'PROJECT_TITLE': 'Title', 'PI_NAMEs': 'PI(s)', 'ORG_NAME': 'Organization', 'FISCAL_YEAR': 'FY'}
                        grants_display.columns = [col_names.get(c, c) for c in display_cols]
                        st.caption("Select a row to view abstract below")
                        organ_selection = st.dataframe(
                            grants_display,
                            hide_index=True,
                            use_container_width=True,
                            height=300,
                            on_select="rerun",
                            selection_mode="single-row"
                        )

                        if len(organ_grants) > 50:
                            st.caption(f"Showing first 50 of {len(organ_grants):,} projects")

                        # Show abstract for selected row
                        if organ_selection and organ_selection.selection and organ_selection.selection.rows:
                            selected_idx = organ_selection.selection.rows[0]
                            grant_row = organ_grants.iloc[selected_idx]
                            st.markdown("---")
                            st.markdown(f"**{grant_row['PROJECT_TITLE']}**")
                            st.markdown(f"*PI:* {clean_pi_names(grant_row.get('PI_NAMEs', 'Unknown'))} | *Org:* {grant_row.get('ORG_NAME', 'Unknown')} | *FY:* {int(grant_row.get('FISCAL_YEAR', 0))}")
                            abstract = grant_row.get('ABSTRACT_TEXT', 'No abstract available')
                            if pd.isna(abstract):
                                abstract = 'No abstract available'
                            st.markdown("**Abstract:**")
                            st.write(abstract)

        # ---- RESEARCH PHASES ----
        elif stomp_category == "🔬 Research Phase":
            st.markdown("#### Where are projects in the research pipeline?")

            phase_data = stomp_results['phases']
            if phase_data:
                sorted_phases = sorted(phase_data.items(), key=lambda x: x[1]['count'], reverse=True)

                col1, col2 = st.columns([2, 1])
                with col1:
                    chart_data = {k: v['count'] for k, v in sorted_phases if v['count'] > 0}
                    chart = create_horizontal_bar_chart(chart_data, value_label="Projects")
                    st.altair_chart(chart, use_container_width=True)

                with col2:
                    phase_table = [{'Research Phase': k, 'Projects': v['count'], '%': f"{v['pct']}%"}
                                  for k, v in sorted_phases]
                    st.dataframe(pd.DataFrame(phase_table), hide_index=True, use_container_width=True)

                # Drill-down with selectbox
                st.markdown("---")
                phase_options = [f"{name} ({info['count']})" for name, info in sorted_phases if info['count'] > 0]
                if phase_options:
                    selected_phase_option = st.selectbox(
                        "Explore projects by research phase",
                        ["Select a research phase..."] + phase_options,
                        key="phase_select"
                    )
                    selected_phase = selected_phase_option.rsplit(" (", 1)[0] if selected_phase_option != "Select a research phase..." else None
                else:
                    selected_phase = None

                if selected_phase:
                    phase_pattern = None
                    for key, (label, pattern) in RESEARCH_PHASES.items():
                        if label == selected_phase:
                            phase_pattern = pattern
                            break

                    if phase_pattern:
                        text = filtered_stomp['PROJECT_TITLE'].fillna('') + ' ' + filtered_stomp['ABSTRACT_TEXT'].fillna('')
                        phase_matches = text.str.contains(phase_pattern, regex=True, flags=re.IGNORECASE, na=False)
                        phase_grants = filtered_stomp[phase_matches].copy()

                        st.markdown(f"### {selected_phase}")
                        st.markdown(f"**{len(phase_grants):,} projects** in this phase")

                        display_cols = ['PROJECT_TITLE', 'PI_NAMEs', 'ORG_NAME', 'FISCAL_YEAR']
                        display_cols = [c for c in display_cols if c in phase_grants.columns]
                        if display_cols:
                            grants_display = phase_grants[display_cols].head(50).copy()
                            if 'PI_NAMEs' in grants_display.columns:
                                grants_display['PI_NAMEs'] = grants_display['PI_NAMEs'].apply(clean_pi_names)
                            col_names = {'PROJECT_TITLE': 'Title', 'PI_NAMEs': 'PI(s)', 'ORG_NAME': 'Organization', 'FISCAL_YEAR': 'FY'}
                            grants_display.columns = [col_names.get(c, c) for c in display_cols]
                            st.caption("Select a row to view abstract below")
                            phase_selection = st.dataframe(
                                grants_display,
                                hide_index=True,
                                use_container_width=True,
                                height=300,
                                on_select="rerun",
                                selection_mode="single-row"
                            )

                            if len(phase_grants) > 50:
                                st.caption(f"Showing first 50 of {len(phase_grants):,} projects")

                            # Show abstract for selected row
                            if phase_selection and phase_selection.selection and phase_selection.selection.rows:
                                selected_idx = phase_selection.selection.rows[0]
                                grant_row = phase_grants.iloc[selected_idx]
                                st.markdown("---")
                                st.markdown(f"**{grant_row['PROJECT_TITLE']}**")
                                st.markdown(f"*PI:* {clean_pi_names(grant_row.get('PI_NAMEs', 'Unknown'))} | *Org:* {grant_row.get('ORG_NAME', 'Unknown')} | *FY:* {int(grant_row.get('FISCAL_YEAR', 0))}")
                                abstract = grant_row.get('ABSTRACT_TEXT', 'No abstract available')
                                if pd.isna(abstract):
                                    abstract = 'No abstract available'
                                st.markdown("**Abstract:**")
                                st.write(abstract)

        # ---- RESEARCH TYPES ----
        elif stomp_category == "📊 Research Type":
            st.markdown("#### What type of research methodology?")

            type_data = stomp_results['research_types']
            if type_data:
                sorted_types = sorted(type_data.items(), key=lambda x: x[1]['count'], reverse=True)

                col1, col2 = st.columns([2, 1])
                with col1:
                    chart_data = {k: v['count'] for k, v in sorted_types if v['count'] > 0}
                    chart = create_horizontal_bar_chart(chart_data, value_label="Projects")
                    st.altair_chart(chart, use_container_width=True)

                with col2:
                    type_table = [{'Research Type': k, 'Projects': v['count'], '%': f"{v['pct']}%"}
                                  for k, v in sorted_types]
                    st.dataframe(pd.DataFrame(type_table), hide_index=True, use_container_width=True)

                # Drill-down with selectbox
                st.markdown("---")
                type_options = [f"{name} ({info['count']})" for name, info in sorted_types if info['count'] > 0]
                if type_options:
                    selected_type_option = st.selectbox(
                        "Explore projects by research type",
                        ["Select a research type..."] + type_options,
                        key="type_select"
                    )
                    selected_type = selected_type_option.rsplit(" (", 1)[0] if selected_type_option != "Select a research type..." else None
                else:
                    selected_type = None

                if selected_type:
                    type_pattern = None
                    type_col = None
                    type_col_map = {
                        'TYPE_EPIDEMIOLOGY': None,
                        'TYPE_MECHANISTIC': 'TYPE_MECHANISTIC',
                        'TYPE_CLINICAL': None,
                        'TYPE_METHODS': 'TYPE_METHODS',
                    }
                    for key, (label, pattern) in RESEARCH_TYPES.items():
                        if label == selected_type:
                            type_pattern = pattern
                            type_col = type_col_map.get(key)
                            break

                    if type_pattern:
                        text = filtered_stomp['PROJECT_TITLE'].fillna('') + ' ' + filtered_stomp['ABSTRACT_TEXT'].fillna('')
                        keyword_matches = text.str.contains(type_pattern, regex=True, flags=re.IGNORECASE, na=False)
                        # Also check pre-classified column
                        if type_col and type_col in filtered_stomp.columns:
                            preclassified = filtered_stomp[type_col] == 1
                            type_matches = keyword_matches | preclassified
                        else:
                            type_matches = keyword_matches
                        type_grants = filtered_stomp[type_matches].copy()

                        st.markdown(f"### {selected_type}")
                        st.markdown(f"**{len(type_grants):,} projects** of this type")

                        display_cols = ['PROJECT_TITLE', 'PI_NAMEs', 'ORG_NAME', 'FISCAL_YEAR']
                        display_cols = [c for c in display_cols if c in type_grants.columns]
                        if display_cols:
                            grants_display = type_grants[display_cols].head(50).copy()
                            if 'PI_NAMEs' in grants_display.columns:
                                grants_display['PI_NAMEs'] = grants_display['PI_NAMEs'].apply(clean_pi_names)
                            col_names = {'PROJECT_TITLE': 'Title', 'PI_NAMEs': 'PI(s)', 'ORG_NAME': 'Organization', 'FISCAL_YEAR': 'FY'}
                            grants_display.columns = [col_names.get(c, c) for c in display_cols]
                            st.caption("Select a row to view abstract below")
                            type_selection = st.dataframe(
                                grants_display,
                                hide_index=True,
                                use_container_width=True,
                                height=300,
                                on_select="rerun",
                                selection_mode="single-row"
                            )

                            if len(type_grants) > 50:
                                st.caption(f"Showing first 50 of {len(type_grants):,} projects")

                            # Show abstract for selected row
                            if type_selection and type_selection.selection and type_selection.selection.rows:
                                selected_idx = type_selection.selection.rows[0]
                                grant_row = type_grants.iloc[selected_idx]
                                st.markdown("---")
                                st.markdown(f"**{grant_row['PROJECT_TITLE']}**")
                                st.markdown(f"*PI:* {clean_pi_names(grant_row.get('PI_NAMEs', 'Unknown'))} | *Org:* {grant_row.get('ORG_NAME', 'Unknown')} | *FY:* {int(grant_row.get('FISCAL_YEAR', 0))}")
                                abstract = grant_row.get('ABSTRACT_TEXT', 'No abstract available')
                                if pd.isna(abstract):
                                    abstract = 'No abstract available'
                                st.markdown("**Abstract:**")
                                st.write(abstract)

        # ---- MECHANISMS ----
        elif stomp_category == "⚙️ Mechanisms":
            st.markdown("#### What biological mechanisms are being studied?")
            st.caption("*Using tightened patterns that require mechanism to be a research focus*")

            # Use MECHANISM_SYSTEMS tightened patterns (same approach as ORGAN_SYSTEMS)
            # Build mechanism data using runtime pattern matching
            mech_name_to_key = {}
            mech_data = {}
            n_grants = len(filtered_stomp)
            text = filtered_stomp['PROJECT_TITLE'].fillna('') + ' ' + filtered_stomp['ABSTRACT_TEXT'].fillna('')

            for key, (label, pattern) in MECHANISM_SYSTEMS.items():
                # Use tightened pattern matching (may also check CSV column as fallback)
                keyword_match = text.str.contains(pattern, regex=True, flags=re.IGNORECASE, na=False)
                # Optionally combine with pre-classified column for broader coverage
                # (but prefer pattern match for precision)
                if key in filtered_stomp.columns:
                    # Use pattern match AND CSV column (intersection for precision)
                    # Or use pattern match only for tighter classification
                    matches = keyword_match  # Use tightened pattern only
                else:
                    matches = keyword_match
                count = int(matches.sum())
                pct = round(100 * count / n_grants, 1) if n_grants > 0 else 0
                mech_data[label] = {'count': count, 'pct': pct, 'key': key, 'pattern': pattern}
                mech_name_to_key[label] = key

            if mech_data:

                sorted_mechs = sorted(mech_data.items(), key=lambda x: x[1]['count'], reverse=True)

                col1, col2 = st.columns([2, 1])
                with col1:
                    chart_data = {k: v['count'] for k, v in sorted_mechs if v['count'] > 0}
                    chart = create_horizontal_bar_chart(chart_data, value_label="Projects")
                    st.altair_chart(chart, use_container_width=True)

                with col2:
                    mech_table = [{'Mechanism': k, 'Projects': v['count'], '%': f"{v['pct']}%"}
                                  for k, v in sorted_mechs[:10]]
                    st.dataframe(pd.DataFrame(mech_table), hide_index=True, use_container_width=True)

                # Show how many projects have ANY mechanism (using pattern matching)
                any_mech_mask = pd.Series(False, index=filtered_stomp.index)
                for label, info in mech_data.items():
                    any_mech_mask = any_mech_mask | text.str.contains(info['pattern'], regex=True, flags=re.IGNORECASE, na=False)
                any_mech = any_mech_mask.sum()
                any_pct = round(100 * any_mech / n_grants, 1) if n_grants > 0 else 0
                st.info(f"**{any_mech:,}** projects ({any_pct}%) have at least one mechanism identified")

                # Drill-down with selectbox
                st.markdown("---")
                mech_options = [f"{name} ({info['count']})" for name, info in sorted_mechs if info['count'] > 0]
                if mech_options:
                    selected_mech_option = st.selectbox(
                        "Explore projects by mechanism",
                        ["Select a mechanism..."] + mech_options,
                        key="mech_select"
                    )
                    selected_mech_stomp = selected_mech_option.rsplit(" (", 1)[0] if selected_mech_option != "Select a mechanism..." else None
                else:
                    selected_mech_stomp = None

                if selected_mech_stomp and selected_mech_stomp in mech_name_to_key:
                    mech_key = mech_name_to_key[selected_mech_stomp]
                    mech_pattern = mech_data[selected_mech_stomp]['pattern']
                    # Use pattern matching for drill-down
                    mech_matches = text.str.contains(mech_pattern, regex=True, flags=re.IGNORECASE, na=False)
                    mech_grants = filtered_stomp[mech_matches].copy()

                    st.markdown(f"### {selected_mech_stomp}")
                    st.markdown(f"**{len(mech_grants):,} projects** studying this mechanism")

                    display_cols = ['PROJECT_TITLE', 'PI_NAMEs', 'ORG_NAME', 'FISCAL_YEAR']
                    display_cols = [c for c in display_cols if c in mech_grants.columns]
                    if display_cols:
                        grants_display = mech_grants[display_cols].head(50).copy()
                        if 'PI_NAMEs' in grants_display.columns:
                            grants_display['PI_NAMEs'] = grants_display['PI_NAMEs'].apply(clean_pi_names)
                        col_names = {'PROJECT_TITLE': 'Title', 'PI_NAMEs': 'PI(s)', 'ORG_NAME': 'Organization', 'FISCAL_YEAR': 'FY'}
                        grants_display.columns = [col_names.get(c, c) for c in display_cols]
                        st.caption("Select a row to view abstract below")
                        mech_selection = st.dataframe(
                            grants_display,
                            hide_index=True,
                            use_container_width=True,
                            height=300,
                            on_select="rerun",
                            selection_mode="single-row"
                        )

                        if len(mech_grants) > 50:
                            st.caption(f"Showing first 50 of {len(mech_grants):,} projects")

                        # Show abstract for selected row
                        if mech_selection and mech_selection.selection and mech_selection.selection.rows:
                            selected_idx = mech_selection.selection.rows[0]
                            grant_row = mech_grants.iloc[selected_idx]
                            st.markdown("---")
                            st.markdown(f"**{grant_row['PROJECT_TITLE']}**")
                            st.markdown(f"*PI:* {clean_pi_names(grant_row.get('PI_NAMEs', 'Unknown'))} | *Org:* {grant_row.get('ORG_NAME', 'Unknown')} | *FY:* {int(grant_row.get('FISCAL_YEAR', 0))}")
                            abstract = grant_row.get('ABSTRACT_TEXT', 'No abstract available')
                            if pd.isna(abstract):
                                abstract = 'No abstract available'
                            st.markdown("**Abstract:**")
                            st.write(abstract)

                # Funding Institute breakdown within mechanisms tab
                st.markdown("---")
                st.markdown("#### Funding Institutes")

                ic_col = None
                for possible_col in ['ADMINISTERING_IC', 'IC_NAME', 'ADMIN_IC']:
                    if possible_col in filtered_stomp.columns:
                        ic_col = possible_col
                        break

                if ic_col:
                    ic_counts = filtered_stomp[ic_col].value_counts().head(10)
                    if len(ic_counts) > 0:
                        col1, col2 = st.columns([2, 1])
                        with col1:
                            chart_data = dict(ic_counts)
                            chart = create_horizontal_bar_chart(chart_data, value_label="Projects")
                            st.altair_chart(chart, use_container_width=True)
                        with col2:
                            ic_table = [{'Institute': ic, 'Projects': count}
                                       for ic, count in ic_counts.head(8).items()]
                            st.dataframe(pd.DataFrame(ic_table), hide_index=True, use_container_width=True)
                else:
                    st.info("Funding institute data not available.")
            else:
                st.info("No mechanism patterns matched in the current dataset.")

        # ---- CROSS-EXPOSURE COMPARISON (outside tabs, at bottom) ----
        if selected_exposures and len(selected_exposures) > 1:
            st.markdown("---")
            st.markdown("### Compare STOMP Categories Across Selected Exposures")

            comparison_data = []
            for exp in selected_exposures:
                if exp in filtered_stomp.columns:
                    exp_grants = filtered_stomp[filtered_stomp[exp] == 1]
                    if len(exp_grants) > 0:
                        exp_stomp = classify_stomp_categories(exp_grants)

                        # Get top organ, phase, and research type for this exposure
                        top_organ = max(exp_stomp['organs'].items(), key=lambda x: x[1]['count'])[0] if exp_stomp['organs'] else 'N/A'
                        top_phase = max(exp_stomp['phases'].items(), key=lambda x: x[1]['count'])[0] if exp_stomp['phases'] else 'N/A'
                        top_type = max(exp_stomp['research_types'].items(), key=lambda x: x[1]['count'])[0] if exp_stomp['research_types'] else 'N/A'

                        comparison_data.append({
                            'Exposure': EXPOSURES.get(exp, exp)[:25],
                            'Projects': len(exp_grants),
                            'Top Organ': top_organ[:20],
                            'Top Phase': top_phase[:18],
                            'Research Type': top_type[:20],
                        })

            if comparison_data:
                st.dataframe(pd.DataFrame(comparison_data), hide_index=True, use_container_width=True)

    else:
        st.info("Filter grants to see STOMP-style analysis.")

with tab6:
    st.subheader("Microplastics Research Landscape")
    st.markdown("""
    <div style="background: linear-gradient(135deg, rgba(70,179,169,0.1) 0%, rgba(13,59,60,0.05) 100%);
                padding: 1rem; border-radius: 8px; border-left: 4px solid #46B3A9; margin-bottom: 1.5rem;">
        <p style="margin: 0; color: #0D3B3C;">
            Analysis of <strong>247 microplastic-focused grants</strong> using high-confidence keyword matching.
            Categories are based on explicit terms in abstracts (e.g., "lung", "zebrafish", "ingestion").
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Filter to microplastics grants
    mp_grants = df[df['EXP_MICROPLASTICS'] == 1].copy() if 'EXP_MICROPLASTICS' in df.columns else pd.DataFrame()

    if len(mp_grants) > 0:
        mp_text = mp_grants['PROJECT_TITLE'].fillna('') + ' ' + mp_grants['ABSTRACT_TEXT'].fillna('')
        n_mp = len(mp_grants)

        # High-confidence classifications
        MP_TISSUES = {
            'Blood/Cardiovascular': r'\bblood\b|\bcardiovasc|\bheart\b',
            'Brain': r'\bbrain\b',
            'Gut/Intestine': r'\bgut\b|\bintestin',
            'Placenta': r'\bplacent',
            'Lung': r'\blung\b|\bpulmonary\b',
            'Liver': r'\bliver\b|\bhepat',
            'Reproductive': r'\btestes\b|\bovary\b|\bsperm\b|\buterus\b|\bfertil',
            'Kidney': r'\bkidney\b|\brenal\b',
        }

        MP_MODELS = {
            'Mouse/Rat': r'\bmouse\b|\bmice\b|\brat\b|\brodent\b|\bmurine\b',
            'Cell culture': r'cell\s+line|cell\s+culture|in\s+vitro|primary\s+cell',
            'Human cohort': r'cohort|epidemiol|NHANES|population.based|human\s+subject',
            'Zebrafish': r'\bzebrafish\b|\bdanio\b',
            'C. elegans': r'c\.\s*elegans|caenorhabditis',
        }

        MP_ROUTES = {
            'Ingestion/Oral': r'ingest|oral\s+exposure|dietary|drinking\s+water|food',
            'Inhalation': r'inhal|airborne|air\s+pollution',
            'Dermal': r'dermal|skin\s+exposure',
        }

        MP_MECHANISMS = {
            'Oxidative stress': r'oxidative|ROS|reactive\s+oxygen|antioxidant',
            'Inflammation': r'inflamm|cytokine|IL-\d|TNF',
            'Barrier disruption': r'barrier|permeab|tight\s+junction|BBB',
            'Microbiome effects': r'microbiome|dysbiosis|microbiota',
            'Apoptosis/Cell death': r'apoptos|cell\s+death|necrosis',
            'Endocrine disruption': r'endocrine|hormone|estrogen|thyroid',
        }

        # Calculate counts
        def count_matches(patterns_dict):
            results = {}
            for name, pattern in patterns_dict.items():
                count = mp_text.str.contains(pattern, flags=re.IGNORECASE, na=False).sum()
                results[name] = count
            return results

        tissue_counts = count_matches(MP_TISSUES)
        model_counts = count_matches(MP_MODELS)
        route_counts = count_matches(MP_ROUTES)
        mech_counts = count_matches(MP_MECHANISMS)

        # Particle size
        nano_count = mp_text.str.contains(r'nanoplastic|nano-plastic|nano.sized\s+plastic', flags=re.IGNORECASE, na=False).sum()
        micro_count = mp_text.str.contains(r'microplastic|micro-plastic', flags=re.IGNORECASE, na=False).sum()

        # Display metrics row
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total MP Grants", f"{n_mp}")
        with col2:
            st.metric("Mention Microplastic", f"{micro_count} ({100*micro_count//n_mp}%)")
        with col3:
            st.metric("Mention Nanoplastic", f"{nano_count} ({100*nano_count//n_mp}%)")

        st.markdown("---")

        # Charts in 2x2 grid
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Tissues Studied")
            chart = create_horizontal_bar_chart(tissue_counts, value_label="Grants")
            st.altair_chart(chart, use_container_width=True)

            st.markdown("#### Exposure Routes")
            chart = create_horizontal_bar_chart(route_counts, value_label="Grants")
            st.altair_chart(chart, use_container_width=True)

        with col2:
            st.markdown("#### Model Systems")
            chart = create_horizontal_bar_chart(model_counts, value_label="Grants")
            st.altair_chart(chart, use_container_width=True)

            st.markdown("#### Mechanisms (medium confidence)")
            chart = create_horizontal_bar_chart(mech_counts, value_label="Grants")
            st.altair_chart(chart, use_container_width=True)

        # Grant browser
        st.markdown("---")
        st.markdown("### Browse Microplastic Grants")

        # Filter options
        filter_col1, filter_col2 = st.columns(2)
        with filter_col1:
            tissue_filter = st.selectbox("Filter by tissue:", ["All"] + list(MP_TISSUES.keys()), key='mp_tissue_filter')
        with filter_col2:
            model_filter = st.selectbox("Filter by model:", ["All"] + list(MP_MODELS.keys()), key='mp_model_filter')

        # Apply filters
        mp_filtered = mp_grants.copy()
        if tissue_filter != "All":
            pattern = MP_TISSUES[tissue_filter]
            mask = mp_text.str.contains(pattern, flags=re.IGNORECASE, na=False)
            mp_filtered = mp_grants[mask.values]
        if model_filter != "All":
            pattern = MP_MODELS[model_filter]
            mp_text_filtered = mp_filtered['PROJECT_TITLE'].fillna('') + ' ' + mp_filtered['ABSTRACT_TEXT'].fillna('')
            mask = mp_text_filtered.str.contains(pattern, flags=re.IGNORECASE, na=False)
            mp_filtered = mp_filtered[mask.values]

        st.markdown(f"**{len(mp_filtered)} grants** match filters")

        display_cols = ['PROJECT_TITLE', 'PI_NAMEs', 'ORG_NAME', 'FISCAL_YEAR']
        display_cols = [c for c in display_cols if c in mp_filtered.columns]
        if display_cols and len(mp_filtered) > 0:
            mp_display = mp_filtered[display_cols].head(50).copy()
            if 'PI_NAMEs' in mp_display.columns:
                mp_display['PI_NAMEs'] = mp_display['PI_NAMEs'].apply(clean_pi_names)
            col_names = {'PROJECT_TITLE': 'Title', 'PI_NAMEs': 'PI(s)', 'ORG_NAME': 'Organization', 'FISCAL_YEAR': 'FY'}
            mp_display.columns = [col_names.get(c, c) for c in display_cols]
            st.caption("Select a row to view abstract below")
            mp_selection = st.dataframe(
                mp_display,
                hide_index=True,
                use_container_width=True,
                on_select="rerun",
                selection_mode="single-row"
            )

            if len(mp_filtered) > 50:
                st.caption(f"Showing first 50 of {len(mp_filtered):,} grants")

            # Show abstract for selected row
            if mp_selection and mp_selection.selection and mp_selection.selection.rows:
                selected_idx = mp_selection.selection.rows[0]
                row = mp_filtered.iloc[selected_idx]
                st.markdown("---")
                st.markdown(f"**{row['PROJECT_TITLE']}**")
                st.markdown(f"*PI:* {clean_pi_names(row.get('PI_NAMEs', 'Unknown'))} | *Org:* {row.get('ORG_NAME', 'Unknown')}")
                abstract = row.get('ABSTRACT_TEXT', 'No abstract available')
                if pd.isna(abstract):
                    abstract = 'No abstract available'
                st.markdown("**Abstract:**")
                st.write(abstract)
    else:
        st.info("No microplastic grants found. Make sure EXP_MICROPLASTICS column exists in the data.")

