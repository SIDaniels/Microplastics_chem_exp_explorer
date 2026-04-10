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
# import anthropic  # Commented out to avoid API costs
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

# ============== PAGINATION HELPER ==============

def paginated_dataframe(df: pd.DataFrame, key: str, page_size: int = 25) -> pd.DataFrame:
    """
    Display a paginated dataframe with navigation controls.
    Returns the current page's dataframe slice.

    Args:
        df: The full dataframe to paginate
        key: Unique key for session state (to track page number)
        page_size: Number of rows per page (default 25)

    Returns:
        The slice of dataframe for the current page
    """
    total_rows = len(df)
    total_pages = max(1, (total_rows + page_size - 1) // page_size)

    # Initialize page state
    page_key = f"page_{key}"
    if page_key not in st.session_state:
        st.session_state[page_key] = 0

    # Ensure page is within bounds
    st.session_state[page_key] = max(0, min(st.session_state[page_key], total_pages - 1))
    current_page = st.session_state[page_key]

    # Calculate slice indices
    start_idx = current_page * page_size
    end_idx = min(start_idx + page_size, total_rows)

    # Display navigation if more than one page
    if total_pages > 1:
        nav_cols = st.columns([1, 2, 2, 2, 1])

        with nav_cols[0]:
            if st.button("◀◀", key=f"first_{key}", disabled=current_page == 0):
                st.session_state[page_key] = 0
                st.rerun()

        with nav_cols[1]:
            if st.button("◀ Prev", key=f"prev_{key}", disabled=current_page == 0):
                st.session_state[page_key] = current_page - 1
                st.rerun()

        with nav_cols[2]:
            st.markdown(f"<div style='text-align: center; padding-top: 5px;'>Page {current_page + 1} of {total_pages} ({total_rows} results total)</div>", unsafe_allow_html=True)

        with nav_cols[3]:
            if st.button("Next ▶", key=f"next_{key}", disabled=current_page >= total_pages - 1):
                st.session_state[page_key] = current_page + 1
                st.rerun()

        with nav_cols[4]:
            if st.button("▶▶", key=f"last_{key}", disabled=current_page >= total_pages - 1):
                st.session_state[page_key] = total_pages - 1
                st.rerun()

    return df.iloc[start_idx:end_idx]


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

    # Map keywords to stem/prefix for partial matching
    keyword_stems = {
        'reproduction': 'reproduct',
        'reproductive': 'reproduct',
        'microplastics': 'microplastic',
        'microplastic': 'microplastic',
        'nanoplastics': 'nanoplastic',
        'nanoplastic': 'nanoplastic',
        'inflammation': 'inflammat',
        'inflammatory': 'inflammat',
        'oxidative': 'oxidat',
        'fertility': 'fertil',
        'pregnant': 'pregnan',
        'pregnancy': 'pregnan',
    }

    # Expand keywords with related terms
    keyword_expansions = {
        'reproduct': ['fertility', 'fertil', 'pregnant', 'pregnan', 'prenatal', 'fetal', 'embryo', 'ovary', 'ovarian', 'sperm', 'testis', 'testicular', 'uterus', 'uterine', 'placenta', 'gestation'],
        'gut': ['intestin', 'gastrointest', 'digestive', 'microbiome', 'colon', 'bowel'],
        'brain': ['neural', 'neuron', 'cognitive', 'neurolog', 'cerebr', 'neurodev'],
        'heart': ['cardiac', 'cardiovascular', 'cardio'],
        'lung': ['pulmonary', 'respiratory', 'airway'],
        'liver': ['hepat', 'hepatic'],
        'kidney': ['renal', 'nephro'],
    }

    # Score each grant by how many keywords it matches
    df = df.copy()
    df['_match_score'] = 0

    for keyword in keywords:
        # Use stem if available
        search_term = keyword_stems.get(keyword, keyword)

        # Check main keyword/stem
        matches = text.str.contains(search_term, regex=False).astype(int)

        # Also check expanded terms if available
        for base, expansions in keyword_expansions.items():
            if search_term.startswith(base) or base.startswith(search_term):
                for exp in expansions:
                    matches = matches | text.str.contains(exp, regex=False).astype(int)
                break

        df['_match_score'] += matches

    num_keywords = len(keywords)

    # Identify health/biology keywords vs chemical keywords
    chemical_keywords = {'microplastic', 'nanoplastic', 'plastic', 'pfas', 'phthalate', 'bpa',
                        'pesticide', 'metal', 'lead', 'mercury', 'arsenic', 'cadmium',
                        'pollution', 'pollutant', 'chemical', 'exposure', 'contaminant'}

    # Find which keywords are health topics (not chemicals)
    health_keywords = []
    for kw in keywords:
        stem = keyword_stems.get(kw, kw)
        if stem not in chemical_keywords and not any(c in stem for c in chemical_keywords):
            health_keywords.append(stem)

    # Prioritize grants matching ALL keywords, then most keywords
    if num_keywords > 1:
        # First try to get grants matching all keywords
        all_match = df[df['_match_score'] >= num_keywords]
        if len(all_match) >= 5:
            primary_results = all_match.sort_values('_match_score', ascending=False).head(max_results // 2)
        else:
            # Get grants matching at least half the keywords, sorted by score
            min_score = max(1, num_keywords // 2)
            primary_results = df[df['_match_score'] >= min_score].sort_values('_match_score', ascending=False).head(max_results // 2)

        # Also get "related research" - grants matching health topic but with OTHER chemicals
        if health_keywords:
            # Find grants matching health keywords but not already in primary results
            related_mask = pd.Series([False] * len(df), index=df.index)
            for hk in health_keywords:
                related_mask = related_mask | text.str.contains(hk, regex=False)
                # Also check expansions
                for base, expansions in keyword_expansions.items():
                    if hk.startswith(base) or base.startswith(hk):
                        for exp in expansions:
                            related_mask = related_mask | text.str.contains(exp, regex=False)
                        break

            # Exclude primary results
            related_mask = related_mask & ~df.index.isin(primary_results.index)
            related_results = df[related_mask].head(max_results // 2)

            # Combine: primary first, then related
            results = pd.concat([primary_results, related_results]).drop(columns=['_match_score'], errors='ignore')
        else:
            results = primary_results.drop(columns=['_match_score'])
    else:
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
    """Get AI response for a chat query using Claude.

    NOTE: Anthropic API integration is currently disabled to avoid API costs.
    To re-enable, uncomment the import and the code below.
    """
    # Chat feature temporarily disabled to avoid API costs
    return "⚠️ AI Chat is temporarily disabled. Please use the search filters and keyword search below to explore the grants database."

    # --- COMMENTED OUT ANTHROPIC API CODE ---
    # try:
    #     # Check for API key
    #     api_key = st.secrets.get("ANTHROPIC_API_KEY", None)
    #     if not api_key:
    #         return "⚠️ Chat is not configured. Please add ANTHROPIC_API_KEY to Streamlit secrets."
    #
    #     client = anthropic.Anthropic(api_key=api_key)
    #
    #     # Search for relevant grants
    #     relevant_grants = search_grants_for_chat(df, query)
    #     context = format_grants_for_context(relevant_grants)
    #
    #     # Build the prompt
    #     system_prompt = """You are a research assistant for the Microplastics & Chemical Exposure Grant Explorer database.
    # You help users discover NIH-funded research grants and conference abstracts.
    #
    # IMPORTANT INSTRUCTIONS:
    # - You will be given grants from the database - some directly matching the query, some related
    # - Organize your response into TWO sections:
    #   1. **Direct Matches**: Grants that specifically match ALL aspects of the query (e.g., microplastics AND placenta)
    #   2. **Related Research**: Grants on the same health topic but with different chemicals/exposures
    # - ALWAYS cite specific PI names, institutions, and grant titles
    # - Be specific and detailed about what each researcher is studying
    # - Only say "no research found" if the grants list is truly empty"""
    #
    #     user_prompt = f"""User question: {query}
    #
    # DATABASE RESULTS:
    # {context}
    #
    # Based on these grants from our database, answer the user's question. Cite specific PIs, institutions, and grant titles."""
    #
    #     response = client.messages.create(
    #         model="claude-sonnet-4-20250514",
    #         max_tokens=500,
    #         system=system_prompt,
    #         messages=[
    #             {"role": "user", "content": user_prompt}
    #         ]
    #     )
    #
    #     return response.content[0].text
    #
    # except Exception as e:
    #     return f"⚠️ Error: {str(e)}"


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
                axis=alt.Axis(labelLimit=500, tickCount=len(data))),
        color=alt.Color('value:Q',
                       scale=alt.Scale(range=[ER_COLORS['soft_teal'], ER_COLORS['dark_teal']]),
                       legend=None),
        tooltip=[
            alt.Tooltip('category:N', title='Category'),
            alt.Tooltip('value:Q', title=value_label, format=',')
        ]
    ).properties(
        height=max(len(data) * 35, 150),
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
    page_title="Microplastic Research Trendspotter (BETA) | Engineered Resilience",
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

    /* Hide sidebar collapse button (keyboard_double_arrow icon) */
    [data-testid="stSidebar"] [data-testid="stSidebarCollapseButton"],
    [data-testid="collapsedControl"] {
        display: none !important;
    }

    /* Reduce spacing around info boxes and make lighter blue */
    .stAlert {
        margin-top: 0.5rem !important;
        margin-bottom: 0.5rem !important;
    }

    /* Lighter blue background for info boxes */
    .stAlert [data-baseweb="notification"] {
        background-color: #f0f7fa !important;
    }

    /* Tabs - styled as obvious clickable buttons spanning full width */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #e8ebe9;
        padding: 8px 10px;
        border-radius: 12px;
        border: 2px solid #d0d5d2;
        box-shadow: inset 0 2px 4px rgba(0,0,0,0.05);
        width: 100%;
        display: flex;
        justify-content: stretch;
    }

    .stTabs [data-baseweb="tab"] {
        background-color: #ffffff;
        color: #0D3B3C;
        border-radius: 8px;
        font-family: 'Source Sans Pro', sans-serif;
        font-weight: 700;
        font-size: 1.3rem;
        padding: 14px 20px;
        border: 2px solid #c5cbc7;
        transition: all 0.2s ease;
        cursor: pointer;
        box-shadow: 0 2px 4px rgba(0,0,0,0.08);
        flex: 1;
        text-align: center;
    }

    .stTabs [data-baseweb="tab"]:hover {
        background-color: #f0f7f6;
        border-color: #46B3A9;
        transform: translateY(-1px);
        box-shadow: 0 4px 8px rgba(70, 179, 169, 0.2);
    }

    .stTabs [aria-selected="true"] {
        background-color: #0D3B3C !important;
        color: white !important;
        border-color: #0D3B3C !important;
        box-shadow: 0 4px 12px rgba(13, 59, 60, 0.35);
        transform: translateY(-1px);
    }

    /* Remove the default bottom border/highlight */
    .stTabs [data-baseweb="tab-highlight"] {
        display: none;
    }

    .stTabs [data-baseweb="tab-border"] {
        display: none;
    }

    /* Hide tab scroll arrows */
    .stTabs button[data-testid="stTabsScrollButton"],
    .stTabs [data-baseweb="button-group"] button {
        display: none !important;
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

    /* Hide Streamlit branding - comprehensive selectors for Cloud deployment */
    footer {visibility: hidden !important; height: 0 !important; position: fixed !important;}
    #MainMenu {visibility: hidden !important; height: 0 !important;}
    header {visibility: hidden !important; height: 0 !important;}

    /* Reduce top padding for embedded iframe view */
    [data-testid="stAppViewContainer"] {
        padding-top: 0 !important;
    }
    [data-testid="stAppViewContainer"] > .main {
        padding-top: 0 !important;
    }
    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 1rem !important;
    }

    .stDeployButton {display: none !important;}
    [data-testid="stDecoration"] {visibility: hidden !important; height: 0 !important; position: fixed !important;}

    /* Hide the running/loading progress bar - comprehensive selectors */
    .stApp > div[data-testid="stDecoration"] {display: none !important;}
    .stSpinner {display: none !important;}
    div[data-testid="stAppViewBlockContainer"] > div:first-child > div[style*="position: fixed"] {display: none !important;}

    /* Target the green running indicator bar at top of page */
    .stApp header {display: none !important;}
    .stApp > header {display: none !important;}
    div[data-testid="stHeader"] {display: none !important;}
    .stAppHeader {display: none !important;}

    /* Hide running/rerunning status indicators */
    .stRunning, .stRerun {display: none !important;}
    [data-testid="stNotification"] {display: none !important;}

    /* Hide the colored line at top (both green running and red error) */
    .stApp > div:first-child > div:first-child {
        background: transparent !important;
        height: 0 !important;
    }
    .stApp [style*="background-color: rgb(0, 128, 0)"] {display: none !important;}
    .stApp [style*="background-color: rgb(46, 134, 193)"] {display: none !important;}
    .stApp [style*="background: linear-gradient"] {display: none !important;}
    [data-testid="stToolbar"] {visibility: hidden !important; height: 0 !important; position: fixed !important;}
    [data-testid="stStatusWidget"] {visibility: hidden !important; height: 0 !important; position: fixed !important;}
    .stApp > footer {display: none !important;}

    /* Hide "Made with Streamlit" badge variants */
    .viewerBadge_container__r5tak {display: none !important;}
    .viewerBadge_link__qRIco {display: none !important;}
    [class*="viewerBadge"] {display: none !important;}
    [class*="stBadge"] {display: none !important;}
    a[href*="streamlit.io"] {display: none !important;}

    /* Hide bottom right corner elements */
    .stApp > div:last-child > div:last-child a[href*="streamlit"] {display: none !important;}
    div[data-testid="stAppViewContainer"] ~ div a {display: none !important;}

    /* Hide any fixed position badges */
    div[style*="position: fixed"][style*="bottom"] a {display: none !important;}
    div[style*="position: fixed"][style*="right"] {display: none !important;}

    /* Additional Streamlit Cloud branding removal (2024-2026 versions) */
    [data-testid="stBottomBlockContainer"] {display: none !important;}
    .st-emotion-cache-h4xjwg {display: none !important;}
    .st-emotion-cache-1wbqy5l {display: none !important;}
    iframe[title="streamlit_badge"] {display: none !important;}
    .stAppDeployButton {display: none !important;}
    div[class*="stAppDeployButton"] {display: none !important;}
    [data-testid="manage-app-button"] {display: none !important;}

    /* Wrap text in dataframe tables so titles are fully visible */
    [data-testid="stDataFrame"] td {
        white-space: normal !important;
        word-wrap: break-word !important;
        max-width: 400px;
    }
    [data-testid="stDataFrame"] th {
        white-space: normal !important;
        word-wrap: break-word !important;
    }
    .stDataFrame td div {
        white-space: normal !important;
        word-wrap: break-word !important;
        overflow-wrap: break-word !important;
    }
</style>
""", unsafe_allow_html=True)

# Data path - use microplastic grants with pre-classified columns for STOMP Analysis
DATA_PATH = Path(__file__).parent / 'data' / 'microplastic_grants_cleaned.csv'
# Cross-field data path - use full chemical exposure dataset for Cross-Field Insights
CROSSFIELD_DATA_PATH = Path(__file__).parent / 'data' / 'chemical_exposure_grants_filtered.csv'

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

# Mechanism categories - using LLM classifications (more accurate than regex)
# LLM classifications done via Claude Sonnet in April 2026
MECHANISMS = {
    'LLM_MECH_INFLAMMATION': 'Inflammation',
    'LLM_MECH_BARRIER': 'Barrier Disruption (e.g. blood-brain)',
    'LLM_MECH_OXIDATIVE': 'Oxidative Stress',
    'LLM_MECH_NEURODEGENERATION': 'Neurodegeneration',
    'LLM_MECH_METABOLIC': 'Metabolic / Cardiovascular',
    'LLM_MECH_CELL_DEATH': 'Cell Death / Senescence',
    'LLM_MECH_MICROBIOME': 'Microbiome / Gut-Brain',
    'LLM_MECH_IMMUNE': 'Immune Dysfunction',
    'LLM_MECH_RECEPTOR': 'Receptor Signaling',
    'LLM_MECH_DNA_DAMAGE': 'DNA Damage / Genotoxicity',
    'LLM_MECH_ENDOCRINE': 'Endocrine Disruption',
}

# Pre-written summaries for each research category (for Cross-Field Insights expander)
# Updated to use LLM classification keys (April 2026) - based on actual grant content
CATEGORY_SUMMARIES = {
    'LLM_MECH_INFLAMMATION': """<strong>Cardiovascular & GI Inflammation</strong> (25 grants): Research spans cardiovascular inflammatory responses to environmental pollutants, gastrointestinal inflammation from microplastic ingestion, and systemic inflammatory effects. Key themes include biodistribution-triggered inflammation, plastic food container leaching effects, and PET-based tracking of inflammatory responses. Studies use comparative models (Xenopus, organoids, rodents) to assess tissue-specific inflammatory pathways.""",

    'LLM_MECH_OXIDATIVE': """<strong>ROS Generation & Mitochondrial Stress</strong> (19 grants): Research examines oxidative damage from environmentally relevant nanoplastics, mitochondrial dysfunction in liver and neural tissues, and ROS-mediated colorectal tumor progression. Key projects investigate weathered/photoaged plastics, hepatic oxidative stress using cell models, and airborne microplastic-induced oxidative pathways. Studies link alpha-synuclein membrane disruption to oxidative mechanisms.""",

    'LLM_MECH_NEURODEGENERATION': """<strong>Brain Aging, Alzheimer's & Parkinson's</strong> (19 grants): Research focuses on nasal-to-brain uptake pathways, alpha-synuclein membrane disruption (Parkinson's), amyloid-β aggregation (Alzheimer's), and PINK1/Parkin-mediated mitophagy. Key projects track nanoplastic lifecycle in brain tissue, kinesin-mediated axonal transport, and neurotoxicity modeling. Studies span whole-body to cellular scales using murine and human neural models.""",

    'LLM_MECH_METABOLIC': """<strong>Atherosclerosis & Vascular Dysfunction</strong> (19 grants): Research examines atherosclerotic lesion development in ApoE-deficient mice, sex-specific cardiovascular effects, and ocean microplastic-accelerated atherosclerosis. Key themes include PXR-mediated cardiovascular disease, uteroplacental vascular effects, and metabolic reprogramming in gut microbiota. Studies link dietary plastic exposure to dyslipidemia and arterial remodeling.""",

    'LLM_MECH_ENDOCRINE': """<strong>Hormone Disruption & Skeletal Effects</strong> (4 grants): Research examines PXR-mediated endocrine disruption, differential impacts of polystyrene vs PET nanoplastics on hormone systems, and DDT-nanoplastic co-exposure effects. Emerging theme: chronic dietary microplastic exposure weakening skeletal integrity through hormonal pathways.""",

    'LLM_MECH_MICROBIOME': """<strong>Gut-Brain Axis & Dysbiosis</strong> (9 grants): Research investigates microplastic-induced colorectal cancer triggers via pks+ E. coli interactions, gut-brain axis effects on cognition, and metabolic reprogramming in gut bacteria. Key projects use synthetic microbiome models, intestinal organoids, and examine microbiome ratio changes from nanoplastic exposure.""",

    'LLM_MECH_IMMUNE': """<strong>Macrophage Activation & Immunotoxicity</strong> (8 grants): Research examines cumulative environmental exposures on immune function, macrophage-mediated pro-inflammatory responses, and developmental immunotoxicity (Xenopus models). Key themes include airborne microplastic immunotoxicology, mucosal barrier immune effects, and biodistribution-triggered immune responses.""",

    'LLM_MECH_DNA_DAMAGE': """<strong>Colorectal Cancer & Genotoxicity</strong> (5 grants): Research focuses on microplastic interactions with genotoxic gut bacteria (pks+ E. coli) in early-onset colorectal cancer, genomic integrity effects from ingested nanoplastic mixtures, and airborne microplastic genotoxicity. Studies use whole-animal models to assess carcinogenic potential.""",

    'LLM_MECH_RECEPTOR': """<strong>PXR, Ion Channels & Signaling</strong> (7 grants): Research examines PXR-mediated cardiovascular effects, mechanosensitive endothelial ion channel impairment from photoaged microplastics, and intracellular signaling dynamics. Key themes include calcium flux disruption, Notch signaling effects, and colorectal tumor receptor mechanisms.""",

    'LLM_MECH_CELL_DEATH': """<strong>Cytotoxicity & Amyloid Aggregation</strong> (15 grants): Research examines alpha-synuclein membrane disruption, nanoplastic effects on amyloid-β aggregation, and dose-dependent cytotoxicity across placental and immune cell types. Key themes include liver hepatocyte toxicity, coffee cup leachate toxicity, and airborne microplastic-induced cell death.""",

    'LLM_MECH_BARRIER': """<strong>Gut Permeability & BBB Penetration</strong> (26 grants): Research investigates intestinal barrier compromise and colorectal cancer progression, BBB penetration via nasal uptake, and uteroplacental barrier effects. Key themes include tampon nanoplastic effects on gynecological barriers, tight junction disruption, and particle translocation/biodistribution.""",

    'TYPE_METHODS': """<strong>Spectroscopy & ML Detection</strong> (variable): Research develops Raman spectroscopy, Py-GC/MS, and FTIR techniques for biological samples. Key projects include machine learning quantification pipelines, blank-corrected methods for reproductive tissues, and nanoparticle tracking analysis. Studies address detection limits in CSF, lung lavage, brain tissue, and atmospheric samples.""",

    'TYPE_EXPOSURE': """<strong>Biomonitoring & Dose Assessment</strong> (variable): Research develops biomonitoring frameworks for dietary, inhalation, and dermal exposure routes. Studies measure MNPs in human tissues/fluids, develop exposure biomarkers, and quantify doses from food containers and ambient air. Population-based studies and exposomic frameworks are key approaches.""",
}

# Non-mechanism TYPE categories (for conference abstracts without mechanism focus)
# Note: These are separate from RESEARCH_TYPES which are regex-based
CONF_TYPE_CATEGORIES = {
    'TYPE_METHODS': 'Detection Methods',
    'TYPE_EXPOSURE': 'Exposure Assessment',
}

# Mapping from LLM column names to regex column names (for Cross-Field compatibility)
# Cross-Field Insights uses chemical_exposure_grants_filtered.csv which has regex columns
LLM_TO_REGEX_COL = {
    'LLM_MECH_INFLAMMATION': 'MECH_INFLAMMATION',  # Not in crossfield - will use regex pattern
    'LLM_MECH_BARRIER': 'MECH_BARRIER_DISRUPTION',
    'LLM_MECH_OXIDATIVE': 'MECH_OXIDATIVE_MITOCHONDRIAL',
    'LLM_MECH_NEURODEGENERATION': 'MECH_NEURODEGENERATION',
    'LLM_MECH_METABOLIC': None,  # New category - not in crossfield
    'LLM_MECH_CELL_DEATH': 'MECH_SENESCENCE_CELL_DEATH',
    'LLM_MECH_MICROBIOME': 'MECH_MICROBIOME',
    'LLM_MECH_IMMUNE': 'MECH_IMMUNE_DYSFUNCTION',
    'LLM_MECH_RECEPTOR': 'MECH_RECEPTOR_SIGNALING',
    'LLM_MECH_DNA_DAMAGE': 'MECH_DNA_DAMAGE',
    'LLM_MECH_ENDOCRINE': 'MECH_ENDOCRINE',
}

# Regex patterns for TYPE_ categories (used for dynamic classification in Cross-Field Insights)
# TYPE_METHODS tightened to detection-only (reduced from 83 to ~47 matches)
TYPE_PATTERNS = {
    'TYPE_METHODS': (
        r'detection\s+(?:method|limit|technique|system)|'
        r'detect\w+\s+(?:microplastic|nanoplastic|plastic\s+particle|MNP)|'
        r'sensor\s+(?:development|design|for\s+(?:microplastic|plastic))|'
        r'biosensor|spectro\w+\s+(?:detection|identification)|'
        r'FTIR\s+(?:spectro|analys|identif|detection)|Raman\s+spectro|'
        r'particle\s+(?:detection|identification|characterization)|'
        r'nanoparticle\s+track|pyrolysis.GC|Py-GC'
    ),
    'TYPE_EXPOSURE': (
        r'exposure\s+(?:assessment|pathway|route|scenario)|'
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
    'reproductive': r'reproduct(?:ive|ion)|ovary|ovarian|testes|testicular|fertility|infertility|sperm|oocyte|uterus|uterine|endometri',
    'cardiovascular': r'cardiovasc|heart|vascul|atheroscler|blood\s+vessel',
    'neurotoxicity': r'neuro|brain|cognitive|nervous\s+system|BBB',
    'cancer': r'cancer|tumor|carcinogen|oncogen|malignant',
    'immune': r'immun(?:e\s+system|odeficien|osuppress|otoxic)|lymphocyte|macrophage\s+activation|T\s*cell|B\s*cell|inflammasome|NLRP3',
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
    'MECH_BARRIER_DISRUPTION': ('Barrier Disruption (e.g. blood-brain)',
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
    }

    # Organ systems - use pre-classified columns ONLY (no runtime keyword matching)
    for key, (label, pattern) in ORGAN_SYSTEMS.items():
        col_name = organ_col_map.get(key)
        if col_name and col_name in df_unique.columns:
            # Use pre-classified column only
            matches = df_unique[col_name] == 1
        else:
            # Fallback to keyword matching only if pre-classified column not available
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

    # Research types - use pre-classified columns ONLY (no runtime keyword matching)
    for key, (label, pattern) in RESEARCH_TYPES.items():
        col_name = type_col_map.get(key)
        if col_name and col_name in df_unique.columns:
            # Use pre-classified column only
            matches = df_unique[col_name] == 1
        else:
            # Fallback to keyword matching only if pre-classified column not available
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
        # Supports regex patterns - user can enter patterns like "inflam.*" or "gut|intestin"
        if keyword_filter and keyword_filter.strip():
            try:
                # Try to use input as regex pattern directly
                pattern = re.compile(keyword_filter.strip(), re.IGNORECASE)
                match = pattern.search(text)
                if match:
                    score += 10
                    matches.insert(0, f"Keyword: {match.group()}")
            except re.error:
                # If invalid regex, fall back to literal comma-separated keywords
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
def load_data(_cache_version: str = "v16_llm_mechanisms") -> pd.DataFrame:
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


@st.cache_data
def load_crossfield_data(_cache_version: str = "v3_llm_mechanisms") -> pd.DataFrame:
    """Load full chemical exposure dataset for Cross-Field Insights comparison."""
    if not CROSSFIELD_DATA_PATH.exists():
        return pd.DataFrame()

    cf_df = pd.read_csv(CROSSFIELD_DATA_PATH, low_memory=False)

    # Create combined text field for searching
    cf_df['_text'] = cf_df['PROJECT_TITLE'].fillna('') + ' ' + cf_df['ABSTRACT_TEXT'].fillna('')

    # Reset index to avoid alignment issues
    cf_df = cf_df.reset_index(drop=True)

    return cf_df


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

# # Initialize chat state - COMMENTED OUT (Anthropic API disabled)
# init_chat_state()
#
# # ============== CHAT INTERFACE ==============
# st.markdown("""
# <div style="background: linear-gradient(135deg, #0D3B3C 0%, #1a5455 100%); padding: 1.5rem; border-radius: 12px; margin-bottom: 1.5rem;">
#     <h3 style="color: #D4A84B !important; margin: 0 0 0.5rem 0; font-family: 'Spectral', serif;">💬 Ask about the research</h3>
#     <p style="color: #FAFAF8; opacity: 0.9; margin: 0; font-size: 0.9rem;">
#         Ask questions like "Who is studying microplastics and gut health?" or "What research focuses on reproductive effects?"
#     </p>
# </div>
# """, unsafe_allow_html=True)
#
# # Chat input
# chat_col1, chat_col2 = st.columns([5, 1])
# with chat_col1:
#     user_question = st.text_input(
#         "Your question",
#         placeholder="e.g., Who is researching microplastics in drinking water?",
#         label_visibility="collapsed",
#         key="chat_input"
#     )
# with chat_col2:
#     ask_button = st.button("Ask", type="primary", use_container_width=True)
#
# # Handle chat submission
# if ask_button and user_question:
#     allowed, limit_msg = check_rate_limit()
#     if not allowed:
#         st.warning(limit_msg)
#     else:
#         st.session_state.question_count += 1
#         with st.spinner("Searching grants and thinking..."):
#             response = get_chat_response(user_question, df)
#             st.session_state.chat_messages.append({"role": "user", "content": user_question})
#             st.session_state.chat_messages.append({"role": "assistant", "content": response})
#
# # Display chat history (most recent first, limit to last 3 exchanges)
# if st.session_state.chat_messages:
#     with st.expander(f"💬 Chat history ({len(st.session_state.chat_messages)//2} questions)", expanded=True):
#         # Show messages in reverse order (most recent first)
#         messages = st.session_state.chat_messages[-6:]  # Last 3 Q&A pairs
#         for i in range(len(messages) - 1, -1, -2):
#             if i >= 1:
#                 # Assistant response
#                 st.markdown(f"**🤖 Assistant:** {messages[i]['content']}")
#                 # User question
#                 st.markdown(f"**You:** {messages[i-1]['content']}")
#                 st.markdown("---")
#
#         remaining = MAX_QUESTIONS_PER_SESSION - st.session_state.question_count
#         st.caption(f"💡 {remaining} questions remaining this session")
#
# st.markdown("<div style='height: 1rem'></div>", unsafe_allow_html=True)

# Count totals for header display
total_entries = len(df)
nih_grants_total = (~df['CORE_PROJECT_NUM'].astype(str).str.startswith('CONF_')).sum() if len(df) > 0 else 0
conf_abstracts_total = (df['CORE_PROJECT_NUM'].astype(str).str.startswith('CONF_')).sum() if len(df) > 0 else 0



if len(df) == 0:
    st.error("No data found. Make sure chemical_exposure_grants.csv exists in the data/ folder.")
    st.stop()

# Sidebar filters
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

# Group by project to show 204 unique projects (not 217 fiscal year records)
group_by_project = True

# Count NIH grants and conference abstracts (used elsewhere)
nih_count = (~df['CORE_PROJECT_NUM'].astype(str).str.startswith('CONF_')).sum()
conf_count = (df['CORE_PROJECT_NUM'].astype(str).str.startswith('CONF_')).sum()

# Calculate filtered_display for use elsewhere
if group_by_project:
    filtered_display = filtered.drop_duplicates(subset=['PROJECT_TITLE'], keep='first')
else:
    filtered_display = filtered

# Compute co-occurrence stats (full dataset for suggestions)
cooccur = compute_cooccurrence(df)

# Compute dynamic stats on filtered data
cooccur_filtered = compute_cooccurrence(filtered) if len(filtered) > 0 else {}

# ============== SUMMARY CARD & ACTIVE FILTERS ==============
# Get deduplicated data for summary
filtered_unique = filtered.drop_duplicates(subset=['PROJECT_TITLE'], keep='first') if group_by_project else filtered


# Main content - tabs
tab1, tab_organ, tab_model, tab_mech, tab4 = st.tabs(["Projects", "Organ Systems", "Model Organisms", "Mechanisms", "Cross-Field Insights"])

with tab1:
    # About this database info box with hyperlinks
    st.markdown("""
    <div style="background-color: #f0f7f7; border-left: 4px solid #0D3B3C; padding: 12px 16px; margin-bottom: 16px; border-radius: 0 8px 8px 0;">
        <strong>About this database:</strong> Staying ahead of the curve on research for emerging pollutants, like microplastics, is a challenge. Explore the latest on funded microplastics research from <a href="https://reporter.nih.gov/" target="_blank" style="color: #0D3B3C;">NIH grants</a> (FY2022-2025) and the inaugural <a href="https://hsc.unm.edu/pharmacy/research/areas/cmbm/mnp-conf/_docs/full-digital-program.pdf" target="_blank" style="color: #0D3B3C;">UNM Micro- and Nanoplastics Conference</a>. Check out details based on the affected organ, model organism, molecular mechanism, or your own query. Use <strong>Cross-Field Insights</strong> to find similar approaches for other pollutants and research experts who could directly apply their work to microplastics.
    </div>
    """, unsafe_allow_html=True)

    # Show title first
    st.subheader(f"Recent Projects and Funded Grants: {len(filtered):,}")

    # Text search box with regex support - use columns to put help icon closer to label
    st.markdown("""
    <style>
    .regex-label {
        font-size: 14px;
        font-weight: 400;
        margin-bottom: 0.25rem;
    }
    .regex-label .help-icon {
        display: inline-block;
        width: 16px;
        height: 16px;
        background: #808495;
        color: white;
        border-radius: 50%;
        text-align: center;
        font-size: 11px;
        line-height: 16px;
        margin-left: 4px;
        cursor: help;
        position: relative;
    }
    .regex-label .help-icon:hover::after {
        content: attr(data-tooltip);
        position: absolute;
        left: 50%;
        transform: translateX(-50%);
        bottom: 125%;
        background: #333;
        color: white;
        padding: 8px 12px;
        border-radius: 6px;
        font-size: 12px;
        white-space: nowrap;
        z-index: 1000;
        font-weight: normal;
    }
    </style>
    <div class="regex-label">Search by keyword for titles and abstracts (regex supported)<span class="help-icon" data-tooltip="'gut|intestin' (OR), 'inflam.*' (wildcard), 'NF.?kB' (optional char)">?</span></div>
    """, unsafe_allow_html=True)
    search_query = st.text_input(
        "Search by keyword for titles and abstracts (regex supported):",
        placeholder="e.g., -omic.*, gut|intestin, inflam.*",
        key="grant_search",
        label_visibility="collapsed"
    )

    # Use global toggle from sidebar
    show_unique = group_by_project

    # Apply text search filter
    search_filtered = filtered.copy()
    if search_query:
        # Create combined text for searching
        search_text = filtered['PROJECT_TITLE'].fillna('') + ' ' + filtered['ABSTRACT_TEXT'].fillna('')
        try:
            mask = search_text.str.contains(search_query, case=False, na=False, regex=True)
            search_filtered = filtered[mask]
        except Exception as e:
            st.error(f"Invalid search pattern: {e}")
            search_filtered = filtered

    if len(search_filtered) > 0:
        # Calculate tag count for each grant (exposures + mechanisms)
        exp_cols = [c for c in EXPOSURES.keys() if c in search_filtered.columns]
        mech_cols = [c for c in MECHANISMS.keys() if c in search_filtered.columns]
        tag_cols = exp_cols + mech_cols

        # Sort by total tag count (most prevalent first)
        filtered_sorted = search_filtered.copy()
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
            display_cols = ['Source', 'Years', 'PROJECT_TITLE', 'PI_NAMEs', 'ORG_NAME']
            unique_projects = len(grouped)
            total_records = len(filtered_sorted)
        else:
            display_df = filtered_sorted
            display_cols = ['Source', 'FISCAL_YEAR', 'PROJECT_TITLE', 'PI_NAMEs', 'ORG_NAME']

        display_cols = [c for c in display_cols if c in display_df.columns]

        # Prepare full display dataframe with nice column names
        full_table_df = display_df[display_cols].copy()
        if 'PI_NAMEs' in full_table_df.columns:
            full_table_df['PI_NAMEs'] = full_table_df['PI_NAMEs'].apply(clean_pi_names)
        col_names = {'PROJECT_TITLE': 'Title', 'PI_NAMEs': 'Contact Researcher Name', 'ORG_NAME': 'Organization', 'FISCAL_YEAR': 'FY', 'Source': 'Source', 'Years': 'Years'}
        full_table_df.columns = [col_names.get(c, c) for c in display_cols]

        # Show grants with row selection enabled
        st.markdown("<p style='color: #666; font-size: 0.95rem; margin-bottom: 0.5rem;'><strong>Click a row</strong> to view the full abstract below.</p>", unsafe_allow_html=True)

        # Paginate the results (25 per page)
        table_df = paginated_dataframe(full_table_df, key="mechanisms_table", page_size=25)

        selection = st.dataframe(
            table_df,
            use_container_width=True,
            hide_index=True,
            on_select="rerun",
            selection_mode="single-row",
            height=300,
            column_config={
                "Title": st.column_config.TextColumn("Title", width="large"),
            }
        )

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
                <p style="color: #666; margin: 0; font-size: 0.9rem;">Contact Researcher: {clean_pi_names(grant.get('PI_NAMEs', 'Unknown'))}</p>
            </div>
            """, unsafe_allow_html=True)

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
    st.markdown("#### Who is studying similar topics with other pollutants?")

    st.markdown("""
    <div style="background-color: #f0f7f7; border-left: 4px solid #0D3B3C; padding: 12px 16px; margin-bottom: 16px; border-radius: 0 8px 8px 0;">
Microplastics research is just getting started. Leverage existing biotech expertise from those working in adjacent spaces. Search 2,400+ NIH-funded researchers working on similar problems for other pollutants who have transferable approaches for microplastics.
        <br><br>
        <strong>Step 1:</strong> Choose a microplastics research category to view a summary of what's currently studied in this subfield.
        <br>
        <strong>Step 2:</strong> View the populated list of experts in the same broader category who study other pollutants. Refine your query and export CSVs.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); padding: 1rem; border-radius: 8px; margin-bottom: 1rem; border-left: 4px solid #D4A84B;">
        <p style="margin: 0; font-size: 0.95rem; color: #333;">
            Select a research category below to see NIH-funded experts studying that topic with other pollutants (pesticides, heavy metals, air pollution, etc.).
            Their validated methods, model organisms, and mechanistic insights can be adapted for microplastics work—saving years of method development.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Load the full chemical exposure dataset for cross-field comparison
    cf_df = load_crossfield_data()

    # Fixed chemical exposure to Microplastics
    my_exposure = 'EXP_MICROPLASTICS'

    # Pre-compute microplastics grant counts for each category using the crossfield dataset
    text_combined = cf_df['PROJECT_TITLE'].fillna('') + ' ' + cf_df['ABSTRACT_TEXT'].fillna('')
    mp_mask = (cf_df[my_exposure] == 1)

    category_mp_counts = {}
    for cat_key in MECHANISMS_AND_TYPES.keys():
        if cat_key.startswith('TYPE_') and cat_key in TYPE_PATTERNS:
            # TYPE_ categories use regex
            pattern = TYPE_PATTERNS[cat_key]
            cat_mask = mp_mask & text_combined.str.contains(pattern, regex=True, flags=re.IGNORECASE, na=False)
        elif cat_key in cf_df.columns:
            # LLM_MECH_ or MECH_ columns - use directly
            cat_mask = mp_mask & (cf_df[cat_key] == 1)
        else:
            cat_mask = mp_mask
        category_mp_counts[cat_key] = cat_mask.sum()

    # Category selection with dropdown + summary card
    category_options_list = list(MECHANISMS_AND_TYPES.keys())
    category_labels = [f"{MECHANISMS_AND_TYPES[k]} ({category_mp_counts.get(k, 0)} grants)" for k in category_options_list]

    # Two-column layout: selector on left, summary card on right
    sel_col, summary_col = st.columns([1, 2])

    with sel_col:
        st.markdown("##### Microplastics Research Category")
        # Create a mapping for the selectbox
        label_to_key = {label: key for key, label in zip(category_options_list, category_labels)}

        selected_label = st.selectbox(
            "Select category:",
            ["Select a category..."] + category_labels,
            key="cf_category_selector"
        )

    # Get the mechanism key from the selected label (outside the with block)
    my_mechanism = label_to_key.get(selected_label) if selected_label != "Select a category..." else None
    mech_label = MECHANISMS_AND_TYPES.get(my_mechanism, my_mechanism) if my_mechanism else None
    mp_count = category_mp_counts.get(my_mechanism, 0) if my_mechanism else 0

    with summary_col:
        if my_mechanism:
            # Get the static summary for this category
            summary = CATEGORY_SUMMARIES.get(my_mechanism, "")

            # Build organ/model tags for current category (compute here for display)
            if my_mechanism.startswith('TYPE_') and my_mechanism in TYPE_PATTERNS:
                type_pattern = TYPE_PATTERNS[my_mechanism]
                cat_mask = mp_mask & text_combined.str.contains(type_pattern, regex=True, flags=re.IGNORECASE, na=False)
            elif my_mechanism in cf_df.columns:
                cat_mask = mp_mask & (cf_df[my_mechanism] == 1)
            else:
                cat_mask = mp_mask
            cat_grants = cf_df[cat_mask]
            cat_text = cat_grants['PROJECT_TITLE'].fillna('') + ' ' + cat_grants['ABSTRACT_TEXT'].fillna('')

            # Find organ systems
            cat_organs = []
            for organ_key, (organ_label, pattern) in ORGAN_SYSTEMS.items():
                matches = cat_text.str.contains(pattern, regex=True, flags=re.IGNORECASE, na=False)
                pct = 100 * matches.sum() / len(cat_text) if len(cat_text) > 0 else 0
                if pct >= 10:
                    cat_organs.append((organ_label, round(pct, 0)))
            cat_organs.sort(key=lambda x: x[1], reverse=True)

            # Find model systems
            cat_models = []
            for model_name, pattern in MODEL_SYSTEMS.items():
                matches = cat_text.str.contains(pattern, regex=True, flags=re.IGNORECASE, na=False)
                pct = 100 * matches.sum() / len(cat_text) if len(cat_text) > 0 else 0
                if pct >= 10:
                    cat_models.append((model_name, round(pct, 0)))
            cat_models.sort(key=lambda x: x[1], reverse=True)

            # Styled summary card
            summary_text = summary if summary else f"{mp_count} microplastics grants studying {mech_label}."

            # Build tags HTML
            tags_html = ""
            if cat_organs:
                organ_tags = ' '.join([f'<span style="background: #e8f4f5; color: #0D3B3C; padding: 2px 8px; border-radius: 12px; font-size: 0.75rem;">{org[0]}</span>' for org in cat_organs[:4]])
                tags_html += f'<div style="margin: 0.5rem 0;"><strong style="color: #666; font-size: 0.8rem;">Organs:</strong> {organ_tags}</div>'
            if cat_models:
                model_tags = ' '.join([f'<span style="background: #f5f0e8; color: #8B6914; padding: 2px 8px; border-radius: 12px; font-size: 0.75rem;">{mod[0]}</span>' for mod in cat_models[:4]])
                tags_html += f'<div style="margin: 0.5rem 0;"><strong style="color: #666; font-size: 0.8rem;">Models:</strong> {model_tags}</div>'

            st.markdown(f"""<div style="background: #f8f9fa; padding: 1rem; border-radius: 10px; border-left: 4px solid #0D3B3C;">
<h5 style="margin: 0 0 0.5rem 0; color: #0D3B3C; font-size: 1.1rem;">{mech_label} <span style="background: #0D3B3C; color: white; padding: 2px 8px; border-radius: 10px; font-size: 0.8rem; font-weight: normal;">{mp_count} grants</span></h5>
{tags_html}
<p style="margin: 0; color: #444; font-size: 0.85rem; line-height: 1.5;">{summary_text}</p>
</div>""", unsafe_allow_html=True)
        else:
            # No category selected - show prompt
            st.markdown("""<div style="background: #f8f9fa; padding: 1rem; border-radius: 10px; border-left: 4px solid #ccc;">
<p style="margin: 0; color: #666; font-size: 0.9rem; font-style: italic;">Select a research category to view the summary and find related experts in adjacent fields.</p>
</div>""", unsafe_allow_html=True)

    # Only show results if a category is selected
    if my_mechanism:
        # Always run - exposure is fixed to Microplastics
        exp_label = EXPOSURES.get(my_exposure, my_exposure)
        mech_label = MECHANISMS_AND_TYPES.get(my_mechanism, my_mechanism) if my_mechanism else "All Categories"

        # Check if this is a TYPE_ category that needs regex-based filtering
        use_regex_filter = my_mechanism and my_mechanism.startswith('TYPE_') and my_mechanism in TYPE_PATTERNS

        # Build combined text column for regex matching (used for TYPE_ categories)
        text_combined = cf_df['PROJECT_TITLE'].fillna('') + ' ' + cf_df['ABSTRACT_TEXT'].fillna('')

        # Get grants from MY field (with mechanism filter)
        my_field_mask = (cf_df[my_exposure] == 1)
        if my_mechanism:
            if use_regex_filter:
                # TYPE_ categories: Use regex pattern matching
                type_pattern = TYPE_PATTERNS[my_mechanism]
                my_field_mask = my_field_mask & text_combined.str.contains(type_pattern, regex=True, flags=re.IGNORECASE, na=False)
            elif my_mechanism in cf_df.columns:
                # LLM_MECH_ columns now exist in crossfield
                my_field_mask = my_field_mask & (cf_df[my_mechanism] == 1)
        my_grants = cf_df[my_field_mask]

        # Get grants from OTHER fields (with mechanism filter)
        # Note: Non-microplastics grants don't have LLM classifications, so we use regex columns for them
        other_exp_cols = [e for e in EXPOSURES.keys() if e != my_exposure and e in cf_df.columns]
        other_field_mask = (cf_df[other_exp_cols].max(axis=1) > 0) & (cf_df[my_exposure] == 0)
        if my_mechanism:
            if use_regex_filter:
                type_pattern = TYPE_PATTERNS[my_mechanism]
                other_field_mask = other_field_mask & text_combined.str.contains(type_pattern, regex=True, flags=re.IGNORECASE, na=False)
            elif my_mechanism.startswith('LLM_MECH_'):
                # For LLM mechanisms, use corresponding regex column for non-MP grants
                regex_col = LLM_TO_REGEX_COL.get(my_mechanism)
                if regex_col and regex_col in cf_df.columns:
                    other_field_mask = other_field_mask & (cf_df[regex_col] == 1)
                # If no regex column (e.g., LLM_MECH_METABOLIC), other_grants will be empty
            elif my_mechanism in cf_df.columns:
                other_field_mask = other_field_mask & (cf_df[my_mechanism] == 1)
        other_grants = cf_df[other_field_mask]

        # Count by chemical field for the gap ratio display
        # Note: Non-microplastics grants use regex columns since they don't have LLM classifications
        chemical_counts = {}
        for exp_col in other_exp_cols:
            if my_mechanism:
                if use_regex_filter:
                    # TYPE_ categories: Use regex pattern matching
                    type_pattern = TYPE_PATTERNS[my_mechanism]
                    type_mask = text_combined.str.contains(type_pattern, regex=True, flags=re.IGNORECASE, na=False)
                    count = ((cf_df[exp_col] == 1) & type_mask & (cf_df[my_exposure] == 0)).sum()
                elif my_mechanism.startswith('LLM_MECH_'):
                    # For LLM mechanisms, use corresponding regex column for non-MP grants
                    regex_col = LLM_TO_REGEX_COL.get(my_mechanism)
                    if regex_col and regex_col in cf_df.columns:
                        count = ((cf_df[exp_col] == 1) & (cf_df[regex_col] == 1) & (cf_df[my_exposure] == 0)).sum()
                    else:
                        count = 0  # No regex equivalent (e.g., LLM_MECH_METABOLIC)
                elif my_mechanism in cf_df.columns:
                    count = ((cf_df[exp_col] == 1) & (cf_df[my_mechanism] == 1) & (cf_df[my_exposure] == 0)).sum()
                else:
                    count = ((cf_df[exp_col] == 1) & (cf_df[my_exposure] == 0)).sum()
            else:
                count = ((cf_df[exp_col] == 1) & (cf_df[my_exposure] == 0)).sum()
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
            chem_items = []
            for name, count in top_chems:
                grant_word = "grant" if count == 1 else "grants"
                chem_items.append(f'<span style="display: inline-block; background: rgba(70,179,169,0.2); padding: 6px 12px; border-radius: 16px; margin: 3px 6px 3px 0; font-size: 0.9rem;"><strong>{name}</strong> <span style="background: rgba(70,179,169,0.4); padding: 2px 6px; border-radius: 10px; margin-left: 4px; font-size: 0.8rem;">{count:,} {grant_word}</span></span>')
            chem_badges = " ".join(chem_items)
            st.markdown(f'<div style="margin-bottom: 1rem; padding: 12px 16px; background: #f8f9fa; border-radius: 8px; border-left: 4px solid #46B3A9;"><div style="font-weight: 600; margin-bottom: 8px;">Top research fields also studying {mech_label}:</div><div>{chem_badges}</div></div>', unsafe_allow_html=True)

        # Secondary filters row - moved below "Top research fields"
        st.markdown("<div style='height: 12px'></div>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            model_options = ["All Models", "In Vitro (Cells)", "Animal (Rodent)", "Animal (Zebrafish)", "Human (Cohort/Epi)", "Human (Clinical)"]
            model_filter = st.selectbox(
                "Filter by model organism:",
                model_options,
                key='crossfield_model'
            )
        with col2:
            # Keyword search field
            keyword_search = st.text_input(
                "Search by keyword for titles and abstracts (regex supported):",
                placeholder="e.g., -omic.*, gut|intestin, inflam.*",
                help="Supports regex patterns: 'gut|intestin' (OR), 'inflam.*' (wildcard), '-omic.*' (omics terms). Plain text also works.",
                key='crossfield_keyword'
            )

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

            if len(other_grants) > 0:
                # Compute similarity scores with enhanced weighting
                scored_grants = compute_grant_similarity(
                    my_grants, other_grants,
                    selected_category=my_mechanism,
                    keyword_filter=keyword_search
                )

                # Apply keyword filter - exclude grants that don't match the keyword regex
                if keyword_search and keyword_search.strip():
                    # Filter to only grants that matched the keyword (have "Keyword:" in matching_features)
                    scored_grants = scored_grants[scored_grants['matching_features'].str.contains('Keyword:', na=False)]
                    if len(scored_grants) == 0:
                        st.warning(f"No grants matched keyword filter: '{keyword_search}'")

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
                                # Get short name: everything before ( or &
                                full_name = EXPOSURES.get(exp, exp)
                                if '(' in full_name:
                                    exp_name = full_name.split('(')[0].strip()
                                elif '&' in full_name:
                                    exp_name = full_name.split('&')[0].strip()
                                else:
                                    exp_name = full_name
                                exps.append(exp_name)
                        return ', '.join(exps)

                    inspiring['Chemical(s)'] = inspiring.apply(get_exposures, axis=1)

                    # Initialize selection variable before conditional
                    inspiring_selection = None

                    if len(inspiring) == 0:
                        st.info(f"No grants found with model organism: {model_filter}")
                    else:
                        # Format for display - PI first, then chemical badge, title, model
                        display_cols = ['PI_NAMEs', 'Chemical(s)', 'PROJECT_TITLE', 'model_system', 'ORG_NAME']
                        display_cols = [c for c in display_cols if c in inspiring.columns]

                        # Prepare full dataframe for pagination
                        full_display_df = inspiring[display_cols].copy()
                        if 'PI_NAMEs' in full_display_df.columns:
                            full_display_df['PI_NAMEs'] = full_display_df['PI_NAMEs'].apply(clean_pi_names)
                        col_rename = {
                            'PI_NAMEs': 'Expert / Contact Researcher',
                            'Chemical(s)': 'Pollutants',
                            'PROJECT_TITLE': 'Project Title',
                            'model_system': 'Model',
                            'ORG_NAME': 'Institution'
                        }
                        full_display_df.columns = [col_rename.get(c, c) for c in display_cols]

                        st.caption(f"Found **{len(inspiring):,}** experts - click a row to view details")

                        # Paginate the results (25 per page)
                        display_df = paginated_dataframe(full_display_df, key="crossfield_experts", page_size=25)

                        inspiring_selection = st.dataframe(
                            display_df,
                            hide_index=True,
                            use_container_width=True,
                            on_select="rerun",
                            selection_mode="single-row",
                            column_config={
                                "Project Title": st.column_config.TextColumn("Project Title", width="large"),
                                "Pollutants": st.column_config.TextColumn("Pollutants", width="medium"),
                                "Expert / Contact Researcher": st.column_config.TextColumn("Expert / Contact Researcher", width="medium"),
                            }
                        )

                        # CSV download button for all expert studies
                        export_cols = ['PI_NAMEs', 'ORG_NAME', 'PROJECT_TITLE', 'Chemical(s)', 'model_system', 'ABSTRACT_TEXT', 'similarity_score']
                        export_cols = [c for c in export_cols if c in inspiring.columns]
                        export_df = inspiring[export_cols].copy()
                        export_df.columns = ['PI', 'Institution', 'Project Title', 'Pollutants', 'Model Organism', 'Abstract', 'Relevance Score'][:len(export_cols)]
                        csv_data = export_df.to_csv(index=False)
                        cat_slug = mech_label.lower().replace(' ', '_').replace('/', '_')[:20]
                        st.download_button(
                            label=f"Download All {len(inspiring):,} Expert Studies (CSV)",
                            data=csv_data,
                            file_name=f"crossfield_experts_{cat_slug}.csv",
                            mime="text/csv",
                            key=f"cf_download_{cat_slug}"
                        )

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

# Organ Systems Tab
with tab_organ:
    if len(filtered) > 0:
        # Run STOMP classification
        stomp_results = classify_stomp_categories(filtered)

        # Deduplicate for all STOMP views
        filtered_stomp = filtered.drop_duplicates(subset=['PROJECT_TITLE'], keep='first')

        organ_data = stomp_results['organs']
        if organ_data:
            # Calculate projects with any organ system identified using CSV columns directly
            n_grants = len(filtered_stomp)
            organ_cols = ['ORGAN_BRAIN_NERVOUS', 'ORGAN_GI_GUT', 'ORGAN_RESPIRATORY', 'ORGAN_CARDIOVASCULAR',
                         'ORGAN_REPRODUCTIVE', 'ORGAN_LIVER', 'ORGAN_KIDNEY', 'ORGAN_IMMUNE', 'ORGAN_SKIN', 'ORGAN_ENDOCRINE']
            any_organ_mask = pd.Series([False] * len(filtered_stomp), index=filtered_stomp.index)
            for col in organ_cols:
                if col in filtered_stomp.columns:
                    any_organ_mask = any_organ_mask | (filtered_stomp[col] == 1)
            any_organ = any_organ_mask.sum()
            any_organ_pct = round(100 * any_organ / n_grants, 1) if n_grants > 0 else 0
            not_categorized = n_grants - any_organ
            not_categorized_pct = round(100 * not_categorized / n_grants, 1) if n_grants > 0 else 0
            # Sort by count
            sorted_organs = sorted(organ_data.items(), key=lambda x: x[1]['count'], reverse=True)

            st.markdown("#### Which body systems are being studied?")

            col1, col2 = st.columns([2, 1])
            with col1:
                chart_data = {k: v['count'] for k, v in sorted_organs if v['count'] > 0}
                chart = create_horizontal_bar_chart(chart_data, value_label="Projects")
                st.altair_chart(chart, use_container_width=True)

            with col2:
                organ_table = [{'Organ System': k, 'Projects': v['count'], '%': f"{v['pct']}%"}
                              for k, v in sorted_organs[:8]]
                st.dataframe(pd.DataFrame(organ_table), hide_index=True, use_container_width=True)

            st.info(f"{any_organ:,} projects ({any_organ_pct}%) have at least one organ system identified (remaining {not_categorized_pct}% are general toxicity, environmental monitoring, or methods development studies)")
            st.markdown("---")

            # Drill-down with selectbox
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

                    display_cols = ['PROJECT_TITLE', 'PI_NAMEs', 'ORG_NAME', 'FISCAL_YEAR']
                    display_cols = [c for c in display_cols if c in organ_grants.columns]
                    if display_cols:
                        # Prepare full display dataframe
                        full_grants_display = organ_grants[display_cols].copy()
                        if 'PI_NAMEs' in full_grants_display.columns:
                            full_grants_display['PI_NAMEs'] = full_grants_display['PI_NAMEs'].apply(clean_pi_names)
                        col_names = {'PROJECT_TITLE': 'Title', 'PI_NAMEs': 'Contact Researcher Name', 'ORG_NAME': 'Organization', 'FISCAL_YEAR': 'FY'}
                        full_grants_display.columns = [col_names.get(c, c) for c in display_cols]
                        st.caption("Select a row to view abstract below")

                        # Paginate the results (25 per page)
                        grants_display = paginated_dataframe(full_grants_display, key="organ_systems_table", page_size=25)

                        organ_selection = st.dataframe(
                            grants_display,
                            hide_index=True,
                            use_container_width=True,
                            height=300,
                            on_select="rerun",
                            selection_mode="single-row",
                            column_config={
                                "Title": st.column_config.TextColumn("Title", width="large"),
                            }
                        )

                        # Download button for organ system results
                        organ_csv = organ_grants.to_csv(index=False)
                        st.download_button(
                            f"Download {selected_organ} Projects (CSV)",
                            organ_csv,
                            f"{selected_organ.lower().replace(' ', '_')}_projects.csv",
                            "text/csv",
                            key="organ_download"
                        )

                        # Show abstract for selected row
                        if organ_selection and organ_selection.selection and organ_selection.selection.rows:
                            selected_idx = organ_selection.selection.rows[0]
                            grant_row = organ_grants.iloc[selected_idx]
                            st.markdown("---")
                            st.markdown(f"**{grant_row['PROJECT_TITLE']}**")
                            st.markdown(f"*Contact Researcher:* {clean_pi_names(grant_row.get('PI_NAMEs', 'Unknown'))} | *Org:* {grant_row.get('ORG_NAME', 'Unknown')} | *FY:* {int(grant_row.get('FISCAL_YEAR', 0))}")
                            abstract = grant_row.get('ABSTRACT_TEXT', 'No abstract available')
                            if pd.isna(abstract):
                                abstract = 'No abstract available'
                            st.markdown("**Abstract:**")
                            st.write(abstract)

    else:
        st.info("Filter grants to see organ system analysis.")

# Model Systems Tab
with tab_model:
    if len(filtered) > 0:
        # Run STOMP classification
        stomp_results = classify_stomp_categories(filtered)

        # Deduplicate for all STOMP views
        filtered_stomp = filtered.drop_duplicates(subset=['PROJECT_TITLE'], keep='first')

        # Calculate projects with any model system identified using pre-classified columns
        n_grants = len(filtered_stomp)
        model_cols = ['MODEL_INVITRO', 'MODEL_RODENT', 'MODEL_ZEBRAFISH', 'MODEL_OTHER_ANIMAL', 'MODEL_HUMAN', 'MODEL_ENVIRONMENTAL']
        any_model_mask = pd.Series([False] * len(filtered_stomp))
        for col in model_cols:
            if col in filtered_stomp.columns:
                any_model_mask = any_model_mask | (filtered_stomp[col] == 1)
        any_model = any_model_mask.sum()
        any_model_pct = round(100 * any_model / n_grants, 1) if n_grants > 0 else 0

        st.markdown("#### What model organisms are being used?")

        # Model systems - use pre-classified columns from CSV
        MODEL_COL_MAP = {
            'In Vitro': 'MODEL_INVITRO',
            'Rodent': 'MODEL_RODENT',
            'Zebrafish': 'MODEL_ZEBRAFISH',
            'Human': 'MODEL_HUMAN',
            'Environmental': 'MODEL_ENVIRONMENTAL',
            'Other Animal': 'MODEL_OTHER_ANIMAL',
        }

        n_total = len(filtered_stomp)

        model_counts = {}
        for name, col in MODEL_COL_MAP.items():
            if col in filtered_stomp.columns:
                count = (filtered_stomp[col] == 1).sum()
                model_counts[name] = {'count': count, 'pct': round(100 * count / n_total, 1) if n_total > 0 else 0, 'col': col}

        sorted_models = sorted(model_counts.items(), key=lambda x: x[1]['count'], reverse=True)

        col1, col2 = st.columns([2, 1])
        with col1:
            # Include all model organisms (even with 0 count) for complete view
            chart_data = {k: v['count'] for k, v in sorted_models}
            if chart_data:
                chart = create_horizontal_bar_chart(chart_data, value_label="Projects")
                st.altair_chart(chart, use_container_width=True)

        with col2:
            # Show all model organisms in table
            model_table = [{'Model Organism': k, 'Projects': v['count'], '%': f"{v['pct']}%"}
                          for k, v in sorted_models]
            st.dataframe(pd.DataFrame(model_table), hide_index=True, use_container_width=True)

        st.info(f"{any_model:,} projects ({any_model_pct}%) have at least one model organism identified (many projects use multiple model systems)")
        st.markdown("---")

        # Drill-down
        model_options = [f"{name} ({info['count']})" for name, info in sorted_models if info['count'] > 0]
        if model_options:
            selected_model_option = st.selectbox(
                "Explore projects by model organism",
                ["Select a model organism..."] + model_options,
                key="model_select"
            )
            selected_model = selected_model_option.rsplit(" (", 1)[0] if selected_model_option != "Select a model organism..." else None
        else:
            selected_model = None

        if selected_model and selected_model in MODEL_COL_MAP:
            model_col = MODEL_COL_MAP[selected_model]
            model_grants = filtered_stomp[filtered_stomp[model_col] == 1].copy()

            st.markdown(f"### {selected_model}")
            st.markdown(f"**{len(model_grants):,} projects** using this model")

            display_cols = ['PROJECT_TITLE', 'PI_NAMEs', 'ORG_NAME', 'FISCAL_YEAR']
            display_cols = [c for c in display_cols if c in model_grants.columns]
            if display_cols:
                # Prepare full display dataframe
                full_grants_display = model_grants[display_cols].copy()
                if 'PI_NAMEs' in full_grants_display.columns:
                    full_grants_display['PI_NAMEs'] = full_grants_display['PI_NAMEs'].apply(clean_pi_names)
                col_names = {'PROJECT_TITLE': 'Title', 'PI_NAMEs': 'Contact Researcher Name', 'ORG_NAME': 'Organization', 'FISCAL_YEAR': 'FY'}
                full_grants_display.columns = [col_names.get(c, c) for c in display_cols]
                st.caption("Select a row to view abstract below")

                # Paginate the results (25 per page)
                grants_display = paginated_dataframe(full_grants_display, key="model_organisms_table", page_size=25)

                model_selection = st.dataframe(
                    grants_display,
                    hide_index=True,
                    use_container_width=True,
                    height=300,
                    on_select="rerun",
                    selection_mode="single-row",
                    column_config={
                        "Title": st.column_config.TextColumn("Title", width="large"),
                    }
                )

                # Download button for model system results
                model_csv = model_grants.to_csv(index=False)
                st.download_button(
                    f"Download {selected_model} Projects (CSV)",
                    model_csv,
                    f"{selected_model.lower().replace(' ', '_').replace('/', '_')}_projects.csv",
                    "text/csv",
                    key="model_download"
                )

                if model_selection and model_selection.selection and model_selection.selection.rows:
                    selected_idx = model_selection.selection.rows[0]
                    grant_row = model_grants.iloc[selected_idx]
                    st.markdown("---")
                    st.markdown(f"**{grant_row['PROJECT_TITLE']}**")
                    st.markdown(f"*Contact Researcher:* {clean_pi_names(grant_row.get('PI_NAMEs', 'Unknown'))} | *Org:* {grant_row.get('ORG_NAME', 'Unknown')} | *FY:* {int(grant_row.get('FISCAL_YEAR', 0))}")
                    abstract = grant_row.get('ABSTRACT_TEXT', 'No abstract available')
                    if pd.isna(abstract):
                        abstract = 'No abstract available'
                    st.markdown("**Abstract:**")
                    st.write(abstract)

    with tab_mech:
        # Use pre-classified MECH_* columns from CSV (same as Cross-Field Insights tab)
        # This provides consistent categorization across both tabs
        mech_name_to_key = {}
        mech_data = {}
        n_grants = len(filtered_stomp)

        # Only show mechanisms from MECHANISMS dict (excludes oxidative, developmental, epigenetic)
        for key, label in MECHANISMS.items():
            # Use pre-classified column
            if key in filtered_stomp.columns:
                matches = filtered_stomp[key] == 1
                count = int(matches.sum())
            else:
                count = 0
            pct = round(100 * count / n_grants, 1) if n_grants > 0 else 0
            mech_data[label] = {'count': count, 'pct': pct, 'key': key}
            mech_name_to_key[label] = key

        if mech_data:
            # Calculate mechanism stats for header
            any_mech_mask = pd.Series(False, index=filtered_stomp.index)
            for label, info in mech_data.items():
                key = info['key']
                if key in filtered_stomp.columns:
                    any_mech_mask = any_mech_mask | (filtered_stomp[key] == 1)
            any_mech = any_mech_mask.sum()
            any_pct = round(100 * any_mech / n_grants, 1) if n_grants > 0 else 0

            st.markdown("#### What biological mechanisms are being studied?")

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

            st.info(f"{any_mech:,} projects ({any_pct}%) have at least one mechanism identified (many projects study multiple mechanisms)")
            st.markdown("---")

            # Drill-down with selectbox
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
                # Use pre-classified column for drill-down (same as Cross-Field Insights)
                if mech_key in filtered_stomp.columns:
                    mech_grants = filtered_stomp[filtered_stomp[mech_key] == 1].copy()
                else:
                    mech_grants = pd.DataFrame()

                st.markdown(f"### {selected_mech_stomp}")
                st.markdown(f"**{len(mech_grants):,} projects** studying this mechanism")

                display_cols = ['PROJECT_TITLE', 'PI_NAMEs', 'ORG_NAME', 'FISCAL_YEAR']
                display_cols = [c for c in display_cols if c in mech_grants.columns]
                if display_cols:
                    # Prepare full display dataframe
                    full_grants_display = mech_grants[display_cols].copy()
                    if 'PI_NAMEs' in full_grants_display.columns:
                        full_grants_display['PI_NAMEs'] = full_grants_display['PI_NAMEs'].apply(clean_pi_names)
                    col_names = {'PROJECT_TITLE': 'Title', 'PI_NAMEs': 'Contact Researcher Name', 'ORG_NAME': 'Organization', 'FISCAL_YEAR': 'FY'}
                    full_grants_display.columns = [col_names.get(c, c) for c in display_cols]
                    st.caption("Select a row to view abstract below")

                    # Paginate the results (25 per page)
                    grants_display = paginated_dataframe(full_grants_display, key="mechanisms_stomp_table", page_size=25)

                    mech_selection = st.dataframe(
                        grants_display,
                        hide_index=True,
                        use_container_width=True,
                        height=300,
                        on_select="rerun",
                        selection_mode="single-row",
                        column_config={
                            "Title": st.column_config.TextColumn("Title", width="large"),
                        }
                    )

                    # Download button for mechanism results
                    mech_csv = mech_grants.to_csv(index=False)
                    st.download_button(
                        f"Download {selected_mech_stomp} Projects (CSV)",
                        mech_csv,
                        f"{selected_mech_stomp.lower().replace(' ', '_').replace('/', '_')}_projects.csv",
                        "text/csv",
                        key="mech_download"
                    )

                    # Show abstract for selected row
                    if mech_selection and mech_selection.selection and mech_selection.selection.rows:
                        selected_idx = mech_selection.selection.rows[0]
                        grant_row = mech_grants.iloc[selected_idx]
                        st.markdown("---")
                        st.markdown(f"**{grant_row['PROJECT_TITLE']}**")
                        st.markdown(f"*Contact Researcher:* {clean_pi_names(grant_row.get('PI_NAMEs', 'Unknown'))} | *Org:* {grant_row.get('ORG_NAME', 'Unknown')} | *FY:* {int(grant_row.get('FISCAL_YEAR', 0))}")
                        abstract = grant_row.get('ABSTRACT_TEXT', 'No abstract available')
                        if pd.isna(abstract):
                            abstract = 'No abstract available'
                        st.markdown("**Abstract:**")
                        st.write(abstract)

        else:
            st.info("No mechanism patterns matched in the current dataset.")

