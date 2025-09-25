import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import numpy as np
from datetime import datetime
import os
import requests
import zipfile
from io import BytesIO
import warnings
from textblob import TextBlob
import re
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(
    page_title="CORD-19 Data Explorer",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
.main-header {
    font-size: 3rem;
    color: #1e88e5;
    text-align: center;
    margin-bottom: 2rem;
}
.metric-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 20px;
    border-radius: 10px;
    color: white;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=3600)
def load_data():
    """Load data from GitHub Releases with multiple fallback options"""
    
    st.info("🔬 Loading CORD-19 research data from GitHub Releases...")
    
    # Try GitHub Releases first
    release_data = load_from_github_releases()
    if release_data is not None:
        return release_data, "GitHub Releases"
    
    # Try direct CSV from repo
    repo_data = load_from_github_repo()
    if repo_data is not None:
        return repo_data, "GitHub Repository"
    
    # Final fallback to comprehensive demo data
    st.warning("⚠️ Using comprehensive demo data - contains realistic COVID-19 research patterns")
    demo_data = create_comprehensive_demo_data()
    return process_data(demo_data), "Comprehensive Demo Data"

def load_from_github_releases():
    """Load data from GitHub Releases"""
    try:
        # Your GitHub Releases URL
        release_url = "https://github.com/kimenjuivy/CORD-19-DATA-ANALYSIS/releases/download/v1.0-data/dataset.zip"
        
        st.info("📦 Downloading from GitHub Releases...")
        response = requests.get(release_url, timeout=30)
        response.raise_for_status()
        
        # Extract zip file
        with zipfile.ZipFile(BytesIO(response.content)) as zip_ref:
            file_list = zip_ref.namelist()
            st.info(f"Files in release: {file_list}")
            
            # Look for CSV files
            csv_files = [f for f in file_list if f.endswith('.csv')]
            if csv_files:
                with zip_ref.open(csv_files[0]) as csv_file:
                    df = pd.read_csv(csv_file)
                st.success(f"✅ Successfully loaded {len(df):,} records from GitHub Releases")
                return process_data(df)
        
        return None
    except Exception as e:
        st.warning(f"GitHub Releases unavailable: {str(e)}")
        return None

def load_from_github_repo():
    """Try loading from GitHub repository directly"""
    try:
        # Try different possible file paths in your repo
        possible_paths = [
            "data/dataset.csv",
            "dataset.csv",
            "cord19_data.csv",
            "data/cord19.csv"
        ]
        
        base_url = "https://raw.githubusercontent.com/kimenjuivy/CORD-19-DATA-ANALYSIS/main/"
        
        for path in possible_paths:
            try:
                url = base_url + path
                st.info(f"Trying: {url}")
                df = pd.read_csv(url)
                if len(df) > 0:
                    st.success(f"✅ Successfully loaded {len(df):,} records from {path}")
                    return process_data(df)
            except:
                continue
        return None
    except Exception as e:
        st.warning(f"GitHub repo access failed: {str(e)}")
        return None

def create_comprehensive_demo_data():
    """Create realistic comprehensive demo data"""
    np.random.seed(42)
    n_records = 2500
    
    # Expanded lists for more variety
    journals = [
        'Lancet', 'Nature Medicine', 'JAMA', 'BMJ', 'New England Journal of Medicine',
        'Science', 'Cell', 'Nature Communications', 'PNAS', 'BioRxiv', 'MedRxiv',
        'Journal of Medical Virology', 'Clinical Infectious Diseases', 'Nature',
        'Science Advances', 'Cell Reports', 'PLOS One', 'BMC Medicine'
    ]
    
    diseases = ['COVID-19', 'SARS-CoV-2', 'Coronavirus', 'Pandemic', 'Vaccine', 
                'Treatment', 'Epidemiology', 'Transmission', 'Variants']
    
    research_types = ['Clinical Trial', 'Systematic Review', 'Case Study', 'Epidemiological Study',
                     'Basic Research', 'Public Health Analysis', 'Meta-Analysis', 'Observational Study']
    
    countries = ['USA', 'China', 'UK', 'Italy', 'Germany', 'France', 'Spain', 'India', 'Brazil', 'Japan']
    
    # Generate realistic date range
    dates = pd.date_range('2019-12-01', '2023-12-31', periods=n_records)
    
    data = {
        'cord_uid': [f'UID_{i:06d}' for i in range(n_records)],
        'title': [generate_realistic_title(diseases, research_types) for _ in range(n_records)],
        'authors': [generate_authors() for _ in range(n_records)],
        'journal': np.random.choice(journals, n_records, p=np.linspace(0.2, 0.01, len(journals))),
        'publish_time': dates,
        'abstract': [generate_abstract(diseases, research_types) for _ in range(n_records)],
        'source': np.random.choice(['PubMed', 'WHO', 'CDC', 'BioRxiv', 'MedRxiv', 'Crossref'], n_records),
        'url': [f'https://example.com/paper/{i}' for i in range(n_records)],
        'country': np.random.choice(countries, n_records),
        'citation_count': np.random.poisson(15, n_records),
        'study_type': np.random.choice(research_types, n_records),
        'has_full_text': np.random.choice([True, False], n_records, p=[0.7, 0.3])
    }
    
    return pd.DataFrame(data)

def generate_realistic_title(diseases, research_types):
    """Generate realistic research paper titles"""
    disease = np.random.choice(diseases)
    research_type = np.random.choice(research_types)
    
    templates = [
        f"{research_type} of {disease}: A Comprehensive Analysis",
        f"Impact of {disease} on Global Health Systems",
        f"Novel Approaches to {disease} {research_type}",
        f"{disease} {research_type}: Lessons from the Pandemic",
        f"Advanced {research_type} in {disease} Management"
    ]
    
    return np.random.choice(templates)

def generate_authors():
    """Generate realistic author lists"""
    last_names = ['Smith', 'Johnson', 'Williams', 'Brown', 'Jones', 'Garcia', 'Miller', 'Davis']
    first_initials = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K']
    
    num_authors = np.random.poisson(4) + 1
    authors = []
    
    for i in range(num_authors):
        if i == 0:
            authors.append(f"{np.random.choice(last_names)} {np.random.choice(first_initials)}")
        else:
            authors.append(f"{np.random.choice(last_names)} {np.random.choice(first_initials)}")
    
    return "; ".join(authors)

def generate_abstract(diseases, research_types):
    """Generate realistic abstracts"""
    disease = np.random.choice(diseases)
    research_type = np.random.choice(research_types)
    
    templates = [
        f"This {research_type.lower()} investigates the impact of {disease} on global health systems. Our findings demonstrate significant implications for public health policy and future pandemic preparedness.",
        f"Background: {disease} has emerged as a major global health challenge. Methods: We conducted a {research_type.lower()} to analyze key trends and patterns. Results: Our analysis reveals important insights into {disease} transmission and control.",
        f"Objective: To examine the effectiveness of various interventions for {disease}. Design: {research_type} analyzing data from multiple sources. Conclusion: Our study provides valuable insights for {disease} management strategies."
    ]
    
    return np.random.choice(templates)

def process_data(df):
    """Enhanced data processing with more features"""
    df_processed = df.copy()
    
    # Date processing
    if 'publish_time' in df_processed.columns:
        df_processed['publish_time'] = pd.to_datetime(df_processed['publish_time'], errors='coerce')
        df_processed['publication_year'] = df_processed['publish_time'].dt.year.fillna(2020).astype(int)
        df_processed['publication_month'] = df_processed['publish_time'].dt.month.fillna(1)
        df_processed['publication_quarter'] = df_processed['publish_time'].dt.quarter.fillna(1)
    else:
        df_processed['publication_year'] = 2020
        df_processed['publish_time'] = pd.to_datetime('2020-01-01')
    
    # Text analysis features
    if 'title' in df_processed.columns:
        df_processed['title_word_count'] = df_processed['title'].astype(str).str.split().str.len().fillna(0)
        df_processed['title_length'] = df_processed['title'].astype(str).str.len().fillna(0)
    
    if 'abstract' in df_processed.columns:
        df_processed['abstract_word_count'] = df_processed['abstract'].astype(str).str.split().str.len().fillna(0)
        df_processed['abstract_length'] = df_processed['abstract'].astype(str).str.len().fillna(0)
        df_processed['has_abstract'] = (df_processed['abstract'].astype(str).str.len() > 20) & \
                                      (df_processed['abstract'] != 'No Abstract Available')
    
    # Author analysis
    if 'authors' in df_processed.columns:
        df_processed['author_count'] = df_processed['authors'].astype(str).str.split(';').str.len().fillna(1)
        df_processed['has_multiple_authors'] = df_processed['author_count'] > 1
    
    # Impact metrics
    if 'citation_count' not in df_processed.columns:
        df_processed['citation_count'] = np.random.poisson(10, len(df_processed))
    
    df_processed['citation_category'] = pd.cut(
        df_processed['citation_count'],
        bins=[-1, 0, 5, 20, 100, float('inf')],
        labels=['None', 'Low', 'Medium', 'High', 'Very High']
    )
    
    # Fill missing values
    required_columns = {
        'abstract': 'No Abstract Available',
        'journal': 'Unknown Journal', 
        'source': 'Unknown Source',
        'authors': 'Unknown Authors',
        'country': 'Unknown Country',
        'study_type': 'Unknown Type'
    }
    
    for col, default_value in required_columns.items():
        if col not in df_processed.columns:
            df_processed[col] = default_value
        else:
            df_processed[col] = df_processed[col].fillna(default_value)
    
    return df_processed

def calculate_text_sentiment(text):
    """Calculate sentiment of text using simple rule-based approach"""
    if pd.isna(text) or text == 'No Abstract Available':
        return 0
    
    positive_words = ['effective', 'successful', 'improved', 'beneficial', 'promising', 'significant']
    negative_words = ['limitation', 'challenge', 'difficult', 'risk', 'adverse', 'fatal']
    
    text_lower = text.lower()
    positive_score = sum(1 for word in positive_words if word in text_lower)
    negative_score = sum(1 for word in negative_words if word in text_lower)
    
    return positive_score - negative_score

# Main app
def main():
    st.markdown('<h1 class="main-header">🔬 CORD-19 Comprehensive Research Explorer</h1>', unsafe_allow_html=True)
    st.markdown("### Advanced Analytics for COVID-19 Research Publications")
    
    # Load data
    with st.spinner("Loading comprehensive research data..."):
        data_result = load_data()
    
    if data_result[0] is None:
        st.error("❌ Failed to load data. Please check your data sources.")
        return
    
    df, dataset_info = data_result
    st.success(f"✅ Loaded {len(df):,} research papers from {dataset_info}")
    
    # Manual upload option
    with st.expander("📁 Upload Custom Dataset"):
        uploaded_file = st.file_uploader("Or upload your own CSV file", type=['csv'])
        if uploaded_file is not None:
            try:
                custom_df = pd.read_csv(uploaded_file)
                df = process_data(custom_df)
                dataset_info = "Uploaded File"
                st.success("✅ Using uploaded file!")
            except Exception as e:
                st.error(f"Error reading uploaded file: {e}")
    
    # Sidebar controls
    st.sidebar.header("🎛️ Advanced Controls")
    st.sidebar.info(f"**Source:** {dataset_info}")
    st.sidebar.info(f"**Total Papers:** {len(df):,}")
    
    # Enhanced filters
    years = sorted(df['publication_year'].unique())
    year_range = st.sidebar.slider(
        "Publication Year Range",
        min_value=int(min(years)),
        max_value=int(max(years)),
        value=(int(min(years)), int(max(years)))
    )
    
    filtered_df = df[
        (df['publication_year'] >= year_range[0]) & 
        (df['publication_year'] <= year_range[1])
    ]
    
    # Additional filters
    col1, col2, col3 = st.sidebar.columns(3)
    
    with col1:
        min_citations = st.number_input("Min Citations", value=0)
        filtered_df = filtered_df[filtered_df['citation_count'] >= min_citations]
    
    with col2:
        min_authors = st.number_input("Min Authors", value=1, min_value=1)
        filtered_df = filtered_df[filtered_df['author_count'] >= min_authors]
    
    with col3:
        with_abstract = st.checkbox("With Abstract", value=True)
        if with_abstract:
            filtered_df = filtered_df[filtered_df['has_abstract']]
    
    # Journal filter
    journals = filtered_df['journal'].value_counts().index.tolist()
    selected_journals = st.sidebar.multiselect("Filter Journals", journals, default=[])
    if selected_journals:
        filtered_df = filtered_df[filtered_df['journal'].isin(selected_journals)]
    
    # Enhanced metrics
    st.subheader("📊 Research Overview Metrics")
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    with col1:
        st.metric("Total Papers", f"{len(filtered_df):,}")
    with col2:
        st.metric("Unique Journals", f"{filtered_df['journal'].nunique():,}")
    with col3:
        avg_citations = filtered_df['citation_count'].mean()
        st.metric("Avg Citations", f"{avg_citations:.1f}")
    with col4:
        st.metric("Avg Authors", f"{filtered_df['author_count'].mean():.1f}")
    with col5:
        papers_with_abstracts = filtered_df['has_abstract'].sum()
        st.metric("With Abstracts", f"{papers_with_abstracts:,}")
    with col6:
        st.metric("Time Span", f"{year_range[0]}-{year_range[1]}")
    
    # Main tabs with comprehensive analysis
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📈 Trends", "📚 Journals", "👥 Authors", "🌍 Geographic", "📖 Content", "📋 Data"
    ])
    
    with tab1:
        st.subheader("Publication Trends and Impact")
        col1, col2 = st.columns(2)
        
        with col1:
            # Yearly publications with citations
            yearly_data = filtered_df.groupby('publication_year').agg({
                'cord_uid': 'count',
                'citation_count': 'mean'
            }).reset_index()
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=yearly_data['publication_year'],
                y=yearly_data['cord_uid'],
                name='Publications',
                marker_color='blue'
            ))
            fig.add_trace(go.Scatter(
                x=yearly_data['publication_year'],
                y=yearly_data['citation_count'],
                name='Avg Citations',
                yaxis='y2',
                line=dict(color='red', width=3)
            ))
            
            fig.update_layout(
                title='Publications and Citation Impact Over Time',
                xaxis_title='Year',
                yaxis_title='Number of Publications',
                yaxis2=dict(title='Average Citations', overlaying='y', side='right'),
                height=400
            )
            st.plotly_chart(fig, width='stretch')
        
        with col2:
            # Monthly trends
            monthly_data = filtered_df.groupby([
                filtered_df['publish_time'].dt.year,
                filtered_df['publish_time'].dt.month
            ]).size().reset_index(name='count')
            monthly_data['date'] = pd.to_datetime(
                monthly_data['publish_time'].astype(str) + '-' + monthly_data['publish_time'].astype(str)
            )
            
            fig = px.line(monthly_data, x='date', y='count', 
                         title='Monthly Publication Trends',
                         height=400)
            st.plotly_chart(fig, width='stretch')
    
    with tab2:
        st.subheader("Journal Analysis")
        col1, col2 = st.columns(2)
        
        with col1:
            # Top journals by publication count
            top_journals = filtered_df['journal'].value_counts().head(15)
            fig = px.bar(top_journals, x=top_journals.values, y=top_journals.index,
                        orientation='h', title='Top 15 Journals by Publication Count',
                        height=500)
            st.plotly_chart(fig, width='stretch')
        
        with col2:
            # Journal impact (citations)
            journal_impact = filtered_df.groupby('journal').agg({
                'citation_count': 'mean',
                'cord_uid': 'count'
            }).nlargest(15, 'cord_uid')
            
            fig = px.scatter(journal_impact, x='cord_uid', y='citation_count',
                           size='citation_count', hover_name=journal_impact.index,
                           title='Journal Impact: Publications vs Citations',
                           height=500)
            st.plotly_chart(fig, width='stretch')
    
    with tab3:
        st.subheader("Author and Collaboration Analysis")
        col1, col2 = st.columns(2)
        
        with col1:
            # Author count distribution
            fig = px.histogram(filtered_df, x='author_count', 
                             title='Distribution of Authors per Paper',
                             nbins=20, height=400)
            st.plotly_chart(fig, width='stretch')
        
        with col2:
            # Collaboration trends over time
            collaboration_trends = filtered_df.groupby('publication_year').agg({
                'author_count': 'mean',
                'has_multiple_authors': 'mean'
            }).reset_index()
            
            fig = px.line(collaboration_trends, x='publication_year', 
                         y=['author_count', 'has_multiple_authors'],
                         title='Collaboration Trends Over Time',
                         height=400)
            st.plotly_chart(fig, width='stretch')
    
    with tab4:
        st.subheader("Geographic and Source Analysis")
        col1, col2 = st.columns(2)
        
        with col1:
            # Country distribution
            country_data = filtered_df['country'].value_counts().head(15)
            fig = px.pie(country_data, values=country_data.values, names=country_data.index,
                        title='Top 15 Countries by Publication Count',
                        height=500)
            st.plotly_chart(fig, width='stretch')
        
        with col2:
            # Source distribution over time
            source_trends = pd.crosstab(filtered_df['publication_year'], filtered_df['source'])
            fig = px.area(source_trends, title='Publication Sources Over Time',
                         height=500)
            st.plotly_chart(fig, width='stretch')
    
    with tab5:
        st.subheader("Content Analysis")
        col1, col2 = st.columns(2)
        
        with col1:
            # Word cloud of titles
            text = ' '.join(filtered_df['title'].dropna().astype(str))
            if text:
                wordcloud = WordCloud(width=600, height=300, background_color='white').generate(text)
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.imshow(wordcloud, interpolation='bilinear')
                ax.axis('off')
                ax.set_title('Common Words in Research Titles', fontsize=16)
                st.pyplot(fig)
        
        with col2:
            # Abstract length vs citations
            fig = px.scatter(filtered_df, x='abstract_word_count', y='citation_count',
                           trendline='lowess', title='Abstract Length vs Citation Impact',
                           height=400)
            st.plotly_chart(fig, width='stretch')
        
        # Study type analysis
        study_type_data = filtered_df['study_type'].value_counts()
        fig = px.bar(study_type_data, x=study_type_data.index, y=study_type_data.values,
                    title='Distribution of Research Types',
                    height=400)
        st.plotly_chart(fig, width='stretch')
    
    with tab6:
        st.subheader("Raw Data and Export")
        
        # Data summary
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Filtered Papers", f"{len(filtered_df):,}")
        with col2:
            st.metric("Columns", f"{len(filtered_df.columns):,}")
        with col3:
            st.metric("Data Size", f"{filtered_df.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB")
        
        # Data preview
        st.dataframe(filtered_df.head(100), width='stretch', height=400)
        
        # Export options
        col1, col2 = st.columns(2)
        with col1:
            csv = filtered_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Filtered Data (CSV)",
                data=csv,
                file_name=f'cord19_analysis_{datetime.now().strftime("%Y%m%d_%H%M")}.csv',
                mime='text/csv'
            )
        
        with col2:
            # Summary statistics
            with st.expander("📊 Dataset Summary Statistics"):
                st.write(filtered_df.describe())
    
    # Footer
    st.markdown("---")
    st.markdown("""
    **Advanced CORD-19 Research Explorer** • 
    *Comprehensive analysis of COVID-19 research publications* • 
    Data updated automatically from multiple sources
    """)

if __name__ == "__main__":
    main()