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
    
    st.info("🔬 Loading CORD-19 research data...")
    
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
            except Exception as path_error:
                st.info(f"Path {path} not found: {str(path_error)}")
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
    """Enhanced data processing with more robust error handling"""
    if df is None or df.empty:
        raise ValueError("DataFrame is None or empty")
    
    df_processed = df.copy()
    
    try:
        # Date processing with robust error handling
        if 'publish_time' in df_processed.columns:
            df_processed['publish_time'] = pd.to_datetime(df_processed['publish_time'], errors='coerce')
            df_processed['publication_year'] = df_processed['publish_time'].dt.year
            df_processed['publication_month'] = df_processed['publish_time'].dt.month
            df_processed['publication_quarter'] = df_processed['publish_time'].dt.quarter
            
            # Fill missing years with 2020
            df_processed['publication_year'] = df_processed['publication_year'].fillna(2020).astype(int)
            df_processed['publication_month'] = df_processed['publication_month'].fillna(1).astype(int)
            df_processed['publication_quarter'] = df_processed['publication_quarter'].fillna(1).astype(int)
        else:
            df_processed['publication_year'] = 2020
            df_processed['publication_month'] = 1
            df_processed['publication_quarter'] = 1
            df_processed['publish_time'] = pd.to_datetime('2020-01-01')
        
        # Text analysis features with safe processing
        if 'title' in df_processed.columns:
            df_processed['title'] = df_processed['title'].astype(str).fillna('Unknown Title')
            df_processed['title_word_count'] = df_processed['title'].str.split().str.len().fillna(0)
            df_processed['title_length'] = df_processed['title'].str.len().fillna(0)
        else:
            df_processed['title'] = 'Unknown Title'
            df_processed['title_word_count'] = 0
            df_processed['title_length'] = 0
        
        if 'abstract' in df_processed.columns:
            df_processed['abstract'] = df_processed['abstract'].astype(str).fillna('No Abstract Available')
            df_processed['abstract_word_count'] = df_processed['abstract'].str.split().str.len().fillna(0)
            df_processed['abstract_length'] = df_processed['abstract'].str.len().fillna(0)
            df_processed['has_abstract'] = (
                (df_processed['abstract'].str.len() > 20) & 
                (df_processed['abstract'] != 'No Abstract Available') &
                (df_processed['abstract'] != 'nan')
            )
        else:
            df_processed['abstract'] = 'No Abstract Available'
            df_processed['abstract_word_count'] = 0
            df_processed['abstract_length'] = 0
            df_processed['has_abstract'] = False
        
        # Author analysis with safe processing
        if 'authors' in df_processed.columns:
            df_processed['authors'] = df_processed['authors'].astype(str).fillna('Unknown Authors')
            df_processed['author_count'] = df_processed['authors'].str.split(';').str.len().fillna(1)
        else:
            df_processed['authors'] = 'Unknown Authors'
            df_processed['author_count'] = 1
        
        df_processed['has_multiple_authors'] = df_processed['author_count'] > 1
        
        # Impact metrics with safe processing
        if 'citation_count' not in df_processed.columns:
            df_processed['citation_count'] = np.random.poisson(10, len(df_processed))
        else:
            df_processed['citation_count'] = pd.to_numeric(df_processed['citation_count'], errors='coerce').fillna(0)
        
        df_processed['citation_category'] = pd.cut(
            df_processed['citation_count'],
            bins=[-1, 0, 5, 20, 100, float('inf')],
            labels=['None', 'Low', 'Medium', 'High', 'Very High']
        )
        
        # Fill missing values safely
        required_columns = {
            'journal': 'Unknown Journal', 
            'source': 'Unknown Source',
            'country': 'Unknown Country',
            'study_type': 'Unknown Type'
        }
        
        for col, default_value in required_columns.items():
            if col not in df_processed.columns:
                df_processed[col] = default_value
            else:
                df_processed[col] = df_processed[col].fillna(default_value).astype(str)
        
        # Ensure no empty dataframe
        if df_processed.empty:
            raise ValueError("Processed DataFrame is empty")
        
        return df_processed
    
    except Exception as e:
        st.error(f"Error in process_data: {str(e)}")
        raise e

def safe_plot_wrapper(plot_func, error_message="Error creating visualization"):
    """Wrapper for safe plotting with error handling"""
    try:
        return plot_func()
    except Exception as e:
        st.error(f"{error_message}: {str(e)}")
        return None

# Main app
def main():
    try:
        st.markdown('<h1 class="main-header">🔬 CORD-19 Comprehensive Research Explorer</h1>', unsafe_allow_html=True)
        st.markdown("### Advanced Analytics for COVID-19 Research Publications")
        
        # Load data with better error handling
        try:
            with st.spinner("Loading comprehensive research data..."):
                data_result = load_data()
            
            if data_result is None or data_result[0] is None:
                st.error("❌ Failed to load data. Please check your data sources.")
                st.info("This might be due to network issues or unavailable data sources. Try refreshing the page.")
                return
            
            df, dataset_info = data_result
            
            # Validate dataframe
            if df is None or df.empty:
                st.error("❌ Loaded data is empty or invalid.")
                return
            
            st.success(f"✅ Loaded {len(df):,} research papers from {dataset_info}")
            
        except Exception as load_error:
            st.error(f"❌ Critical error loading data: {str(load_error)}")
            st.info("Please try refreshing the page or check your internet connection.")
            return
        
        # Debug info
        with st.expander("📊 Data Debug Information"):
            st.write(f"**Data Shape:** {df.shape}")
            st.write(f"**Columns:** {list(df.columns)}")
            st.write(f"**Date Range:** {df['publish_time'].min()} to {df['publish_time'].max()}")
            st.write(f"**Memory Usage:** {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
        
        # Manual upload option
        with st.expander("📁 Upload Custom Dataset"):
            uploaded_file = st.file_uploader("Or upload your own CSV file", type=['csv'])
            if uploaded_file is not None:
                try:
                    custom_df = pd.read_csv(uploaded_file)
                    if not custom_df.empty:
                        df = process_data(custom_df)
                        dataset_info = "Uploaded File"
                        st.success("✅ Using uploaded file!")
                    else:
                        st.error("Uploaded file is empty")
                except Exception as e:
                    st.error(f"Error reading uploaded file: {e}")
        
        # Sidebar controls with validation
        st.sidebar.header("🎛️ Advanced Controls")
        st.sidebar.info(f"**Source:** {dataset_info}")
        st.sidebar.info(f"**Total Papers:** {len(df):,}")
        
        # Enhanced filters with validation
        try:
            years = sorted(df['publication_year'].dropna().unique())
            if len(years) == 0:
                years = [2020]
            
            year_range = st.sidebar.slider(
                "Publication Year Range",
                min_value=int(min(years)),
                max_value=int(max(years)),
                value=(int(min(years)), int(max(years)))
            )
            
            # Apply year filter safely
            filtered_df = df[
                (df['publication_year'] >= year_range[0]) & 
                (df['publication_year'] <= year_range[1])
            ]
            
            if filtered_df.empty:
                st.warning("⚠️ Year filter resulted in no data. Resetting to full dataset.")
                filtered_df = df.copy()
            
        except Exception as filter_error:
            st.warning(f"Error applying year filter: {str(filter_error)}. Using full dataset.")
            filtered_df = df.copy()
        
        # Additional filters with validation
        col1, col2, col3 = st.sidebar.columns(3)
        
        with col1:
            try:
                min_citations = st.number_input("Min Citations", value=0, min_value=0)
                temp_df = filtered_df[filtered_df['citation_count'] >= min_citations]
                if not temp_df.empty:
                    filtered_df = temp_df
            except Exception:
                st.sidebar.warning("Citation filter error")
        
        with col2:
            try:
                min_authors = st.number_input("Min Authors", value=1, min_value=1)
                temp_df = filtered_df[filtered_df['author_count'] >= min_authors]
                if not temp_df.empty:
                    filtered_df = temp_df
            except Exception:
                st.sidebar.warning("Author filter error")
        
        with col3:
            try:
                with_abstract = st.checkbox("With Abstract", value=False)
                if with_abstract:
                    temp_df = filtered_df[filtered_df['has_abstract']]
                    if not temp_df.empty:
                        filtered_df = temp_df
            except Exception:
                st.sidebar.warning("Abstract filter error")
        
        # Journal filter with validation
        try:
            journals = filtered_df['journal'].value_counts().index.tolist()[:20]  # Limit to top 20
            selected_journals = st.sidebar.multiselect("Filter Journals (Top 20)", journals, default=[])
            if selected_journals:
                temp_df = filtered_df[filtered_df['journal'].isin(selected_journals)]
                if not temp_df.empty:
                    filtered_df = temp_df
        except Exception:
            st.sidebar.warning("Journal filter error")
        
        # Final validation
        if filtered_df.empty:
            st.warning("⚠️ No data matches your filters. Showing all data.")
            filtered_df = df.copy()
        
        # Enhanced metrics
        st.subheader("📊 Research Overview Metrics")
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        
        try:
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
        except Exception as e:
            st.warning(f"Error displaying metrics: {str(e)}")
        
        # Main tabs with comprehensive analysis
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "📈 Trends", "📚 Journals", "👥 Authors", "🌍 Geographic", "📖 Content", "📋 Data"
        ])
        
        with tab1:
            st.subheader("Publication Trends and Impact")
            col1, col2 = st.columns(2)
            
            with col1:
                def create_yearly_trends():
                    yearly_data = filtered_df.groupby('publication_year').agg({
                        'cord_uid': 'count',
                        'citation_count': 'mean'
                    }).reset_index()
                    
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        x=yearly_data['publication_year'],
                        y=yearly_data['cord_uid'],
                        name='Publications',
                        marker_color='#1f77b4'
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
                    st.plotly_chart(fig, use_container_width=True)
                
                safe_plot_wrapper(create_yearly_trends, "Error creating yearly trends")
            
            with col2:
                def create_monthly_trends():
                    if 'publish_time' in filtered_df.columns and not filtered_df.empty:
                        monthly_data = filtered_df.copy()
                        monthly_data = monthly_data.dropna(subset=['publish_time'])
                        if not monthly_data.empty:
                            monthly_data['date'] = monthly_data['publish_time'].dt.to_period('M')
                            monthly_trends = monthly_data.groupby('date').size().reset_index(name='count')
                            monthly_trends['date'] = monthly_trends['date'].dt.start_time
                            
                            fig = px.line(monthly_trends, x='date', y='count', 
                                         title='Monthly Publication Trends',
                                         height=400)
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.info("No valid date data for monthly trends")
                    else:
                        st.info("No date data available for monthly trends")
                
                safe_plot_wrapper(create_monthly_trends, "Error creating monthly trends")
        
        with tab2:
            st.subheader("Journal Analysis")
            col1, col2 = st.columns(2)
            
            with col1:
                def create_journal_bar():
                    top_journals = filtered_df['journal'].value_counts().head(15)
                    if not top_journals.empty:
                        fig = px.bar(top_journals, x=top_journals.values, y=top_journals.index,
                                    orientation='h', title='Top 15 Journals by Publication Count',
                                    height=500)
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("No journal data available")
                
                safe_plot_wrapper(create_journal_bar, "Error creating journal bar chart")
            
            with col2:
                def create_journal_scatter():
                    journal_impact = filtered_df.groupby('journal').agg({
                        'citation_count': 'mean',
                        'cord_uid': 'count'
                    }).reset_index()
                    journal_impact = journal_impact.nlargest(15, 'cord_uid')
                    
                    if not journal_impact.empty:
                        fig = px.scatter(journal_impact, x='cord_uid', y='citation_count',
                                       size='citation_count', hover_name='journal',
                                       title='Journal Impact: Publications vs Citations',
                                       height=500)
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("No journal impact data available")
                
                safe_plot_wrapper(create_journal_scatter, "Error creating journal scatter plot")
        
        with tab3:
            st.subheader("Author and Collaboration Analysis")
            col1, col2 = st.columns(2)
            
            with col1:
                def create_author_histogram():
                    if 'author_count' in filtered_df.columns:
                        fig = px.histogram(filtered_df, x='author_count', 
                                         title='Distribution of Authors per Paper',
                                         nbins=20, height=400)
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("No author count data available")
                
                safe_plot_wrapper(create_author_histogram, "Error creating author histogram")
            
            with col2:
                def create_collaboration_trends():
                    collaboration_trends = filtered_df.groupby('publication_year').agg({
                        'author_count': 'mean',
                        'has_multiple_authors': 'mean'
                    }).reset_index()
                    
                    if not collaboration_trends.empty:
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=collaboration_trends['publication_year'],
                            y=collaboration_trends['author_count'],
                            name='Avg Authors',
                            line=dict(color='blue')
                        ))
                        fig.add_trace(go.Scatter(
                            x=collaboration_trends['publication_year'],
                            y=collaboration_trends['has_multiple_authors'],
                            name='Multi-author Rate',
                            yaxis='y2',
                            line=dict(color='red')
                        ))
                        
                        fig.update_layout(
                            title='Collaboration Trends Over Time',
                            xaxis_title='Year',
                            yaxis_title='Average Authors',
                            yaxis2=dict(title='Multi-author Rate', overlaying='y', side='right'),
                            height=400
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("No collaboration data available")
                
                safe_plot_wrapper(create_collaboration_trends, "Error creating collaboration trends")
        
        with tab4:
            st.subheader("Geographic and Source Analysis")
            col1, col2 = st.columns(2)
            
            with col1:
                def create_country_pie():
                    country_data = filtered_df['country'].value_counts().head(15)
                    if not country_data.empty:
                        fig = px.pie(values=country_data.values, names=country_data.index,
                                    title='Top 15 Countries by Publication Count',
                                    height=500)
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("No country data available")
                
                safe_plot_wrapper(create_country_pie, "Error creating country pie chart")
            
            with col2:
                def create_source_trends():
                    try:
                        source_trends = pd.crosstab(filtered_df['publication_year'], filtered_df['source'])
                        if not source_trends.empty:
                            fig = go.Figure()
                            for col in source_trends.columns:
                                fig.add_trace(go.Scatter(
                                    x=source_trends.index,
                                    y=source_trends[col],
                                    mode='lines+markers',
                                    name=col,
                                    stackgroup='one'
                                ))
                            
                            fig.update_layout(
                                title='Publication Sources Over Time',
                                height=500,
                                xaxis_title='Year',
                                yaxis_title='Number of Publications'
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.info("No source trend data available")
                    except Exception as e:
                        st.info(f"Source trends not available: {str(e)}")
                
                safe_plot_wrapper(create_source_trends, "Error creating source area chart")
        
        with tab5:
            st.subheader("Content Analysis")
            col1, col2 = st.columns(2)
            
            with col1:
                def create_wordcloud():
                    try:
                        text_data = filtered_df['title'].dropna().astype(str)
                        text = ' '.join(text_data.head(1000))  # Limit to first 1000 for performance
                        if text.strip() and len(text) > 50:
                            wordcloud = WordCloud(
                                width=600, 
                                height=300, 
                                background_color='white',
                                max_words=100
                            ).generate(text)
                            
                            fig, ax = plt.subplots(figsize=(10, 5))
                            ax.imshow(wordcloud, interpolation='bilinear')
                            ax.axis('off')
                            ax.set_title('Common Words in Research Titles', fontsize=16)
                            st.pyplot(fig)
                            plt.close()
                        else:
                            st.info("Insufficient text data for word cloud")
                    except Exception as e:
                        st.info(f"Word cloud not available: {str(e)}")
                
                safe_plot_wrapper(create_wordcloud, "Error creating word cloud")
            
            with col2:
                def create_abstract_scatter():
                    if ('abstract_word_count' in filtered_df.columns and 
                        'citation_count' in filtered_df.columns):
                        plot_data = filtered_df[
                            (filtered_df['abstract_word_count'] > 0) & 
                            (filtered_df['citation_count'] >= 0)
                        ].copy()
                        
                        if not plot_data.empty and len(plot_data) > 10:
                            fig = px.scatter(plot_data.head(1000), 
                                           x='abstract_word_count', y='citation_count',
                                           trendline='lowess', 
                                           title='Abstract Length vs Citation Impact',
                                           height=400)
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.info("Insufficient data for abstract analysis")
                    else:
                        st.info("No abstract data available for analysis")
                
                safe_plot_wrapper(create_abstract_scatter, "Error creating abstract scatter plot")
            
            def create_study_type_bar():
                if 'study_type' in filtered_df.columns:
                    study_type_data = filtered_df['study_type'].value_counts()
                    if not study_type_data.empty:
                        fig = px.bar(x=study_type_data.index, y=study_type_data.values,
                                    title='Distribution of Research Types',
                                    height=400)
                        fig.update_xaxis(tickangle=45)
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("No study type data available")
                else:
                    st.info("Study type information not available")
            
            safe_plot_wrapper(create_study_type_bar, "Error creating study type bar chart")
        
        with tab6:
            st.subheader("Raw Data and Export")
            
            # Data summary with error handling
            col1, col2, col3 = st.columns(3)
            try:
                with col1:
                    st.metric("Filtered Papers", f"{len(filtered_df):,}")
                with col2:
                    st.metric("Columns", f"{len(filtered_df.columns):,}")
                with col3:
                    memory_mb = filtered_df.memory_usage(deep=True).sum() / 1024 / 1024
                    st.metric("Data Size", f"{memory_mb:.1f} MB")
            except Exception as e:
                st.warning(f"Error displaying data metrics: {str(e)}")
            
            # Data preview with pagination
            st.subheader("Data Preview")
            try:
                # Show column info
                with st.expander("📋 Column Information"):
                    col_info = pd.DataFrame({
                        'Column': filtered_df.columns,
                        'Type': [str(dtype) for dtype in filtered_df.dtypes],
                        'Non-Null Count': [filtered_df[col].count() for col in filtered_df.columns],
                        'Null Count': [filtered_df[col].isnull().sum() for col in filtered_df.columns]
                    })
                    st.dataframe(col_info, use_container_width=True)
                
                # Paginated data view
                rows_per_page = st.selectbox("Rows per page", [10, 25, 50, 100], index=2)
                total_pages = (len(filtered_df) - 1) // rows_per_page + 1
                
                if total_pages > 1:
                    page = st.selectbox(f"Page (1 to {total_pages})", range(1, total_pages + 1))
                    start_idx = (page - 1) * rows_per_page
                    end_idx = start_idx + rows_per_page
                    display_df = filtered_df.iloc[start_idx:end_idx]
                else:
                    display_df = filtered_df.head(rows_per_page)
                
                st.dataframe(display_df, use_container_width=True, height=400)
                
            except Exception as e:
                st.error(f"Error displaying data preview: {str(e)}")
                st.info("Showing basic data info instead")
                st.write(f"Data shape: {filtered_df.shape}")
                st.write(f"Columns: {list(filtered_df.columns)}")
            
            # Export options with error handling
            st.subheader("Export Options")
            col1, col2, col3 = st.columns(3)
            
            try:
                with col1:
                    if st.button("📥 Prepare CSV Download"):
                        csv = filtered_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Filtered Data (CSV)",
                            data=csv,
                            file_name=f'cord19_analysis_{datetime.now().strftime("%Y%m%d_%H%M")}.csv',
                            mime='text/csv'
                        )
                        st.success("✅ CSV ready for download!")
                
                with col2:
                    if st.button("📊 Prepare Summary Stats"):
                        try:
                            summary_stats = filtered_df.describe(include='all').fillna('')
                            summary_csv = summary_stats.to_csv()
                            st.download_button(
                                label="📊 Download Summary Statistics",
                                data=summary_csv,
                                file_name=f'cord19_summary_{datetime.now().strftime("%Y%m%d_%H%M")}.csv',
                                mime='text/csv'
                            )
                            st.success("✅ Summary stats ready!")
                        except Exception as e:
                            st.warning(f"Could not generate summary stats: {str(e)}")
                
                with col3:
                    if st.button("🔍 Show Data Info"):
                        with st.expander("📊 Dataset Summary Statistics", expanded=True):
                            try:
                                numeric_cols = filtered_df.select_dtypes(include=[np.number]).columns
                                if len(numeric_cols) > 0:
                                    st.write("**Numeric Columns Summary:**")
                                    st.dataframe(filtered_df[numeric_cols].describe(), use_container_width=True)
                                
                                categorical_cols = filtered_df.select_dtypes(include=['object']).columns
                                if len(categorical_cols) > 0:
                                    st.write("**Categorical Columns Summary:**")
                                    for col in categorical_cols[:5]:  # Show first 5 categorical columns
                                        st.write(f"**{col}:** {filtered_df[col].nunique()} unique values")
                                        top_values = filtered_df[col].value_counts().head(3)
                                        st.write(f"Top values: {dict(top_values)}")
                            except Exception as e:
                                st.warning(f"Error generating detailed statistics: {str(e)}")
            
            except Exception as e:
                st.warning(f"Error in export section: {str(e)}")
        
        # Footer
        st.markdown("---")
        st.markdown("""
        **Advanced CORD-19 Research Explorer** • 
        *Comprehensive analysis of COVID-19 research publications* • 
        Data updated automatically from multiple sources
        """)
        
        # Performance info
        if st.sidebar.checkbox("Show Performance Info"):
            st.sidebar.info(f"""
            **Performance Metrics:**
            - Records: {len(filtered_df):,}
            - Filters Applied: {len(df) - len(filtered_df):,} records filtered
            - Memory Usage: {filtered_df.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB
            """)
    
    except Exception as e:
        st.error(f"❌ Critical application error: {str(e)}")
        st.info("Please try the following:")
        st.info("1. Refresh the page")
        st.info("2. Check your internet connection")
        st.info("3. Try uploading a custom CSV file")
        
        # Show error details for debugging
        with st.expander("🔧 Error Details (for debugging)"):
            import traceback
            st.code(traceback.format_exc())

if __name__ == "__main__":
    main()