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
warnings.filterwarnings('ignore')

# Page config - updated with proper width settings
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
</style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=3600)
def load_data():
    """Load data with improved error handling and fallbacks"""
    
    st.info("🔬 Loading CORD-19 research data...")
    
    # Try multiple data sources with fallbacks
    data_sources = [
        # Primary: Direct CSV from GitHub
        "https://raw.githubusercontent.com/kimenjuivy/CORD-19-DATA-ANALYSIS/main/sample_data/sample_cord19.csv",
        # Fallback: Smaller sample data
        "https://raw.githubusercontent.com/allenai/cord19/main/sample_data/sample_metadata.csv"
    ]
    
    for i, url in enumerate(data_sources):
        try:
            st.info(f"Attempting to load from source {i+1}...")
            df = pd.read_csv(url)
            if len(df) > 0:
                st.success(f"✅ Successfully loaded {len(df):,} records from source {i+1}")
                return process_data(df), f"Source {i+1}"
        except Exception as e:
            st.warning(f"Source {i+1} failed: {str(e)}")
            continue
    
    # Final fallback: Create demo data
    st.warning("⚠️ Using demo data - upload your own CSV for full analysis")
    demo_data = create_demo_data()
    return process_data(demo_data), "Demo Data"

def create_demo_data():
    """Create realistic demo data when external sources fail"""
    np.random.seed(42)
    n_records = 500
    
    journals = ['Lancet', 'Nature Medicine', 'JAMA', 'BMJ', 'NEJM', 
                'Science', 'Cell', 'Nature Communications', 'PNAS', 'BioRxiv']
    
    diseases = ['COVID-19', 'SARS-CoV-2', 'Coronavirus', 'Pandemic', 'Vaccine']
    research_types = ['Clinical Trial', 'Review', 'Case Study', 'Epidemiology', 'Basic Research']
    
    data = {
        'title': [f'Research on {d} and {rt} Study #{i}' 
                 for i, (d, rt) in enumerate(zip(
                     np.random.choice(diseases, n_records),
                     np.random.choice(research_types, n_records)
                 ))],
        'authors': [f'Researcher {i}; Co-author {i+1}' for i in range(n_records)],
        'journal': np.random.choice(journals, n_records),
        'publish_time': pd.date_range('2020-01-01', periods=n_records, freq='D'),
        'abstract': [f'This study investigates {d} through {rt} methodology. Important findings include significant results that contribute to our understanding of the pandemic.'
                    for d, rt in zip(np.random.choice(diseases, n_records),
                                   np.random.choice(research_types, n_records))],
        'source': np.random.choice(['PubMed', 'WHO', 'CDC', 'BioRxiv', 'MedRxiv'], n_records)
    }
    
    return pd.DataFrame(data)

def process_data(df):
    """Process the dataframe for the app"""
    
    # Create a copy to avoid modifying the original
    df_processed = df.copy()
    
    # Handle date columns
    if 'publish_time' in df_processed.columns:
        try:
            df_processed['publish_time'] = pd.to_datetime(df_processed['publish_time'], errors='coerce')
            # Fill NaT with a reasonable default date
            mask = df_processed['publish_time'].isna()
            df_processed.loc[mask, 'publish_time'] = pd.to_datetime('2020-01-01')
            df_processed['publication_year'] = df_processed['publish_time'].dt.year.fillna(2020).astype(int)
        except Exception as e:
            st.warning(f"Date processing issue: {e}")
            df_processed['publication_year'] = 2020
            df_processed['publish_time'] = pd.to_datetime('2020-01-01')
    else:
        df_processed['publication_year'] = 2020
        df_processed['publish_time'] = pd.to_datetime('2020-01-01')
    
    # Create title word count if title exists
    if 'title' in df_processed.columns:
        df_processed['title_word_count'] = df_processed['title'].astype(str).str.split().str.len().fillna(0)
    else:
        df_processed['title_word_count'] = 0
    
    # Fill missing columns with defaults
    required_columns = {
        'abstract': 'No Abstract Available',
        'journal': 'Unknown Journal', 
        'source': 'Unknown Source',
        'authors': 'Unknown Authors'
    }
    
    for col, default_value in required_columns.items():
        if col not in df_processed.columns:
            df_processed[col] = default_value
        else:
            df_processed[col] = df_processed[col].fillna(default_value)
    
    return df_processed

def safe_plotting(func, *args, **kwargs):
    """Safe plotting wrapper to handle errors"""
    try:
        return func(*args, **kwargs)
    except Exception as e:
        st.error(f"Plotting error: {e}")
        return None

# Main app function
def main():
    # Header
    st.markdown('<h1 class="main-header">🔬 CORD-19 Research Data Explorer</h1>', unsafe_allow_html=True)
    st.markdown("### Explore COVID-19 research publications and discover insights from academic literature")
    
    # Load data
    with st.spinner("Loading research data..."):
        data_result = load_data()
    
    if data_result[0] is None:
        # Manual upload fallback
        st.subheader("🔄 Upload Dataset Manually")
        uploaded_file = st.file_uploader("Upload your CSV file", type=['csv'])
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                df = process_data(df)
                dataset_info = "Uploaded File"
                st.success("✅ Using uploaded file!")
            except Exception as e:
                st.error(f"Error reading uploaded file: {e}")
                st.info("Please try uploading a different CSV file or use the demo data.")
                # Use demo data as final fallback
                df = process_data(create_demo_data())
                dataset_info = "Demo Data (Upload Failed)"
        else:
            # Use demo data if no upload
            df = process_data(create_demo_data())
            dataset_info = "Demo Data"
    else:
        df, dataset_info = data_result
    
    # Sidebar with safe checks
    st.sidebar.header("🎛️ Controls")
    st.sidebar.info(f"**Dataset:** {dataset_info}")
    st.sidebar.info(f"**Papers:** {len(df):,}")
    
    # Safe year filter
    try:
        years = sorted(df['publication_year'].dropna().unique())
        if len(years) > 0:
            year_range = st.sidebar.slider(
                "Select Year Range",
                min_value=int(min(years)),
                max_value=int(max(years)),
                value=(int(min(years)), int(max(years)))
            )
            filtered_df = df[
                (df['publication_year'] >= year_range[0]) & 
                (df['publication_year'] <= year_range[1])
            ]
        else:
            filtered_df = df
            st.sidebar.warning("No year data available")
    except Exception as e:
        filtered_df = df
        st.sidebar.warning("Year filter not available")
    
    # Safe journal filter
    try:
        if 'journal' in filtered_df.columns:
            journal_counts = filtered_df['journal'].value_counts()
            if len(journal_counts) > 0:
                top_journals = journal_counts.head(20).index.tolist()
                selected_journals = st.sidebar.multiselect(
                    "Select Journals (Optional)",
                    options=top_journals,
                    default=[]
                )
                if selected_journals:
                    filtered_df = filtered_df[filtered_df['journal'].isin(selected_journals)]
    except Exception as e:
        st.sidebar.warning("Journal filter not available")
    
    # Safe source filter
    try:
        if 'source' in filtered_df.columns:
            sources = filtered_df['source'].unique().tolist()
            selected_sources = st.sidebar.multiselect(
                "Filter by Source",
                options=sources,
                default=[]
            )
            if selected_sources:
                filtered_df = filtered_df[filtered_df['source'].isin(selected_sources)]
    except Exception as e:
        pass  # Silent fail for optional filter
    
    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        try:
            st.metric("📄 Total Papers", f"{len(filtered_df):,}")
        except:
            st.metric("📄 Total Papers", "N/A")
    
    with col2:
        try:
            st.metric("📚 Unique Journals", f"{filtered_df['journal'].nunique():,}")
        except:
            st.metric("📚 Unique Journals", "N/A")
    
    with col3:
        try:
            avg_title_length = filtered_df['title_word_count'].mean()
            st.metric("📝 Avg Title Length", f"{avg_title_length:.1f} words")
        except:
            st.metric("📝 Avg Title Length", "N/A")
    
    with col4:
        try:
            papers_with_abstracts = (filtered_df['abstract'] != 'No Abstract Available').sum()
            st.metric("📋 Papers with Abstracts", f"{papers_with_abstracts:,}")
        except:
            st.metric("📋 Papers with Abstracts", "N/A")
    
    st.markdown("---")
    
    # Main content tabs with safe plotting
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "📈 Trends", "🔍 Deep Dive", "📋 Data Sample"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Publications by Year")
            try:
                yearly_counts = filtered_df['publication_year'].value_counts().sort_index()
                if len(yearly_counts) > 0:
                    fig = px.bar(
                        x=yearly_counts.index,
                        y=yearly_counts.values,
                        title="Number of Publications per Year",
                        labels={'x': 'Year', 'y': 'Publications'},
                        color=yearly_counts.values,
                        color_continuous_scale='Blues'
                    )
                    fig.update_layout(showlegend=False, height=400)
                    st.plotly_chart(fig, width='stretch')
                else:
                    st.info("No yearly data available")
            except Exception as e:
                st.error(f"Could not create yearly publications chart: {e}")
        
        with col2:
            st.subheader("Top Journals")
            try:
                if 'journal' in filtered_df.columns:
                    top_journals_data = filtered_df['journal'].value_counts().head(10)
                    if len(top_journals_data) > 0:
                        fig = px.pie(
                            values=top_journals_data.values,
                            names=[str(journal)[:30] + '...' if len(str(journal)) > 30 else str(journal) 
                                   for journal in top_journals_data.index],
                            title="Distribution of Top 10 Journals"
                        )
                        fig.update_layout(height=400)
                        st.plotly_chart(fig, width='stretch')
                    else:
                        st.info("No journal data available")
                else:
                    st.info("Journal column not found")
            except Exception as e:
                st.error(f"Could not create journals chart: {e}")
    
    with tab2:
        st.subheader("Publication Trends Over Time")
        try:
            if 'publish_time' in filtered_df.columns:
                # Create monthly aggregation
                filtered_df['month_year'] = filtered_df['publish_time'].dt.to_period('M').astype(str)
                monthly_counts = filtered_df['month_year'].value_counts().sort_index()
                
                if len(monthly_counts) > 0:
                    monthly_df = pd.DataFrame({
                        'Date': [pd.to_datetime(period) for period in monthly_counts.index],
                        'Publications': monthly_counts.values
                    })
                    
                    fig = px.line(
                        monthly_df,
                        x='Date',
                        y='Publications',
                        title='Monthly Publication Trends',
                        markers=True
                    )
                    fig.update_layout(height=500)
                    st.plotly_chart(fig, width='stretch')
                else:
                    st.info("No date data available for trends")
            else:
                st.info("Date information not available for trends")
        except Exception as e:
            st.error(f"Could not create trends chart: {e}")
    
    with tab3:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Title Word Count Distribution")
            try:
                if 'title_word_count' in filtered_df.columns:
                    fig = px.histogram(
                        filtered_df,
                        x='title_word_count',
                        nbins=30,
                        title='Distribution of Title Lengths',
                        labels={'title_word_count': 'Words in Title', 'count': 'Frequency'}
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, width='stretch')
                else:
                    st.info("Title word count data not available")
            except Exception as e:
                st.error(f"Could not create word count chart: {e}")
        
        with col2:
            st.subheader("Research Source Analysis")
            try:
                if 'source' in filtered_df.columns:
                    source_counts = filtered_df['source'].value_counts()
                    if len(source_counts) > 0:
                        fig = px.pie(
                            values=source_counts.values[:8],
                            names=source_counts.index[:8],
                            title='Distribution by Research Source'
                        )
                        fig.update_layout(height=400)
                        st.plotly_chart(fig, width='stretch')
                    else:
                        st.info("No source data available")
                else:
                    st.info("Source column not found")
            except Exception as e:
                st.error(f"Could not create source analysis chart: {e}")
        
        # Additional analysis
        st.subheader("Word Cloud of Paper Titles")
        try:
            if 'title' in filtered_df.columns:
                # Combine all titles
                text = ' '.join(filtered_df['title'].dropna().astype(str))
                if len(text) > 0:
                    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
                    
                    fig, ax = plt.subplots(figsize=(10, 5))
                    ax.imshow(wordcloud, interpolation='bilinear')
                    ax.axis('off')
                    ax.set_title('Common Words in Paper Titles', fontsize=16)
                    st.pyplot(fig)
                else:
                    st.info("No title data available for word cloud")
            else:
                st.info("Title column not found")
        except Exception as e:
            st.error(f"Could not create word cloud: {e}")
    
    with tab4:
        st.subheader("Data Sample")
        try:
            st.write(f"Showing sample of {len(filtered_df):,} papers")
            
            # Select available columns
            available_columns = []
            for col in ['title', 'authors', 'journal', 'publication_year', 'source', 'title_word_count']:
                if col in filtered_df.columns:
                    available_columns.append(col)
            
            if available_columns:
                sample_df = filtered_df.head(20)[available_columns]
                st.dataframe(sample_df, width='stretch', height=400)
                
                # Download button
                csv = filtered_df[available_columns].to_csv(index=False)
                st.download_button(
                    label="📥 Download Filtered Data as CSV",
                    data=csv,
                    file_name=f'cord19_filtered_data_{datetime.now().strftime("%Y%m%d")}.csv',
                    mime='text/csv'
                )
            else:
                st.info("No columns available for display")
        except Exception as e:
            st.error(f"Could not display data sample: {e}")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    **About this app:** Interactive exploration of COVID-19 research data from the CORD-19 dataset.
    
    *Note: This app uses demo data when external sources are unavailable. Upload your own CSV for custom analysis.*
    """)

# Health check endpoint (for deployment)
def health_check():
    return "OK"

if __name__ == "__main__":
    main()