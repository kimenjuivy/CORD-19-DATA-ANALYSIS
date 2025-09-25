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
</style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=3600)
def load_data():
    """Load data from GitHub release"""
    
    st.info("🌐 Loading dataset from GitHub release...")
    
    try:
        # GitHub release URL
        url = "https://github.com/kimenjuivy/CORD-19-DATA-ANALYSIS/releases/download/v1.0-data/dataset.zip"
        
        # Download the file
        response = requests.get(url)
        response.raise_for_status()
        
        # Extract zip file
        with zipfile.ZipFile(BytesIO(response.content)) as zip_ref:
            # Get list of files in zip
            file_list = zip_ref.namelist()
            st.info(f"Files in zip: {file_list}")
            
            # Look for CSV files
            csv_files = [f for f in file_list if f.endswith('.csv')]
            
            if not csv_files:
                st.error("❌ No CSV file found in the zip")
                return None, None
            
            # Read the first CSV file found
            with zip_ref.open(csv_files[0]) as csv_file:
                df = pd.read_csv(csv_file)
        
        st.success(f"✅ Successfully loaded: {csv_files[0]}")
        return process_data(df), "GitHub Release"
        
    except Exception as e:
        st.error(f"❌ Error loading dataset: {str(e)}")
        return None, None

def process_data(df):
    """Process the dataframe for the app"""
    
    # Handle date columns
    if 'publish_time' in df.columns:
        try:
            df['publish_time'] = pd.to_datetime(df['publish_time'], errors='coerce')
            df['publication_year'] = df['publish_time'].dt.year.fillna(2020).astype(int)
        except:
            df['publication_year'] = 2020
            df['publish_time'] = pd.to_datetime('2020-01-01')
    else:
        df['publication_year'] = 2020
        df['publish_time'] = pd.to_datetime('2020-01-01')
    
    # Create title word count if title exists
    if 'title' in df.columns:
        df['title_word_count'] = df['title'].str.split().str.len().fillna(0)
    else:
        df['title_word_count'] = 0
    
    # Fill missing columns with defaults
    if 'abstract' not in df.columns:
        df['abstract'] = 'No Abstract Available'
    else:
        df['abstract'] = df['abstract'].fillna('No Abstract Available')
    
    if 'journal' not in df.columns:
        df['journal'] = 'Unknown Journal'
    else:
        df['journal'] = df['journal'].fillna('Unknown Journal')
    
    if 'source' not in df.columns:
        df['source'] = 'Unknown Source'
    
    if 'authors' not in df.columns:
        df['authors'] = 'Unknown Authors'
    
    return df

def safe_plotting(func, *args, **kwargs):
    """Safe plotting wrapper to handle errors"""
    try:
        return func(*args, **kwargs)
    except Exception as e:
        st.error(f"Plotting error: {e}")
        return None

# Header
st.markdown('<h1 class="main-header">🔬 CORD-19 Research Data Explorer</h1>', unsafe_allow_html=True)
st.markdown("### Explore COVID-19 research publications and discover insights from academic literature")

# Load data
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
            st.error(f"Error: {e}")
            st.stop()
    else:
        st.stop()
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
except:
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
except:
    st.sidebar.warning("Journal filter not available")

# Safe metrics
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
            fig = px.bar(
                x=yearly_counts.index,
                y=yearly_counts.values,
                title="Number of Publications per Year",
                labels={'x': 'Year', 'y': 'Publications'},
                color=yearly_counts.values,
                color_continuous_scale='Blues'
            )
            fig.update_layout(showlegend=False, height=400)
            st.plotly_chart(fig, use_container_width=True)
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
                    st.plotly_chart(fig, use_container_width=True)
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
            filtered_df['month_year'] = filtered_df['publish_time'].dt.to_period('M')
            monthly_counts = filtered_df['month_year'].value_counts().sort_index()
            
            monthly_df = pd.DataFrame({
                'Date': [pd.to_datetime(str(period)) for period in monthly_counts.index],
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
            st.plotly_chart(fig, use_container_width=True)
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
                st.plotly_chart(fig, use_container_width=True)
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
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No source data available")
            else:
                st.info("Source column not found")
        except Exception as e:
            st.error(f"Could not create source analysis chart: {e}")

with tab4:
    st.subheader("Data Sample")
    try:
        st.write(f"Showing sample of {len(filtered_df):,} papers")
        
        # Select available columns
        available_columns = []
        for col in ['title', 'authors', 'journal', 'publication_year', 'title_word_count']:
            if col in filtered_df.columns:
                available_columns.append(col)
        
        if available_columns:
            sample_df = filtered_df.head(20)[available_columns]
            st.dataframe(sample_df, use_container_width=True, height=400)
            
            # Download button
            csv = filtered_df.to_csv(index=False)
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
""")