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
    """Load data from GitHub release - SIMPLIFIED VERSION"""
    
    st.info("🌐 Loading dataset from GitHub release...")
    
    try:
        # GitHub release URL
        url = "https://github.com/kimenjuivy/CORD-19-DATA-ANALYSIS/releases/download/v1.0-data/dataset.zip"
        
        # Download the file
        response = requests.get(url)
        response.raise_for_status()
        
        # Check if we got a valid response
        if len(response.content) == 0:
            st.error("❌ Downloaded file is empty")
            return None, None
        
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
        
    except zipfile.BadZipFile:
        st.error("❌ The file is not a valid zip file. Please reupload the zip file.")
        return None, None
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        return None, None

def process_data(df):
    """Process the dataframe for the app"""
    
    # Handle date columns
    if 'publish_time' in df.columns:
        df['publish_time'] = pd.to_datetime(df['publish_time'], errors='coerce')
        df['publication_year'] = df['publish_time'].dt.year.fillna(2020).astype(int)
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
    
    if 'journal' not in df.columns:
        df['journal'] = 'Unknown Journal'
    
    if 'source' not in df.columns:
        df['source'] = 'Unknown Source'
    
    if 'authors' not in df.columns:
        df['authors'] = 'Unknown Authors'
    
    return df

# Header
st.markdown('<h1 class="main-header">🔬 CORD-19 Research Data Explorer</h1>', unsafe_allow_html=True)
st.markdown("### Explore COVID-19 research publications and discover insights from academic literature")

# Load data
data_result = load_data()

if data_result[0] is None:
    st.error("""
    **Unable to load data. Please:**
    1. Reupload dataset.zip to GitHub releases with the CSV file directly in the zip root
    2. Ensure the zip file is not empty
    3. Use standard zip compression (not WinRAR)
    """)
    
    # Manual upload fallback
    st.subheader("🔄 Upload Dataset Manually")
    uploaded_file = st.file_uploader("Or upload your CSV file directly", type=['csv'])
    
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

# Rest of your app remains the same...
st.sidebar.header("🎛️ Controls")
st.sidebar.info(f"**Dataset:** {dataset_info}")
st.sidebar.info(f"**Papers:** {len(df):,}")

# Year filter
years = sorted(df['publication_year'].dropna().unique())
year_range = st.sidebar.slider(
    "Select Year Range",
    min_value=int(min(years)),
    max_value=int(max(years)),
    value=(int(min(years)), int(max(years)))
)

# Filter data
filtered_df = df[
    (df['publication_year'] >= year_range[0]) & 
    (df['publication_year'] <= year_range[1])
]

# Journal filter
top_journals = df['journal'].value_counts().head(20).index.tolist()
selected_journals = st.sidebar.multiselect(
    "Select Journals (Optional)",
    options=top_journals,
    default=[]
)

if selected_journals:
    filtered_df = filtered_df[filtered_df['journal'].isin(selected_journals)]

# Metrics
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("📄 Total Papers", f"{len(filtered_df):,}")

with col2:
    st.metric("📚 Unique Journals", f"{filtered_df['journal'].nunique():,}")

with col3:
    avg_title_length = filtered_df['title_word_count'].mean()
    st.metric("📝 Avg Title Length", f"{avg_title_length:.1f} words")

with col4:
    papers_with_abstracts = (filtered_df['abstract'] != 'No Abstract Available').sum()
    st.metric("📋 Papers with Abstracts", f"{papers_with_abstracts:,}")

st.markdown("---")

# Main content tabs
tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "📈 Trends", "🔍 Deep Dive", "📋 Data Sample"])

with tab1:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Publications by Year")
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
    
    with col2:
        st.subheader("Top Journals")
        top_journals_data = filtered_df['journal'].value_counts().head(10)
        
        fig = px.pie(
            values=top_journals_data.values,
            names=[journal[:30] + '...' if len(journal) > 30 else journal 
                   for journal in top_journals_data.index],
            title="Distribution of Top 10 Journals"
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("Publication Trends Over Time")
    
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

with tab3:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Title Word Count Distribution")
        fig = px.histogram(
            filtered_df,
            x='title_word_count',
            nbins=30,
            title='Distribution of Title Lengths',
            labels={'title_word_count': 'Words in Title', 'count': 'Frequency'}
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Research Source Analysis")
        source_counts = filtered_df['source'].value_counts()
        fig = px.pie(
            values=source_counts.values[:8],
            names=source_counts.index[:8],
            title='Distribution by Research Source'
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.subheader("Data Sample")
    st.write(f"Showing sample of {len(filtered_df):,} papers")
    
    sample_df = filtered_df.head(20)[
        ['title', 'authors', 'journal', 'publication_year', 'title_word_count']
    ]
    
    st.dataframe(sample_df, use_container_width=True, height=400)
    
    csv = filtered_df.to_csv(index=False)
    st.download_button(
        label="📥 Download Filtered Data as CSV",
        data=csv,
        file_name=f'cord19_filtered_data_{datetime.now().strftime("%Y%m%d")}.csv',
        mime='text/csv'
    )

st.markdown("---")
st.markdown("""
**About this app:** Interactive exploration of COVID-19 research data from the CORD-19 dataset.
""")