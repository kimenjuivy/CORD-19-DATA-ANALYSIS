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
.metric-card {
    background-color: #f8f9fa;
    padding: 1rem;
    border-radius: 0.5rem;
    border-left: 4px solid #1e88e5;
}
</style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=3600)
def load_data():
    """Load data from GitHub release - FIXED VERSION"""
    
    st.info("🌐 Loading dataset from GitHub release...")
    
    try:
        # GitHub release URL
        url = "https://github.com/kimenjuivy/CORD-19-DATA-ANALYSIS/releases/download/v1.0-data/dataset.zip"
        
        # Download the file
        response = requests.get(url)
        response.raise_for_status()
        
        # Extract zip file
        with zipfile.ZipFile(BytesIO(response.content)) as zip_ref:
            # Get list of all files in zip (including folder structure)
            file_list = zip_ref.namelist()
            st.info(f"Files in release: {file_list}")
            
            # Look for the CSV file in the raw_data folder
            csv_files = [f for f in file_list if f.endswith('.csv')]
            
            if not csv_files:
                st.error("❌ No CSV file found in the release")
                return None, None
            
            # Use the first CSV file found (should be raw_data/metadata_cleaned.csv)
            csv_file_path = csv_files[0]
            st.info(f"Loading CSV file: {csv_file_path}")
            
            # Read the CSV file from the zip
            with zip_ref.open(csv_file_path) as csv_file:
                df = pd.read_csv(csv_file)
        
        st.success(f"✅ Successfully loaded dataset: {csv_file_path}")
        st.info(f"Dataset shape: {df.shape}")
        
        return process_data(df), "GitHub Release"
        
    except zipfile.BadZipFile:
        st.error("❌ The downloaded file is not a valid zip file")
        st.info("This usually means the GitHub release file is corrupted or not properly uploaded")
        return None, None
    except Exception as e:
        st.error(f"❌ Failed to load dataset: {str(e)}")
        return None, None

def process_data(df):
    """Process the dataframe for the app"""
    
    # Handle date columns
    date_columns = ['publish_time', 'date', 'publication_date', 'publish_date', 'time']
    for col in date_columns:
        if col in df.columns:
            try:
                df['publish_time'] = pd.to_datetime(df[col], errors='coerce')
                df['publication_year'] = df['publish_time'].dt.year.fillna(2020).astype(int)
                break
            except:
                continue
    else:
        # If no date column found, create a dummy one
        df['publication_year'] = 2020
        df['publish_time'] = pd.to_datetime('2020-01-01')
    
    # Create title word count if title exists
    if 'title' in df.columns:
        df['title_word_count'] = df['title'].str.split().str.len().fillna(0)
    else:
        df['title_word_count'] = 0
    
    # Fill missing columns with defaults
    default_values = {
        'abstract': 'No Abstract Available',
        'journal': 'Unknown Journal', 
        'source': 'Unknown Source',
        'authors': 'Unknown Authors'
    }
    
    for col, default_val in default_values.items():
        if col not in df.columns:
            df[col] = default_val
        else:
            df[col] = df[col].fillna(default_val)
    
    return df

# Header
st.markdown('<h1 class="main-header">🔬 CORD-19 Research Data Explorer</h1>', unsafe_allow_html=True)
st.markdown("### Explore COVID-19 research publications and discover insights from academic literature")

# Load data
with st.spinner("Loading dataset from GitHub..."):
    data_result = load_data()

if data_result[0] is None:
    st.error("""
    **Application cannot start without data. Please check:**
    
    1. **GitHub Release**: Ensure v1.0-data release exists at https://github.com/kimenjuivy/CORD-19-DATA-ANALYSIS/releases
    2. **File Content**: The release should contain dataset.zip with a CSV file inside
    3. **File Size**: Check if the zip file is not empty or corrupted
    """)
    
    # Test the file directly
    st.markdown("---")
    st.subheader("🔧 Debug Information")
    
    try:
        # Test if we can download the file
        url = "https://github.com/kimenjuivy/CORD-19-DATA-ANALYSIS/releases/download/v1.0-data/dataset.zip"
        response = requests.head(url)
        st.info(f"HTTP Status: {response.status_code}")
        st.info(f"Content-Type: {response.headers.get('content-type', 'Unknown')}")
        st.info(f"Content-Length: {response.headers.get('content-length', 'Unknown')} bytes")
        
        if response.status_code == 200:
            # Try to download a small portion to check if it's a valid zip
            test_response = requests.get(url, stream=True)
            first_chunk = next(test_response.iter_content(chunk_size=1024))
            
            # Check if it starts with PK (zip file signature)
            if first_chunk.startswith(b'PK'):
                st.success("✅ File is a valid zip file (starts with PK signature)")
            else:
                st.error("❌ File does not appear to be a valid zip file")
                st.info(f"First bytes: {first_chunk[:10]}")
                
    except Exception as e:
        st.error(f"Debug error: {e}")
    
    # Add manual upload as fallback
    st.markdown("---")
    st.subheader("🔄 Alternative: Upload Dataset")
    uploaded_file = st.file_uploader("Upload your dataset file (CSV or ZIP)", type=['csv', 'zip'])
    
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith('.zip'):
                with zipfile.ZipFile(uploaded_file) as zip_ref:
                    # Look for CSV files in any folder structure
                    csv_files = [f for f in zip_ref.namelist() if f.endswith('.csv')]
                    if csv_files:
                        with zip_ref.open(csv_files[0]) as f:
                            df = pd.read_csv(f)
                    else:
                        st.error("No CSV file found in ZIP")
                        st.stop()
            
            df = process_data(df)
            dataset_info = "Uploaded File"
            st.success("✅ Using uploaded file!")
        except Exception as e:
            st.error(f"Error processing uploaded file: {e}")
            st.stop()
    else:
        st.stop()

else:
    df, dataset_info = data_result

# Rest of your app code continues exactly as before...
# Sidebar
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
    
    # Monthly trend
    filtered_df['month_year'] = filtered_df['publish_time'].dt.to_period('M')
    monthly_counts = filtered_df['month_year'].value_counts().sort_index()
    
    # Convert to datetime for plotly
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
    
    # Display sample
    sample_df = filtered_df.head(20)[
        ['title', 'authors', 'journal', 'publication_year', 'title_word_count']
    ]
    
    st.dataframe(
        sample_df,
        use_container_width=True,
        height=400
    )
    
    # Download button
    csv = filtered_df.to_csv(index=False)
    st.download_button(
        label="📥 Download Filtered Data as CSV",
        data=csv,
        file_name=f'cord19_filtered_data_{datetime.now().strftime("%Y%m%d")}.csv',
        mime='text/csv'
    )

# Footer
st.markdown("---")
st.markdown("""
**About this app:** Interactive exploration of COVID-19 research data from the CORD-19 dataset.

**Data Source:** [CORD-19 Dataset](https://www.kaggle.com/allen-institute-for-ai/CORD-19-research-challenge)
""")