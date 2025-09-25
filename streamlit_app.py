import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import numpy as np
from datetime import datetime
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

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv('raw_data/metadata_cleaned.csv')
    df['publish_time'] = pd.to_datetime(df['publish_time'])
    return df

# Header
st.markdown('<h1 class="main-header">🔬 CORD-19 Research Data Explorer</h1>', unsafe_allow_html=True)
st.markdown("### Explore COVID-19 research publications and discover insights from academic literature")

try:
    df = load_data()
    
    # Sidebar
    st.sidebar.header("🎛️ Controls")
    
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
        
        # Growth analysis
        st.subheader("Year-over-Year Growth Analysis")
        yearly_counts = filtered_df['publication_year'].value_counts().sort_index()
        growth_rates = yearly_counts.pct_change() * 100
        
        growth_df = pd.DataFrame({
            'Year': growth_rates.index,
            'Growth_Rate': growth_rates.values
        }).dropna()
        
        fig = px.bar(
            growth_df,
            x='Year',
            y='Growth_Rate',
            title='Year-over-Year Growth Rate (%)',
            color='Growth_Rate',
            color_continuous_scale='RdYlGn'
        )
        fig.update_layout(height=400)
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
            
            st.subheader("Author Collaboration Analysis")
            author_counts = filtered_df['author_count'].value_counts().head(15)
            fig = px.bar(
                x=author_counts.index,
                y=author_counts.values,
                title='Distribution of Author Counts',
                labels={'x': 'Number of Authors', 'y': 'Number of Papers'}
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
            
            # Word frequency analysis
            st.subheader("Most Common Title Words")
            if st.button("Generate Word Analysis"):
                all_titles = ' '.join(filtered_df['title'].dropna().astype(str))
                words = all_titles.lower().split()
                
                # Filter out common stop words
                stop_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
                filtered_words = [word for word in words if word not in stop_words and len(word) > 3]
                
                word_freq = pd.Series(filtered_words).value_counts().head(20)
                
                fig = px.bar(
                    x=word_freq.values,
                    y=word_freq.index,
                    orientation='h',
                    title='Top 20 Most Common Words in Titles',
                    labels={'x': 'Frequency', 'y': 'Words'}
                )
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.subheader("Data Sample")
        st.write(f"Showing sample of {len(filtered_df):,} papers")
        
        # Display options
        col1, col2 = st.columns(2)
        with col1:
            sample_size = st.slider("Sample Size", 10, 100, 20)
        with col2:
            sort_by = st.selectbox("Sort By", ['publication_year', 'title_word_count', 'author_count'])
        
        # Display sample
        sample_df = filtered_df.nlargest(sample_size, sort_by)[
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
    **About this app:** This Streamlit application provides an interactive exploration of COVID-19 research data 
    from the CORD-19 dataset. Use the sidebar controls to filter and explore different aspects of the research landscape.
    
    **Data Source:** [CORD-19 Dataset](https://www.kaggle.com/allen-institute-for-ai/CORD-19-research-challenge)
    """)

except FileNotFoundError:
    st.error("""
    **Data file not found!** 
    
    Please ensure that `raw_data/metadata_cleaned.csv` exists in your project directory.
    
    Run the data cleaning notebook first to generate this file.
    """)
except Exception as e:
    st.error(f"An error occurred: {str(e)}")
    st.info("Please check your data files and try again.")