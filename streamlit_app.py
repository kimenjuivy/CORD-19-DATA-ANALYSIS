import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Page config - simplified to avoid conflicts
st.set_page_config(
    page_title="CORD-19 Data Explorer",
    page_icon="🔬",
    layout="wide"
)

# Simplified CSS to avoid iframe issues
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    color: #1e88e5;
    text-align: center;
    margin-bottom: 1rem;
}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def create_demo_data():
    """Create comprehensive demo data without external dependencies"""
    np.random.seed(42)
    n_records = 1000  # Reduced for better performance
    
    journals = [
        'Nature Medicine', 'The Lancet', 'JAMA', 'BMJ', 'NEJM',
        'Science', 'Cell', 'Nature Communications', 'PNAS', 'PLOS ONE'
    ]
    
    countries = ['USA', 'China', 'UK', 'Italy', 'Germany', 'France', 'Spain', 'India']
    sources = ['PubMed', 'bioRxiv', 'medRxiv', 'WHO', 'CDC']
    study_types = ['Clinical Trial', 'Observational', 'Review', 'Case Study', 'Meta-Analysis']
    
    # Generate realistic data
    dates = pd.date_range('2020-01-01', '2023-12-31', periods=n_records)
    
    data = {
        'cord_uid': [f'UID_{i:06d}' for i in range(n_records)],
        'title': [f'COVID-19 Research Study {i+1}: Analysis and Impact' for i in range(n_records)],
        'authors': [f'Author{i} A; Author{i+1} B' for i in range(n_records)],
        'journal': np.random.choice(journals, n_records),
        'publish_time': dates,
        'abstract': [f'This study investigates COVID-19 impacts. Research findings show significant results for public health policy and pandemic response.' for _ in range(n_records)],
        'source': np.random.choice(sources, n_records),
        'country': np.random.choice(countries, n_records),
        'citation_count': np.random.poisson(20, n_records),
        'study_type': np.random.choice(study_types, n_records),
    }
    
    return pd.DataFrame(data)

def process_demo_data(df):
    """Process the demo data safely"""
    df = df.copy()
    
    # Date processing
    df['publish_time'] = pd.to_datetime(df['publish_time'])
    df['publication_year'] = df['publish_time'].dt.year
    df['publication_month'] = df['publish_time'].dt.month
    
    # Text features
    df['title_length'] = df['title'].str.len()
    df['abstract_length'] = df['abstract'].str.len()
    df['has_abstract'] = df['abstract_length'] > 50
    
    # Author features
    df['author_count'] = df['authors'].str.split(';').str.len()
    df['has_multiple_authors'] = df['author_count'] > 1
    
    # Citation categories
    df['citation_category'] = pd.cut(
        df['citation_count'],
        bins=[-1, 0, 10, 25, 50, float('inf')],
        labels=['None', 'Low', 'Medium', 'High', 'Very High']
    )
    
    return df

def safe_plot(plot_function, error_message="Error creating plot"):
    """Safely execute plotting functions"""
    try:
        plot_function()
    except Exception as e:
        st.error(f"{error_message}: {str(e)}")

def main():
    try:
        # Header
        st.markdown('<h1 class="main-header">🔬 CORD-19 Research Data Explorer</h1>', unsafe_allow_html=True)
        st.markdown("### COVID-19 Research Analytics Dashboard")
        
        # Load data
        with st.spinner("Loading research data..."):
            try:
                df = create_demo_data()
                df = process_demo_data(df)
                st.success(f"✅ Successfully loaded {len(df):,} research papers")
            except Exception as e:
                st.error(f"Failed to load data: {str(e)}")
                st.stop()
        
        # Sidebar filters
        st.sidebar.header("🎛️ Filters")
        
        # Year filter
        years = sorted(df['publication_year'].unique())
        year_range = st.sidebar.slider(
            "Publication Years",
            min_value=min(years),
            max_value=max(years),
            value=(min(years), max(years))
        )
        
        # Apply year filter
        filtered_df = df[
            (df['publication_year'] >= year_range[0]) & 
            (df['publication_year'] <= year_range[1])
        ]
        
        # Journal filter
        journals = st.sidebar.multiselect(
            "Select Journals",
            options=df['journal'].unique(),
            default=[]
        )
        
        if journals:
            filtered_df = filtered_df[filtered_df['journal'].isin(journals)]
        
        # Country filter
        countries = st.sidebar.multiselect(
            "Select Countries",
            options=df['country'].unique(),
            default=[]
        )
        
        if countries:
            filtered_df = filtered_df[filtered_df['country'].isin(countries)]
        
        # Citation filter
        min_citations = st.sidebar.slider(
            "Minimum Citations",
            min_value=0,
            max_value=int(df['citation_count'].max()),
            value=0
        )
        
        filtered_df = filtered_df[filtered_df['citation_count'] >= min_citations]
        
        # Check if we have data after filtering
        if filtered_df.empty:
            st.warning("No data matches your filters. Please adjust the criteria.")
            st.stop()
        
        # Metrics
        st.subheader("📊 Overview")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Papers", f"{len(filtered_df):,}")
        with col2:
            st.metric("Journals", f"{filtered_df['journal'].nunique()}")
        with col3:
            st.metric("Countries", f"{filtered_df['country'].nunique()}")
        with col4:
            st.metric("Avg Citations", f"{filtered_df['citation_count'].mean():.1f}")
        
        # Tabs for analysis
        tab1, tab2, tab3, tab4 = st.tabs(["📈 Trends", "📚 Journals", "🌍 Geographic", "📋 Data"])
        
        with tab1:
            st.subheader("Publication Trends")
            
            col1, col2 = st.columns(2)
            
            with col1:
                def plot_yearly():
                    yearly_data = filtered_df.groupby('publication_year').size()
                    fig = px.bar(
                        x=yearly_data.index,
                        y=yearly_data.values,
                        title="Publications by Year",
                        labels={'x': 'Year', 'y': 'Number of Publications'}
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                safe_plot(plot_yearly, "Error creating yearly trend")
            
            with col2:
                def plot_monthly():
                    monthly_data = filtered_df.groupby(filtered_df['publish_time'].dt.month).size()
                    fig = px.line(
                        x=monthly_data.index,
                        y=monthly_data.values,
                        title="Publications by Month",
                        labels={'x': 'Month', 'y': 'Number of Publications'}
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                safe_plot(plot_monthly, "Error creating monthly trend")
            
            # Citation trends
            def plot_citations():
                citation_trend = filtered_df.groupby('publication_year')['citation_count'].mean()
                fig = px.line(
                    x=citation_trend.index,
                    y=citation_trend.values,
                    title="Average Citations Over Time",
                    labels={'x': 'Year', 'y': 'Average Citations'}
                )
                st.plotly_chart(fig, use_container_width=True)
            
            safe_plot(plot_citations, "Error creating citation trends")
        
        with tab2:
            st.subheader("Journal Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                def plot_journal_count():
                    journal_counts = filtered_df['journal'].value_counts().head(10)
                    fig = px.bar(
                        x=journal_counts.values,
                        y=journal_counts.index,
                        orientation='h',
                        title="Top 10 Journals by Publication Count"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                safe_plot(plot_journal_count, "Error creating journal count chart")
            
            with col2:
                def plot_journal_citations():
                    journal_citations = filtered_df.groupby('journal')['citation_count'].mean().sort_values(ascending=False).head(10)
                    fig = px.bar(
                        x=journal_citations.values,
                        y=journal_citations.index,
                        orientation='h',
                        title="Top 10 Journals by Average Citations"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                safe_plot(plot_journal_citations, "Error creating journal citation chart")
        
        with tab3:
            st.subheader("Geographic Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                def plot_country_pie():
                    country_counts = filtered_df['country'].value_counts()
                    fig = px.pie(
                        values=country_counts.values,
                        names=country_counts.index,
                        title="Publications by Country"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                safe_plot(plot_country_pie, "Error creating country chart")
            
            with col2:
                def plot_country_citations():
                    country_citations = filtered_df.groupby('country')['citation_count'].mean().sort_values(ascending=False)
                    fig = px.bar(
                        x=country_citations.values,
                        y=country_citations.index,
                        orientation='h',
                        title="Average Citations by Country"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                safe_plot(plot_country_citations, "Error creating country citation chart")
            
            # Study types
            def plot_study_types():
                study_counts = filtered_df['study_type'].value_counts()
                fig = px.pie(
                    values=study_counts.values,
                    names=study_counts.index,
                    title="Distribution of Study Types"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            safe_plot(plot_study_types, "Error creating study type chart")
        
        with tab4:
            st.subheader("Data Explorer")
            
            # Summary statistics
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Dataset Summary:**")
                st.write(f"- Total records: {len(filtered_df):,}")
                st.write(f"- Date range: {filtered_df['publish_time'].min().date()} to {filtered_df['publish_time'].max().date()}")
                st.write(f"- Citation range: {filtered_df['citation_count'].min()} - {filtered_df['citation_count'].max()}")
                st.write(f"- Average authors per paper: {filtered_df['author_count'].mean():.1f}")
            
            with col2:
                st.write("**Citation Statistics:**")
                st.write(filtered_df['citation_count'].describe())
            
            # Data preview
            st.subheader("Data Preview")
            
            # Select columns to display
            display_columns = st.multiselect(
                "Select columns to display:",
                options=filtered_df.columns.tolist(),
                default=['title', 'journal', 'publication_year', 'country', 'citation_count']
            )
            
            if display_columns:
                # Show data with pagination
                page_size = st.selectbox("Rows per page:", [10, 25, 50], index=1)
                total_pages = (len(filtered_df) - 1) // page_size + 1
                
                if total_pages > 1:
                    page = st.number_input("Page:", min_value=1, max_value=total_pages, value=1)
                    start_idx = (page - 1) * page_size
                    end_idx = start_idx + page_size
                    display_data = filtered_df[display_columns].iloc[start_idx:end_idx]
                else:
                    display_data = filtered_df[display_columns].head(page_size)
                
                st.dataframe(display_data, use_container_width=True)
            
            # Export functionality
            st.subheader("Export Data")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("📥 Prepare CSV Export"):
                    csv_data = filtered_df.to_csv(index=False)
                    st.download_button(
                        label="Download CSV",
                        data=csv_data,
                        file_name=f"cord19_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                    st.success("CSV ready for download!")
            
            with col2:
                if st.button("📊 Show Full Statistics"):
                    with st.expander("Full Dataset Statistics", expanded=True):
                        numeric_cols = filtered_df.select_dtypes(include=[np.number])
                        if not numeric_cols.empty:
                            st.write("**Numeric Columns:**")
                            st.dataframe(numeric_cols.describe())
                        
                        categorical_cols = filtered_df.select_dtypes(include=['object', 'category'])
                        if not categorical_cols.empty:
                            st.write("**Categorical Columns:**")
                            for col in categorical_cols.columns:
                                if col != 'abstract':  # Skip long text
                                    unique_count = filtered_df[col].nunique()
                                    st.write(f"**{col}**: {unique_count} unique values")
        
        # Footer
        st.markdown("---")
        st.markdown("**CORD-19 Research Explorer** | COVID-19 Research Data Analysis Dashboard")
        
    except Exception as e:
        st.error(f"Application Error: {str(e)}")
        st.info("Please refresh the page to restart the application.")
        
        # Debug information
        with st.expander("🔧 Debug Information"):
            import traceback
            st.code(traceback.format_exc())

if __name__ == "__main__":
    main()