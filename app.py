"""
COVID-Era Earnings Manipulation Analysis
MINIMAL VERSION - Works on Streamlit Cloud
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import io

# Page Configuration
st.set_page_config(
    page_title="COVID Earnings Analysis",
    page_icon="📊",
    layout="wide"
)

# Initialize session state
if 'df' not in st.session_state:
    st.session_state.df = None

# Title
st.title("📊 COVID-Era Earnings Manipulation Analysis")
st.markdown("A Difference-in-Differences (DiD) Analysis of Financial Reporting")

# ============================================================================
# SIDEBAR - DATA UPLOAD
# ============================================================================
with st.sidebar:
    st.header("📁 Data Upload")
    
    # Upload CSV file
    uploaded_file = st.file_uploader(
        "Upload your CSV file",
        type=['csv'],
        help="Upload CSV with: company_id, year, treatment_group, post_covid, net_profit"
    )
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            
            # Check required columns
            required = ['company_id', 'year', 'treatment_group', 'post_covid']
            missing = [col for col in required if col not in df.columns]
            
            if missing:
                st.error(f"Missing columns: {missing}")
            else:
                st.session_state.df = df
                st.success(f"✅ Loaded {len(df)} rows")
                st.info(f"Companies: {df['company_id'].nunique()}")
                
        except Exception as e:
            st.error(f"Error: {str(e)}")
    
    # Or use sample data
    st.markdown("---")
    st.header("📊 Sample Data")
    
    if st.button("Generate Sample Data"):
        np.random.seed(42)
        
        # Generate simple sample data
        data = []
        for i in range(150):
            treatment = np.random.choice([0, 1], p=[0.6, 0.4])
            for year in [2019, 2020]:
                base = 1000
                if year == 2020 and treatment == 1:
                    profit = base * np.random.uniform(0.9, 1.3)  # Possible manipulation
                else:
                    profit = base * np.random.uniform(0.8, 1.1)
                
                data.append({
                    'company_id': f'C{i:03d}',
                    'year': year,
                    'treatment_group': treatment,
                    'post_covid': 1 if year == 2020 else 0,
                    'net_profit': round(profit, 2),
                    'revenue': round(profit * np.random.uniform(5, 10), 2)
                })
        
        df = pd.DataFrame(data)
        st.session_state.df = df
        st.success(f"✅ Generated {len(df)} rows")
        st.rerun()
    
    # Download template
    st.markdown("---")
    st.header("📥 Template")
    
    template_data = {
        'company_id': ['CMP001', 'CMP001', 'CMP002', 'CMP002'],
        'year': [2019, 2020, 2019, 2020],
        'treatment_group': [1, 1, 0, 0],
        'post_covid': [0, 1, 0, 1],
        'net_profit': [100.0, 120.0, 200.0, 210.0],
        'revenue': [1000.0, 1100.0, 2000.0, 2050.0]
    }
    template_df = pd.DataFrame(template_data)
    
    csv = template_df.to_csv(index=False)
    st.download_button(
        "Download CSV Template",
        data=csv,
        file_name="template.csv",
        mime="text/csv"
    )

# ============================================================================
# MAIN CONTENT
# ============================================================================
if st.session_state.df is None:
    # Welcome screen
    st.markdown("---")
    st.markdown("""
    ## 👈 Get Started
    
    1. **Upload your CSV file** using the sidebar
    2. **Or generate sample data** for testing
    3. **Required columns in CSV:**
       - `company_id` - Unique company identifier
       - `year` - Year of observation
       - `treatment_group` - 1 for treatment, 0 for control
       - `post_covid` - 1 for 2020+, 0 for pre-2020
       - `net_profit` - Net profit amount
       
    ## 📊 Example CSV Format:
    """)
    
    example_df = pd.DataFrame({
        'company_id': ['CMP001', 'CMP001', 'CMP002', 'CMP002'],
        'year': [2019, 2020, 2019, 2020],
        'treatment_group': [1, 1, 0, 0],
        'post_covid': [0, 1, 0, 1],
        'net_profit': [100.0, 120.0, 200.0, 210.0],
        'revenue': [1000.0, 1100.0, 2000.0, 2050.0]
    })
    st.dataframe(example_df)
    
    st.stop()

# Data is loaded - show analysis
df = st.session_state.df

# Create tabs
tab1, tab2, tab3, tab4 = st.tabs(["📊 Data", "📈 DiD Analysis", "📉 Charts", "📋 Report"])

with tab1:
    st.header("Data Overview")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Companies", df['company_id'].nunique())
    with col2:
        st.metric("Rows", len(df))
    with col3:
        st.metric("Years", df['year'].nunique())
    
    st.subheader("Data Preview")
    st.dataframe(df.head(100))
    
    st.subheader("Summary Statistics")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        st.write(df[numeric_cols].describe())

with tab2:
    st.header("Difference-in-Differences Analysis")
    
    # Select outcome variable
    numeric_cols = [col for col in df.select_dtypes(include=[np.number]).columns 
                   if col not in ['year', 'treatment_group', 'post_covid']]
    
    if not numeric_cols:
        st.warning("No numeric columns found for analysis")
        st.stop()
    
    outcome_var = st.selectbox("Select variable to analyze:", numeric_cols)
    
    # Manual DiD calculation
    st.subheader("Manual DiD Calculation")
    
    # Prepare data
    df_analysis = df.copy()
    df_analysis['period'] = df_analysis['year'].apply(lambda x: 'pre' if x < 2020 else 'post')
    
    # Calculate group means
    means = df_analysis.groupby(['treatment_group', 'period'])[outcome_var].mean().unstack()
    
    # Calculate DiD
    control_change = means.loc[0, 'post'] - means.loc[0, 'pre']
    treatment_change = means.loc[1, 'post'] - means.loc[1, 'pre']
    did_effect = treatment_change - control_change
    
    # Display results
    results_df = pd.DataFrame({
        'Group': ['Control', 'Treatment'],
        'Pre-2019': [means.loc[0, 'pre'], means.loc[1, 'pre']],
        'Post-2020': [means.loc[0, 'post'], means.loc[1, 'post']],
        'Change': [control_change, treatment_change]
    })
    
    st.dataframe(results_df.style.format("{:.2f}"))
    
    # DiD result
    st.markdown(f"""
    ### 🎯 DiD Result
    
    **Treatment Effect:** {treatment_change:.2f}
    
    **Control Effect:** {control_change:.2f}
    
    **DiD Coefficient (Treatment - Control):** **{did_effect:.2f}**
    """)
    
    # Interpretation
    if did_effect > 0:
        st.success(f"✅ The treatment group shows {did_effect:.2f} higher change in {outcome_var}. This suggests potential earnings manipulation.")
    else:
        st.info(f"📊 The treatment group shows {did_effect:.2f} change in {outcome_var}. No strong evidence of earnings manipulation.")

with tab3:
    st.header("Visualizations")
    
    # Select variable for charts
    numeric_cols = [col for col in df.select_dtypes(include=[np.number]).columns 
                   if col not in ['year', 'treatment_group', 'post_covid']]
    
    if not numeric_cols:
        st.warning("No numeric columns for charts")
        st.stop()
    
    chart_var = st.selectbox("Select variable for charts:", numeric_cols)
    
    # Time trend chart
    st.subheader("Time Trends")
    
    trend_data = df.groupby(['year', 'treatment_group'])[chart_var].mean().reset_index()
    trend_data['Group'] = trend_data['treatment_group'].apply(lambda x: 'Treatment' if x == 1 else 'Control')
    
    fig1 = px.line(
        trend_data,
        x='year',
        y=chart_var,
        color='Group',
        markers=True,
        title=f"{chart_var.replace('_', ' ').title()} Over Time"
    )
    
    if 2020 in df['year'].values:
        fig1.add_vrect(x0=2019.5, x1=2020.5, fillcolor="red", opacity=0.1, annotation_text="COVID")
    
    st.plotly_chart(fig1, use_container_width=True)
    
    # Box plot
    st.subheader("Distribution by Period")
    
    df_chart = df.copy()
    df_chart['Period'] = df_chart['year'].apply(lambda x: 'Post-COVID' if x >= 2020 else 'Pre-COVID')
    df_chart['Group'] = df_chart['treatment_group'].apply(lambda x: 'Treatment' if x == 1 else 'Control')
    
    fig2 = px.box(
        df_chart,
        x='Period',
        y=chart_var,
        color='Group',
        title=f"Distribution of {chart_var.replace('_', ' ').title()}"
    )
    
    st.plotly_chart(fig2, use_container_width=True)
    
    # Simple t-test simulation (without scipy)
    st.subheader("Group Comparison")
    
    pre_data = df[df['year'] < 2020]
    post_data = df[df['year'] >= 2020]
    
    if len(pre_data) > 0 and len(post_data) > 0:
        # Calculate means
        treatment_pre = pre_data[pre_data['treatment_group'] == 1][chart_var].mean()
        treatment_post = post_data[post_data['treatment_group'] == 1][chart_var].mean()
        control_pre = pre_data[pre_data['treatment_group'] == 0][chart_var].mean()
        control_post = post_data[post_data['treatment_group'] == 0][chart_var].mean()
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Treatment Pre", f"{treatment_pre:.2f}")
            st.metric("Treatment Post", f"{treatment_post:.2f}")
            st.metric("Treatment Change", f"{treatment_post - treatment_pre:.2f}")
        
        with col2:
            st.metric("Control Pre", f"{control_pre:.2f}")
            st.metric("Control Post", f"{control_post:.2f}")
            st.metric("Control Change", f"{control_post - control_pre:.2f}")

with tab4:
    st.header("Analysis Report")
    
    # Generate report
    report_date = datetime.now().strftime("%Y-%m-%d %H:%M")
    total_companies = df['company_id'].nunique()
    treatment_count = df[df['treatment_group'] == 1]['company_id'].nunique()
    control_count = df[df['treatment_group'] == 0]['company_id'].nunique()
    
    # Calculate DiD if available
    did_result = ""
    if 'outcome_var' in st.session_state:
        df_analysis = df.copy()
        df_analysis['period'] = df_analysis['year'].apply(lambda x: 'pre' if x < 2020 else 'post')
        means = df_analysis.groupby(['treatment_group', 'period'])[st.session_state.outcome_var].mean().unstack()
        did_result = (means.loc[1, 'post'] - means.loc[1, 'pre']) - (means.loc[0, 'post'] - means.loc[0, 'pre'])
    
    report_content = f"""
# COVID Earnings Analysis Report

**Date:** {report_date}
**Dataset:** {total_companies} companies, {len(df)} rows
**Treatment Group:** {treatment_count} companies
**Control Group:** {control_count} companies

## Summary
Difference-in-Differences analysis completed on uploaded financial data.

## Key Metrics
- Total observations: {len(df)}
- Years covered: {df['year'].min()} to {df['year'].max()}
- Companies analyzed: {total_companies}

## DiD Result
DiD Coefficient: {did_result:.2f if did_result != '' else 'Run analysis in DiD tab'}

## Interpretation
Analysis suggests {'potential earnings manipulation' if did_result > 0 else 'no significant evidence'} in treatment group.

## Recommendations
1. Verify data quality
2. Include more financial metrics
3. Conduct robustness checks
"""
    
    st.markdown(report_content)
    
    # Export options
    st.subheader("Export")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Export report
        st.download_button(
            "Download Report",
            data=report_content,
            file_name="covid_report.txt",
            mime="text/plain"
        )
    
    with col2:
        # Export data
        csv = df.to_csv(index=False)
        st.download_button(
            "Download Data",
            data=csv,
            file_name="analysis_data.csv",
            mime="text/csv"
        )

# Footer
st.markdown("---")
st.markdown("COVID Earnings Analysis Tool | Upload your CSV data for DiD analysis")
