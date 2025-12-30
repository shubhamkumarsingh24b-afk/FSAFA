"""
COVID-Era Earnings Manipulation Analysis
Streamlit Application - Compatible with Streamlit Cloud Python 3.13
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import io
import base64

# Try to import stats libraries (handle gracefully if not available)
try:
    import statsmodels.formula.api as smf
    import scipy.stats as stats
    STATS_AVAILABLE = True
except ImportError:
    STATS_AVAILABLE = False
    st.warning("Note: Some statistical features require statsmodels and scipy")

# Page Configuration
st.set_page_config(
    page_title="COVID Earnings Manipulation Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid #3B82F6;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
    }
    .upload-box {
        border: 2px dashed #4CAF50;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        background-color: #f8fff8;
        margin: 2rem 0;
    }
    .success-box {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 5px;
        border-left: 5px solid #28a745;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #d1ecf1;
        color: #0c5460;
        padding: 1rem;
        border-radius: 5px;
        border-left: 5px solid #17a2b8;
        margin: 1rem 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #F0F2F6;
        border-radius: 5px 5px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1E3A8A;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'df' not in st.session_state:
    st.session_state.df = None
    st.session_state.data_loaded = False
    st.session_state.analysis_complete = False

# Title
st.markdown('<h1 class="main-header">📊 COVID-Era Earnings Manipulation Analysis</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">A Difference-in-Differences (DiD) Analysis of Financial Reporting During the Pandemic</p>', unsafe_allow_html=True)

# ============================================================================
# SIDEBAR - DATA UPLOAD & CONFIGURATION
# ============================================================================
with st.sidebar:
    st.markdown("## 📁 Data Configuration")
    
    # Data source selection
    data_option = st.radio(
        "Choose data source:",
        ["📤 Upload CSV File", "🎲 Generate Sample Data", "🚀 Use Demo Data"]
    )
    
    if data_option == "📤 Upload CSV File":
        st.markdown("### Upload Your CSV")
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type=['csv'],
            help="CSV should contain: company_id, year, treatment_group, post_covid, net_profit"
        )
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                
                # Validate required columns
                required_cols = ['company_id', 'year', 'treatment_group', 'post_covid']
                missing_cols = [col for col in required_cols if col not in df.columns]
                
                if missing_cols:
                    st.error(f"❌ Missing required columns: {missing_cols}")
                else:
                    st.session_state.df = df
                    st.session_state.data_loaded = True
                    st.success(f"✅ Data loaded successfully!")
                    st.info(f"""
                    **File Info:**
                    - Rows: {len(df):,}
                    - Columns: {len(df.columns)}
                    - Companies: {df['company_id'].nunique()}
                    - Years: {df['year'].min()} to {df['year'].max()}
                    """)
                    
            except Exception as e:
                st.error(f"❌ Error reading file: {str(e)}")
    
    elif data_option == "🎲 Generate Sample Data":
        st.markdown("### Generate Sample Data")
        
        sample_size = st.slider("Number of companies", 50, 500, 200)
        
        if st.button("Generate Data", type="primary"):
            with st.spinner("Generating sample data..."):
                np.random.seed(42)
                
                # Generate realistic sample data
                n_companies = sample_size
                industries = ['Aviation', 'Hospitality', 'Real Estate', 'IT', 'FMCG', 'Pharma']
                years = [2018, 2019, 2020, 2021]
                
                data = []
                for company_id in range(n_companies):
                    industry = np.random.choice(industries)
                    treatment = 1 if industry in ['Aviation', 'Hospitality', 'Real Estate'] else 0
                    
                    base_revenue = np.random.lognormal(12, 0.5)
                    growth_rate = np.random.normal(0.08, 0.03)
                    
                    for year_idx, year in enumerate(years):
                        # Revenue calculation
                        time_trend = (1 + growth_rate) ** year_idx
                        revenue = base_revenue * time_trend * np.random.lognormal(0, 0.1)
                        
                        # COVID impact
                        if year >= 2020 and treatment == 1:
                            revenue *= np.random.uniform(0.5, 0.8)  # Significant reduction
                        
                        # Profit margin with potential manipulation
                        base_margin = np.random.uniform(0.05, 0.2)
                        
                        # Simulate earnings manipulation in treatment group post-COVID
                        if year >= 2020 and treatment == 1:
                            if np.random.random() < 0.4:  # 40% chance of manipulation
                                manipulation = np.random.uniform(0.05, 0.15)
                                profit_margin = base_margin + manipulation
                            else:
                                profit_margin = base_margin
                        else:
                            profit_margin = base_margin
                        
                        net_profit = revenue * profit_margin
                        
                        data.append({
                            'company_id': f'COMP{company_id:04d}',
                            'company_name': f'{industry} Company {company_id}',
                            'industry': industry,
                            'year': year,
                            'treatment_group': treatment,
                            'post_covid': 1 if year >= 2020 else 0,
                            'revenue': round(revenue, 2),
                            'net_profit': round(net_profit, 2),
                            'cfo': round(net_profit * np.random.uniform(0.8, 1.2), 2),
                            'total_assets': round(revenue * np.random.uniform(1.5, 3.0), 2),
                            'receivables': round(revenue * np.random.uniform(0.1, 0.3), 2),
                            'debt': round(revenue * np.random.uniform(0.5, 1.5), 2),
                            'profit_margin': round(profit_margin * 100, 2)
                        })
                
                df = pd.DataFrame(data)
                st.session_state.df = df
                st.session_state.data_loaded = True
                st.success(f"✅ Generated {len(df):,} rows of sample data!")
    
    else:  # Use Demo Data
        st.markdown("### Demo Dataset")
        
        if st.button("Load Demo Data", type="primary"):
            # Create simple demo data
            np.random.seed(123)
            
            data = []
            for i in range(150):
                treatment = np.random.choice([0, 1], p=[0.6, 0.4])
                industry = 'Aviation' if treatment == 1 else 'IT'
                
                for year in [2018, 2019, 2020, 2021]:
                    base = 1000 + i * 10
                    
                    # Revenue affected by COVID for treatment group
                    if year >= 2020 and treatment == 1:
                        revenue = base * (1 + 0.1 * (year - 2018)) * np.random.uniform(0.6, 0.9)
                        # Potential earnings manipulation
                        profit = revenue * np.random.uniform(0.12, 0.25)
                    else:
                        revenue = base * (1 + 0.1 * (year - 2018)) * np.random.uniform(0.9, 1.1)
                        profit = revenue * np.random.uniform(0.08, 0.15)
                    
                    data.append({
                        'company_id': f'DEMO{i:03d}',
                        'industry': industry,
                        'year': year,
                        'treatment_group': treatment,
                        'post_covid': 1 if year >= 2020 else 0,
                        'revenue': round(revenue, 2),
                        'net_profit': round(profit, 2),
                        'cfo': round(profit * np.random.uniform(0.9, 1.1), 2)
                    })
            
            df = pd.DataFrame(data)
            st.session_state.df = df
            st.session_state.data_loaded = True
            st.success("✅ Demo data loaded!")
    
    # Analysis Configuration
    if st.session_state.data_loaded:
        st.markdown("---")
        st.markdown("## ⚙️ Analysis Settings")
        
        df = st.session_state.df
        
        # Select outcome variable
        numeric_cols = [col for col in df.select_dtypes(include=[np.number]).columns 
                       if col not in ['year', 'treatment_group', 'post_covid', 'company_id']]
        
        if numeric_cols:
            outcome_var = st.selectbox(
                "Select outcome variable:",
                numeric_cols,
                index=numeric_cols.index('net_profit') if 'net_profit' in numeric_cols else 0
            )
            st.session_state.outcome_var = outcome_var
        
        # Time period selection
        years = sorted(df['year'].unique())
        if len(years) >= 2:
            pre_covid_end = st.selectbox(
                "Pre-COVID period (end year):",
                [y for y in years if y < 2020] or years[:-1],
                index=0
            )
            
            post_covid_start = st.selectbox(
                "Post-COVID period (start year):",
                [y for y in years if y > pre_covid_end],
                index=0
            )
            
            st.session_state.pre_covid_end = pre_covid_end
            st.session_state.post_covid_start = post_covid_start
        
        if st.button("🚀 Run DiD Analysis", type="primary", use_container_width=True):
            st.session_state.analysis_complete = True
            st.rerun()
    
    # Download template
    st.markdown("---")
    st.markdown("## 📥 Download Template")
    
    template_data = {
        'company_id': ['CMP001', 'CMP001', 'CMP002', 'CMP002'],
        'year': [2019, 2020, 2019, 2020],
        'treatment_group': [1, 1, 0, 0],
        'post_covid': [0, 1, 0, 1],
        'net_profit': [100.0, 120.0, 200.0, 210.0],
        'revenue': [1000.0, 1100.0, 2000.0, 2050.0],
        'cfo': [90.0, 95.0, 180.0, 190.0],
        'industry': ['Aviation', 'Aviation', 'IT', 'IT']
    }
    template_df = pd.DataFrame(template_data)
    
    csv = template_df.to_csv(index=False)
    st.download_button(
        label="Download CSV Template",
        data=csv,
        file_name="financial_data_template.csv",
        mime="text/csv",
        use_container_width=True
    )

# ============================================================================
# MAIN CONTENT
# ============================================================================
if not st.session_state.data_loaded:
    # Welcome screen
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown('<div class="upload-box">', unsafe_allow_html=True)
        st.markdown("## 📤 Get Started")
        st.markdown("""
        1. **Upload your CSV file** using the sidebar
        2. **Or generate sample data** for testing
        3. **Configure analysis settings**
        4. **Run DiD analysis**
        
        **Required columns:** company_id, year, treatment_group, post_covid
        """)
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Project overview
    st.markdown("---")
    st.markdown("## 📋 Project Overview")
    
    st.markdown("""
    This application performs **Difference-in-Differences (DiD) analysis** to detect potential earnings manipulation during the COVID-19 pandemic.
    
    ### Key Features:
    - **CSV Data Upload**: Upload your financial data
    - **DiD Analysis**: Treatment vs Control group comparison
    - **Interactive Visualizations**: Plotly charts and graphs
    - **Statistical Results**: Manual and regression DiD calculations
    - **Report Generation**: Exportable analysis reports
    
    ### Methodology:
    - **Treatment Group**: COVID-affected industries (Aviation, Hospitality, Real Estate)
    - **Control Group**: Less affected industries (IT, FMCG, Pharma)
    - **Pre-Period**: Before 2020
    - **Post-Period**: 2020 and later
    """)
    
    st.stop()

# ============================================================================
# DATA LOADED - SHOW ANALYSIS TABS
# ============================================================================
df = st.session_state.df

# Create tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Dashboard", 
    "📈 DiD Analysis", 
    "📉 Visualizations", 
    "📋 Statistical Tests",
    "📄 Report"
])

with tab1:
    st.header("Data Dashboard")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Companies", 
            df['company_id'].nunique(),
            help="Unique companies in dataset"
        )
    
    with col2:
        treatment_count = df[df['treatment_group'] == 1]['company_id'].nunique()
        st.metric(
            "Treatment Group", 
            treatment_count,
            delta=f"{treatment_count/df['company_id'].nunique()*100:.1f}%"
        )
    
    with col3:
        years = df['year'].nunique()
        st.metric(
            "Years", 
            years,
            help="Years covered in data"
        )
    
    with col4:
        if 'net_profit' in df.columns:
            avg_profit = df['net_profit'].mean()
            st.metric(
                "Avg. Net Profit", 
                f"${avg_profit:,.0f}",
                help="Average net profit across all companies"
            )
    
    # Data preview
    st.subheader("Data Preview")
    
    show_cols = st.multiselect(
        "Select columns to display:",
        df.columns.tolist(),
        default=['company_id', 'year', 'treatment_group', 'post_covid', 'net_profit', 'revenue'][:5]
    )
    
    if show_cols:
        st.dataframe(df[show_cols].head(100), use_container_width=True, height=300)
    
    # Summary statistics
    st.subheader("Summary Statistics")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        summary = df[numeric_cols].describe().T
        summary['missing'] = df[numeric_cols].isnull().sum()
        summary['missing_pct'] = (summary['missing'] / len(df) * 100).round(1)
        
        st.dataframe(summary, use_container_width=True)
    
    # Industry distribution
    if 'industry' in df.columns:
        st.subheader("Industry Distribution")
        
        industry_counts = df[['company_id', 'industry']].drop_duplicates()['industry'].value_counts()
        
        fig = px.pie(
            values=industry_counts.values,
            names=industry_counts.index,
            title="Companies by Industry",
            hole=0.3
        )
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.header("Difference-in-Differences Analysis")
    
    if not st.session_state.analysis_complete:
        st.info("👈 Configure analysis settings in sidebar and click 'Run DiD Analysis'")
        st.stop()
    
    outcome_var = st.session_state.get('outcome_var', 'net_profit')
    pre_end = st.session_state.get('pre_covid_end', 2019)
    post_start = st.session_state.get('post_covid_start', 2020)
    
    # Prepare data for DiD
    analysis_df = df.copy()
    analysis_df['period'] = analysis_df['year'].apply(
        lambda x: 'pre' if x <= pre_end else 'post'
    )
    analysis_df['post_dummy'] = (analysis_df['period'] == 'post').astype(int)
    
    # 1. Manual DiD Calculation
    st.subheader("1. Manual DiD Calculation")
    
    # Calculate group means
    group_means = analysis_df.groupby(['treatment_group', 'period'])[outcome_var].mean().unstack()
    
    # Calculate changes
    control_change = group_means.loc[0, 'post'] - group_means.loc[0, 'pre']
    treatment_change = group_means.loc[1, 'post'] - group_means.loc[1, 'pre']
    did_effect = treatment_change - control_change
    
    # Display results
    results_df = pd.DataFrame({
        'Group': ['Control (0)', 'Treatment (1)'],
        'Pre-Period Mean': [group_means.loc[0, 'pre'], group_means.loc[1, 'pre']],
        'Post-Period Mean': [group_means.loc[0, 'post'], group_means.loc[1, 'post']],
        'Change': [control_change, treatment_change]
    })
    
    st.dataframe(
        results_df.style.format({
            'Pre-Period Mean': '{:.2f}',
            'Post-Period Mean': '{:.2f}',
            'Change': '{:.2f}'
        }).apply(
            lambda x: ['background-color: #e8f5e9' if x.name == 1 else '' for _ in x],
            axis=1
        ),
        use_container_width=True
    )
    
    # DiD Result
    st.markdown(f"""
    <div class="success-box">
    <h4>🎯 DiD Result</h4>
    <p><strong>Control Group Change:</strong> {control_change:.2f}</p>
    <p><strong>Treatment Group Change:</strong> {treatment_change:.2f}</p>
    <p><strong>DiD Effect (Treatment - Control):</strong> <span style="color: #28a745; font-weight: bold;">{did_effect:.2f}</span></p>
    </div>
    """, unsafe_allow_html=True)
    
    # Interpretation
    interpretation = """
    <div class="info-box">
    <h4>📊 Interpretation</h4>
    """
    
    if did_effect > 0:
        interpretation += f"""
        <p>The treatment group shows <strong>{did_effect:.2f}</strong> higher change in {outcome_var} compared to the control group.</p>
        <p>This suggests <strong>potential earnings manipulation</strong> in COVID-affected industries post-pandemic.</p>
        """
    else:
        interpretation += f"""
        <p>The treatment group shows <strong>{did_effect:.2f}</strong> change in {outcome_var} compared to the control group.</p>
        <p>This suggests <strong>no significant evidence of earnings manipulation</strong> in the treatment group.</p>
        """
    
    interpretation += "</div>"
    st.markdown(interpretation, unsafe_allow_html=True)
    
    # 2. Regression DiD Analysis
    st.subheader("2. Regression DiD Analysis")
    
    if STATS_AVAILABLE:
        try:
            # Create interaction term
            analysis_df['treatment_post'] = analysis_df['treatment_group'] * analysis_df['post_dummy']
            
            # Run regression
            model = smf.ols(f"{outcome_var} ~ treatment_group + post_dummy + treatment_post", 
                          data=analysis_df).fit()
            
            # Display results
            coef_df = pd.DataFrame({
                'Variable': model.params.index,
                'Coefficient': model.params.values,
                'Std Error': model.bse.values,
                't-value': model.tvalues.values,
                'P-value': model.pvalues.values
            })
            
            st.dataframe(
                coef_df.style.format({
                    'Coefficient': '{:.4f}',
                    'Std Error': '{:.4f}',
                    't-value': '{:.2f}',
                    'P-value': '{:.4f}'
                }).apply(
                    lambda x: ['background-color: #e8f5e9' if 'treatment_post' in str(x['Variable']) else '' for _ in x],
                    axis=1
                ),
                use_container_width=True
            )
            
            # Key finding
            did_coef = model.params.get('treatment_post', 0)
            did_pval = model.pvalues.get('treatment_post', 1)
            
            if did_pval < 0.05:
                sig_text = "statistically significant"
                sig_color = "#28a745"
            else:
                sig_text = "not statistically significant"
                sig_color = "#dc3545"
            
            st.markdown(f"""
            <div style="padding: 1rem; background-color: #f8f9fa; border-radius: 5px; border-left: 5px solid {sig_color};">
            <h4>🔑 Key Finding</h4>
            <p><strong>DiD Coefficient (treatment × post):</strong> {did_coef:.4f}</p>
            <p><strong>P-value:</strong> {did_pval:.4f}</p>
            <p>This effect is <strong style="color: {sig_color}">{sig_text}</strong> at the 5% level.</p>
            </div>
            """, unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"Regression error: {str(e)}")
    else:
        st.warning("Statsmodels not available. Regression analysis disabled.")

with tab3:
    st.header("Visualizations")
    
    outcome_var = st.session_state.get('outcome_var', 'net_profit')
    
    # Visualization selector
    viz_type = st.selectbox(
        "Select visualization type:",
        ["Time Trends", "Group Comparison", "Distribution Analysis", "Scatter Plot"]
    )
    
    if viz_type == "Time Trends":
        st.subheader(f"{outcome_var} Trends Over Time")
        
        # Calculate average by year and group
        trend_data = df.groupby(['year', 'treatment_group'])[outcome_var].mean().reset_index()
        trend_data['Group'] = trend_data['treatment_group'].apply(lambda x: 'Treatment' if x == 1 else 'Control')
        
        fig = px.line(
            trend_data,
            x='year',
            y=outcome_var,
            color='Group',
            markers=True,
            title=f"Average {outcome_var} Over Time",
            line_shape='linear'
        )
        
        # Add COVID period shading if 2020 exists
        if 2020 in df['year'].unique():
            fig.add_vrect(
                x0=2019.5, x1=2021.5,
                fillcolor="red",
                opacity=0.1,
                line_width=0,
                annotation_text="COVID Period",
                annotation_position="top left"
            )
        
        fig.update_layout(
            xaxis_title="Year",
            yaxis_title=outcome_var.replace('_', ' ').title(),
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    elif viz_type == "Group Comparison":
        st.subheader("Pre vs Post COVID Comparison")
        
        # Create period column
        df_viz = df.copy()
        df_viz['Period'] = df_viz['year'].apply(
            lambda x: 'Post-COVID' if x >= 2020 else 'Pre-COVID'
        )
        df_viz['Group'] = df_viz['treatment_group'].apply(
            lambda x: 'Treatment' if x == 1 else 'Control'
        )
        
        fig = px.box(
            df_viz,
            x='Period',
            y=outcome_var,
            color='Group',
            title=f"Distribution of {outcome_var.replace('_', ' ').title()}",
            points=False
        )
        
        fig.update_layout(
            boxmode='group',
            xaxis_title="",
            yaxis_title=outcome_var.replace('_', ' ').title()
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    elif viz_type == "Distribution Analysis":
        st.subheader("Distribution Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Histogram
            fig1 = px.histogram(
                df,
                x=outcome_var,
                color='treatment_group',
                nbins=30,
                title=f"Distribution of {outcome_var.replace('_', ' ').title()}",
                barmode='overlay',
                opacity=0.7,
                labels={'treatment_group': 'Treatment Group'}
            )
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            # Violin plot
            fig2 = px.violin(
                df,
                y=outcome_var,
                x='treatment_group',
                color='treatment_group',
                box=True,
                points=False,
                title=f"Violin Plot by Group",
                labels={'treatment_group': 'Group (0=Control, 1=Treatment)'}
            )
            st.plotly_chart(fig2, use_container_width=True)
    
    else:  # Scatter Plot
        st.subheader("Scatter Plot")
        
        # Select X variable
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        x_options = [col for col in numeric_cols if col != outcome_var]
        
        if x_options:
            x_var = st.selectbox(
                "Select X-axis variable:",
                x_options,
                index=0
            )
            
            fig = px.scatter(
                df,
                x=x_var,
                y=outcome_var,
                color='treatment_group',
                hover_data=['year', 'industry'] if 'industry' in df.columns else None,
                title=f"{outcome_var.replace('_', ' ').title()} vs {x_var.replace('_', ' ').title()}",
                labels={'treatment_group': 'Treatment Group'}
            )
            
            st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.header("Statistical Tests")
    
    if not STATS_AVAILABLE:
        st.warning("Statistical tests require statsmodels and scipy. Install via requirements.txt")
        st.stop()
    
    outcome_var = st.session_state.get('outcome_var', 'net_profit')
    
    st.subheader("T-tests for Group Differences")
    
    # Prepare data
    pre_data = df[df['year'] < 2020]
    post_data = df[df['year'] >= 2020]
    
    results = []
    
    # Test 1: Treatment vs Control in Pre-period
    if len(pre_data) > 0:
        treatment_pre = pre_data[pre_data['treatment_group'] == 1][outcome_var]
        control_pre = pre_data[pre_data['treatment_group'] == 0][outcome_var]
        
        if len(treatment_pre) > 1 and len(control_pre) > 1:
            t_stat, p_val = stats.ttest_ind(treatment_pre, control_pre, equal_var=False)
            results.append({
                'Test': 'Pre-COVID: Treatment vs Control',
                'Treatment Mean': treatment_pre.mean(),
                'Control Mean': control_pre.mean(),
                't-statistic': t_stat,
                'p-value': p_val,
                'Significant': p_val < 0.05
            })
    
    # Test 2: Treatment vs Control in Post-period
    if len(post_data) > 0:
        treatment_post = post_data[post_data['treatment_group'] == 1][outcome_var]
        control_post = post_data[post_data['treatment_group'] == 0][outcome_var]
        
        if len(treatment_post) > 1 and len(control_post) > 1:
            t_stat, p_val = stats.ttest_ind(treatment_post, control_post, equal_var=False)
            results.append({
                'Test': 'Post-COVID: Treatment vs Control',
                'Treatment Mean': treatment_post.mean(),
                'Control Mean': control_post.mean(),
                't-statistic': t_stat,
                'p-value': p_val,
                'Significant': p_val < 0.05
            })
    
    if results:
        results_df = pd.DataFrame(results)
        st.dataframe(
            results_df.style.format({
                'Treatment Mean': '{:.2f}',
                'Control Mean': '{:.2f}',
                't-statistic': '{:.3f}',
                'p-value': '{:.4f}'
            }).apply(
                lambda x: ['background-color: #e8f5e9' if x['Significant'] else '' for _ in x],
                axis=1
            ),
            use_container_width=True
        )

with tab5:
    st.header("Analysis Report")
    
    # Generate report
    report_date = datetime.now().strftime("%Y-%m-%d %H:%M")
    total_companies = df['company_id'].nunique()
    treatment_count = df[df['treatment_group'] == 1]['company_id'].nunique()
    control_count = df[df['treatment_group'] == 0]['company_id'].nunique()
    years = sorted(df['year'].unique())
    
    # Get DiD results if available
    did_result = ""
    if st.session_state.analysis_complete:
        # Recalculate or get from session
        analysis_df = df.copy()
        pre_end = st.session_state.get('pre_covid_end', 2019)
        analysis_df['period'] = analysis_df['year'].apply(
            lambda x: 'pre' if x <= pre_end else 'post'
        )
        
        group_means = analysis_df.groupby(['treatment_group', 'period'])[st.session_state.outcome_var].mean().unstack()
        control_change = group_means.loc[0, 'post'] - group_means.loc[0, 'pre']
        treatment_change = group_means.loc[1, 'post'] - group_means.loc[1, 'pre']
        did_result = treatment_change - control_change
    
    report_content = f"""
# COVID-Era Earnings Manipulation Analysis Report

**Generated:** {report_date}
**Dataset:** {total_companies} companies, {len(df)} observations
**Analysis Variable:** {st.session_state.get('outcome_var', 'Not specified')}

## Executive Summary

This report presents a Difference-in-Differences (DiD) analysis of potential earnings manipulation during the COVID-19 pandemic.

### Key Findings:
1. **Dataset Composition**: {total_companies} companies ({treatment_count} treatment, {control_count} control)
2. **Time Period**: {years[0]} to {years[-1]}
3. **DiD Result**: {did_result:.2f if did_result != '' else 'Not calculated'}
4. **Interpretation**: {'Potential earnings manipulation detected' if did_result > 0 else 'No significant evidence found' if did_result != '' else 'Analysis not completed'}

## Methodology

### Research Design
- **Treatment Group**: COVID-affected industries
- **Control Group**: Less affected industries
- **Statistical Method**: Difference-in-Differences (DiD)

### Data Description
- **Total Observations**: {len(df):,}
- **Companies by Group**: Treatment={treatment_count}, Control={control_count}
- **Years Covered**: {', '.join(map(str, years))}

## Recommendations

1. **Data Verification**: Verify data quality and completeness
2. **Extended Analysis**: Include more financial metrics
3. **Robustness Checks**: Test alternative model specifications
4. **Real-World Validation**: Compare with industry reports

---
*Generated by COVID Earnings Analysis Tool*
"""
    
    st.markdown(report_content)
    
    # Export options
    st.subheader("Export Options")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Export report as text
        st.download_button(
            label="📥 Download Report",
            data=report_content,
            file_name="covid_analysis_report.txt",
            mime="text/plain",
            use_container_width=True
        )
    
    with col2:
        # Export data as CSV
        csv_data = df.to_csv(index=False)
        st.download_button(
            label="📊 Download Data",
            data=csv_data,
            file_name="analysis_data.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with col3:
        # Export summary statistics
        if len(df.select_dtypes(include=[np.number]).columns) > 0:
            summary = df.describe().to_csv()
            st.download_button(
                label="📈 Download Summary",
                data=summary,
                file_name="summary_statistics.csv",
                mime="text/csv",
                use_container_width=True
            )

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p>COVID-Era Earnings Manipulation Analysis Tool | Built with Streamlit</p>
    <p>For educational and research purposes | Data from user uploads</p>
</div>
""", unsafe_allow_html=True)
