"""
COVID-Era Earnings Manipulation Analysis
Streamlit Application for Difference-in-Differences Analysis
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
try:
    import statsmodels.formula.api as smf
    import scipy.stats as stats
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    st.warning("Some statistical features disabled")

# Page Configuration
st.set_page_config(
    page_title="COVID Earnings Manipulation Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.8rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid #3B82F6;
    }
    .sub-header {
        color: #374151;
        font-size: 1.4rem;
        margin-bottom: 2rem;
        text-align: center;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
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
    .upload-section {
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
    .warning-box {
        background-color: #fff3cd;
        color: #856404;
        padding: 1rem;
        border-radius: 5px;
        border-left: 5px solid #ffc107;
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
</style>
""", unsafe_allow_html=True)

# Title and Header
st.markdown('<h1 class="main-header">📊 COVID-Era Earnings Manipulation Analysis</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">A Difference-in-Differences (DiD) Analysis of Financial Reporting During the Pandemic</p>', unsafe_allow_html=True)

# Initialize session state for data persistence
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
    st.session_state.df = None
    st.session_state.analysis_complete = False
    st.session_state.did_results = None

# ============================================================================
# SIDEBAR - DATA UPLOAD & CONFIGURATION
# ============================================================================
with st.sidebar:
    st.markdown("## 📁 Data Configuration")
    
    # Option selection
    data_option = st.radio(
        "Choose data source:",
        ["Upload CSV File", "Generate Sample Data", "Use Demo Dataset"],
        index=0
    )
    
    if data_option == "Upload CSV File":
        st.markdown("### Upload Your CSV")
        uploaded_file = st.file_uploader(
            "Choose a CSV file with financial data",
            type=['csv'],
            help="File should contain columns: company_id, year, treatment_group, post_covid, revenue, net_profit, etc."
        )
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                
                # Validate required columns
                required_cols = ['company_id', 'year', 'treatment_group', 'post_covid']
                missing_cols = [col for col in required_cols if col not in df.columns]
                
                if missing_cols:
                    st.error(f"❌ Missing required columns: {missing_cols}")
                    st.stop()
                
                # Store in session state
                st.session_state.df = df
                st.session_state.data_loaded = True
                st.success(f"✅ Data loaded successfully!")
                
                # Show file info
                st.info(f"""
                **File Information:**
                - Filename: {uploaded_file.name}
                - Size: {uploaded_file.size / 1024:.1f} KB
                - Rows: {len(df):,}
                - Columns: {len(df.columns)}
                - Companies: {df['company_id'].nunique()}
                """)
                
            except Exception as e:
                st.error(f"Error reading file: {str(e)}")
    
    elif data_option == "Generate Sample Data":
        st.markdown("### Generate Sample Data")
        
        # Configuration for sample data
        n_companies = st.slider("Number of companies", 50, 500, 200)
        years = st.multiselect(
            "Select years",
            [2017, 2018, 2019, 2020, 2021, 2022],
            [2018, 2019, 2020, 2021]
        )
        
        if st.button("Generate Data", type="primary"):
            with st.spinner("Generating sample data..."):
                # Generate sample data
                np.random.seed(42)
                
                industries = {
                    'Aviation': {'treatment': 1, 'covid_impact': 0.4},
                    'Hospitality': {'treatment': 1, 'covid_impact': 0.5},
                    'Real Estate': {'treatment': 1, 'covid_impact': 0.3},
                    'IT': {'treatment': 0, 'covid_impact': 0.1},
                    'FMCG': {'treatment': 0, 'covid_impact': -0.05},
                    'Pharma': {'treatment': 0, 'covid_impact': -0.1}
                }
                
                data = []
                for company_id in range(n_companies):
                    industry = np.random.choice(list(industries.keys()))
                    industry_info = industries[industry]
                    
                    base_revenue = np.random.lognormal(mean=12, sigma=0.8)
                    growth_rate = np.random.normal(0.08, 0.03)
                    volatility = np.random.uniform(0.05, 0.15)
                    
                    for year_idx, year in enumerate(years):
                        # Time trend with company-specific growth
                        time_trend = (1 + growth_rate) ** year_idx
                        
                        # Calculate base revenue
                        revenue = base_revenue * time_trend * np.random.lognormal(0, volatility)
                        
                        # Apply COVID impact
                        if year >= 2020:
                            covid_effect = np.random.normal(1 - industry_info['covid_impact'], 0.1)
                            revenue *= max(covid_effect, 0.5)
                        
                        # Profit with potential manipulation
                        base_margin = np.random.uniform(0.05, 0.25)
                        
                        # Simulate earnings manipulation in treatment group post-COVID
                        if industry_info['treatment'] == 1 and year >= 2020:
                            manipulation = np.random.uniform(0, 0.2)  # Positive earnings manipulation
                            profit_margin = base_margin + manipulation
                        else:
                            profit_margin = base_margin + np.random.normal(0, 0.02)
                        
                        net_profit = revenue * profit_margin
                        
                        # Generate other financial metrics
                        cfo = net_profit * np.random.uniform(0.7, 1.3)
                        total_assets = revenue * np.random.uniform(1.5, 3.0)
                        receivables = revenue * np.random.uniform(0.15, 0.35)
                        debt = total_assets * np.random.uniform(0.2, 0.6)
                        
                        data.append({
                            'company_id': f'COMP{company_id:04d}',
                            'company_name': f'{industry} Company {company_id}',
                            'industry': industry,
                            'year': year,
                            'treatment_group': industry_info['treatment'],
                            'post_covid': 1 if year >= 2020 else 0,
                            'revenue': round(revenue, 2),
                            'net_profit': round(net_profit, 2),
                            'cfo': round(cfo, 2),
                            'total_assets': round(total_assets, 2),
                            'receivables': round(receivables, 2),
                            'debt': round(debt, 2),
                            'profit_margin': round(profit_margin * 100, 2)
                        })
                
                df = pd.DataFrame(data)
                st.session_state.df = df
                st.session_state.data_loaded = True
                st.success(f"✅ Generated {len(df)} rows of sample data!")
    
    else:  # Use Demo Dataset
        if st.button("Load Demo Dataset", type="primary"):
            # Create a comprehensive demo dataset
            np.random.seed(123)
            n_companies = 150
            years = [2018, 2019, 2020, 2021]
            
            industries = ['Aviation', 'Hospitality', 'Real Estate', 'IT', 'FMCG', 'Pharma']
            treatment_map = {ind: 1 if ind in ['Aviation', 'Hospitality', 'Real Estate'] else 0 for ind in industries}
            
            data = []
            for i in range(n_companies):
                industry = np.random.choice(industries)
                treatment = treatment_map[industry]
                
                base_rev = np.random.uniform(100, 10000)
                
                for year in years:
                    # Revenue calculation
                    growth = np.random.normal(0.1, 0.05)
                    rev = base_rev * (1 + growth) ** (year - 2018)
                    
                    # COVID impact
                    if year >= 2020 and treatment == 1:
                        rev *= np.random.uniform(0.5, 0.9)
                    
                    # Net profit with potential manipulation
                    margin = np.random.uniform(0.05, 0.25)
                    if year >= 2020 and treatment == 1:
                        # Add earnings manipulation
                        margin += np.random.uniform(0.05, 0.15)
                    
                    profit = rev * margin
                    
                    # Other metrics
                    cfo = profit * np.random.uniform(0.8, 1.2)
                    assets = rev * np.random.uniform(1.5, 3)
                    
                    data.append({
                        'company_id': f'DEMO{i:03d}',
                        'company_name': f'{industry} Demo {i}',
                        'industry': industry,
                        'year': year,
                        'treatment_group': treatment,
                        'post_covid': 1 if year >= 2020 else 0,
                        'revenue': round(rev, 2),
                        'net_profit': round(profit, 2),
                        'cfo': round(cfo, 2),
                        'total_assets': round(assets, 2),
                        'receivables': round(rev * np.random.uniform(0.1, 0.3), 2),
                        'debt': round(assets * np.random.uniform(0.1, 0.5), 2)
                    })
            
            df = pd.DataFrame(data)
            st.session_state.df = df
            st.session_state.data_loaded = True
            st.success("✅ Demo dataset loaded!")
    
    # Analysis Configuration
    st.markdown("---")
    st.markdown("## ⚙️ Analysis Settings")
    
    if st.session_state.data_loaded:
        outcome_options = ['net_profit', 'cfo', 'revenue', 'profit_margin', 'receivables', 'debt']
        available_cols = [col for col in outcome_options if col in st.session_state.df.columns]
        
        outcome_var = st.selectbox(
            "Select outcome variable for DiD analysis:",
            available_cols,
            index=0
        )
        
        st.session_state.outcome_var = outcome_var
        
        # Time period selection
        years = sorted(st.session_state.df['year'].unique())
        pre_period = st.selectbox(
            "Pre-COVID period (end year):",
            years,
            index=min(1, len(years)-1)
        )
        
        post_period = st.selectbox(
            "Post-COVID period (start year):",
            [y for y in years if y > pre_period],
            index=0
        )
        
        st.session_state.pre_period = pre_period
        st.session_state.post_period = post_period
        
        # Additional analysis options
        include_controls = st.checkbox("Include control variables", value=False)
        robust_se = st.checkbox("Use robust standard errors", value=True)
        
        if st.button("🚀 Run Analysis", type="primary"):
            st.session_state.analysis_complete = True
            st.rerun()

# ============================================================================
# MAIN CONTENT AREA
# ============================================================================
if not st.session_state.data_loaded:
    # Welcome screen with upload instructions
    st.markdown("""
    <div style="text-align: center; padding: 3rem;">
        <h2>📤 Welcome to COVID Earnings Analysis</h2>
        <p style="font-size: 1.2rem; margin: 2rem 0;">
            Upload your financial data or generate sample data to begin analysis
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown('<div class="upload-section">', unsafe_allow_html=True)
        st.markdown("### 📎 Ready to Upload?")
        st.markdown("""
        1. **Prepare your CSV file** with financial data
        2. **Required columns:** company_id, year, treatment_group, post_covid
        3. **Optional columns:** revenue, net_profit, cfo, total_assets, etc.
        4. **Use the sidebar** to upload your file
        """)
        
        # Download template
        template_data = {
            'company_id': ['CMP001', 'CMP001', 'CMP002', 'CMP002'],
            'year': [2019, 2020, 2019, 2020],
            'treatment_group': [1, 1, 0, 0],
            'post_covid': [0, 1, 0, 1],
            'revenue': [1000.0, 850.0, 2000.0, 2100.0],
            'net_profit': [100.0, 120.0, 200.0, 210.0],
            'cfo': [90.0, 85.0, 180.0, 190.0],
            'industry': ['Aviation', 'Aviation', 'IT', 'IT']
        }
        template_df = pd.DataFrame(template_data)
        
        csv = template_df.to_csv(index=False)
        st.download_button(
            label="📥 Download CSV Template",
            data=csv,
            file_name="financial_data_template.csv",
            mime="text/csv",
            key="template_download"
        )
        st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Project overview
    st.markdown("""
    ## 📋 Project Overview
    
    This application performs **Difference-in-Differences (DiD) analysis** to detect potential earnings manipulation during the COVID-19 pandemic.
    
    ### Key Features:
    
    1. **Data Upload & Validation** - Upload CSV files with financial data
    2. **Difference-in-Differences Analysis** - Compare treatment vs control groups
    3. **Visualization Dashboard** - Interactive charts and graphs
    4. **Statistical Testing** - Hypothesis tests and regression analysis
    5. **Report Generation** - Exportable analysis reports
    
    ### Methodology:
    
    - **Treatment Group**: Industries heavily affected by COVID-19 (Aviation, Hospitality, Real Estate)
    - **Control Group**: Less affected industries (IT, FMCG, Pharma)
    - **Analysis Period**: Pre-COVID (2018-2019) vs Post-COVID (2020-2021)
    - **Outcome Variables**: Net profit, Cash flow, Receivables, etc.
    """)
    
    st.stop()

# ============================================================================
# DATA LOADED - SHOW ANALYSIS TABS
# ============================================================================
df = st.session_state.df

# Create tabs for different sections
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Data Overview", 
    "📈 DiD Analysis", 
    "📉 Visualizations", 
    "🧪 Statistical Tests",
    "📋 Report"
])

with tab1:
    st.header("Data Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Companies",
            df['company_id'].nunique(),
            help="Unique companies in dataset"
        )
    
    with col2:
        treatment_pct = (df[df['treatment_group'] == 1]['company_id'].nunique() / 
                        df['company_id'].nunique() * 100)
        st.metric(
            "Treatment Group",
            f"{treatment_pct:.1f}%",
            help="Percentage of companies in treatment group"
        )
    
    with col3:
        years = df['year'].nunique()
        st.metric(
            "Years Covered",
            years,
            help="Number of years in dataset"
        )
    
    with col4:
        st.metric(
            "Total Observations",
            f"{len(df):,}",
            help="Total rows in dataset"
        )
    
    # Data preview
    st.subheader("Data Preview")
    
    show_all = st.checkbox("Show all columns", value=False)
    if show_all:
        st.dataframe(df, use_container_width=True, height=400)
    else:
        # Show only key columns
        key_cols = ['company_id', 'company_name', 'industry', 'year', 
                   'treatment_group', 'post_covid', 'revenue', 'net_profit']
        available_cols = [col for col in key_cols if col in df.columns]
        st.dataframe(df[available_cols], use_container_width=True, height=400)
    
    # Data summary statistics
    st.subheader("Summary Statistics")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        summary_stats = df[numeric_cols].describe().T
        summary_stats['missing'] = df[numeric_cols].isnull().sum()
        summary_stats['missing_pct'] = (summary_stats['missing'] / len(df) * 100).round(2)
        
        st.dataframe(summary_stats, use_container_width=True)
    
    # Industry distribution
    if 'industry' in df.columns:
        st.subheader("Industry Distribution")
        
        industry_counts = df[['company_id', 'industry']].drop_duplicates()['industry'].value_counts()
        
        fig = px.pie(
            values=industry_counts.values,
            names=industry_counts.index,
            title="Company Distribution by Industry",
            hole=0.3
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.header("Difference-in-Differences Analysis")
    
    if not st.session_state.analysis_complete:
        st.info("👈 Configure analysis settings in sidebar and click 'Run Analysis'")
        st.stop()
    
    outcome_var = st.session_state.outcome_var
    
    # Prepare data for DiD analysis
    analysis_df = df.copy()
    
    # Define pre and post periods
    analysis_df['period'] = analysis_df['year'].apply(
        lambda x: 'pre' if x <= st.session_state.pre_period else 'post'
    )
    analysis_df['post_dummy'] = (analysis_df['period'] == 'post').astype(int)
    
    # 1. Manual DiD Calculation
    st.subheader("1. Manual DiD Calculation")
    
    # Calculate group means
    did_table = analysis_df.groupby(['treatment_group', 'period'])[outcome_var].mean().unstack()
    
    # Calculate differences
    did_table['Change'] = did_table['post'] - did_table['pre']
    
    # Calculate DiD effect
    treatment_effect = did_table.loc[1, 'Change']
    control_effect = did_table.loc[0, 'Change']
    did_effect = treatment_effect - control_effect
    
    # Format table for display
    display_table = pd.DataFrame({
        'Group': ['Control (0)', 'Treatment (1)'],
        'Pre-Period Mean': [did_table.loc[0, 'pre'], did_table.loc[1, 'pre']],
        'Post-Period Mean': [did_table.loc[0, 'post'], did_table.loc[1, 'post']],
        'Change': [control_effect, treatment_effect]
    })
    
    # Display table
    st.dataframe(
        display_table.style.format({
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
    <p><strong>Treatment Effect (Group 1 Change):</strong> {treatment_effect:.2f}</p>
    <p><strong>Control Effect (Group 0 Change):</strong> {control_effect:.2f}</p>
    <p><strong>DiD Coefficient (Treatment - Control):</strong> <span style="color: #28a745; font-weight: bold;">{did_effect:.2f}</span></p>
    </div>
    """, unsafe_allow_html=True)
    
    # Interpretation
    if did_effect > 0:
        interpretation = f"""
        <div class="info-box">
        <h4>📊 Interpretation</h4>
        <p>The treatment group shows a <strong>{did_effect:.2f}</strong> higher change in {outcome_var} compared to the control group.</p>
        <p>This suggests <strong>potential earnings manipulation</strong> in COVID-affected industries post-pandemic.</p>
        </div>
        """
    else:
        interpretation = f"""
        <div class="info-box">
        <h4>📊 Interpretation</h4>
        <p>The treatment group shows a <strong>{did_effect:.2f}</strong> change in {outcome_var} compared to the control group.</p>
        <p>This suggests <strong>no significant evidence of earnings manipulation</strong> in the treatment group.</p>
        </div>
        """
    
    st.markdown(interpretation, unsafe_allow_html=True)
    
    # 2. Regression DiD Analysis
    st.subheader("2. Regression DiD Analysis")
    
    # Create interaction term
    analysis_df['treatment_post'] = analysis_df['treatment_group'] * analysis_df['post_dummy']
    
    # Run regression
    formula = f"{outcome_var} ~ treatment_group + post_dummy + treatment_post"
    
    try:
        model = smf.ols(formula, data=analysis_df).fit()
        
        # Display regression results
        st.write("**Regression Results:**")
        
        # Create summary table
        coef_df = pd.DataFrame({
            'Variable': model.params.index,
            'Coefficient': model.params.values,
            'Std Error': model.bse.values,
            't-value': model.tvalues.values,
            'P-value': model.pvalues.values,
            'Significance': ['***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else '' for p in model.pvalues.values]
        })
        
        # Format the table
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
        
        # Key findings
        did_coef = model.params.get('treatment_post', 0)
        did_pval = model.pvalues.get('treatment_post', 1)
        
        if did_pval < 0.05:
            significance = "statistically significant"
            sig_color = "#28a745"
        else:
            significance = "not statistically significant"
            sig_color = "#dc3545"
        
        st.markdown(f"""
        <div style="padding: 1.5rem; background-color: #f8f9fa; border-radius: 10px; border-left: 5px solid {sig_color};">
        <h4>🔑 Key Finding</h4>
        <p>The DiD coefficient (treatment × post interaction) is <strong>{did_coef:.4f}</strong></p>
        <p>P-value: <strong>{did_pval:.4f}</strong></p>
        <p>This effect is <strong style="color: {sig_color}">{significance}</strong> at the 5% level.</p>
        </div>
        """, unsafe_allow_html=True)
        
    except Exception as e:
        st.error(f"Regression error: {str(e)}")

with tab3:
    st.header("Visualizations")
    
    if not st.session_state.data_loaded:
        st.info("Load data to see visualizations")
        st.stop()
    
    # Visualization type selector
    viz_type = st.selectbox(
        "Select visualization type:",
        ["Time Trends", "Group Comparison", "Parallel Trends", "Distribution", "Scatter Plot"]
    )
    
    if viz_type == "Time Trends":
        st.subheader(f"{st.session_state.outcome_var} Trends Over Time")
        
        # Calculate group averages by year
        trend_data = df.groupby(['year', 'treatment_group'])[st.session_state.outcome_var].mean().reset_index()
        trend_data['Group'] = trend_data['treatment_group'].apply(lambda x: 'Treatment' if x == 1 else 'Control')
        
        fig = px.line(
            trend_data,
            x='year',
            y=st.session_state.outcome_var,
            color='Group',
            markers=True,
            title=f"Average {st.session_state.outcome_var} Over Time by Group",
            line_shape='linear'
        )
        
        # Add COVID period shading
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
            yaxis_title=st.session_state.outcome_var.replace('_', ' ').title(),
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
            y=st.session_state.outcome_var,
            color='Group',
            title=f"Distribution of {st.session_state.outcome_var.replace('_', ' ').title()} by Period and Group",
            points=False
        )
        
        fig.update_layout(
            boxmode='group',
            xaxis_title="",
            yaxis_title=st.session_state.outcome_var.replace('_', ' ').title()
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    elif viz_type == "Parallel Trends":
        st.subheader("Parallel Trends Assumption Check")
        
        # Check pre-COVID trends
        pre_covid_data = df[df['year'] < 2020].copy()
        
        if len(pre_covid_data) > 0:
            pre_trends = pre_covid_data.groupby(['year', 'treatment_group'])[st.session_state.outcome_var].mean().unstack()
            
            fig = go.Figure()
            
            colors = ['#1f77b4', '#ff7f0e']
            for i, group in enumerate([0, 1]):
                group_name = 'Control' if group == 0 else 'Treatment'
                fig.add_trace(go.Scatter(
                    x=pre_trends.index,
                    y=pre_trends[group],
                    mode='lines+markers',
                    name=group_name,
                    line=dict(color=colors[i], width=3),
                    marker=dict(size=8)
                ))
            
            fig.update_layout(
                title="Pre-COVID Trends (Parallel Trends Assumption)",
                xaxis_title="Year",
                yaxis_title=st.session_state.outcome_var.replace('_', ' ').title(),
                showlegend=True,
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Statistical test for parallel trends
            st.markdown("""
            <div class="info-box">
            <h4>📝 Parallel Trends Assumption</h4>
            <p>For DiD to be valid, treatment and control groups should have similar trends before the treatment.</p>
            <p>If lines are approximately parallel before COVID (2018-2019), the assumption holds.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.warning("Insufficient pre-COVID data for parallel trends check")
            
    elif viz_type == "Distribution":
        st.subheader("Distribution Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Histogram
            fig1 = px.histogram(
                df,
                x=st.session_state.outcome_var,
                color='treatment_group',
                nbins=30,
                title=f"Distribution of {st.session_state.outcome_var.replace('_', ' ').title()}",
                labels={'treatment_group': 'Treatment Group'},
                barmode='overlay',
                opacity=0.7
            )
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            # Violin plot
            fig2 = px.violin(
                df,
                y=st.session_state.outcome_var,
                x='treatment_group',
                color='treatment_group',
                box=True,
                points="all",
                title=f"Violin Plot by Treatment Group",
                labels={'treatment_group': 'Group (0=Control, 1=Treatment)'}
            )
            st.plotly_chart(fig2, use_container_width=True)
    
    else:  # Scatter Plot
        st.subheader("Scatter Plot Analysis")
        
        # Select variables for scatter plot
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        x_var = st.selectbox(
            "Select X-axis variable:",
            numeric_cols,
            index=numeric_cols.index('revenue') if 'revenue' in numeric_cols else 0
        )
        
        fig = px.scatter(
            df,
            x=x_var,
            y=st.session_state.outcome_var,
            color='treatment_group',
            size='total_assets' if 'total_assets' in df.columns else None,
            hover_data=['company_name', 'industry', 'year'],
            title=f"{st.session_state.outcome_var.replace('_', ' ').title()} vs {x_var.replace('_', ' ').title()}",
            labels={'treatment_group': 'Treatment Group'}
        )
        
        # Add trend lines
        fig.update_traces(marker=dict(opacity=0.6))
        
        st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.header("Statistical Tests")
    
    if not st.session_state.data_loaded:
        st.info("Load data to perform statistical tests")
        st.stop()
    
    st.subheader("1. T-tests for Group Differences")
    
    # Prepare data
    pre_data = df[df['year'] < 2020]
    post_data = df[df['year'] >= 2020]
    
    # Perform t-tests
    results = []
    
    # Test 1: Treatment vs Control in Pre-period
    if len(pre_data) > 0:
        treatment_pre = pre_data[pre_data['treatment_group'] == 1][st.session_state.outcome_var]
        control_pre = pre_data[pre_data['treatment_group'] == 0][st.session_state.outcome_var]
        
        if len(treatment_pre) > 1 and len(control_pre) > 1:
            t_stat, p_val = stats.ttest_ind(treatment_pre, control_pre, equal_var=False)
            results.append({
                'Test': 'Pre-COVID: Treatment vs Control',
                'Treatment Mean': treatment_pre.mean(),
                'Control Mean': control_pre.mean(),
                'Difference': treatment_pre.mean() - control_pre.mean(),
                't-statistic': t_stat,
                'p-value': p_val,
                'Significant': p_val < 0.05
            })
    
    # Test 2: Treatment vs Control in Post-period
    if len(post_data) > 0:
        treatment_post = post_data[post_data['treatment_group'] == 1][st.session_state.outcome_var]
        control_post = post_data[post_data['treatment_group'] == 0][st.session_state.outcome_var]
        
        if len(treatment_post) > 1 and len(control_post) > 1:
            t_stat, p_val = stats.ttest_ind(treatment_post, control_post, equal_var=False)
            results.append({
                'Test': 'Post-COVID: Treatment vs Control',
                'Treatment Mean': treatment_post.mean(),
                'Control Mean': control_post.mean(),
                'Difference': treatment_post.mean() - control_post.mean(),
                't-statistic': t_stat,
                'p-value': p_val,
                'Significant': p_val < 0.05
            })
    
    # Test 3: Treatment Group Pre vs Post
    if len(pre_data) > 0 and len(post_data) > 0:
        treatment_pre = pre_data[pre_data['treatment_group'] == 1][st.session_state.outcome_var]
        treatment_post = post_data[post_data['treatment_group'] == 1][st.session_state.outcome_var]
        
        if len(treatment_pre) > 1 and len(treatment_post) > 1:
            t_stat, p_val = stats.ttest_ind(treatment_pre, treatment_post, equal_var=False)
            results.append({
                'Test': 'Treatment Group: Pre vs Post',
                'Pre Mean': treatment_pre.mean(),
                'Post Mean': treatment_post.mean(),
                'Difference': treatment_post.mean() - treatment_pre.mean(),
                't-statistic': t_stat,
                'p-value': p_val,
                'Significant': p_val < 0.05
            })
    
    # Display results
    if results:
        results_df = pd.DataFrame(results)
        st.dataframe(
            results_df.style.format({
                'Treatment Mean': '{:.2f}',
                'Control Mean': '{:.2f}',
                'Pre Mean': '{:.2f}',
                'Post Mean': '{:.2f}',
                'Difference': '{:.2f}',
                't-statistic': '{:.3f}',
                'p-value': '{:.4f}'
            }).apply(
                lambda x: ['background-color: #e8f5e9' if x['Significant'] else '' for _ in x],
                axis=1
            ),
            use_container_width=True
        )
    
    # Statistical assumptions check
    st.subheader("2. Statistical Assumptions Check")
    
    # Normality test
    if st.checkbox("Check normality assumption"):
        col1, col2 = st.columns(2)
        
        with col1:
            # Treatment group normality
            treatment_data = df[df['treatment_group'] == 1][st.session_state.outcome_var]
            if len(treatment_data) > 3:
                stat, p = stats.shapiro(treatment_data)
                st.metric(
                    "Treatment Group Normality",
                    f"p = {p:.4f}",
                    "Normal" if p > 0.05 else "Not Normal"
                )
        
        with col2:
            # Control group normality
            control_data = df[df['treatment_group'] == 0][st.session_state.outcome_var]
            if len(control_data) > 3:
                stat, p = stats.shapiro(control_data)
                st.metric(
                    "Control Group Normality",
                    f"p = {p:.4f}",
                    "Normal" if p > 0.05 else "Not Normal"
                )
    
    # Homogeneity of variance test
    if st.checkbox("Check homogeneity of variance"):
        if 'treatment_pre' in locals() and 'control_pre' in locals():
            stat, p = stats.levene(treatment_pre, control_pre)
            st.metric(
                "Homogeneity of Variance (Levene's Test)",
                f"p = {p:.4f}",
                "Equal Variances" if p > 0.05 else "Unequal Variances"
            )

with tab5:
    st.header("Analysis Report")
    
    if not st.session_state.data_loaded:
        st.info("Load data to generate report")
        st.stop()
    
    # Generate report content
    report_date = datetime.now().strftime("%Y-%m-%d %H:%M")
    
    # Calculate key metrics for report
    total_companies = df['company_id'].nunique()
    treatment_companies = df[df['treatment_group'] == 1]['company_id'].nunique()
    control_companies = df[df['treatment_group'] == 0]['company_id'].nunique()
    years_covered = sorted(df['year'].unique())
    
    # Get DiD results if available
    did_result = None
    if st.session_state.analysis_complete:
        # Recalculate or get from session state
        analysis_df = df.copy()
        analysis_df['period'] = analysis_df['year'].apply(
            lambda x: 'pre' if x <= st.session_state.pre_period else 'post'
        )
        analysis_df['post_dummy'] = (analysis_df['period'] == 'post').astype(int)
        
        did_table = analysis_df.groupby(['treatment_group', 'period'])[st.session_state.outcome_var].mean().unstack()
        treatment_effect = did_table.loc[1, 'post'] - did_table.loc[1, 'pre']
        control_effect = did_table.loc[0, 'post'] - did_table.loc[0, 'pre']
        did_result = treatment_effect - control_effect
    
    # Create report
    report_content = f"""
# COVID-Era Earnings Manipulation Analysis Report

**Generated:** {report_date}  
**Dataset:** {total_companies} companies, {len(df)} observations  
**Analysis Variable:** {st.session_state.outcome_var if st.session_state.analysis_complete else 'Not specified'}

## Executive Summary

This report presents a Difference-in-Differences (DiD) analysis of potential earnings manipulation during the COVID-19 pandemic.

### Key Findings:
1. **Dataset Composition**: Analyzed {total_companies} companies ({treatment_companies} treatment, {control_companies} control)
2. **Time Period**: {min(years_covered)} to {max(years_covered)}
3. **DiD Result**: {'Not calculated' if did_result is None else f'{did_result:.2f}'}
4. **Interpretation**: {'Analysis not completed' if did_result is None else 'Potential earnings manipulation detected' if did_result > 0 else 'No significant evidence found'}

## Methodology

### Research Design
- **Treatment Group**: COVID-affected industries (Aviation, Hospitality, Real Estate)
- **Control Group**: Less affected industries (IT, FMCG, Pharma)
- **Pre-Period**: Years before 2020
- **Post-Period**: Years 2020 and later
- **Statistical Method**: Difference-in-Differences (DiD) with OLS regression

### Data Description
- **Total Observations**: {len(df):,}
- **Companies by Group**: Treatment={treatment_companies}, Control={control_companies}
- **Years Covered**: {', '.join(map(str, years_covered))}
- **Industries**: {', '.join(df['industry'].unique()) if 'industry' in df.columns else 'Not specified'}

## Detailed Results

### 1. Descriptive Statistics
{df[st.session_state.outcome_var].describe().to_string() if st.session_state.analysis_complete else 'Not available'}

### 2. DiD Analysis Results
**Treatment Effect**: {treatment_effect:.2f if 'treatment_effect' in locals() else 'N/A'}  
**Control Effect**: {control_effect:.2f if 'control_effect' in locals() else 'N/A'}  
**DiD Coefficient**: {did_result:.2f if did_result is not None else 'N/A'}

### 3. Statistical Significance
{'Significant at 5% level' if 'did_pval' in locals() and did_pval < 0.05 else 'Not significant' if 'did_pval' in locals() else 'Not tested'}

## Limitations

1. **Data Quality**: Dependent on uploaded data accuracy
2. **Model Assumptions**: Parallel trends assumption may not hold
3. **Sample Size**: Limited to uploaded dataset
4. **Time Period**: Restricted to available years

## Recommendations

1. **Further Analysis**: Extend to more years and companies
2. **Additional Variables**: Include more financial metrics
3. **Robustness Checks**: Test alternative specifications
4. **Industry Analysis**: Conduct sector-specific studies

---
*Report generated by COVID Earnings Manipulation Analysis Tool*
"""
    
    # Display report
    st.markdown(report_content)
    
    # Export options
    st.subheader("Export Report")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Export as text
        st.download_button(
            label="📥 Download as Text",
            data=report_content,
            file_name="covid_earnings_report.txt",
            mime="text/plain"
        )
    
    with col2:
        # Export data as CSV
        csv = df.to_csv(index=False)
        st.download_button(
            label="📊 Download Data CSV",
            data=csv,
            file_name="analysis_data.csv",
            mime="text/csv"
        )
    
    with col3:
        # Export summary statistics
        if st.session_state.analysis_complete:
            summary_stats = df.describe().T
            summary_csv = summary_stats.to_csv()
            st.download_button(
                label="📈 Download Summary Stats",
                data=summary_csv,
                file_name="summary_statistics.csv",
                mime="text/csv"
            )

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p>COVID-Era Earnings Manipulation Analysis Tool | Built with Streamlit</p>
    <p>For educational and research purposes only | Data from user uploads</p>
    <p style="font-size: 0.8rem;">© 2024 Financial Analytics Project</p>
</div>
""", unsafe_allow_html=True)
