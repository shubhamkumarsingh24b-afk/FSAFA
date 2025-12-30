"""
COVID Earnings Data Generator
Generate sample data for testing
"""

import pandas as pd
import numpy as np
import argparse

def generate_sample_data(n_companies=200, output_file='covid_financial_data.csv'):
    """
    Generate synthetic financial data for COVID earnings analysis
    """
    np.random.seed(42)
    
    print(f"Generating data for {n_companies} companies...")
    
    industries = ['Aviation', 'Hospitality', 'Real Estate', 'IT', 'FMCG', 'Pharma']
    years = [2018, 2019, 2020, 2021]
    
    data = []
    for i in range(n_companies):
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
                revenue *= np.random.uniform(0.5, 0.8)
            
            # Profit with potential manipulation
            base_margin = np.random.uniform(0.05, 0.2)
            
            if year >= 2020 and treatment == 1:
                if np.random.random() < 0.4:  # 40% chance of manipulation
                    profit_margin = base_margin + np.random.uniform(0.05, 0.15)
                else:
                    profit_margin = base_margin
            else:
                profit_margin = base_margin
            
            net_profit = revenue * profit_margin
            
            data.append({
                'company_id': f'C{i:04d}',
                'company_name': f'{industry} Company {i}',
                'industry': industry,
                'year': year,
                'treatment_group': treatment,
                'post_covid': 1 if year >= 2020 else 0,
                'revenue': round(revenue, 2),
                'net_profit': round(net_profit, 2),
                'cfo': round(net_profit * np.random.uniform(0.8, 1.2), 2),
                'total_assets': round(revenue * np.random.uniform(1.5, 3.0), 2),
                'receivables': round(revenue * np.random.uniform(0.1, 0.3), 2),
                'debt': round(revenue * np.random.uniform(0.5, 1.5), 2)
            })
    
    df = pd.DataFrame(data)
    df.to_csv(output_file, index=False)
    
    print(f"✅ Generated {len(df)} rows")
    print(f"📁 Saved to: {output_file}")
    print(f"\n📊 Summary:")
    print(f"  Companies: {n_companies}")
    print(f"  Treatment group: {df[df['treatment_group']==1]['company_id'].nunique()}")
    print(f"  Control group: {df[df['treatment_group']==0]['company_id'].nunique()}")
    
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate COVID financial data')
    parser.add_argument('--companies', '-n', type=int, default=200, help='Number of companies')
    parser.add_argument('--output', '-o', default='covid_financial_data.csv', help='Output file')
    
    args = parser.parse_args()
    generate_sample_data(args.companies, args.output)
