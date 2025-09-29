import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from scipy import stats
import plotly.figure_factory as ff

# Handle optional statsmodels import
try:
    import statsmodels.api as sm
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    st.warning("⚠️ Some advanced statistical features require statsmodels. Install with: pip install statsmodels")

st.set_page_config(page_title="Statistical Analysis", page_icon="📈", layout="wide")

@st.cache_data
def load_data():
    return pd.read_csv('vendor_sales_summary.csv')

def perform_statistical_tests(df):
    """Perform various statistical tests on the data"""
    results = {}
    
    # Clean data for analysis
    clean_df = df[df['ProfitMargin'] != np.inf].copy()
    
    # 1. Correlation analysis
    numeric_cols = ['PurchasePrice', 'ActualPrice', 'TotalPurchaseQuantity', 
                   'TotalSalesDollars', 'GrossProfit', 'ProfitMargin', 'StockTurnover']
    correlation_matrix = clean_df[numeric_cols].corr()
    results['correlation'] = correlation_matrix
    
    # 2. Vendor performance comparison
    vendor_stats = clean_df.groupby('VendorName').agg({
        'TotalSalesDollars': 'sum',
        'ProfitMargin': 'mean',
        'StockTurnover': 'mean'
    }).reset_index()
    
    # Classify vendors into performance groups
    vendor_stats['Performance_Class'] = pd.cut(
        vendor_stats['TotalSalesDollars'],
        bins=3,
        labels=['Low Performers', 'Medium Performers', 'High Performers']
    )
    
    # T-test between high and low performers
    high_performers = vendor_stats[vendor_stats['Performance_Class'] == 'High Performers']['ProfitMargin']
    low_performers = vendor_stats[vendor_stats['Performance_Class'] == 'Low Performers']['ProfitMargin']
    
    if len(high_performers) > 1 and len(low_performers) > 1:
        t_stat, p_value = stats.ttest_ind(high_performers, low_performers)
        results['ttest'] = {'t_stat': t_stat, 'p_value': p_value, 
                           'high_mean': high_performers.mean(), 'low_mean': low_performers.mean()}
    
    # 3. Normality tests
    results['normality'] = {}
    for col in ['ProfitMargin', 'StockTurnover', 'TotalSalesDollars']:
        if col in clean_df.columns:
            data = clean_df[col].dropna()
            if len(data) > 8:  # Minimum sample size for Shapiro-Wilk
                shapiro_stat, shapiro_p = stats.shapiro(data[:5000])  # Limit sample size
                results['normality'][col] = {'shapiro_stat': shapiro_stat, 'shapiro_p': shapiro_p}
    
    # 4. ANOVA for vendor groups
    vendor_groups = [group['ProfitMargin'].values for name, group in clean_df.groupby('VendorName') 
                    if len(group) >= 3]  # Only vendors with at least 3 products
    
    if len(vendor_groups) >= 3:
        f_stat, anova_p = stats.f_oneway(*vendor_groups[:10])  # Limit to top 10 vendors
        results['anova'] = {'f_stat': f_stat, 'p_value': anova_p}
    
    return results, clean_df, vendor_stats

def main():
    st.title("📈 Statistical Analysis Dashboard")
    
    df = load_data()
    
    # Perform statistical analysis
    with st.spinner("Performing statistical analysis..."):
        stats_results, clean_df, vendor_stats = perform_statistical_tests(df)
    
    # Overview
    st.header("Statistical Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Sample Size", f"{len(clean_df):,}")
    with col2:
        st.metric("Variables Analyzed", "7")
    with col3:
        st.metric("Vendors Analyzed", f"{clean_df['VendorName'].nunique():,}")
    with col4:
        confidence_level = 95
        st.metric("Confidence Level", f"{confidence_level}%")
    
    # Correlation Analysis
    st.header("🔗 Correlation Analysis")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if 'correlation' in stats_results:
            fig = px.imshow(
                stats_results['correlation'],
                text_auto=True,
                aspect="auto",
                title="Correlation Matrix of Key Business Metrics",
                color_continuous_scale='RdBu_r',
                zmin=-1, zmax=1
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Key Correlations")
        corr_matrix = stats_results['correlation']
        
        # Find strongest correlations
        correlations = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                correlations.append({
                    'var1': corr_matrix.columns[i],
                    'var2': corr_matrix.columns[j],
                    'correlation': corr_val
                })
        
        correlations_df = pd.DataFrame(correlations)
        correlations_df['abs_corr'] = abs(correlations_df['correlation'])
        top_correlations = correlations_df.nlargest(5, 'abs_corr')
        
        for _, row in top_correlations.iterrows():
            strength = "Strong" if abs(row['correlation']) > 0.7 else "Moderate" if abs(row['correlation']) > 0.3 else "Weak"
            direction = "Positive" if row['correlation'] > 0 else "Negative"
            st.write(f"**{row['var1'][:15]}... vs {row['var2'][:15]}...**")
            st.write(f"{strength} {direction}: {row['correlation']:.3f}")
            st.write("---")
    
    # Distribution Analysis
    st.header("📊 Distribution Analysis")
    
    tab1, tab2, tab3 = st.tabs(["Profit Margin", "Stock Turnover", "Revenue"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            # Histogram with normal curve overlay
            fig = px.histogram(
                clean_df, 
                x='ProfitMargin', 
                nbins=50,
                title="Profit Margin Distribution",
                marginal="box"
            )
            
            # Add normal distribution overlay
            mean_margin = clean_df['ProfitMargin'].mean()
            std_margin = clean_df['ProfitMargin'].std()
            x_range = np.linspace(clean_df['ProfitMargin'].min(), clean_df['ProfitMargin'].max(), 100)
            normal_curve = stats.norm.pdf(x_range, mean_margin, std_margin)
            
            # Scale normal curve to match histogram
            hist_max = len(clean_df) * (clean_df['ProfitMargin'].max() - clean_df['ProfitMargin'].min()) / 50
            normal_curve_scaled = normal_curve * hist_max / max(normal_curve)
            
            fig.add_trace(go.Scatter(x=x_range, y=normal_curve_scaled, 
                                   mode='lines', name='Normal Distribution',
                                   line=dict(color='red', dash='dash')))
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Q-Q plot
            fig = go.Figure()
            
            # Generate Q-Q plot data
            sorted_data = np.sort(clean_df['ProfitMargin'])
            n = len(sorted_data)
            theoretical_quantiles = stats.norm.ppf(np.arange(1, n+1) / (n+1))
            
            fig.add_trace(go.Scatter(
                x=theoretical_quantiles,
                y=sorted_data,
                mode='markers',
                name='Data Points',
                marker=dict(size=4)
            ))
            
            # Add reference line
            fig.add_trace(go.Scatter(
                x=[theoretical_quantiles.min(), theoretical_quantiles.max()],
                y=[sorted_data.min(), sorted_data.max()],
                mode='lines',
                name='Perfect Normal',
                line=dict(color='red', dash='dash')
            ))
            
            fig.update_layout(
                title="Q-Q Plot: Profit Margin vs Normal Distribution",
                xaxis_title="Theoretical Quantiles",
                yaxis_title="Sample Quantiles"
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Normality test results
        if 'ProfitMargin' in stats_results['normality']:
            shapiro_p = stats_results['normality']['ProfitMargin']['shapiro_p']
            st.write(f"**Shapiro-Wilk Normality Test p-value: {shapiro_p:.6f}**")
            if shapiro_p < 0.05:
                st.warning("⚠️ Data significantly deviates from normal distribution")
            else:
                st.success("✅ Data appears to be normally distributed")
    
    with tab2:
        # Stock turnover analysis
        turnover_data = clean_df[clean_df['StockTurnover'] <= 10]  # Filter extreme outliers
        
        fig = px.histogram(
            turnover_data,
            x='StockTurnover',
            nbins=30,
            title="Stock Turnover Distribution (Filtered)",
            marginal="violin"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Summary statistics
        st.subheader("Stock Turnover Statistics")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Mean", f"{turnover_data['StockTurnover'].mean():.2f}")
        with col2:
            st.metric("Median", f"{turnover_data['StockTurnover'].median():.2f}")
        with col3:
            st.metric("Std Dev", f"{turnover_data['StockTurnover'].std():.2f}")
        with col4:
            st.metric("Skewness", f"{stats.skew(turnover_data['StockTurnover']):.2f}")
    
    with tab3:
        # Revenue distribution (log scale)
        fig = px.histogram(
            clean_df,
            x='TotalSalesDollars',
            nbins=50,
            title="Revenue Distribution",
            marginal="box"
        )
        fig.update_xaxes(type="log", title="Revenue (Log Scale)")
        st.plotly_chart(fig, use_container_width=True)
    
    # Hypothesis Testing
    st.header("🧪 Hypothesis Testing")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Vendor Performance Comparison")
        
        if 'ttest' in stats_results:
            ttest_results = stats_results['ttest']
            
            st.write("**Research Question:** Do high-performing vendors have significantly different profit margins than low-performing vendors?")
            st.write("**H₀:** No difference in profit margins between vendor groups")
            st.write("**H₁:** Significant difference exists between vendor groups")
            
            st.write("**Results:**")
            st.write(f"• High Performers Mean Margin: {ttest_results['high_mean']:.2f}%")
            st.write(f"• Low Performers Mean Margin: {ttest_results['low_mean']:.2f}%")
            st.write(f"• T-statistic: {ttest_results['t_stat']:.4f}")
            st.write(f"• P-value: {ttest_results['p_value']:.6f}")
            
            if ttest_results['p_value'] < 0.05:
                st.success("✅ **Reject H₀:** Significant difference found!")
                st.write("High and low performing vendors have statistically different profit margins.")
            else:
                st.info("ℹ️ **Fail to reject H₀:** No significant difference found.")
        
        # Box plot for visual comparison
        fig = px.box(
            vendor_stats,
            x='Performance_Class',
            y='ProfitMargin',
            title="Profit Margin by Vendor Performance Class",
            points="all"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("ANOVA: Vendor Group Analysis")
        
        if 'anova' in stats_results:
            anova_results = stats_results['anova']
            
            st.write("**Research Question:** Do different vendors have significantly different profit margins?")
            st.write("**H₀:** All vendor groups have equal mean profit margins")
            st.write("**H₁:** At least one vendor group has different mean profit margin")
            
            st.write("**Results:**")
            st.write(f"• F-statistic: {anova_results['f_stat']:.4f}")
            st.write(f"• P-value: {anova_results['p_value']:.6f}")
            
            if anova_results['p_value'] < 0.05:
                st.success("✅ **Reject H₀:** Significant differences found between vendors!")
            else:
                st.info("ℹ️ **Fail to reject H₀:** No significant differences between vendors.")
        
        # Vendor comparison violin plot
        top_vendors = clean_df['VendorName'].value_counts().head(8).index
        vendor_subset = clean_df[clean_df['VendorName'].isin(top_vendors)]
        
        fig = px.violin(
            vendor_subset,
            x='VendorName',
            y='ProfitMargin',
            title="Profit Margin Distribution by Top Vendors",
            box=True
        )
        fig.update_xaxes(tickangle=45)
        st.plotly_chart(fig, use_container_width=True)
    
    # Advanced Analytics
    st.header("🔬 Advanced Analytics")
    
    tab1, tab2, tab3 = st.tabs(["Regression Analysis", "Outlier Detection", "Confidence Intervals"])
    
    with tab1:
        st.subheader("Linear Regression: Price vs Revenue")
        
        # Simple linear regression
        x = clean_df['ActualPrice']
        y = clean_df['TotalSalesDollars']
        
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        
        # Create scatter plot with manual trendline
        fig = px.scatter(
            clean_df,
            x='ActualPrice',
            y='TotalSalesDollars',
            title=f"Price vs Revenue (R² = {r_value**2:.3f})"
        )
        
        # Add manual trendline
        x_range = np.linspace(clean_df['ActualPrice'].min(), clean_df['ActualPrice'].max(), 100)
        y_trend = slope * x_range + intercept
        
        fig.add_trace(go.Scatter(
            x=x_range,
            y=y_trend,
            mode='lines',
            name=f'Trendline (R² = {r_value**2:.3f})',
            line=dict(color='red', dash='dash')
        ))
        st.plotly_chart(fig, use_container_width=True)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("R-squared", f"{r_value**2:.3f}")
        with col2:
            st.metric("P-value", f"{p_value:.6f}")
        with col3:
            significance = "Significant" if p_value < 0.05 else "Not Significant"
            st.metric("Relationship", significance)
    
    with tab2:
        st.subheader("Outlier Detection")
        
        # Z-score method for outlier detection
        z_scores = np.abs(stats.zscore(clean_df['ProfitMargin']))
        outliers = clean_df[z_scores > 3]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Outliers Detected", len(outliers))
            st.metric("Outlier Percentage", f"{len(outliers)/len(clean_df)*100:.1f}%")
            
            if len(outliers) > 0:
                st.write("**Top Outliers by Profit Margin:**")
                top_outliers = outliers.nlargest(5, 'ProfitMargin')[['Description', 'VendorName', 'ProfitMargin']]
                st.dataframe(top_outliers, hide_index=True)
        
        with col2:
            # Box plot highlighting outliers
            fig = px.box(
                clean_df,
                y='ProfitMargin',
                title="Profit Margin Outliers",
                points="outliers"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("Confidence Intervals")
        
        # Calculate confidence intervals for key metrics
        confidence_level = 0.95
        alpha = 1 - confidence_level
        
        metrics = ['ProfitMargin', 'StockTurnover', 'TotalSalesDollars']
        ci_results = []
        
        for metric in metrics:
            data = clean_df[metric].dropna()
            mean = data.mean()
            sem = stats.sem(data)  # Standard error of mean
            ci = stats.t.interval(confidence_level, len(data)-1, loc=mean, scale=sem)
            
            ci_results.append({
                'Metric': metric,
                'Mean': mean,
                'Lower CI': ci[0],
                'Upper CI': ci[1],
                'Margin of Error': ci[1] - mean
            })
        
        ci_df = pd.DataFrame(ci_results)
        
        # Format the dataframe for display
        for col in ['Mean', 'Lower CI', 'Upper CI', 'Margin of Error']:
            ci_df[col] = ci_df[col].apply(lambda x: f"{x:.2f}")
        
        st.dataframe(ci_df, hide_index=True, use_container_width=True)
        
        st.write(f"**Interpretation:** We are {confidence_level*100:.0f}% confident that the true population mean lies within these intervals.")
    
    # Statistical Summary
    st.header("📋 Statistical Summary & Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Key Findings")
        
        findings = []
        
        # Correlation insights
        if 'correlation' in stats_results:
            corr_matrix = stats_results['correlation']
            strongest_corr = correlations_df.loc[correlations_df['abs_corr'].idxmax()]
            findings.append(f"Strongest correlation: {strongest_corr['var1']} vs {strongest_corr['var2']} ({strongest_corr['correlation']:.3f})")
        
        # Distribution insights
        profit_skew = stats.skew(clean_df['ProfitMargin'])
        if abs(profit_skew) > 1:
            skew_direction = "right" if profit_skew > 0 else "left"
            findings.append(f"Profit margin distribution is highly skewed {skew_direction} (skewness: {profit_skew:.2f})")
        
        # Performance insights
        if 'ttest' in stats_results and stats_results['ttest']['p_value'] < 0.05:
            findings.append("High and low performing vendors have statistically different profit margins")
        
        for finding in findings:
            st.write(f"• {finding}")
    
    with col2:
        st.subheader("Recommendations")
        
        recommendations = [
            "Focus on vendors with consistently high profit margins",
            "Investigate products with extreme outlier performance",
            "Consider price optimization for products with weak price-revenue correlation",
            "Implement statistical process control for key metrics",
            "Regular monitoring of vendor performance differences"
        ]
        
        for rec in recommendations:
            st.write(f"• {rec}")

if __name__ == "__main__":
    main()
