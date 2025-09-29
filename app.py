import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Vendor Performance Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #1f77b4;
    }
    .insight-box {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and cache the vendor sales summary data"""
    try:
        df = pd.read_csv('vendor_sales_summary.csv')
        return df
    except FileNotFoundError:
        st.error("Data file not found. Please ensure 'vendor_sales_summary.csv' is in the project directory.")
        return None

def create_summary_metrics(df):
    """Create summary metrics for the dashboard"""
    total_vendors = df['VendorNumber'].nunique()
    total_brands = df['Brand'].nunique()
    total_revenue = df['TotalSalesDollars'].sum()
    total_profit = df['GrossProfit'].sum()
    avg_profit_margin = df[df['ProfitMargin'] != np.inf]['ProfitMargin'].mean()
    
    return {
        'total_vendors': total_vendors,
        'total_brands': total_brands,
        'total_revenue': total_revenue,
        'total_profit': total_profit,
        'avg_profit_margin': avg_profit_margin
    }

def filter_data(df):
    """Create sidebar filters for the data"""
    st.sidebar.header("🔍 Filters")
    
    # Vendor filter
    vendors = st.sidebar.multiselect(
        "Select Vendors",
        options=sorted(df['VendorName'].unique()),
        default=[]
    )
    
    # Profit margin filter
    profit_range = st.sidebar.slider(
        "Profit Margin Range (%)",
        min_value=float(df[df['ProfitMargin'] != -np.inf]['ProfitMargin'].min()),
        max_value=float(df[df['ProfitMargin'] != np.inf]['ProfitMargin'].max()),
        value=(0.0, 50.0),
        step=1.0
    )
    
    # Revenue filter
    revenue_range = st.sidebar.slider(
        "Revenue Range ($)",
        min_value=0,
        max_value=int(df['TotalSalesDollars'].max()),
        value=(0, int(df['TotalSalesDollars'].max())),
        step=10000
    )
    
    # Apply filters
    filtered_df = df.copy()
    
    if vendors:
        filtered_df = filtered_df[filtered_df['VendorName'].isin(vendors)]
    
    filtered_df = filtered_df[
        (filtered_df['ProfitMargin'] >= profit_range[0]) & 
        (filtered_df['ProfitMargin'] <= profit_range[1]) &
        (filtered_df['TotalSalesDollars'] >= revenue_range[0]) &
        (filtered_df['TotalSalesDollars'] <= revenue_range[1])
    ]
    
    return filtered_df

def main():
    # Load data
    df = load_data()
    if df is None:
        return
    
    # Main title
    st.markdown('<h1 class="main-header">📊 Vendor Performance Analytics Dashboard</h1>', unsafe_allow_html=True)
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page",
        ["Overview", "Vendor Analysis", "Brand Performance", "Profitability Analysis", "Inventory Insights", "Statistical Analysis"]
    )
    
    # Apply filters
    filtered_df = filter_data(df)
    
    if page == "Overview":
        show_overview(filtered_df)
    elif page == "Vendor Analysis":
        show_vendor_analysis(filtered_df)
    elif page == "Brand Performance":
        show_brand_performance(filtered_df)
    elif page == "Profitability Analysis":
        show_profitability_analysis(filtered_df)
    elif page == "Inventory Insights":
        show_inventory_insights(filtered_df)
    elif page == "Statistical Analysis":
        show_statistical_analysis(filtered_df)

def show_overview(df):
    """Display overview dashboard"""
    st.header("📈 Business Overview")
    
    # Summary metrics
    metrics = create_summary_metrics(df)
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Total Vendors", f"{metrics['total_vendors']:,}")
    with col2:
        st.metric("Total Brands", f"{metrics['total_brands']:,}")
    with col3:
        st.metric("Total Revenue", f"${metrics['total_revenue']:,.0f}")
    with col4:
        st.metric("Total Profit", f"${metrics['total_profit']:,.0f}")
    with col5:
        st.metric("Avg Profit Margin", f"{metrics['avg_profit_margin']:.1f}%")
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Top 10 Vendors by Revenue")
        top_vendors = df.groupby('VendorName')['TotalSalesDollars'].sum().sort_values(ascending=False).head(10)
        fig = px.bar(
            x=top_vendors.values,
            y=top_vendors.index,
            orientation='h',
            title="Revenue by Vendor",
            labels={'x': 'Revenue ($)', 'y': 'Vendor'}
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Profit Margin Distribution")
        clean_margins = df[df['ProfitMargin'] != np.inf]['ProfitMargin']
        fig = px.histogram(
            clean_margins,
            nbins=30,
            title="Distribution of Profit Margins",
            labels={'value': 'Profit Margin (%)', 'count': 'Frequency'}
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Key insights
    st.markdown('<div class="insight-box">', unsafe_allow_html=True)
    st.subheader("🔍 Key Insights")
    
    # Calculate insights
    high_margin_brands = df[df['ProfitMargin'] > 40].shape[0]
    low_sales_brands = df[df['TotalSalesQuantity'] == 0].shape[0]
    top_vendor_contribution = df.groupby('VendorName')['TotalSalesDollars'].sum().sort_values(ascending=False).head(10).sum() / df['TotalSalesDollars'].sum() * 100
    
    st.write(f"• **{high_margin_brands}** brands have profit margins above 40%")
    st.write(f"• **{low_sales_brands}** brands have zero sales (potential dead stock)")
    st.write(f"• Top 10 vendors contribute **{top_vendor_contribution:.1f}%** of total revenue")
    st.write(f"• Average stock turnover ratio: **{df['StockTurnover'].mean():.2f}**")
    st.markdown('</div>', unsafe_allow_html=True)

def show_vendor_analysis(df):
    """Display vendor analysis page"""
    st.header("🏢 Vendor Performance Analysis")
    
    # Vendor selection
    selected_vendor = st.selectbox(
        "Select a vendor for detailed analysis",
        options=sorted(df['VendorName'].unique())
    )
    
    vendor_data = df[df['VendorName'] == selected_vendor]
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Brands", len(vendor_data))
    with col2:
        st.metric("Total Revenue", f"${vendor_data['TotalSalesDollars'].sum():,.0f}")
    with col3:
        st.metric("Total Profit", f"${vendor_data['GrossProfit'].sum():,.0f}")
    with col4:
        avg_margin = vendor_data[vendor_data['ProfitMargin'] != np.inf]['ProfitMargin'].mean()
        st.metric("Avg Profit Margin", f"{avg_margin:.1f}%")
    
    # Vendor brand performance
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Brand Performance by Revenue")
        fig = px.bar(
            vendor_data.sort_values('TotalSalesDollars', ascending=True).tail(10),
            x='TotalSalesDollars',
            y='Description',
            orientation='h',
            title=f"Top Brands for {selected_vendor}"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Profit Margin vs Revenue")
        clean_data = vendor_data[vendor_data['ProfitMargin'] != np.inf]
        fig = px.scatter(
            clean_data,
            x='TotalSalesDollars',
            y='ProfitMargin',
            size='TotalSalesQuantity',
            hover_data=['Description'],
            title="Profitability Analysis"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Detailed table
    st.subheader("Detailed Brand Data")
    display_cols = ['Description', 'TotalSalesDollars', 'GrossProfit', 'ProfitMargin', 'StockTurnover']
    st.dataframe(
        vendor_data[display_cols].sort_values('TotalSalesDollars', ascending=False),
        use_container_width=True
    )

def show_brand_performance(df):
    """Display brand performance analysis"""
    st.header("🏷️ Brand Performance Analysis")
    
    # Performance categories
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("High Margin, Low Sales Brands")
        st.write("*Candidates for promotional campaigns*")
        
        high_margin_low_sales = df[
            (df['ProfitMargin'] > 35) & 
            (df['TotalSalesQuantity'] < df['TotalSalesQuantity'].median()) &
            (df['ProfitMargin'] != np.inf)
        ].sort_values('ProfitMargin', ascending=False).head(10)
        
        fig = px.scatter(
            high_margin_low_sales,
            x='TotalSalesQuantity',
            y='ProfitMargin',
            size='GrossProfit',
            hover_data=['Description', 'VendorName'],
            title="Promotion Candidates"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("High Sales, Low Margin Brands")
        st.write("*Candidates for cost optimization*")
        
        high_sales_low_margin = df[
            (df['ProfitMargin'] < 25) & 
            (df['TotalSalesQuantity'] > df['TotalSalesQuantity'].median()) &
            (df['ProfitMargin'] > 0)
        ].sort_values('TotalSalesQuantity', ascending=False).head(10)
        
        fig = px.scatter(
            high_sales_low_margin,
            x='TotalSalesQuantity',
            y='ProfitMargin',
            size='TotalSalesDollars',
            hover_data=['Description', 'VendorName'],
            title="Cost Optimization Candidates"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Brand comparison
    st.subheader("Brand Comparison Tool")
    selected_brands = st.multiselect(
        "Select brands to compare",
        options=df['Description'].unique(),
        default=df.nlargest(5, 'TotalSalesDollars')['Description'].tolist()
    )
    
    if selected_brands:
        comparison_data = df[df['Description'].isin(selected_brands)]
        
        metrics = ['TotalSalesDollars', 'GrossProfit', 'ProfitMargin', 'StockTurnover']
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Revenue', 'Gross Profit', 'Profit Margin (%)', 'Stock Turnover'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        for i, metric in enumerate(metrics):
            row = (i // 2) + 1
            col = (i % 2) + 1
            
            clean_data = comparison_data[comparison_data[metric] != np.inf] if metric == 'ProfitMargin' else comparison_data
            
            fig.add_trace(
                go.Bar(x=clean_data['Description'], y=clean_data[metric], name=metric),
                row=row, col=col
            )
        
        fig.update_layout(height=600, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

def show_profitability_analysis(df):
    """Display profitability analysis"""
    st.header("💰 Profitability Analysis")
    
    # Clean data for analysis
    clean_df = df[df['ProfitMargin'] != np.inf].copy()
    
    # Profitability segments
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Profitability Segments")
        
        # Create segments
        clean_df['Profitability_Segment'] = pd.cut(
            clean_df['ProfitMargin'],
            bins=[-np.inf, 0, 15, 30, np.inf],
            labels=['Loss Making', 'Low Profit', 'Medium Profit', 'High Profit']
        )
        
        segment_counts = clean_df['Profitability_Segment'].value_counts()
        
        fig = px.pie(
            values=segment_counts.values,
            names=segment_counts.index,
            title="Distribution of Profitability Segments"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Revenue vs Profit Correlation")
        
        fig = px.scatter(
            clean_df,
            x='TotalSalesDollars',
            y='GrossProfit',
            color='ProfitMargin',
            size='TotalSalesQuantity',
            hover_data=['Description', 'VendorName'],
            title="Revenue vs Profit Analysis"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Top performers analysis
    st.subheader("Top Performers Analysis")
    
    tab1, tab2, tab3 = st.tabs(["By Revenue", "By Profit", "By Margin"])
    
    with tab1:
        top_revenue = df.nlargest(10, 'TotalSalesDollars')
        fig = px.bar(
            top_revenue,
            x='Description',
            y='TotalSalesDollars',
            color='VendorName',
            title="Top 10 Brands by Revenue"
        )
        fig.update_xaxes(tickangle=45)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        top_profit = df.nlargest(10, 'GrossProfit')
        fig = px.bar(
            top_profit,
            x='Description',
            y='GrossProfit',
            color='VendorName',
            title="Top 10 Brands by Gross Profit"
        )
        fig.update_xaxes(tickangle=45)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        top_margin = clean_df.nlargest(10, 'ProfitMargin')
        fig = px.bar(
            top_margin,
            x='Description',
            y='ProfitMargin',
            color='VendorName',
            title="Top 10 Brands by Profit Margin"
        )
        fig.update_xaxes(tickangle=45)
        st.plotly_chart(fig, use_container_width=True)

def show_inventory_insights(df):
    """Display inventory management insights"""
    st.header("📦 Inventory Management Insights")
    
    # Stock turnover analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Stock Turnover Distribution")
        
        fig = px.histogram(
            df[df['StockTurnover'] <= 5],  # Filter extreme outliers
            x='StockTurnover',
            nbins=30,
            title="Stock Turnover Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Stock turnover categories
        df['Turnover_Category'] = pd.cut(
            df['StockTurnover'],
            bins=[0, 0.5, 1.0, 2.0, np.inf],
            labels=['Slow Moving', 'Normal', 'Fast Moving', 'Very Fast']
        )
        
        turnover_summary = df['Turnover_Category'].value_counts()
        st.write("**Stock Turnover Categories:**")
        for category, count in turnover_summary.items():
            st.write(f"• {category}: {count} brands")
    
    with col2:
        st.subheader("Unsold Inventory Analysis")
        
        # Calculate unsold inventory value
        df['Unsold_Quantity'] = df['TotalPurchaseQuantity'] - df['TotalSalesQuantity']
        df['Unsold_Value'] = df['Unsold_Quantity'] * df['PurchasePrice']
        
        # Top unsold inventory by value
        top_unsold = df[df['Unsold_Value'] > 0].nlargest(10, 'Unsold_Value')
        
        fig = px.bar(
            top_unsold,
            x='Description',
            y='Unsold_Value',
            color='VendorName',
            title="Top 10 Brands by Unsold Inventory Value"
        )
        fig.update_xaxes(tickangle=45)
        st.plotly_chart(fig, use_container_width=True)
    
    # Inventory recommendations
    st.subheader("📋 Inventory Recommendations")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**🔴 Slow Moving Items**")
        slow_moving = df[df['StockTurnover'] < 0.5].nlargest(5, 'Unsold_Value')
        for _, item in slow_moving.iterrows():
            st.write(f"• {item['Description'][:30]}...")
            st.write(f"  Unsold Value: ${item['Unsold_Value']:,.0f}")
    
    with col2:
        st.markdown("**🟡 Overstock Items**")
        overstock = df[df['Unsold_Quantity'] > df['TotalSalesQuantity']].nlargest(5, 'Unsold_Value')
        for _, item in overstock.iterrows():
            st.write(f"• {item['Description'][:30]}...")
            st.write(f"  Excess Qty: {item['Unsold_Quantity']:,.0f}")
    
    with col3:
        st.markdown("**🟢 Fast Moving Items**")
        fast_moving = df[df['StockTurnover'] > 2.0].nlargest(5, 'TotalSalesDollars')
        for _, item in fast_moving.iterrows():
            st.write(f"• {item['Description'][:30]}...")
            st.write(f"  Turnover: {item['StockTurnover']:.2f}x")

def show_statistical_analysis(df):
    """Display statistical analysis"""
    st.header("📊 Statistical Analysis")
    
    # Clean data
    clean_df = df[df['ProfitMargin'] != np.inf].copy()
    
    # Correlation analysis
    st.subheader("Correlation Analysis")
    
    numeric_cols = ['PurchasePrice', 'ActualPrice', 'TotalPurchaseQuantity', 
                   'TotalSalesDollars', 'GrossProfit', 'ProfitMargin', 'StockTurnover']
    
    corr_matrix = clean_df[numeric_cols].corr()
    
    fig = px.imshow(
        corr_matrix,
        text_auto=True,
        aspect="auto",
        title="Correlation Matrix of Key Metrics"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Vendor performance comparison
    st.subheader("Vendor Performance Comparison")
    
    # Group vendors by performance
    vendor_stats = clean_df.groupby('VendorName').agg({
        'TotalSalesDollars': 'sum',
        'ProfitMargin': 'mean',
        'StockTurnover': 'mean'
    }).reset_index()
    
    # Classify vendors
    vendor_stats['Performance_Class'] = pd.cut(
        vendor_stats['TotalSalesDollars'],
        bins=3,
        labels=['Low Performers', 'Medium Performers', 'High Performers']
    )
    
    # Statistical test
    high_performers = vendor_stats[vendor_stats['Performance_Class'] == 'High Performers']['ProfitMargin']
    low_performers = vendor_stats[vendor_stats['Performance_Class'] == 'Low Performers']['ProfitMargin']
    
    if len(high_performers) > 0 and len(low_performers) > 0:
        t_stat, p_value = stats.ttest_ind(high_performers, low_performers)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Performance Distribution")
            fig = px.box(
                vendor_stats,
                x='Performance_Class',
                y='ProfitMargin',
                title="Profit Margin by Performance Class"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Statistical Test Results")
            st.write("**T-test: High vs Low Performers**")
            st.write(f"T-statistic: {t_stat:.4f}")
            st.write(f"P-value: {p_value:.4f}")
            
            if p_value < 0.05:
                st.success("✅ Significant difference in profit margins between high and low performers")
            else:
                st.info("ℹ️ No significant difference found")
            
            st.write("**Summary Statistics:**")
            st.write(f"High Performers Avg Margin: {high_performers.mean():.2f}%")
            st.write(f"Low Performers Avg Margin: {low_performers.mean():.2f}%")
    
    # Key insights
    st.subheader("📈 Statistical Insights")
    
    insights = []
    
    # Price correlation
    price_corr = clean_df['PurchasePrice'].corr(clean_df['TotalSalesDollars'])
    insights.append(f"Purchase price correlation with sales: {price_corr:.3f}")
    
    # Margin distribution
    margin_std = clean_df['ProfitMargin'].std()
    insights.append(f"Profit margin standard deviation: {margin_std:.2f}%")
    
    # Turnover insights
    avg_turnover = clean_df['StockTurnover'].mean()
    insights.append(f"Average stock turnover: {avg_turnover:.2f}x")
    
    for insight in insights:
        st.write(f"• {insight}")

if __name__ == "__main__":
    main()
