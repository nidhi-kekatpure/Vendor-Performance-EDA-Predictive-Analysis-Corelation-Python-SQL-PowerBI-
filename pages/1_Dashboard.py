import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="Dashboard", page_icon="📊", layout="wide")

@st.cache_data
def load_data():
    return pd.read_csv('vendor_sales_summary.csv')

def main():
    st.title("📊 Executive Dashboard")
    
    df = load_data()
    
    # Key Performance Indicators
    st.header("Key Performance Indicators")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_revenue = df['TotalSalesDollars'].sum()
        st.metric(
            label="Total Revenue",
            value=f"${total_revenue:,.0f}",
            delta=f"{(total_revenue/1e6):.1f}M"
        )
    
    with col2:
        total_profit = df['GrossProfit'].sum()
        profit_margin = (total_profit / total_revenue) * 100
        st.metric(
            label="Total Gross Profit",
            value=f"${total_profit:,.0f}",
            delta=f"{profit_margin:.1f}% margin"
        )
    
    with col3:
        active_vendors = df['VendorNumber'].nunique()
        st.metric(
            label="Active Vendors",
            value=f"{active_vendors:,}",
            delta=f"{df['Brand'].nunique():,} brands"
        )
    
    with col4:
        avg_turnover = df['StockTurnover'].mean()
        st.metric(
            label="Avg Stock Turnover",
            value=f"{avg_turnover:.2f}x",
            delta="Annual"
        )
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Revenue by Top 15 Vendors")
        vendor_revenue = df.groupby('VendorName')['TotalSalesDollars'].sum().sort_values(ascending=False).head(15)
        
        fig = px.bar(
            x=vendor_revenue.values,
            y=vendor_revenue.index,
            orientation='h',
            title="Top Vendors by Revenue",
            labels={'x': 'Revenue ($)', 'y': 'Vendor'},
            color=vendor_revenue.values,
            color_continuous_scale='Blues'
        )
        fig.update_layout(height=500, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Profitability Distribution")
        clean_margins = df[df['ProfitMargin'] != float('inf')]['ProfitMargin']
        
        fig = px.histogram(
            clean_margins,
            nbins=40,
            title="Profit Margin Distribution",
            labels={'value': 'Profit Margin (%)', 'count': 'Number of Products'},
            color_discrete_sequence=['#1f77b4']
        )
        fig.add_vline(x=clean_margins.mean(), line_dash="dash", line_color="red", 
                     annotation_text=f"Mean: {clean_margins.mean():.1f}%")
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
    
    # Performance Matrix
    st.header("Performance Matrix")
    
    # Create performance quadrants
    median_sales = df['TotalSalesDollars'].median()
    median_margin = df[df['ProfitMargin'] != float('inf')]['ProfitMargin'].median()
    
    df_clean = df[df['ProfitMargin'] != float('inf')].copy()
    
    fig = px.scatter(
        df_clean,
        x='TotalSalesDollars',
        y='ProfitMargin',
        size='TotalSalesQuantity',
        color='VendorName',
        hover_data=['Description'],
        title="Sales vs Profit Margin Performance Matrix",
        labels={'TotalSalesDollars': 'Total Sales ($)', 'ProfitMargin': 'Profit Margin (%)'}
    )
    
    # Add quadrant lines
    fig.add_hline(y=median_margin, line_dash="dash", line_color="gray", opacity=0.5)
    fig.add_vline(x=median_sales, line_dash="dash", line_color="gray", opacity=0.5)
    
    # Add quadrant labels
    fig.add_annotation(x=df_clean['TotalSalesDollars'].max()*0.8, y=df_clean['ProfitMargin'].max()*0.9,
                      text="Stars<br>(High Sales, High Margin)", showarrow=False, bgcolor="lightgreen", opacity=0.7)
    fig.add_annotation(x=df_clean['TotalSalesDollars'].max()*0.2, y=df_clean['ProfitMargin'].max()*0.9,
                      text="Question Marks<br>(Low Sales, High Margin)", showarrow=False, bgcolor="lightyellow", opacity=0.7)
    fig.add_annotation(x=df_clean['TotalSalesDollars'].max()*0.8, y=df_clean['ProfitMargin'].min()*1.1,
                      text="Cash Cows<br>(High Sales, Low Margin)", showarrow=False, bgcolor="lightblue", opacity=0.7)
    fig.add_annotation(x=df_clean['TotalSalesDollars'].max()*0.2, y=df_clean['ProfitMargin'].min()*1.1,
                      text="Dogs<br>(Low Sales, Low Margin)", showarrow=False, bgcolor="lightcoral", opacity=0.7)
    
    fig.update_layout(height=600, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    
    # Business Insights
    st.header("🔍 Key Business Insights")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Revenue Concentration")
        top_10_revenue = df.groupby('VendorName')['TotalSalesDollars'].sum().sort_values(ascending=False).head(10).sum()
        concentration = (top_10_revenue / df['TotalSalesDollars'].sum()) * 100
        st.metric("Top 10 Vendors Share", f"{concentration:.1f}%")
        
        if concentration > 70:
            st.warning("⚠️ High vendor concentration risk")
        else:
            st.success("✅ Good vendor diversification")
    
    with col2:
        st.subheader("Inventory Health")
        zero_sales = df[df['TotalSalesQuantity'] == 0].shape[0]
        total_products = df.shape[0]
        dead_stock_pct = (zero_sales / total_products) * 100
        st.metric("Dead Stock %", f"{dead_stock_pct:.1f}%")
        
        if dead_stock_pct > 10:
            st.warning(f"⚠️ {zero_sales} products with zero sales")
        else:
            st.success("✅ Healthy inventory turnover")
    
    with col3:
        st.subheader("Profitability Health")
        profitable_products = df[df['GrossProfit'] > 0].shape[0]
        profitability_rate = (profitable_products / total_products) * 100
        st.metric("Profitable Products", f"{profitability_rate:.1f}%")
        
        if profitability_rate > 80:
            st.success("✅ Strong profitability")
        else:
            st.warning("⚠️ Review unprofitable products")

if __name__ == "__main__":
    main()
