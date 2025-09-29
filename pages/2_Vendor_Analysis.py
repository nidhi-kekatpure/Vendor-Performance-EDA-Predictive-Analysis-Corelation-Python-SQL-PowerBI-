import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

st.set_page_config(page_title="Vendor Analysis", page_icon="🏢", layout="wide")

@st.cache_data
def load_data():
    return pd.read_csv('vendor_sales_summary.csv')

def main():
    st.title("🏢 Vendor Performance Analysis")
    
    df = load_data()
    
    # Vendor Overview
    st.header("Vendor Portfolio Overview")
    
    # Aggregate vendor data
    vendor_summary = df.groupby(['VendorNumber', 'VendorName']).agg({
        'TotalSalesDollars': 'sum',
        'GrossProfit': 'sum',
        'TotalPurchaseDollars': 'sum',
        'Brand': 'count',
        'ProfitMargin': lambda x: (x[x != np.inf]).mean() if len(x[x != np.inf]) > 0 else 0,
        'StockTurnover': 'mean'
    }).reset_index()
    
    vendor_summary.columns = ['VendorNumber', 'VendorName', 'TotalRevenue', 'TotalProfit', 
                             'TotalPurchases', 'BrandCount', 'AvgProfitMargin', 'AvgStockTurnover']
    
    # Top metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Vendors", len(vendor_summary))
    with col2:
        top_vendor_revenue = vendor_summary['TotalRevenue'].max()
        st.metric("Top Vendor Revenue", f"${top_vendor_revenue:,.0f}")
    with col3:
        avg_brands_per_vendor = vendor_summary['BrandCount'].mean()
        st.metric("Avg Brands/Vendor", f"{avg_brands_per_vendor:.1f}")
    with col4:
        vendor_profit_margin = (vendor_summary['TotalProfit'].sum() / vendor_summary['TotalRevenue'].sum()) * 100
        st.metric("Overall Profit Margin", f"{vendor_profit_margin:.1f}%")
    
    # Vendor Selection for Detailed Analysis
    st.header("Detailed Vendor Analysis")
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        selected_vendor = st.selectbox(
            "Select Vendor",
            options=sorted(df['VendorName'].unique()),
            index=0
        )
        
        # Vendor summary stats
        vendor_data = df[df['VendorName'] == selected_vendor]
        vendor_stats = vendor_summary[vendor_summary['VendorName'] == selected_vendor].iloc[0]
        
        st.subheader("Vendor Summary")
        st.metric("Total Revenue", f"${vendor_stats['TotalRevenue']:,.0f}")
        st.metric("Total Profit", f"${vendor_stats['TotalProfit']:,.0f}")
        st.metric("Number of Brands", int(vendor_stats['BrandCount']))
        st.metric("Avg Profit Margin", f"{vendor_stats['AvgProfitMargin']:.1f}%")
        st.metric("Avg Stock Turnover", f"{vendor_stats['AvgStockTurnover']:.2f}x")
    
    with col2:
        # Vendor brand performance
        st.subheader(f"Brand Performance - {selected_vendor}")
        
        # Create tabs for different views
        tab1, tab2, tab3 = st.tabs(["Revenue Analysis", "Profitability", "Inventory"])
        
        with tab1:
            fig = px.bar(
                vendor_data.sort_values('TotalSalesDollars', ascending=True).tail(15),
                x='TotalSalesDollars',
                y='Description',
                orientation='h',
                title="Top 15 Brands by Revenue",
                labels={'TotalSalesDollars': 'Revenue ($)', 'Description': 'Brand'}
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            clean_vendor_data = vendor_data[vendor_data['ProfitMargin'] != np.inf]
            if len(clean_vendor_data) > 0:
                fig = px.scatter(
                    clean_vendor_data,
                    x='TotalSalesDollars',
                    y='ProfitMargin',
                    size='TotalSalesQuantity',
                    hover_data=['Description'],
                    title="Revenue vs Profit Margin",
                    labels={'TotalSalesDollars': 'Revenue ($)', 'ProfitMargin': 'Profit Margin (%)'}
                )
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No valid profit margin data available for this vendor.")
        
        with tab3:
            fig = px.scatter(
                vendor_data,
                x='TotalPurchaseQuantity',
                y='TotalSalesQuantity',
                size='TotalSalesDollars',
                hover_data=['Description'],
                title="Purchase vs Sales Quantity",
                labels={'TotalPurchaseQuantity': 'Purchased Qty', 'TotalSalesQuantity': 'Sold Qty'}
            )
            # Add diagonal line for reference
            max_qty = max(vendor_data['TotalPurchaseQuantity'].max(), vendor_data['TotalSalesQuantity'].max())
            fig.add_shape(
                type="line",
                x0=0, y0=0, x1=max_qty, y1=max_qty,
                line=dict(color="red", dash="dash"),
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
    
    # Vendor Comparison
    st.header("Vendor Comparison")
    
    # Multi-select for vendor comparison
    comparison_vendors = st.multiselect(
        "Select vendors to compare (max 5)",
        options=sorted(df['VendorName'].unique()),
        default=vendor_summary.nlargest(3, 'TotalRevenue')['VendorName'].tolist(),
        max_selections=5
    )
    
    if comparison_vendors:
        comparison_data = vendor_summary[vendor_summary['VendorName'].isin(comparison_vendors)]
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Revenue comparison
            fig = px.bar(
                comparison_data.sort_values('TotalRevenue', ascending=True),
                x='TotalRevenue',
                y='VendorName',
                orientation='h',
                title="Revenue Comparison",
                labels={'TotalRevenue': 'Revenue ($)', 'VendorName': 'Vendor'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Profit margin comparison
            fig = px.bar(
                comparison_data.sort_values('AvgProfitMargin', ascending=True),
                x='AvgProfitMargin',
                y='VendorName',
                orientation='h',
                title="Average Profit Margin Comparison",
                labels={'AvgProfitMargin': 'Avg Profit Margin (%)', 'VendorName': 'Vendor'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Detailed comparison table
        st.subheader("Detailed Comparison")
        display_cols = ['VendorName', 'TotalRevenue', 'TotalProfit', 'BrandCount', 
                       'AvgProfitMargin', 'AvgStockTurnover']
        comparison_display = comparison_data[display_cols].copy()
        comparison_display['TotalRevenue'] = comparison_display['TotalRevenue'].apply(lambda x: f"${x:,.0f}")
        comparison_display['TotalProfit'] = comparison_display['TotalProfit'].apply(lambda x: f"${x:,.0f}")
        comparison_display['AvgProfitMargin'] = comparison_display['AvgProfitMargin'].apply(lambda x: f"{x:.1f}%")
        comparison_display['AvgStockTurnover'] = comparison_display['AvgStockTurnover'].apply(lambda x: f"{x:.2f}x")
        
        st.dataframe(comparison_display, use_container_width=True, hide_index=True)
    
    # Vendor Performance Matrix
    st.header("Vendor Performance Matrix")
    
    # Create performance categories
    revenue_median = vendor_summary['TotalRevenue'].median()
    margin_median = vendor_summary['AvgProfitMargin'].median()
    
    fig = px.scatter(
        vendor_summary,
        x='TotalRevenue',
        y='AvgProfitMargin',
        size='BrandCount',
        hover_data=['VendorName'],
        title="Vendor Performance Matrix: Revenue vs Profit Margin",
        labels={'TotalRevenue': 'Total Revenue ($)', 'AvgProfitMargin': 'Average Profit Margin (%)'}
    )
    
    # Add quadrant lines
    fig.add_hline(y=margin_median, line_dash="dash", line_color="gray", opacity=0.5)
    fig.add_vline(x=revenue_median, line_dash="dash", line_color="gray", opacity=0.5)
    
    # Add quadrant labels
    fig.add_annotation(x=vendor_summary['TotalRevenue'].max()*0.8, y=vendor_summary['AvgProfitMargin'].max()*0.9,
                      text="Star Vendors<br>(High Revenue, High Margin)", showarrow=False, 
                      bgcolor="lightgreen", opacity=0.7)
    fig.add_annotation(x=vendor_summary['TotalRevenue'].max()*0.2, y=vendor_summary['AvgProfitMargin'].max()*0.9,
                      text="Niche Vendors<br>(Low Revenue, High Margin)", showarrow=False, 
                      bgcolor="lightyellow", opacity=0.7)
    fig.add_annotation(x=vendor_summary['TotalRevenue'].max()*0.8, y=vendor_summary['AvgProfitMargin'].min()*1.5,
                      text="Volume Vendors<br>(High Revenue, Low Margin)", showarrow=False, 
                      bgcolor="lightblue", opacity=0.7)
    fig.add_annotation(x=vendor_summary['TotalRevenue'].max()*0.2, y=vendor_summary['AvgProfitMargin'].min()*1.5,
                      text="Underperforming<br>(Low Revenue, Low Margin)", showarrow=False, 
                      bgcolor="lightcoral", opacity=0.7)
    
    fig.update_layout(height=600, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    
    # Vendor Recommendations
    st.header("📋 Vendor Recommendations")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("🌟 Top Performers")
        top_performers = vendor_summary.nlargest(5, 'TotalRevenue')
        for _, vendor in top_performers.iterrows():
            st.write(f"**{vendor['VendorName']}**")
            st.write(f"Revenue: ${vendor['TotalRevenue']:,.0f}")
            st.write(f"Margin: {vendor['AvgProfitMargin']:.1f}%")
            st.write("---")
    
    with col2:
        st.subheader("⚠️ Review Required")
        low_performers = vendor_summary[
            (vendor_summary['TotalRevenue'] < revenue_median) & 
            (vendor_summary['AvgProfitMargin'] < margin_median)
        ].head(5)
        for _, vendor in low_performers.iterrows():
            st.write(f"**{vendor['VendorName']}**")
            st.write(f"Revenue: ${vendor['TotalRevenue']:,.0f}")
            st.write(f"Margin: {vendor['AvgProfitMargin']:.1f}%")
            st.write("---")
    
    with col3:
        st.subheader("🎯 Growth Opportunities")
        growth_opportunities = vendor_summary[
            (vendor_summary['TotalRevenue'] < revenue_median) & 
            (vendor_summary['AvgProfitMargin'] > margin_median)
        ].head(5)
        for _, vendor in growth_opportunities.iterrows():
            st.write(f"**{vendor['VendorName']}**")
            st.write(f"Revenue: ${vendor['TotalRevenue']:,.0f}")
            st.write(f"Margin: {vendor['AvgProfitMargin']:.1f}%")
            st.write("---")

if __name__ == "__main__":
    main()
