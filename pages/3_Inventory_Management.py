import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

st.set_page_config(page_title="Inventory Management", page_icon="📦", layout="wide")

@st.cache_data
def load_data():
    return pd.read_csv('vendor_sales_summary.csv')

def main():
    st.title("📦 Inventory Management Dashboard")
    
    df = load_data()
    
    # Calculate inventory metrics
    df['Unsold_Quantity'] = df['TotalPurchaseQuantity'] - df['TotalSalesQuantity']
    df['Unsold_Value'] = df['Unsold_Quantity'] * df['PurchasePrice']
    df['Inventory_Days'] = np.where(df['TotalSalesQuantity'] > 0, 
                                   (df['Unsold_Quantity'] / df['TotalSalesQuantity']) * 365, 
                                   np.inf)
    
    # Inventory Health Overview
    st.header("Inventory Health Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_unsold_value = df['Unsold_Value'].sum()
        st.metric("Total Unsold Inventory", f"${total_unsold_value:,.0f}")
    
    with col2:
        zero_sales_count = df[df['TotalSalesQuantity'] == 0].shape[0]
        st.metric("Dead Stock Items", f"{zero_sales_count:,}")
    
    with col3:
        avg_turnover = df['StockTurnover'].mean()
        st.metric("Average Turnover", f"{avg_turnover:.2f}x")
    
    with col4:
        overstock_items = df[df['Unsold_Quantity'] > df['TotalSalesQuantity']].shape[0]
        st.metric("Overstock Items", f"{overstock_items:,}")
    
    # Stock Turnover Analysis
    st.header("Stock Turnover Analysis")
    
    # Create turnover categories
    df['Turnover_Category'] = pd.cut(
        df['StockTurnover'],
        bins=[0, 0.5, 1.0, 2.0, np.inf],
        labels=['Slow Moving', 'Normal', 'Fast Moving', 'Very Fast'],
        include_lowest=True
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Turnover distribution
        turnover_counts = df['Turnover_Category'].value_counts()
        fig = px.pie(
            values=turnover_counts.values,
            names=turnover_counts.index,
            title="Stock Turnover Distribution",
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Turnover vs Revenue
        fig = px.scatter(
            df[df['StockTurnover'] <= 10],  # Filter extreme outliers
            x='StockTurnover',
            y='TotalSalesDollars',
            size='TotalSalesQuantity',
            color='Turnover_Category',
            hover_data=['Description', 'VendorName'],
            title="Stock Turnover vs Revenue",
            labels={'StockTurnover': 'Stock Turnover Ratio', 'TotalSalesDollars': 'Revenue ($)'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Dead Stock Analysis
    st.header("Dead Stock Analysis")
    
    dead_stock = df[df['TotalSalesQuantity'] == 0].copy()
    
    if len(dead_stock) > 0:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Dead Stock by Value")
            top_dead_stock = dead_stock.nlargest(15, 'Unsold_Value')
            
            fig = px.bar(
                top_dead_stock,
                x='Unsold_Value',
                y='Description',
                orientation='h',
                title="Top 15 Dead Stock Items by Value",
                labels={'Unsold_Value': 'Unsold Value ($)', 'Description': 'Product'},
                color='Unsold_Value',
                color_continuous_scale='Reds'
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Dead Stock by Vendor")
            dead_stock_by_vendor = dead_stock.groupby('VendorName').agg({
                'Unsold_Value': 'sum',
                'Brand': 'count'
            }).reset_index()
            dead_stock_by_vendor.columns = ['VendorName', 'Total_Dead_Value', 'Dead_Items_Count']
            dead_stock_by_vendor = dead_stock_by_vendor.sort_values('Total_Dead_Value', ascending=False).head(10)
            
            fig = px.bar(
                dead_stock_by_vendor,
                x='Total_Dead_Value',
                y='VendorName',
                orientation='h',
                title="Dead Stock Value by Vendor",
                labels={'Total_Dead_Value': 'Dead Stock Value ($)', 'VendorName': 'Vendor'},
                color='Total_Dead_Value',
                color_continuous_scale='Oranges'
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
        
        # Dead stock details
        st.subheader("Dead Stock Details")
        dead_stock_display = dead_stock[['VendorName', 'Description', 'TotalPurchaseQuantity', 
                                        'PurchasePrice', 'Unsold_Value']].sort_values('Unsold_Value', ascending=False)
        dead_stock_display['PurchasePrice'] = dead_stock_display['PurchasePrice'].apply(lambda x: f"${x:.2f}")
        dead_stock_display['Unsold_Value'] = dead_stock_display['Unsold_Value'].apply(lambda x: f"${x:,.2f}")
        
        st.dataframe(dead_stock_display.head(20), use_container_width=True, hide_index=True)
    else:
        st.success("🎉 No dead stock found! All products have sales.")
    
    # Overstock Analysis
    st.header("Overstock Analysis")
    
    overstock = df[df['Unsold_Quantity'] > df['TotalSalesQuantity']].copy()
    
    if len(overstock) > 0:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Overstock Items by Value")
            top_overstock = overstock.nlargest(15, 'Unsold_Value')
            
            fig = px.bar(
                top_overstock,
                x='Unsold_Value',
                y='Description',
                orientation='h',
                title="Top 15 Overstock Items by Value",
                labels={'Unsold_Value': 'Overstock Value ($)', 'Description': 'Product'},
                color='Unsold_Value',
                color_continuous_scale='Blues'
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Overstock Ratio Analysis")
            overstock['Overstock_Ratio'] = overstock['Unsold_Quantity'] / overstock['TotalSalesQuantity']
            
            fig = px.scatter(
                overstock[overstock['Overstock_Ratio'] <= 10],  # Filter extreme outliers
                x='TotalSalesQuantity',
                y='Overstock_Ratio',
                size='Unsold_Value',
                hover_data=['Description', 'VendorName'],
                title="Sales Quantity vs Overstock Ratio",
                labels={'TotalSalesQuantity': 'Sales Quantity', 'Overstock_Ratio': 'Overstock Ratio'}
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.success("🎉 No overstock issues detected!")
    
    # Fast Moving Items
    st.header("Fast Moving Items")
    
    fast_moving = df[df['StockTurnover'] > 2.0].copy()
    
    if len(fast_moving) > 0:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Top Fast Moving Items")
            top_fast_moving = fast_moving.nlargest(15, 'TotalSalesDollars')
            
            fig = px.bar(
                top_fast_moving,
                x='StockTurnover',
                y='Description',
                orientation='h',
                title="Top 15 Fast Moving Items by Turnover",
                labels={'StockTurnover': 'Stock Turnover Ratio', 'Description': 'Product'},
                color='StockTurnover',
                color_continuous_scale='Greens'
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Fast Moving Items Revenue Impact")
            fast_moving_revenue = fast_moving.groupby('VendorName')['TotalSalesDollars'].sum().sort_values(ascending=False).head(10)
            
            fig = px.bar(
                x=fast_moving_revenue.values,
                y=fast_moving_revenue.index,
                orientation='h',
                title="Fast Moving Items Revenue by Vendor",
                labels={'x': 'Revenue ($)', 'y': 'Vendor'},
                color=fast_moving_revenue.values,
                color_continuous_scale='Viridis'
            )
            fig.update_layout(height=500, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
    
    # Inventory Recommendations
    st.header("📋 Inventory Management Recommendations")
    
    tab1, tab2, tab3, tab4 = st.tabs(["Immediate Actions", "Reorder Suggestions", "Clearance Items", "Optimization"])
    
    with tab1:
        st.subheader("🚨 Immediate Actions Required")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**High Value Dead Stock (>$1000)**")
            high_value_dead = dead_stock[dead_stock['Unsold_Value'] > 1000].nlargest(5, 'Unsold_Value')
            for _, item in high_value_dead.iterrows():
                st.write(f"• {item['Description'][:40]}...")
                st.write(f"  Value: ${item['Unsold_Value']:,.0f}")
                st.write(f"  Vendor: {item['VendorName']}")
                st.write("---")
        
        with col2:
            st.write("**Critical Overstock Items**")
            if len(overstock) > 0:
                critical_overstock = overstock[overstock['Overstock_Ratio'] > 3].nlargest(5, 'Unsold_Value')
                for _, item in critical_overstock.iterrows():
                    st.write(f"• {item['Description'][:40]}...")
                    st.write(f"  Excess Qty: {item['Unsold_Quantity']:,.0f}")
                    st.write(f"  Value: ${item['Unsold_Value']:,.0f}")
                    st.write("---")
            else:
                st.success("No critical overstock items!")
    
    with tab2:
        st.subheader("🔄 Reorder Suggestions")
        
        # Items with high turnover and low stock
        reorder_candidates = df[
            (df['StockTurnover'] > 1.5) & 
            (df['Unsold_Quantity'] < df['TotalSalesQuantity'] * 0.2)
        ].nlargest(10, 'TotalSalesDollars')
        
        if len(reorder_candidates) > 0:
            st.write("**High Priority Reorders**")
            for _, item in reorder_candidates.iterrows():
                st.write(f"• **{item['Description']}**")
                st.write(f"  Current Stock: {item['Unsold_Quantity']:,.0f}")
                st.write(f"  Turnover: {item['StockTurnover']:.2f}x")
                st.write(f"  Vendor: {item['VendorName']}")
                st.write("---")
        else:
            st.info("No immediate reorder requirements identified.")
    
    with tab3:
        st.subheader("🏷️ Clearance Sale Candidates")
        
        # Slow moving items with significant inventory
        clearance_candidates = df[
            (df['StockTurnover'] < 0.5) & 
            (df['Unsold_Value'] > 500)
        ].nlargest(10, 'Unsold_Value')
        
        if len(clearance_candidates) > 0:
            for _, item in clearance_candidates.iterrows():
                discount_suggestion = min(50, max(20, (1 - item['StockTurnover']) * 30))
                st.write(f"• **{item['Description']}**")
                st.write(f"  Unsold Value: ${item['Unsold_Value']:,.0f}")
                st.write(f"  Suggested Discount: {discount_suggestion:.0f}%")
                st.write(f"  Vendor: {item['VendorName']}")
                st.write("---")
        else:
            st.success("No clearance candidates identified!")
    
    with tab4:
        st.subheader("⚡ Optimization Opportunities")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Vendor Consolidation Opportunities**")
            # Find vendors with low performance
            vendor_performance = df.groupby('VendorName').agg({
                'TotalSalesDollars': 'sum',
                'Brand': 'count',
                'StockTurnover': 'mean'
            }).reset_index()
            
            low_performance_vendors = vendor_performance[
                (vendor_performance['TotalSalesDollars'] < vendor_performance['TotalSalesDollars'].median()) &
                (vendor_performance['Brand'] < 3)
            ].head(5)
            
            for _, vendor in low_performance_vendors.iterrows():
                st.write(f"• **{vendor['VendorName']}**")
                st.write(f"  Revenue: ${vendor['TotalSalesDollars']:,.0f}")
                st.write(f"  Brands: {vendor['Brand']}")
                st.write("---")
        
        with col2:
            st.write("**Category Optimization**")
            # Analyze by price ranges
            df['Price_Category'] = pd.cut(
                df['ActualPrice'],
                bins=[0, 15, 30, 50, np.inf],
                labels=['Budget', 'Mid-Range', 'Premium', 'Luxury']
            )
            
            category_performance = df.groupby('Price_Category').agg({
                'StockTurnover': 'mean',
                'ProfitMargin': lambda x: x[x != np.inf].mean() if len(x[x != np.inf]) > 0 else 0,
                'TotalSalesDollars': 'sum'
            }).reset_index()
            
            for _, cat in category_performance.iterrows():
                st.write(f"• **{cat['Price_Category']} Category**")
                st.write(f"  Avg Turnover: {cat['StockTurnover']:.2f}x")
                st.write(f"  Avg Margin: {cat['ProfitMargin']:.1f}%")
                st.write("---")

if __name__ == "__main__":
    main()
