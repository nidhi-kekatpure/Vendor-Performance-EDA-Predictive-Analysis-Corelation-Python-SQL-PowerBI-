# Vendor Performance Analytics - Streamlit App

An interactive web application for comprehensive vendor performance analysis, built with Streamlit and deployed on Streamlit Cloud.

## Live Demo

**Try the app now:** 

## 📋 Features

### Main Dashboard (`app.py`)
- **Multi-page Navigation**: Seamless navigation between different analysis modules
- **Interactive Filters**: Dynamic filtering by vendor, profit margin, and revenue ranges
- **Real-time Metrics**: Key performance indicators with live updates
- **Responsive Design**: Optimized for desktop and mobile viewing

### Executive Dashboard
- **KPI Overview**: Total revenue, profit, vendors, and stock turnover metrics
- **Performance Matrix**: BCG-style matrix categorizing products into Stars, Cash Cows, Question Marks, and Dogs
- **Business Insights**: Automated insights on revenue concentration, inventory health, and profitability

### Vendor Analysis
- **Vendor Portfolio**: Comprehensive overview of all vendor relationships
- **Detailed Analysis**: Deep-dive into individual vendor performance
- **Comparison Tools**: Side-by-side vendor comparison with multiple metrics
- **Performance Classification**: Automatic categorization of vendors into performance tiers

### Inventory Management
- **Stock Turnover Analysis**: Visual analysis of inventory movement patterns
- **Dead Stock Detection**: Identification of non-moving inventory with value impact
- **Overstock Analysis**: Detection of excess inventory situations
- **Actionable Recommendations**: Specific suggestions for inventory optimization

### Statistical Analysis
- **Correlation Analysis**: Interactive correlation matrix of key business metrics
- **Distribution Analysis**: Statistical distribution analysis with normality tests
- **Hypothesis Testing**: T-tests and ANOVA for vendor performance comparison
- **Advanced Analytics**: Regression analysis, outlier detection, and confidence intervals

## Technology Stack

- **Frontend**: Streamlit
- **Data Processing**: Pandas, NumPy
- **Visualizations**: Plotly, Seaborn, Matplotlib
- **Statistics**: SciPy
- **Deployment**: Streamlit Cloud

## Project Structure

```
├── app.py                          # Main application entry point
├── requirements.txt                # Python dependencies
├── vendor_sales_summary.csv        # Data source
├── pages/
│   ├── 1_Dashboard.py          # Executive dashboard
│   ├── 2_Vendor_Analysis.py    # Vendor performance analysis
│   ├── 3_Inventory_Management.py # Inventory insights
│   └── 4_Statistical_Analysis.py # Statistical analysis
└── README_Streamlit.md            # This file
```

##  Quick Start

### Local Development

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd vendor-performance-analytics
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   streamlit run app.py
   ```

4. **Open your browser**
   Navigate to `http://localhost:8501`

### Streamlit Cloud Deployment

1. **Fork this repository** to your GitHub account

2. **Connect to Streamlit Cloud**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Sign in with GitHub
   - Click "New app"

3. **Configure deployment**
   - Repository: `your-username/vendor-performance-analytics`
   - Branch: `main`
   - Main file path: `app.py`

4. **Deploy**
   - Click "Deploy!"
   - Your app will be available at `https://your-app-name.streamlit.app`

## Data Requirements

The application expects a CSV file named `vendor_sales_summary.csv` with the following columns:

| Column | Description |
|--------|-------------|
| VendorNumber | Unique vendor identifier |
| VendorName | Vendor company name |
| Brand | Product brand identifier |
| Description | Product description |
| PurchasePrice | Cost price per unit |
| ActualPrice | Selling price per unit |
| Volume | Product volume/size |
| TotalPurchaseQuantity | Total units purchased |
| TotalPurchaseDollars | Total purchase cost |
| TotalSalesQuantity | Total units sold |
| TotalSalesDollars | Total sales revenue |
| TotalSalesPrice | Total selling price |
| TotalExciseTax | Total tax amount |
| FreightCost | Shipping and handling costs |
| GrossProfit | Revenue minus costs |
| ProfitMargin | Profit as percentage of revenue |
| StockTurnover | Sales quantity / Purchase quantity |
| SalesToPurchaseRatio | Sales dollars / Purchase dollars |

## Key Metrics & KPIs

### Financial Metrics
- **Total Revenue**: Sum of all sales dollars
- **Gross Profit**: Revenue minus purchase costs
- **Profit Margin**: Profit as percentage of revenue
- **ROI**: Return on investment per vendor/product

### Operational Metrics
- **Stock Turnover**: Inventory movement efficiency
- **Dead Stock**: Products with zero sales
- **Overstock**: Excess inventory situations
- **Vendor Concentration**: Revenue distribution across vendors

### Performance Categories
- **Stars**: High revenue, high margin products
- **Cash Cows**: High revenue, low margin products
- **Question Marks**: Low revenue, high margin products
- **Dogs**: Low revenue, low margin products

## Customization

### Adding New Pages
1. Create a new Python file in the `pages/` directory
2. Follow the naming convention: `N_Page_Name.py`
3. Use the same structure as existing pages

### Modifying Visualizations
- All charts use Plotly for interactivity
- Color schemes can be customized in each page
- Chart types can be easily swapped

### Adding New Metrics
1. Calculate new metrics in the data processing functions
2. Add to the appropriate dashboard sections
3. Update filters and selections as needed

## Business Value

### For Executives
- **Strategic Overview**: High-level KPIs and performance trends
- **Risk Assessment**: Vendor concentration and inventory risks
- **Growth Opportunities**: Identification of high-potential areas

### For Operations Managers
- **Inventory Optimization**: Dead stock and overstock identification
- **Vendor Management**: Performance-based vendor evaluation
- **Cost Reduction**: Identification of inefficient processes

### For Analysts
- **Statistical Insights**: Correlation analysis and hypothesis testing
- **Predictive Analytics**: Trend analysis and forecasting capabilities
- **Data Quality**: Outlier detection and data validation

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

##� Support

If you encounter any issues or have questions:

1. Check the [Issues](https://github.com/your-username/vendor-performance-analytics/issues) page
2. Create a new issue with detailed description
3. Include screenshots and error messages if applicable

## Future Enhancements

- [ ] Real-time data integration
- [ ] Machine learning predictions
- [ ] Advanced forecasting models
- [ ] Export functionality for reports
- [ ] User authentication and role-based access
- [ ] Integration with ERP systems
- [ ] Mobile app version

---
