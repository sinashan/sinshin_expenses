import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os
import requests

# Page configuration
st.set_page_config(
    page_title="Noushin & Sina Expense Tracker",
    page_icon="💰",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
    }
    [data-testid="stMetricLabel"] {
        font-size: 16px !important;
        font-weight: bold !important;
        color: #333333 !important;
    }
    [data-testid="stMetricValue"] {
        font-size: 24px !important;
    }
</style>
""", unsafe_allow_html=True)

# ============ SPLITWISE API CONFIGURATION ============
# Get your API key from: https://secure.splitwise.com/apps
# Store in Streamlit secrets (for cloud) or environment variable (for local)

def get_splitwise_config():
    """Get Splitwise API configuration from secrets or environment"""
    api_key = None
    group_id = None
    
    # Try Streamlit secrets first (for Streamlit Cloud)
    try:
        api_key = st.secrets.get("SPLITWISE_API_KEY", None)
        group_id = st.secrets.get("SPLITWISE_GROUP_ID", None)
    except:
        pass
    
    # Fall back to environment variables
    if not api_key:
        api_key = os.environ.get("SPLITWISE_API_KEY", "")
    if not group_id:
        group_id = os.environ.get("SPLITWISE_GROUP_ID", "")
    
    return api_key, group_id

SPLITWISE_API_KEY, GROUP_ID = get_splitwise_config()

def fetch_splitwise_expenses():
    """Fetch expenses directly from Splitwise API"""
    if not SPLITWISE_API_KEY:
        return None
    
    headers = {"Authorization": f"Bearer {SPLITWISE_API_KEY}"}
    
    all_expenses = []
    offset = 0
    limit = 100
    
    try:
        while True:
            params = {
                "limit": limit,
                "offset": offset,
            }
            if GROUP_ID:
                params["group_id"] = GROUP_ID
                
            response = requests.get(
                "https://secure.splitwise.com/api/v3.0/get_expenses",
                headers=headers,
                params=params,
                timeout=30
            )
            
            if response.status_code != 200:
                st.sidebar.error(f"API Error: {response.status_code}")
                return None
                
            data = response.json()
            expenses = data.get("expenses", [])
            
            if not expenses:
                break
                
            all_expenses.extend(expenses)
            offset += limit
            
            if len(expenses) < limit:
                break
        
        return all_expenses
    except Exception as e:
        st.sidebar.error(f"API Error: {str(e)}")
        return None

def parse_splitwise_data(expenses):
    """Convert Splitwise API response to DataFrame"""
    records = []
    
    for exp in expenses:
        # Skip deleted expenses
        if exp.get("deleted_at"):
            continue
        
        # Skip payments (transfers between users)
        if exp.get("payment", False):
            # Still include payments for balance tracking
            pass
            
        # Find Noushin and Sina in users
        noushin_share = 0
        sina_share = 0
        
        for user in exp.get("users", []):
            first_name = user.get('user', {}).get('first_name', '').lower()
            last_name = user.get('user', {}).get('last_name', '').lower()
            full_name = f"{first_name} {last_name}"
            
            owed = float(user.get("owed_share", 0) or 0)
            paid = float(user.get("paid_share", 0) or 0)
            net = paid - owed  # Positive = they paid more than their share
            
            if "noushin" in first_name or "noushin" in full_name:
                noushin_share = net
            elif "sina" in first_name or "sina" in full_name:
                sina_share = net
        
        # Get category
        category = exp.get("category", {})
        if isinstance(category, dict):
            category_name = category.get("name", "Uncategorized")
        else:
            category_name = "Uncategorized"
        
        records.append({
            "Date": exp.get("date", "")[:10] if exp.get("date") else "",  # Get just the date part
            "Description": exp.get("description", ""),
            "Category": category_name,
            "Cost": float(exp.get("cost", 0) or 0),
            "Currency": exp.get("currency_code", "USD"),
            "noushin haddad": noushin_share,
            "Sina Ahmadi": sina_share,
        })
    
    df = pd.DataFrame(records)
    return df

def process_dataframe(df):
    """Common processing for both API and CSV data"""
    if df.empty:
        return df
    
    # Convert date column
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date'])
    
    # Convert numeric columns
    df['Cost'] = pd.to_numeric(df['Cost'], errors='coerce')
    df['noushin haddad'] = pd.to_numeric(df['noushin haddad'], errors='coerce')
    df['Sina Ahmadi'] = pd.to_numeric(df['Sina Ahmadi'], errors='coerce')
    
    # Add useful columns
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['MonthName'] = df['Date'].dt.strftime('%B')
    df['YearMonth'] = df['Date'].dt.to_period('M').astype(str)
    df['WeekDay'] = df['Date'].dt.day_name()
    
    # Determine who paid
    df['Paid By'] = df.apply(
        lambda row: 'Noushin' if row['noushin haddad'] > 0 else ('Sina' if row['Sina Ahmadi'] > 0 else 'Split'), 
        axis=1
    )
    
    return df

def load_from_csv():
    """Load data from local CSV file"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, "Splitwise expenses Jan 19.csv")
    
    if not os.path.exists(csv_path):
        return None
    
    df = pd.read_csv(csv_path)
    
    # Clean up column names
    df.columns = df.columns.str.strip()
    
    # Remove the total balance row and any empty rows
    df = df[df['Date'].notna() & (df['Date'] != '')]
    df = df[~df['Description'].str.contains('Total balance', case=False, na=False)]
    
    return df

def save_to_csv(df):
    """Save API data to CSV as backup"""
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(script_dir, "Splitwise expenses Jan 19.csv")
        
        # Select and order columns for CSV
        csv_df = df[['Date', 'Description', 'Category', 'Cost', 'Currency', 'noushin haddad', 'Sina Ahmadi']].copy()
        csv_df['Date'] = csv_df['Date'].dt.strftime('%Y-%m-%d')
        csv_df.to_csv(csv_path, index=False)
        return True
    except Exception as e:
        st.sidebar.warning(f"Could not save backup: {str(e)}")
        return False

@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_data(_force_refresh=False):
    """Load expense data - try API first, fall back to CSV"""
    
    data_source = "csv"
    
    # Try to fetch from Splitwise API
    if SPLITWISE_API_KEY:
        expenses = fetch_splitwise_expenses()
        if expenses:
            df = parse_splitwise_data(expenses)
            if len(df) > 0:
                df = process_dataframe(df)
                data_source = "api"
                # Save as backup
                save_to_csv(df)
                return df, data_source
    
    # Fall back to CSV file
    df = load_from_csv()
    if df is not None:
        df = process_dataframe(df)
        return df, data_source
    
    # Return empty dataframe if nothing works
    return pd.DataFrame(), data_source

# Sidebar - Data source info and refresh
st.sidebar.header("📡 Data Source")

# Add a refresh button
if st.sidebar.button("🔄 Refresh Data from Splitwise"):
    st.cache_data.clear()
    st.rerun()

# Load data
df, data_source = load_data()

# Show data source status
if data_source == "api":
    st.sidebar.success("✅ Live data from Splitwise API")
    st.sidebar.caption("Data refreshes every hour automatically")
else:
    if SPLITWISE_API_KEY:
        st.sidebar.warning("⚠️ Using cached CSV (API unavailable)")
    else:
        st.sidebar.info("📁 Using local CSV file")
        st.sidebar.caption("Set up Splitwise API for live data")

# Show last update time
st.sidebar.caption(f"Last loaded: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

st.sidebar.markdown("---")

# Title and header
st.title("� Noushin & Sina Expense Dashboard")
st.markdown("---")

# Sidebar filters
st.sidebar.header("🔍 Filters")

# Date range filter
min_date = df['Date'].min().date()
max_date = df['Date'].max().date()

col_start, col_end = st.sidebar.columns(2)
with col_start:
    start_date = st.date_input(
        "Start Date",
        value=min_date,
        min_value=min_date,
        max_value=max_date
    )
with col_end:
    end_date = st.date_input(
        "End Date",
        value=max_date,
        min_value=min_date,
        max_value=max_date
    )

# Ensure start_date is before end_date
if start_date > end_date:
    st.sidebar.error("Start date must be before end date!")
    start_date, end_date = min_date, max_date

# Category filter
categories = ['All'] + sorted(df['Category'].dropna().unique().tolist())
selected_categories = st.sidebar.multiselect(
    "Select Categories",
    options=categories,
    default=['All']
)

# Paid by filter
paid_by_options = ['All', 'Noushin', 'Sina']
selected_paid_by = st.sidebar.selectbox("Filter by Who Paid", paid_by_options)

# Apply filters
filtered_df = df.copy()

# Convert dates to pandas Timestamp for comparison
start_ts = pd.Timestamp(start_date)
end_ts = pd.Timestamp(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)

filtered_df = filtered_df[(filtered_df['Date'] >= start_ts) & 
                          (filtered_df['Date'] <= end_ts)]

if 'All' not in selected_categories and selected_categories:
    filtered_df = filtered_df[filtered_df['Category'].isin(selected_categories)]

if selected_paid_by != 'All':
    filtered_df = filtered_df[filtered_df['Paid By'] == selected_paid_by]

# Key Metrics Row
st.header("📊 Key Metrics")
col1, col2, col3, col4, col5, col6 = st.columns(6)

with col1:
    total_spent = filtered_df['Cost'].sum()
    st.metric("💰 Total Spent", f"${total_spent:,.2f}")

with col2:
    noushin_paid = filtered_df[filtered_df['noushin haddad'] > 0]['Cost'].sum()
    st.metric("👩 Noushin Paid", f"${noushin_paid:,.2f}")

with col3:
    sina_paid = filtered_df[filtered_df['Sina Ahmadi'] > 0]['Cost'].sum()
    st.metric("👨 Sina Paid", f"${sina_paid:,.2f}")

with col4:
    # Net balance: positive Sina Ahmadi means Noushin owes Sina
    net_balance = filtered_df['Sina Ahmadi'].sum()
    if net_balance > 0:
        st.metric("💸 To Settle Up", f"Noushin → Sina", delta=f"${net_balance:,.2f}", delta_color="off")
    else:
        st.metric("💸 To Settle Up", f"Sina → Noushin", delta=f"${abs(net_balance):,.2f}", delta_color="off")

# Show transaction count separately
st.markdown(f"**📝 {len(filtered_df)} transactions** in selected period")

st.markdown("---")

# Charts Row 1
col1, col2 = st.columns(2)

with col1:
    st.subheader("📈 Spending Over Time")
    monthly_spending = filtered_df.groupby('YearMonth')['Cost'].sum().reset_index()
    monthly_spending = monthly_spending.sort_values('YearMonth')
    
    fig_timeline = px.line(
        monthly_spending, 
        x='YearMonth', 
        y='Cost',
        markers=True,
        title="Monthly Spending Trend"
    )
    fig_timeline.update_layout(xaxis_title="Month", yaxis_title="Amount ($)")
    fig_timeline.update_xaxes(tickangle=45)
    st.plotly_chart(fig_timeline, use_container_width=True)

with col2:
    st.subheader("🏷️ Spending by Category")
    category_spending = filtered_df.groupby('Category')['Cost'].sum().reset_index()
    category_spending = category_spending.sort_values('Cost', ascending=False)
    
    fig_category = px.pie(
        category_spending.head(10), 
        values='Cost', 
        names='Category',
        title="Top 10 Categories",
        hole=0.4
    )
    st.plotly_chart(fig_category, use_container_width=True)

# Charts Row 2
col1, col2 = st.columns(2)

with col1:
    st.subheader("👥 Who Paid More?")
    payment_breakdown = filtered_df.groupby('Paid By')['Cost'].sum().reset_index()
    
    fig_paid = px.bar(
        payment_breakdown,
        x='Paid By',
        y='Cost',
        color='Paid By',
        title="Payment Distribution",
        color_discrete_map={'Noushin': '#FF6B6B', 'Sina': '#4ECDC4', 'Split': '#95E1D3'}
    )
    fig_paid.update_layout(showlegend=False)
    st.plotly_chart(fig_paid, use_container_width=True)

with col2:
    st.subheader("📅 Spending by Day of Week")
    weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    weekday_spending = filtered_df.groupby('WeekDay')['Cost'].sum().reset_index()
    weekday_spending['WeekDay'] = pd.Categorical(weekday_spending['WeekDay'], categories=weekday_order, ordered=True)
    weekday_spending = weekday_spending.sort_values('WeekDay')
    
    fig_weekday = px.bar(
        weekday_spending,
        x='WeekDay',
        y='Cost',
        title="Spending Pattern by Day",
        color='Cost',
        color_continuous_scale='Viridis'
    )
    st.plotly_chart(fig_weekday, use_container_width=True)

st.markdown("---")

# Detailed Category Analysis
st.header("🔎 Category Deep Dive")

col1, col2 = st.columns(2)

with col1:
    # Category breakdown table
    category_stats = filtered_df.groupby('Category').agg({
        'Cost': ['sum', 'mean', 'count']
    }).round(2)
    category_stats.columns = ['Total', 'Average', 'Count']
    category_stats = category_stats.sort_values('Total', ascending=False)
    category_stats['Total'] = category_stats['Total'].apply(lambda x: f"${x:,.2f}")
    category_stats['Average'] = category_stats['Average'].apply(lambda x: f"${x:,.2f}")
    
    st.subheader("Category Statistics")
    st.dataframe(category_stats, use_container_width=True)

with col2:
    # Top expenses
    st.subheader("💸 Top 10 Biggest Expenses")
    top_expenses = filtered_df.nlargest(10, 'Cost')[['Date', 'Description', 'Category', 'Cost', 'Paid By']]
    top_expenses['Date'] = top_expenses['Date'].dt.strftime('%Y-%m-%d')
    top_expenses['Cost'] = top_expenses['Cost'].apply(lambda x: f"${x:,.2f}")
    st.dataframe(top_expenses, use_container_width=True, hide_index=True)

st.markdown("---")

# Balance Analysis
st.header("⚖️ Balance Analysis")

col1, col2, col3 = st.columns(3)

with col1:
    # Running balance over time
    df_sorted = filtered_df.sort_values('Date')
    df_sorted['Running_Balance'] = df_sorted['Sina Ahmadi'].cumsum()
    
    fig_balance = px.line(
        df_sorted,
        x='Date',
        y='Running_Balance',
        title="Running Balance (+ = Noushin owes, - = Sina owes)"
    )
    fig_balance.add_hline(y=0, line_dash="dash", line_color="red")
    st.plotly_chart(fig_balance, use_container_width=True)

with col2:
    # Monthly balance contribution
    monthly_balance = filtered_df.groupby('YearMonth').agg({
        'noushin haddad': 'sum',
        'Sina Ahmadi': 'sum'
    }).reset_index()
    
    fig_monthly_balance = go.Figure()
    fig_monthly_balance.add_trace(go.Bar(
        name='Noushin Net',
        x=monthly_balance['YearMonth'],
        y=monthly_balance['noushin haddad'],
        marker_color='#FF6B6B'
    ))
    fig_monthly_balance.add_trace(go.Bar(
        name='Sina Net',
        x=monthly_balance['YearMonth'],
        y=monthly_balance['Sina Ahmadi'],
        marker_color='#4ECDC4'
    ))
    fig_monthly_balance.update_layout(
        title="Monthly Net Contribution",
        barmode='group',
        xaxis_tickangle=45
    )
    st.plotly_chart(fig_monthly_balance, use_container_width=True)

with col3:
    st.subheader("📋 Summary Stats")
    
    # Total contributions
    total_noushin_contribution = abs(filtered_df[filtered_df['noushin haddad'] > 0]['noushin haddad'].sum())
    total_sina_contribution = abs(filtered_df[filtered_df['Sina Ahmadi'] > 0]['Sina Ahmadi'].sum())
    
    st.write(f"**Noushin's total contribution:** ${total_noushin_contribution:,.2f}")
    st.write(f"**Sina's total contribution:** ${total_sina_contribution:,.2f}")
    
    # Fair share
    fair_share = total_spent / 2
    st.write(f"**Fair share (50/50):** ${fair_share:,.2f} each")
    
    # Who's ahead
    final_balance = filtered_df['Sina Ahmadi'].sum()
    if final_balance > 0:
        st.success(f"✅ Noushin owes Sina: ${final_balance:,.2f}")
    else:
        st.warning(f"⚠️ Sina owes Noushin: ${abs(final_balance):,.2f}")

st.markdown("---")

# Monthly Breakdown
st.header("📅 Monthly Breakdown")

# Year selector
years = sorted(filtered_df['Year'].unique(), reverse=True)
selected_year = st.selectbox("Select Year", years)

yearly_data = filtered_df[filtered_df['Year'] == selected_year]
monthly_summary = yearly_data.groupby('MonthName').agg({
    'Cost': 'sum',
    'Date': 'count'
}).reset_index()
monthly_summary.columns = ['Month', 'Total Spent', 'Transactions']

# Proper month ordering
month_order = ['January', 'February', 'March', 'April', 'May', 'June', 
               'July', 'August', 'September', 'October', 'November', 'December']
monthly_summary['Month'] = pd.Categorical(monthly_summary['Month'], categories=month_order, ordered=True)
monthly_summary = monthly_summary.sort_values('Month')

fig_monthly = px.bar(
    monthly_summary,
    x='Month',
    y='Total Spent',
    title=f"Monthly Spending in {selected_year}",
    text='Total Spent',
    color='Total Spent',
    color_continuous_scale='RdYlGn_r'
)
fig_monthly.update_traces(texttemplate='$%{text:.0f}', textposition='outside')
st.plotly_chart(fig_monthly, use_container_width=True)

st.markdown("---")

# Transaction Search
st.header("🔍 Search Transactions")

search_term = st.text_input("Search by description", "")

if search_term:
    search_results = filtered_df[filtered_df['Description'].str.contains(search_term, case=False, na=False)]
    search_results_display = search_results[['Date', 'Description', 'Category', 'Cost', 'Paid By']].copy()
    search_results_display['Date'] = search_results_display['Date'].dt.strftime('%Y-%m-%d')
    search_results_display['Cost'] = search_results_display['Cost'].apply(lambda x: f"${x:,.2f}")
    
    st.write(f"Found {len(search_results)} transactions matching '{search_term}'")
    st.dataframe(search_results_display, use_container_width=True, hide_index=True)
    
    if len(search_results) > 0:
        total_search = search_results['Cost'].sum()
        st.write(f"**Total for '{search_term}':** ${total_search:,.2f}")

st.markdown("---")

# Full Data Table
with st.expander("📋 View All Transactions"):
    display_df = filtered_df[['Date', 'Description', 'Category', 'Cost', 'Paid By', 'noushin haddad', 'Sina Ahmadi']].copy()
    display_df['Date'] = display_df['Date'].dt.strftime('%Y-%m-%d')
    display_df['Cost'] = display_df['Cost'].apply(lambda x: f"${x:,.2f}")
    st.dataframe(display_df, use_container_width=True, hide_index=True)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #888;'>💕 Expense Tracker for Noushin & Sina | Data up to January 2026</div>",
    unsafe_allow_html=True
)
