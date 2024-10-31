import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from datetime import datetime, timedelta
import requests
import matplotlib.dates as mdates


# Fetch the S&P 500 stock symbols from Wikipedia
@st.cache_data  # Cache to avoid re-downloading data
def load_sp500_symbols():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    html = requests.get(url).text
    sp500 = pd.read_html(html)[0]  # Wikipedia table is the first in the list
    return sp500['Symbol'].tolist(), sp500[['Symbol', 'Security']]

# Load S&P 500 symbols and names
symbols, company_data = load_sp500_symbols()

# Define dashboard layout
st.sidebar.title("📈 Arajem Aboudi - Financial Dashboard📉")

st.sidebar.subheader("Make your selection")
stock_symbol = st.sidebar.selectbox("Select a stock", symbols)
update_button = st.sidebar.button("Update Data")

# Display selected stock name
company_name = company_data[company_data['Symbol'] == stock_symbol]['Security'].values[0]
st.sidebar.write(f"**Selected Company:** {company_name}")

# Load stock data for the selected symbol
stock = yf.Ticker(stock_symbol)

# Create separate tabs for each section
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Summary", "Chart", "Financials", "Monte Carlo Simulation", "My Own Analysis"])

# Summary tab
with tab1:
    st.subheader("Stock Summary")
    
    # Display basic company info
    info = stock.info
    st.write(f"**Company:** {info.get('longName', 'N/A')}")
    st.write(f"**Sector:** {info.get('sector', 'N/A')}")
    st.write(f"**Industry:** {info.get('industry', 'N/A')}")
    st.write(f"**Market Cap:** {info.get('marketCap', 'N/A'):,}")
    st.write(f"**Summary:** {info.get('longBusinessSummary', 'N/A')}")
    
    # Display major shareholders
    st.write("### Major Shareholders")
    shareholders = stock.major_holders
    st.write(shareholders)

# Chart tab
# Chart tab with additional features
with tab2:
    st.subheader("Stock Price Chart")

    # Options for date range
    date_ranges = {
        "1M": timedelta(days=30),
        "3M": timedelta(days=90),
        "6M": timedelta(days=180),
        "YTD": (datetime.now() - datetime(datetime.now().year, 1, 1)).days,
        "1Y": timedelta(days=365),
        "3Y": timedelta(days=3 * 365),
        "5Y": timedelta(days=5 * 365)
    }

    # Select date range and interval
    date_range = st.selectbox("Select Date Range", list(date_ranges.keys()))
    start_date = datetime.now() - date_ranges[date_range] if date_range != "MAX" else None
    end_date = datetime.now()

    interval = st.selectbox("Select Time Interval", ["1d", "1mo", "1y"], index=0)  # Default to "Day"
    chart_type = st.selectbox("Select Chart Type", ["Line", "Candlestick"], index=0)

    # Fetch historical data
    data = stock.history(start=start_date, end=end_date, interval=interval)

    # Calculate the Simple Moving Average (50 days) and add it to the data if it's daily data
    if interval == "1d":
        data["SMA_50"] = data["Close"].rolling(window=50).mean()

    # Plot with Plotly
    fig = go.Figure()

    if chart_type == "Line":
        fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Close Price'))
    else:
        fig.add_trace(go.Candlestick(
            x=data.index,
            open=data['Open'], high=data['High'],
            low=data['Low'], close=data['Close'],
            name="Candlestick"
        ))

    # Add Simple Moving Average (SMA) if interval is daily
    if interval == "1d":
        fig.add_trace(go.Scatter(
            x=data.index, y=data["SMA_50"],
            mode="lines", name="50-Day SMA",
            line=dict(color='orange', width=1.5)
        ))

    # Add trading volume as a bar chart below
    fig.add_trace(go.Bar(
        x=data.index, y=data['Volume'],
        name='Volume', marker=dict(color='blue'),
        opacity=0.3, yaxis="y2"
    ))

    # Set up layout to display volume on a separate y-axis
    fig.update_layout(
        height=600,
        yaxis=dict(title="Price", showgrid=True),
        yaxis2=dict(title="Volume", overlaying="y", side="right", showgrid=False, range=[0, data['Volume'].max()*4]),
        xaxis=dict(title="Date", showgrid=True),
        title=f"{stock_symbol} Price Chart ({date_range} - Interval: {interval})"
    )

    st.plotly_chart(fig)


# Financials tab
with tab3:
    st.subheader("Financial Statements")
    
    statement_type = st.selectbox("Statement Type", ["Income Statement", "Balance Sheet", "Cash Flow"])
    period = st.selectbox("Period", ["Annual", "Quarterly"])
    
    if statement_type == "Income Statement":
        st.write(stock.financials if period == "Annual" else stock.quarterly_financials)
    elif statement_type == "Balance Sheet":
        st.write(stock.balance_sheet if period == "Annual" else stock.quarterly_balance_sheet)
    else:
        st.write(stock.cashflow if period == "Annual" else stock.quarterly_cashflow)

# Monte Carlo Simulation tab
with tab4:
    st.subheader("Monte Carlo Simulation for Future Stock Prices")

    # Number of simulations and time horizon
    n_simulations = st.selectbox("Number of Simulations", [200, 500, 1000])
    time_horizon = st.selectbox("Time Horizon (days)", [30, 60, 90])
    
    # Historical returns for the simulation
    daily_returns = data['Close'].pct_change().dropna()
    mean_return = daily_returns.mean()
    std_dev = daily_returns.std()
    
    # Initialize Monte Carlo simulation
    simulations = np.zeros((time_horizon, n_simulations))
    last_price = data['Close'][-1]
    
    for i in range(n_simulations):
        price = last_price
        for t in range(time_horizon):
            price *= (1 + np.random.normal(mean_return, std_dev))
            simulations[t, i] = price
    
    # Plot simulation results
    plt.figure(figsize=(10, 6))
    plt.plot(simulations)
    plt.title(f"{n_simulations} Monte Carlo Simulations for {stock_symbol} over {time_horizon} Days")
    plt.xlabel("Days")
    plt.ylabel("Price")
    st.pyplot(plt)

    # Calculate and display Value at Risk (VaR)
    VaR_95 = np.percentile(simulations[-1], 5)
    st.write(f"Value at Risk (VaR) at 95% confidence interval: ${VaR_95:.2f}")

# Additional imports
from collections import defaultdict


# Add the new "Your Own Analysis" tab
with tab5:
    st.subheader("Your Own Analysis")

# Create columns for the two tables
    col1, col2 = st.columns(2)

    # Key Financial Metrics
    with col1:
        st.write("### Key Financial Metrics")
        metrics = {
            "Revenue": stock.financials.loc['Total Revenue'].sum(),
            "Net Income": stock.financials.loc['Net Income'].sum(),
            "EPS": stock.info.get('trailingEps', 'N/A'),
            "P/E Ratio": stock.info.get('trailingPE', 'N/A'),
            "Debt-to-Equity Ratio": stock.info.get('debtToEquity', 'N/A'),
            "Return on Equity (ROE)": f"{stock.info.get('returnOnEquity', 'N/A') * 100:.2f}%",
            "Dividend Yield": f"{stock.info.get('dividendYield', 'N/A') * 100:.2f}%"
        }

        # Display the metrics in a table
        metrics_df = pd.DataFrame(list(metrics.items()), columns=['Metric', 'Value'])
        st.table(metrics_df)

    # Stock Performance
    with col2:
        st.write("### Stock Performance")
        performance_data = {
            "1-Year Price Change (%)": ((data['Close'][-1] - data['Close'][0]) / data['Close'][0]) * 100,
            "52-Week High": data['Close'].max(),
            "52-Week Low": data['Close'].min(),
            "Average Trading Volume": data['Volume'].mean(),
            "Beta": stock.info.get('beta', 'N/A')
        }

        # Display the performance data in a table
        performance_df = pd.DataFrame(list(performance_data.items()), columns=['Metric', 'Value'])
        st.table(performance_df)

