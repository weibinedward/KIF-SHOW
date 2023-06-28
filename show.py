import os
import pandas as pd
import streamlit as st

def calculate_bb(df, column='Last', window=20):
    """Calculate Bollinger Bands."""
    rolling_mean = df[column].rolling(window).mean()
    rolling_std = df[column].rolling(window).std()

    df['Upper'] = rolling_mean + (2 * rolling_std)
    df['Middle'] = rolling_mean
    df['Lower'] = rolling_mean - (2 * rolling_std)
    return df

def calculate_macd(df, column='Last', fastperiod=12, slowperiod=26, signalperiod=9):
    """Calculate MACD."""
    df['EMA_Fast'] = df[column].ewm(span=fastperiod).mean()
    df['EMA_Slow'] = df[column].ewm(span=slowperiod).mean()
    df['MACD'] = df['EMA_Fast'] - df['EMA_Slow']
    df['Signal'] = df['MACD'].ewm(span=signalperiod).mean()
    df['Hist'] = df['MACD'] - df['Signal']
    return df

def calculate_adx(df, column_high='High', column_low='Low', column_last='Last', window=14):
    """Calculate ADX."""
    df['TR'] = df[[f'{column_high}', f'{column_last}']].diff().abs().max(axis=1)
    df['TR'] = df[['TR', f'{column_high}']].diff().abs().max(axis=1)
    df['TR'] = df[['TR', f'{column_low}']].diff().abs().max(axis=1)
    df['TR'] = df['TR'].rolling(window).sum()

    df['+DM'] = df[column_high].diff()
    df['+DM'].where(df['+DM'] > 0, 0.0, inplace=True)
    df['-DM'] = -df[column_low].diff()
    df['-DM'].where(df['-DM'] > 0, 0.0, inplace=True)

    df['+DI'] = (df['+DM'] / df['TR']).rolling(window).mean()
    df['-DI'] = (df['-DM'] / df['TR']).rolling(window).mean()

    df['DX'] = (df['+DI'] - df['-DI']).abs() / (df['+DI'] + df['-DI'])
    df['DX'].fillna(0, inplace=True)
    df['DX'] = df['DX'].rolling(window).mean()

    df['ADX'] = df['DX'].rolling(window).mean()
    return df

def generate_technical_analysis_report(stock):
    # Read CSV data
    df = pd.read_csv(f"data/{stock.lower()}.csv")  # The CSV file name is still lowercase

    # Remove the last row
    df = df[:-1]
    df = df.iloc[::-1]

    # Calculate indicators
    df = calculate_bb(df)
    df = calculate_macd(df)
    df = calculate_adx(df)

    # Generate technical analysis report
    report = f"Technical Analysis Report for Stock: {stock}\n\n"

    # Volume-Price Analysis
    if df['Last'].diff().fillna(0).gt(0).rolling(window=3).sum().iloc[-1] >= 3 and \
            df['Volume'].diff().fillna(0).gt(0).rolling(window=3).sum().iloc[-1] >= 3:
        report += "Volume-Price Analysis: Confirmation of an uptrend. The last price has increased for 3 consecutive days with increasing volume.\n"
    elif df['Last'].diff().fillna(0).lt(0).rolling(window=3).sum().iloc[-1] <= -3 and \
            df['Volume'].diff().fillna(0).gt(0).rolling(window=3).sum().iloc[-1] >= 3:
        report += "Volume-Price Analysis: Confirmation of a downtrend. The last price has decreased for 3 consecutive days with increasing volume.\n"
    elif df['Last'].diff().fillna(0).gt(0).rolling(window=3).sum().iloc[-1] <= 1 and \
            df['Volume'].diff().fillna(0).gt(0).rolling(window=3).sum().iloc[-1] >= 3:
        report += "Volume-Price Analysis: Observation of reversal points. The last price has mostly decreased in the last 3 days, but the volume has increased.\n"
    else:
        report += "Volume-Price Analysis: No clear volume-price relationship observed.\n"

    # Volatility Analysis
    upper_band = df['Upper'].values[-1]
    middle_band = df['Middle'].values[-1]
    lower_band = df['Lower'].values[-1]
    if df['Last'].values[-1] > upper_band:
        report += "Volatility Analysis: The stock price has touched or crossed the upper Bollinger Band, indicating that the stock might be overbought.\n"
    elif df['Last'].values[-1] < lower_band:
        report += "Volatility Analysis: The stock price has touched or crossed the lower Bollinger Band, indicating that the stock might be oversold.\n"
    else:
        report += "Volatility Analysis: The stock price is within the Bollinger Bands, indicating normal trading range.\n"

    # Trend Analysis
    ## MACD Analysis
    macd_line = df['MACD'].values[-1]
    signal_line = df['Signal'].values[-1]
    if macd_line > signal_line:
        report += "Trend Analysis (MACD): The MACD line is above the signal line, indicating a bullish signal.\n"
    elif macd_line < signal_line:
        report += "Trend Analysis (MACD): The MACD line is below the signal line, indicating a bearish signal.\n"
    else:
        report += "Trend Analysis (MACD): The MACD line is crossing the signal line, indicating a possible trend change.\n"

    ## ADX Analysis
    adx = df['ADX'].values[-1]
    di_plus = df['+DI'].values[-1]
    di_minus = df['-DI'].values[-1]
    if adx > 25:
        report += "Trend Analysis (ADX): The ADX value is above 25, indicating a strong trend in the market.\n"
        if di_plus > di_minus:
            report += "Moreover, the +DI is above -DI, indicating the trend is more likely to be upwards.\n"
        else:
            report += "Moreover, the -DI is above +DI, indicating the trend is more likely to be downwards.\n"
    else:
        report += "Trend Analysis (ADX): The ADX value is below 25, indicating a weak or no trend in the market.\n"

    return report
@st.cache_data(show_spinner=False)
def load_chart_html(stock):
    # Set the path for the chart HTML file
    chart_file = os.path.join(f"charts/{stock.upper()}output.html")

    # Check if the chart file exists
    if os.path.isfile(chart_file):
        # Read the chart HTML file
        with open(chart_file, 'r') as f:
            chart_html = f.read()

        return chart_html
    else:
        return None


import streamlit.components.v1 as components


def display_chart(stock):
    chart_file = os.path.join(f"charts/{stock.upper()}output.html")

    if os.path.isfile(chart_file):
        with open(chart_file, 'r') as f:
            chart_html = f.read()

        with st.expander("Chart"):
            components.html(chart_html, height=1600)
    else:
        st.write(f"Chart for stock {stock} not found.")


def main():
    # 将页面底色设置为白色
    st.set_page_config(layout="wide", page_title="Technical Analysis Reports", page_icon="📈")
    st.markdown("## KING’S INVESTMENT FUND FINAL REPORT")
    st.sidebar.markdown("**Group 15**")
    st.sidebar.text("To Be Number One Team")
    st.sidebar.markdown("**Author**")
    st.sidebar.text("Weibin Feng (weibin.feng@kcl.ac.uk)")

    st.title("Technical Analysis Reports")
    st.sidebar.title("Select a Stock")

    # Define the list of stocks
    stock_list = ['0htp.ln', '0p2w.ln', '0qb8.ln', '0qqo.ln', '0qzd.ln',
                  '0r0x.ln', '0r2y.ln', 'air.fp', 'amz.d.dx', 'azn.ln',
                  'bbva.e.dx', 'bmw.d.dx', 'iii.ln', 'itpg.ln', 'stmmi.m.dx',
                  'volvb.s.dx', 'wtb.ln', 'ANTO', 'BYIT', 'CCL', 'HLMA',
                  'HYUD', 'MKS', 'OXIG', 'RMS', 'STAN', 'AI.P.DX']
    # Create a drop-down list in the sidebar to select a stock
    selected_stock = st.sidebar.selectbox('Choose a stock for analysis', [stock.upper() for stock in stock_list])

    if st.button('Generate Report'):
        with st.spinner('Generating Report...'):
            report = generate_technical_analysis_report(selected_stock)
            st.write(report)

        # Call the display_chart function
        display_chart(selected_stock)


if __name__ == "__main__":
    main()
