# LSTM Trading Strategy Demonstrator

A simple web application built with Python and Streamlit to demonstrate a basic trading strategy using Long Short-Term Memory (LSTM) networks for stock price prediction.

## Features

*   Select a stock ticker symbol (e.g., AAPL, GOOG).
*   Choose a historical date range for analysis.
*   Adjust LSTM model parameters (lookback period, units, epochs, batch size).
*   Configure simple strategy thresholds (buy/sell based on predicted change).
*   Fetches historical stock data using `yfinance`.
*   Preprocesses data (scaling, sequence creation).
*   Builds and trains an LSTM model to predict the next day's closing price.
*   Generates BUY/SELL signals based on the model's predictions compared to the previous day's close.
*   Estimates simplified Profit & Loss (P&L) based on executing these signals (long-only, no costs).
*   Visualizes:
    *   Historical actual prices.
    *   Predicted prices (on the test set).
    *   Buy/Sell signal markers on the price chart (using Streamlit's native chart).
    *   Estimated portfolio value over the test period.
*   Displays key metrics (P&L, Return %, Signal Counts).

## ⚠️ Disclaimer ⚠️

This application is purely for **educational and demonstration purposes only**. The trading strategy implemented is highly simplified and **does not account for real-world complexities** such as:

*   Transaction Costs (Brokerage Fees)
*   Slippage (Difference between expected and actual execution price)
*   Market Impact of large trades
*   Taxes
*   Accurate Trade Execution Models (e.g., using next day's open instead of signal day's close)
*   Risk Management Rules
*   Short Selling logic
*   Robust Feature Engineering
*   Advanced Model Tuning and Validation

**DO NOT use this application or its signals for actual trading decisions without significant further development, testing, and validation.** Financial markets are complex, and trading involves substantial risk of loss.

## Setup

### Prerequisites

*   Python 3.8+
*   pip (Python package installer)

### Installation

1.  **Clone the repository (Optional):**
    If you have the code in a repository:
    ```bash
    git clone https://github.com/manyan-chan/LSTM-Trading-Strategy-Demo.git
    cd LSTM-Trading-Strategy-Demo
    ```
    If you only have the `app.py` and `requirements.txt` files, just place them in a new project directory and navigate there in your terminal.

2.  **Create a Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    # On Windows
    venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Running the App

Once the dependencies are installed, run the Streamlit application from your terminal:

```bash
streamlit run app.py
```

Your web browser should automatically open to the application's local URL (usually `http://localhost:8501`).

## How It Works

1.  **Data Loading:** Fetches historical stock data (`Open`, `High`, `Low`, `Close`, `Volume`, `Adj Close`) for the selected ticker and date range using `yfinance`.
2.  **Preprocessing:**
    *   Selects the 'Close' price.
    *   Scales the data to a range (0, 1) using `MinMaxScaler`.
    *   Creates input sequences (`X`) of length `lookback` and corresponding target values (`y`) which are the next day's price.
    *   Splits data into training and testing sets.
3.  **LSTM Model:**
    *   Builds a sequential Keras model with LSTM layers.
    *   Trains the model on the training sequences to predict the next scaled price.
4.  **Prediction:**
    *   Uses the trained model to predict prices on the test set sequences.
    *   Inverse transforms the scaled predictions back to actual price values.
5.  **Signal Generation:**
    *   Compares the predicted price for day `i` with the *actual* closing price of day `i-1`.
    *   Generates a BUY signal (1) if `predicted > actual[i-1] * buy_threshold`.
    *   Generates a SELL signal (-1) if `predicted < actual[i-1] * sell_threshold`.
    *   Generates a HOLD signal (0) otherwise.
6.  **P&L Estimation:**
    *   Simulates executing trades based on the signals, starting with a defined initial capital.
    *   Assumes trades occur at the closing price on the signal day.
    *   Calculates the running portfolio value and final P&L.
7.  **Visualization:**
    *   Uses `st.line_chart` to display actual prices, predicted prices, and buy/sell markers.
    *   Displays the portfolio value history.
    *   Shows summary metrics and a table of recent signals.

## Libraries Used

*   [Streamlit](https://streamlit.io/): For creating the web application interface.
*   [Pandas](https://pandas.pydata.org/): For data manipulation and analysis.
*   [NumPy](https://numpy.org/): For numerical operations.
*   [yfinance](https://github.com/ranaroussi/yfinance): For downloading historical stock market data.
*   [Scikit-learn](https://scikit-learn.org/): For data preprocessing (`MinMaxScaler`).
*   [TensorFlow](https://www.tensorflow.org/) / [Keras](https://keras.io/): For building and training the LSTM model.