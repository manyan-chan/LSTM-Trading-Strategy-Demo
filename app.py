from datetime import date, timedelta

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential

st.set_page_config(page_title="LSTM Trading Strategy Demo", layout="wide")

st.title("📈 LSTM Trading Strategy Demonstrator")
st.caption("Demonstrates using LSTM predictions to generate simple trading signals.")


DEFAULT_TICKER = "AAPL"
DEFAULT_START_DATE = date.today() - timedelta(days=3 * 365)
DEFAULT_END_DATE = date.today()
LOOKBACK_PERIOD = 60
TRAIN_SPLIT_RATIO = 0.8
LSTM_UNITS = 50
EPOCHS = 20
BATCH_SIZE = 32
BUY_THRESHOLD = 1.01
SELL_THRESHOLD = 0.99


@st.cache_data
def load_data(ticker, start_date, end_date):
    """Fetches stock data from Yahoo Finance and flattens columns."""
    try:
        data = yf.download(ticker, start=start_date, end=end_date)
        if data.empty:
            st.error(
                f"Could not fetch data for {ticker}. Check the ticker symbol or date range."
            )
            return None

        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
            data = data.loc[:, ~data.columns.duplicated()]

        return data
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None


def preprocess_data(data, lookback):
    if data is None or len(data) < lookback:
        return None, None, None, None, None, None

    scaler = MinMaxScaler(feature_range=(0, 1))
    # Use 'Close' which should now be a simple column name
    scaled_data = scaler.fit_transform(data["Close"].values.reshape(-1, 1))

    X, y = [], []
    for i in range(lookback, len(scaled_data)):
        X.append(scaled_data[i - lookback : i, 0])
        y.append(scaled_data[i, 0])

    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    train_size = int(len(X) * TRAIN_SPLIT_RATIO)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    return X_train, y_train, X_test, y_test, scaler, train_size


def build_and_train_model(X_train, y_train, lstm_units, epochs, batch_size):
    model = Sequential(
        [
            LSTM(
                units=lstm_units,
                return_sequences=True,
                input_shape=(X_train.shape[1], 1),
            ),
            Dropout(0.2),
            LSTM(units=lstm_units, return_sequences=False),
            Dropout(0.2),
            Dense(units=25),
            Dense(units=1),
        ]
    )

    model.compile(optimizer="adam", loss="mean_squared_error")

    with st.spinner(
        f"Training LSTM model for {epochs} epochs... This may take a while."
    ):
        history = model.fit(
            X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0
        )
    st.success("Model training complete!")
    return model


def generate_signals(
    data, predictions, actual_test_prices, lookback, train_size, buy_thresh, sell_thresh
):
    try:
        predictions_np = np.array(predictions).flatten()
        actual_test_prices_np = np.array(actual_test_prices).flatten()
    except Exception as e:
        st.error(
            f"Error converting predictions/actual_prices to flattened numpy arrays: {e}"
        )
        return pd.DataFrame(
            columns=["Actual Close", "Predicted Close", "Signal", "Buy", "Sell"]
        )

    test_data_index = data.index[train_size + lookback :]
    expected_len = len(test_data_index)

    if len(predictions_np) != expected_len:
        st.error(
            f"Length Mismatch Error: Test index ({expected_len}) vs Predictions ({len(predictions_np)}). Cannot generate signals reliably."
        )
        if len(predictions_np) > expected_len:
            st.warning(
                f"Truncating predictions array from {len(predictions_np)} to {expected_len} to match test index."
            )
            predictions_np = predictions_np[:expected_len]
        else:
            st.error(
                "Predictions array is shorter than the test index. Aborting signal generation."
            )
            return pd.DataFrame(
                columns=["Actual Close", "Predicted Close", "Signal", "Buy", "Sell"]
            )

    if len(actual_test_prices_np) != expected_len:
        st.warning(
            f"Length Mismatch Warning: Test index ({expected_len}) vs Actual Prices ({len(actual_test_prices_np)}). Trimming/Padding may occur."
        )
        if len(actual_test_prices_np) > expected_len:
            actual_test_prices_np = actual_test_prices_np[:expected_len]
        elif len(actual_test_prices_np) < expected_len:
            st.error(
                f"Actual test prices array ({len(actual_test_prices_np)}) is shorter than expected ({expected_len}). Aborting."
            )
            return pd.DataFrame(
                columns=["Actual Close", "Predicted Close", "Signal", "Buy", "Sell"]
            )

    signals = pd.DataFrame(index=test_data_index)
    signals["Actual Close"] = actual_test_prices_np
    signals["Predicted Close"] = predictions_np
    signals["Signal"] = 0

    prev_close_start_idx = train_size + lookback - 1
    prev_close_end_idx = prev_close_start_idx + expected_len

    if prev_close_start_idx < 0 or prev_close_end_idx > len(data):
        st.error(
            "Index Error: Calculated previous close indices are out of bounds for the original data."
        )
        return pd.DataFrame(
            columns=["Actual Close", "Predicted Close", "Signal", "Buy", "Sell"]
        )

    try:
        previous_closes_np = (
            data["Close"].values[prev_close_start_idx:prev_close_end_idx].flatten()
        )
    except KeyError:
        st.error(
            "KeyError: Failed to find 'Close' column in data for previous_closes calculation. Check data loading."
        )
        return pd.DataFrame(
            columns=["Actual Close", "Predicted Close", "Signal", "Buy", "Sell"]
        )
    except Exception as e:
        st.error(f"Error extracting or flattening previous_closes: {e}")
        return pd.DataFrame(
            columns=["Actual Close", "Predicted Close", "Signal", "Buy", "Sell"]
        )

    if len(previous_closes_np) != expected_len:
        st.error(
            f"Length Mismatch Error: Expected ({expected_len}) vs Previous Closes ({len(previous_closes_np)}). Aborting."
        )
        return pd.DataFrame(
            columns=["Actual Close", "Predicted Close", "Signal", "Buy", "Sell"]
        )

    try:
        predicted_change_ratio = np.full(expected_len, np.nan, dtype=np.float64)
        safe_division_mask = (previous_closes_np != 0) & (~np.isnan(previous_closes_np))

        if len(predictions_np) == len(previous_closes_np):
            predicted_change_ratio[safe_division_mask] = (
                predictions_np[safe_division_mask]
                / previous_closes_np[safe_division_mask]
            )
        else:
            st.error(
                "Internal Length Mismatch before division. Aborting signal calculation."
            )
            return signals

        buy_mask = (predicted_change_ratio > buy_thresh) & (
            ~np.isnan(predicted_change_ratio)
        )
        sell_mask = (predicted_change_ratio < sell_thresh) & (
            ~np.isnan(predicted_change_ratio)
        )

        if len(buy_mask) == len(signals.index):
            signals.loc[buy_mask, "Signal"] = 1
        else:
            st.warning(
                f"Buy mask length ({len(buy_mask)}) mismatch with signals index ({len(signals.index)}). Skipping buy signal assignment."
            )

        if len(sell_mask) == len(signals.index):
            signals.loc[sell_mask, "Signal"] = -1
        else:
            st.warning(
                f"Sell mask length ({len(sell_mask)}) mismatch with signals index ({len(signals.index)}). Skipping sell signal assignment."
            )

    except IndexError as e:
        st.error(f"Caught IndexError during signal calculation/assignment: {e}")
        return signals
    except Exception as e:
        st.error(
            f"Caught unexpected error during signal calculation: {type(e).__name__} - {e}"
        )
        return pd.DataFrame(
            columns=["Actual Close", "Predicted Close", "Signal", "Buy", "Sell"]
        )

    signals["Buy"] = np.nan
    signals["Sell"] = np.nan
    signals.loc[signals["Signal"] == 1, "Buy"] = signals.loc[
        signals["Signal"] == 1, "Actual Close"
    ]
    signals.loc[signals["Signal"] == -1, "Sell"] = signals.loc[
        signals["Signal"] == -1, "Actual Close"
    ]

    return signals


def calculate_pnl(signals, initial_capital=10000):
    """Calculates simplified Profit/Loss based on trading signals."""
    cash = initial_capital
    shares_held = 0
    portfolio_value = initial_capital
    position_value = 0
    last_price = 0

    portfolio_history = pd.DataFrame(index=signals.index)
    portfolio_history["Value"] = initial_capital

    for index, row in signals.iterrows():
        current_price = row["Actual Close"]
        signal = row["Signal"]

        # Check if current_price is valid before proceeding
        if pd.isna(current_price):
            portfolio_history.loc[index, "Value"] = (
                portfolio_value  # Carry forward last value if price is NaN
            )
            continue  # Skip trading logic for this day

        if shares_held > 0:
            position_value = shares_held * current_price
        else:
            position_value = 0

        portfolio_value = cash + position_value
        portfolio_history.loc[index, "Value"] = portfolio_value

        if signal == 1 and cash > 0:
            shares_to_buy = cash / current_price
            shares_held += shares_to_buy
            cash = 0

        elif signal == -1 and shares_held > 0:
            cash += shares_held * current_price
            shares_held = 0

        if not pd.isna(current_price):
            last_price = current_price

    if shares_held > 0 and last_price > 0:
        final_portfolio_value = cash + (shares_held * last_price)
        if not portfolio_history.empty:
            # Make sure the index exists before trying to assign
            last_valid_index = portfolio_history.index[-1]
            portfolio_history.loc[last_valid_index, "Value"] = final_portfolio_value
    else:
        final_portfolio_value = cash

    total_profit = final_portfolio_value - initial_capital
    total_return_pct = (
        (total_profit / initial_capital) * 100 if initial_capital > 0 else 0
    )

    return total_profit, total_return_pct, final_portfolio_value, portfolio_history


# --- Sidebar Inputs ---
st.sidebar.header("Parameters")

if "ticker" not in st.session_state:
    st.session_state.ticker = DEFAULT_TICKER

st.session_state.ticker = st.sidebar.text_input(
    "Stock Ticker", value=st.session_state.ticker, key="ticker_input"
).upper()
start_date = st.sidebar.date_input("Start Date", value=DEFAULT_START_DATE)
end_date = st.sidebar.date_input("End Date", value=DEFAULT_END_DATE)

st.sidebar.subheader("LSTM Settings")
lookback = st.sidebar.slider("Lookback Period (Days)", 10, 120, LOOKBACK_PERIOD, 5)
lstm_units = st.sidebar.slider("LSTM Units per Layer", 10, 100, LSTM_UNITS, 5)
epochs = st.sidebar.slider("Training Epochs", 5, 100, EPOCHS, 5)
batch_size = st.sidebar.select_slider(
    "Batch Size", options=[16, 32, 64], value=BATCH_SIZE
)

st.sidebar.subheader("Strategy Settings")
buy_thresh_pct = st.sidebar.slider(
    "Buy Threshold (%)", 0.1, 5.0, (BUY_THRESHOLD * 100 - 100), 0.1
)
sell_thresh_pct = st.sidebar.slider(
    "Sell Threshold (%)", 0.1, 5.0, (100 - SELL_THRESHOLD * 100), 0.1
)

buy_thresh = 1 + (buy_thresh_pct / 100)
sell_thresh = 1 - (sell_thresh_pct / 100)

st.sidebar.caption(f"Buy if predicted > {buy_thresh:.2%} of prev close.")
st.sidebar.caption(f"Sell if predicted < {sell_thresh:.2%} of prev close.")

st.sidebar.subheader("P&L Settings")
initial_capital_input = st.sidebar.number_input(
    "Initial Capital ($)", min_value=100, value=10000, step=1000
)


# --- Main Execution Logic ---
if st.sidebar.button("Run Analysis", type="primary"):
    if not st.session_state.ticker:
        st.warning("Please enter a stock ticker.")
    elif start_date >= end_date:
        st.warning("Start Date must be before End Date.")
    else:
        st.markdown("### 1. Loading Data...")
        data = load_data(st.session_state.ticker, start_date, end_date)

        if data is not None:
            st.dataframe(data.tail(), use_container_width=True)

            st.markdown("### 2. Preprocessing Data...")
            X_train, y_train, X_test, y_test, scaler, train_size = preprocess_data(
                data, lookback
            )

            if (
                X_train is None
                or y_train is None
                or X_test is None
                or y_test is None
                or scaler is None
            ):
                st.warning(
                    f"Preprocessing failed. Not enough data ({len(data)} days?) for the selected lookback period ({lookback} days), or other issue."
                )
            else:
                st.text(f"Training data shape: {X_train.shape}")
                st.text(f"Testing data shape: {X_test.shape}")

                if X_test.shape[0] == 0 or y_test.shape[0] == 0:
                    st.error(
                        "Test data is empty after preprocessing. Cannot train/predict. Adjust date range or split ratio."
                    )
                else:
                    st.markdown("### 3. Building & Training LSTM Model...")
                    model = build_and_train_model(
                        X_train, y_train, lstm_units, epochs, batch_size
                    )

                    st.markdown("### 4. Making Predictions...")
                    with st.spinner("Generating predictions on the test set..."):
                        predictions_scaled = model.predict(X_test)
                        predictions = scaler.inverse_transform(predictions_scaled)
                        actual_test_prices = scaler.inverse_transform(
                            y_test.reshape(-1, 1)
                        )
                        predictions = predictions.flatten()
                        actual_test_prices = actual_test_prices.flatten()

                    st.markdown("### 5. Generating Trading Signals...")
                    signals = generate_signals(
                        data,
                        predictions,
                        actual_test_prices,
                        lookback,
                        train_size,
                        buy_thresh,
                        sell_thresh,
                    )

                    st.markdown("### 6. Visualizing Results (Using st.line_chart)")
                    if signals is not None and not signals.empty:
                        if "Close" in data.columns:
                            plot_data = data[["Close"]].copy()
                            plot_data.rename(
                                columns={"Close": "Actual Price"}, inplace=True
                            )
                            plot_data["Predicted Price"] = signals["Predicted Close"]
                            plot_data["Buy Signal Price"] = signals["Buy"]
                            plot_data["Sell Signal Price"] = signals["Sell"]

                            st.line_chart(plot_data)

                            train_split_index_calc = train_size + lookback - 1
                            if (
                                train_split_index_calc >= 0
                                and train_split_index_calc < len(data.index)
                            ):
                                train_end_date = data.index[train_split_index_calc]
                                st.caption(
                                    f"Train/Test split occurs around {train_end_date.strftime('%Y-%m-%d')}. Predictions and signals apply after this date."
                                )
                            else:
                                st.caption(
                                    "Train/Test split indicator could not be calculated (index out of bounds)."
                                )
                        else:
                            st.error(
                                "Could not find 'Close' column in the loaded data to create the plot."
                            )

                        st.subheader("Generated Signals (Test Set)")
                        display_signals = signals[
                            ["Actual Close", "Predicted Close", "Signal"]
                        ].copy()
                        display_signals["Signal"] = display_signals["Signal"].map(
                            {1: "BUY", -1: "SELL", 0: "HOLD"}
                        )
                        st.dataframe(display_signals.tail(20), use_container_width=True)

                        buy_count = (signals["Signal"] == 1).sum()
                        sell_count = (signals["Signal"] == -1).sum()

                        st.markdown("### 7. Profit/Loss Estimation (Simplified)")
                        if not signals["Actual Close"].isnull().all():
                            (
                                total_profit,
                                total_return_pct,
                                final_value,
                                portfolio_history,
                            ) = calculate_pnl(signals, initial_capital_input)

                            col1, col2, col3 = st.columns(3)
                            col1.metric("Est. Total P&L ($)", f"{total_profit:,.2f}")
                            col2.metric(
                                "Est. Total Return (%)", f"{total_return_pct:.2f}%"
                            )
                            col3.metric(
                                "Final Portfolio Value ($)", f"{final_value:,.2f}"
                            )

                            st.subheader("Portfolio Value Over Time (Test Period)")
                            if not portfolio_history.empty:
                                # Ensure the history index is compatible with line_chart
                                if pd.api.types.is_datetime64_any_dtype(
                                    portfolio_history.index
                                ):
                                    st.line_chart(portfolio_history["Value"])
                                else:
                                    st.warning(
                                        "Portfolio history index is not datetime. Cannot plot timeline."
                                    )
                                    st.dataframe(portfolio_history.tail())
                            else:
                                st.warning("Could not generate portfolio history.")

                            st.caption(
                                "Note: This is a simplified simulation assuming trades execute at the closing price on the signal day, with no costs/slippage, investing full available cash on buys."
                            )

                        else:
                            st.warning(
                                "Cannot calculate P&L because 'Actual Close' prices are missing or all NaN in the signals data."
                            )

                        st.metric("Buy Signals Generated", f"{buy_count}")
                        st.metric("Sell Signals Generated", f"{sell_count}")

                    else:
                        st.warning(
                            "Cannot visualize results or calculate P&L because the 'signals' DataFrame is empty or None."
                        )
        else:
            st.error(
                f"Failed to load data for {st.session_state.ticker}. Cannot proceed."
            )
else:
    st.info("Adjust the parameters in the sidebar and click 'Run Analysis' to start.")

st.markdown("---")
st.markdown(
    "**Disclaimer:** This is a simplified demonstration for educational purposes only. It does not account for real-world trading complexities like transaction costs, slippage, or risk management. **Do not use for actual trading.**"
)
