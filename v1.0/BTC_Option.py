# Copyright (c) 2025 Seyed Mohammad Hossein Fasihi (Mhmd Fasihi)
# This file is part of a project licensed under AGPLv3 or a commercial license.
# AGPLv3: https://www.gnu.org/licenses/agpl-3.0.html
# Contact for commercial licensing: mhmd.fasihi@gmail.com

import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime as dt, date, timedelta
import time
import logging

# Setup logging
logging.basicConfig(
    filename="option_data.log",
    level=logging.INFO,
    format="%(asctime)s - %(message)s"
)

# Time conversion utility functions
def datetime_to_timestamp(datetime_obj):
    """Converts a datetime object to a Unix timestamp in milliseconds."""
    if isinstance(datetime_obj, date):
        datetime_obj = dt.combine(datetime_obj, dt.min.time())
    return int(dt.timestamp(datetime_obj) * 1000)

def timestamp_to_datetime(timestamp):
    """Converts a Unix timestamp in milliseconds to a datetime object."""
    return dt.fromtimestamp(timestamp / 1000)

# API request utility function
def get_with_retries(session, url, params, retries=5, backoff_factor=0.3):
    """Send a GET request with retry logic."""
    for i in range(retries):
        try:
            response = session.get(url, params=params, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logging.warning(f"Attempt {i+1} failed: {e}")
            time.sleep(backoff_factor * (2 ** i))

    logging.error(f"Failed after {retries} attempts")
    return None

class OptionData:
    def __init__(self, currency: str, start_date: date, end_date: date):
        """Initialize the OptionData class with currency and date range."""
        self.currency = currency
        self.start_date = start_date
        self.end_date = end_date

        # Validate input arguments
        assert isinstance(currency, str), "currency must be a string"
        assert isinstance(start_date, date), "start_date must be a date object"
        assert isinstance(end_date, date), "end_date must be a date object"
        assert start_date <= end_date, "start_date must be before or equal to end_date"

    def option_data(self) -> pd.DataFrame:
        """Retrieve and process option data from Deribit API."""
        option_list = []
        params = {
            "currency": self.currency,
            "kind": "option",
            "count": 1000,
            "include_old": True,
            "start_timestamp": datetime_to_timestamp(self.start_date),
            "end_timestamp": datetime_to_timestamp(self.end_date)
        }

        url = 'https://history.deribit.com/api/v2/public/get_last_trades_by_currency_and_time'

        with requests.Session() as session:
            while True:
                response_data = get_with_retries(session, url, params)
                if not response_data or "result" not in response_data or "trades" not in response_data["result"]:
                    break

                trades = response_data["result"]["trades"]
                if len(trades) == 0:
                    break

                option_list.extend(trades)
                params["start_timestamp"] = trades[-1]["timestamp"] + 1

                if params["start_timestamp"] >= datetime_to_timestamp(self.end_date):
                    break

                time.sleep(0.2)  # Rate limiting

        # Process the data
        option_data = pd.DataFrame(option_list)
        if option_data.empty:
            return pd.DataFrame()

        # Select and process required columns
        option_data = option_data[["timestamp", "price", "instrument_name",
                                 "index_price", "direction", "amount", "iv"]]

        # Extract information from instrument name
        option_data["kind"] = option_data["instrument_name"].apply(lambda x: str(x).split("-")[0])
        option_data["maturity_date"] = option_data["instrument_name"].apply(
            lambda x: dt.strptime(str(x).split("-")[1], "%d%b%y"))
        option_data["strike_price"] = option_data["instrument_name"].apply(
            lambda x: int(str(x).split("-")[2]))
        option_data["option_type"] = option_data["instrument_name"].apply(
            lambda x: str(x).split("-")[3].lower())

        # Calculate derived metrics
        option_data["moneyness"] = option_data["index_price"] / option_data["strike_price"]
        option_data["price"] = (option_data["price"] * option_data["index_price"]).round(2)
        option_data["date_time"] = option_data["timestamp"].apply(timestamp_to_datetime)
        option_data["time_to_maturity"] = option_data["maturity_date"] - option_data["date_time"]
        option_data["time_to_maturity"] = option_data["time_to_maturity"].apply(
            lambda x: max(round(x.total_seconds() / 31536000, 3), 1e-4) * 365)
        option_data["iv"] = round(option_data["iv"] / 100, 3)

        # Filter for call options
        call_options = option_data[option_data["option_type"] == "c"]

        return call_options[['instrument_name', 'date_time', 'price',
                           'index_price', 'strike_price', 'moneyness',
                           'option_type', 'iv', 'time_to_maturity',
                           'maturity_date']]

def create_visualizations(call_option_data):
    """Create visualizations for the option data analysis."""

    # 1. Price vs Strike Price Scatter Plot
    plt.figure(figsize=(14, 9))
    plt.scatter(call_option_data["strike_price"], call_option_data["price"],
               alpha=0.6, c="blue")
    plt.title("Option Price vs. Strike Price", fontsize=14)
    plt.xlabel("Strike Price", fontsize=12)
    plt.ylabel("Option Price", fontsize=12)
    plt.grid(alpha=0.3)
    plt.show()

    # 2. Implied Volatility Distribution
    plt.figure(figsize=(14, 9))
    plt.hist(call_option_data["iv"], bins=30, color="green",
             alpha=0.7, edgecolor="black")
    plt.title("Distribution of Implied Volatility (IV)", fontsize=14)
    plt.xlabel("Implied Volatility", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.grid(alpha=0.3)
    plt.show()

    # 3. Moneyness vs Time to Maturity Heatmap
    plt.figure(figsize=(20, 15))
    heatmap_data = call_option_data.pivot_table(
        index="moneyness",
        columns="time_to_maturity",
        aggfunc="size",
        fill_value=0
    )
    sns.heatmap(heatmap_data, cmap="coolwarm", annot=False,
                cbar_kws={"label": "Frequency"})
    plt.title("Moneyness vs Time to Maturity", fontsize=14)
    plt.xlabel("Time to Maturity (Days)", fontsize=12)
    plt.ylabel("Moneyness", fontsize=12)
    plt.show()

    # 4. Option Price Over Time
    plt.figure(figsize=(14, 9))
    call_option_data_sorted = call_option_data.sort_values("date_time")
    plt.plot(call_option_data_sorted["date_time"],
            call_option_data_sorted["price"],
            label="Option Price", color="blue", alpha=0.7)
    plt.title("Option Price Over Time", fontsize=14)
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Option Price", fontsize=12)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.show()

    # 5. Strike Price Distribution by Maturity Date
    plt.figure(figsize=(14, 9))
    call_option_data["maturity_date_str"] = call_option_data["maturity_date"].dt.strftime("%Y-%m-%d")
    sns.boxplot(x="maturity_date_str", y="strike_price",
               data=call_option_data, palette="viridis")
    plt.title("Strike Price Distribution by Maturity Date", fontsize=14)
    plt.xlabel("Maturity Date", fontsize=12)
    plt.ylabel("Strike Price", fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(alpha=0.3)
    plt.show()

def main():
    """Main function to execute the analysis."""
    # Define the date range
    start_date = date(2025, 1, 19)
    end_date = date(2025, 1, 21)

    # Initialize the OptionData class and fetch data
    option_data = OptionData(currency="BTC", start_date=start_date, end_date=end_date)
    call_option_data = option_data.option_data()

    # Save the data to an Excel file
    if not call_option_data.empty:
        call_option_data.to_excel("Bitcoin_Call_Options.xlsx", index=False)
        print("Data saved to Bitcoin_Call_Options.xlsx")

        # Create visualizations
        create_visualizations(call_option_data)
    else:
        print("No data retrieved for the specified date range.")

if __name__ == "__main__":
    main()
