import pandas as pd
import os
from pathlib import Path
import json
import numpy as np
import warnings
import matplotlib.pyplot as plt


def load_investment_attributes(file_path='investment_attributes.json'):
    with open(file_path, 'r') as f:
        return json.load(f)


def preprocess(df):
    # Clean up Schwab csv
    df = df.iloc[1:]  # Remove the first row which contains the account info
    df.columns = df.iloc[0]  # Set the second row as column names
    df = df[1:]  # Remove the row that became column names
    df = df.reset_index(drop=True)
    df = df.dropna(axis=1, how='all')
    df = df.dropna(how='all')
    df.drop(df.tail(2).index, inplace=True)  # drop last two rows with account + cash totals

    # Define new column names
    new_column_names = {
        'Symbol': 'symbol',
        'Description': 'description',
        'Qty (Quantity)': 'quantity',
        'Price': 'price',
        'Price Chng % (Price Change %)': 'price_change_pct',
        'Price Chng $ (Price Change $)': 'price_change_usd',
        'Mkt Val (Market Value)': 'market_value',
        'Day Chng % (Day Change %)': 'day_change_pct',
        'Day Chng $ (Day Change $)': 'day_change_usd',
        'Cost Basis': 'cost_basis',
        'Gain $ (Gain/Loss $)': 'gain_loss_usd',
        'Gain % (Gain/Loss %)': 'gain_loss_pct',
        'Ratings': 'ratings',
        'Reinvest?': 'reinvest',
        'Reinvest Capital Gains?': 'reinvest_capital_gains',
        '% of Acct (% of Account)': 'pct_of_account',
        'Security Type': 'security_type'
    }

    # Rename columns
    df = df.rename(columns=new_column_names)

    # Convert numeric columns
    numeric_columns = ['quantity', 'price', 'price_change_pct', 'price_change_usd', 'market_value',
                       'day_change_pct', 'day_change_usd', 'cost_basis', 'gain_loss_usd',
                       'gain_loss_pct', 'pct_of_account']

    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].str.replace('$', '').str.replace('%', ''), errors='coerce')

    # Convert boolean columns
    boolean_columns = ['reinvest', 'reinvest_capital_gains']
    for col in boolean_columns:
        if col in df.columns:
            df[col] = df[col].map({'Yes': True, 'No': False})

    return df


def expand_dataframe(df, investment_attributes):
    unknown_symbols = set()

    def get_attribute(symbol, attribute, default_value):
        if symbol not in investment_attributes:
            unknown_symbols.add(symbol)
            return default_value
        return investment_attributes[symbol].get(attribute, default_value)

    df['category'] = df['symbol'].map(lambda x: get_attribute(x, 'category', 'Unknown'))
    df['risk_score'] = df['symbol'].map(lambda x: get_attribute(x, 'risk_score', np.nan))
    df['sector'] = df['symbol'].map(lambda x: get_attribute(x, 'sector', 'Unknown'))
    df['asset_class'] = df['symbol'].map(lambda x: get_attribute(x, 'asset_class', 'Unknown'))

    if unknown_symbols:
        warnings.warn(f"The following stock symbols were not found in the JSON mapping: {', '.join(unknown_symbols)}. "
                      f"Please add them to your investment_attributes.json file.", UserWarning)

    return df


def load_accounts(directory, investment_attributes):
    accounts = {}
    for file in os.listdir(directory):
        if file.endswith('.csv'):
            file_path = os.path.join(directory, file)
            account_name = Path(file).stem  # Get filename without extension
            df = pd.read_csv(file_path)
            df = preprocess(df)
            df = expand_dataframe(df, investment_attributes)
            accounts[account_name] = df
    return accounts


def analyze_account(df):
    total_value = df['market_value'].sum()
    account_summary = df.groupby('symbol').agg({
        'market_value': 'sum',
        'description': 'first',
        'category': 'first',
        'asset_class': 'first'
    }).reset_index()
    account_summary['percentage'] = account_summary['market_value'] / total_value * 100
    account_summary = account_summary.sort_values('percentage', ascending=False)
    return account_summary


def analyze_all_accounts(accounts):
    all_investments = pd.concat([df for df in accounts.values()])
    total_value = all_investments['market_value'].sum()
    overall_summary = all_investments.groupby('symbol').agg({
        'market_value': 'sum',
        'description': 'first',
        'category': 'first',
        'asset_class': 'first'
    }).reset_index()
    overall_summary['percentage'] = overall_summary['market_value'] / total_value * 100
    overall_summary = overall_summary.sort_values('percentage', ascending=False)
    return overall_summary


def plot_account_composition(summary, title):
    plt.figure(figsize=(10, 6))
    plt.pie(summary['percentage'], labels=summary['symbol'], autopct='%1.1f%%')
    plt.title(title)
    plt.axis('equal')
    plt.show()


def main():
    investment_attributes = load_investment_attributes()

    accounts_directory = 'accounts'
    accounts = load_accounts(accounts_directory, investment_attributes)

    # Analyze each account
    for account_name, df in accounts.items():
        print(f"\nAccount: {account_name}")
        account_summary = analyze_account(df)
        print(account_summary[['symbol', 'description', 'market_value', 'percentage', 'category', 'asset_class']])
        plot_account_composition(account_summary, f"Account Composition: {account_name}")

        print("\n" + "=" * 50)  # Separator between accounts

    # Analyze all accounts together
    overall_summary = analyze_all_accounts(accounts)
    print("\nOverall Investment Summary:")
    print(overall_summary[['symbol', 'description', 'market_value', 'percentage', 'category', 'asset_class']])
    plot_account_composition(overall_summary, "Overall Investment Composition")
    pass

if __name__ == "__main__":
    main()