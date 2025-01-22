import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Data Cleaning Functions
def clean_missing_values(df):
    """Handle missing values by replacing them with column mean."""
    return df.fillna(df.mean())

def standardize_columns(df):
    """Standardize column names by making them lowercase and replacing spaces with underscores."""
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
    return df

def remove_outliers(df, column):
    """Remove outliers from a column based on the IQR method."""
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    return df[(df[column] >= q1 - 1.5 * iqr) & (df[column] <= q3 + 1.5 * iqr)]

def clean_data(df):
    """Combine all data cleaning steps into a single function."""
    df = clean_missing_values(df)
    df = standardize_columns(df)
    return df

# EDA Functions
def generate_summary(df):
    """Generate descriptive statistics for the dataset."""
    return df.describe()

def plot_histogram(df, column):
    """Plot a histogram with KDE for a specific column."""
    sns.histplot(df[column], kde=True)
    plt.title(f'Histogram of {column}')
    plt.show()

def plot_correlation_matrix(df):
    """Plot a heatmap of the correlation matrix."""
    correlation = df.corr()
    sns.heatmap(correlation, annot=True, cmap="coolwarm")
    plt.title('Correlation Matrix')
    plt.show()

# Main Function
def main():
    """Main workflow for the data automation project."""
    # Load dataset
    data_path = 'data/sample_dataset.csv'  # Replace with your dataset path
    df = pd.read_csv(data_path)

    print("Original Data:")
    print(df.head())

    # Clean data
    df_cleaned = clean_data(df)
    df_cleaned.to_csv('output/cleaned_data.csv', index=False)
    print("\nData cleaning completed. Cleaned data saved to 'output/cleaned_data.csv'.")

    # Perform EDA
    print("\nSummary Statistics:")
    print(generate_summary(df_cleaned))

    # Visualizations
    column_name = input("Enter the column name for histogram: ")  # Replace with a valid column name
    if column_name in df_cleaned.columns:
        plot_histogram(df_cleaned, column_name)
    else:
        print(f"Column '{column_name}' not found in the dataset.")

    print("\nGenerating correlation matrix...")
    plot_correlation_matrix(df_cleaned)

if __name__ == "__main__":
    main()
