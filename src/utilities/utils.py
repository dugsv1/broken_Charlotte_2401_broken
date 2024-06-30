import pandas as pd

def describe_df(df, name):
    print(f"Summary Report for {name}")
    print("=" * 50)
    
    # Shape of the DataFrame
    print(f"Shape: {df.shape}")
    
    # Column Names and Data Types
    print("\nColumn Names and Data Types:")
    print(df.dtypes)
    
    # Missing Values
    print("\nMissing Values:")
    print(df.isnull().sum())
    
    # Basic Statistics
    print("\nBasic Statistics:")
    print(df.describe(include='all'))
    
    print("\n\n")