
import pandas as pd
updated_file_path = 'combine all.xlsx'
updated_xls = pd.ExcelFile(updated_file_path)
updated_df = pd.read_excel(updated_xls, sheet_name='Sheet1')
updated_df.head(10)
header_row = updated_df.iloc[0]
year_row = updated_df.iloc[1].fillna(method='ffill')
combined_columns_fixed = [
    f"{header}_{int(year)}" if pd.notna(year) and str(year).isdigit() else header
    for header, year in zip(header_row, year_row)
]
cleaned_column_names = [str(col) for col in combined_columns_fixed]
current_index = -1
unique_columns_fixed = []
for col in cleaned_column_names:
    base_col = col.split('_')[0]  # Extract base name
    if base_col == "drug":
        current_index += 1  # Increment the index when encountering 'drug'
    unique_columns_fixed.append(f"{base_col}_{current_index}")
updated_df.columns = unique_columns_fixed
indexed_cleaned_time_series_df = updated_df[2:].reset_index(drop=True)
indexed_cleaned_time_series_df
olympic_years = [1896, 1900, 1904, 1906, 1908, 1912, 1920, 1924, 1928, 1932, 1936,
                 1948, 1952, 1956, 1960, 1964, 1968, 1972, 1976, 1980, 1984,
                 1988, 1992, 1996, 2000, 2004, 2008, 2012, 2016, 2020, 2024, 2028, 2032]
def get_new_number(old_number):
    return olympic_years[int(old_number)]
new_columns = []
for col in indexed_cleaned_time_series_df.columns:
    if '_' in col:
        base_name, number = col.rsplit('_', 1)  # Split from the right to handle names like "data_1_test"
        if number.isdigit():  # Ensure it's a valid number
            new_number = get_new_number(number)
            new_columns.append(f"{base_name}_{new_number}")
        else:
            new_columns.append(col)  # Keep the original if no valid number found
    else:
        new_columns.append(col)  # Keep columns without '_'
df2 = indexed_cleaned_time_series_df
df2.columns = new_columns
print(df2.columns)
df2
df2.to_csv("not_final.csv")
def create_label_columns(df):
    total_columns = [col for col in df.columns if col.startswith('total_') and col.split('_')[1].isdigit()]
    for col in total_columns:
        year_suffix = col.split('_')[1]
        new_col_name = f"label_{year_suffix}"
        df[new_col_name] = df[col].apply(lambda x: 1 if x > 0 else 0)
    return df
create_label_columns(df2)
df2
def create_popularity_columns(df):
    total_columns = [col for col in df.columns if col.startswith('total_') and col.split('_')[1].isdigit()]
    for col in total_columns:
        year_suffix = col.split('_')[1]
        year_total_sum = df[col].sum()
        new_col_name = f"popularity_{year_suffix}"
        df[new_col_name] = df[col] / year_total_sum if year_total_sum != 0 else 0
    return df
df3 = create_popularity_columns(df2)
df3
def create_popularity_and_normalized_country_columns(df, total_countries=200):
    total_columns = [col for col in df.columns if col.startswith('total_') and col.split('_')[1].isdigit()]
    country_columns = [col for col in df.columns if col.startswith('country_') and col.split('_')[1].isdigit()]
    for col in total_columns:
        year_suffix = col.split('_')[1]
        year_total_sum = df[col].sum()
        new_col_name = f"popularity_{year_suffix}"
        df[new_col_name] = df[col] / year_total_sum if year_total_sum != 0 else 0
    for col in country_columns:
        year_suffix = col.split('_')[1]
        new_col_name = f"normalizedcountry_{year_suffix}"
        df[new_col_name] = df[col] / total_countries
    return df
df3 = create_popularity_and_normalized_country_columns(df2)
df3
import numpy as np
def normalize_log_scale(df, column_name):
    if column_name in df.columns:
        df[f"{column_name}_log_scaled"] = df[column_name].apply(lambda x: np.log1p(x) if x > 0 else 0)
    else:
        print(f"Column {column_name} not found in the DataFrame.")
    return df
df3 = normalize_log_scale(df3, "estimate per event_1896")
df3
def add_age_representative_columns(df, sport_column):
    age_columns = [col for col in df.columns if col.split('_')[0].replace('.', '').isdigit() and 10.0 <= float(col.split('_')[0]) <= 100.0]
    years = sorted(set(col.split('_')[1] for col in age_columns))
    new_columns = {}
    for year in years:
        year_columns = [col for col in age_columns if f"_{year}" in col]
        def calculate_cv(row):
            mean_value = row.mean()
            sd_value = row.std()
            print(f"Year: {year}, Row Index: {row.name}, Mean: {mean_value}, SD: {sd_value}")
            return sd_value / mean_value if mean_value != 0 and len(row.dropna()) > 1 else 0
        new_columns[f"CV_{year}"] = df[year_columns].apply(calculate_cv, axis=1)
    df = pd.concat([df, pd.DataFrame(new_columns, index=df.index)], axis=1)
    return df
print(df3.columns)
df3 = add_age_representative_columns(df3,"Sport_1896")
df3
df3.to_csv("not_final_no_CV.csv")
