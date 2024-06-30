# %%
import pandas as pd
import geopandas as gpd
from pyproj import CRS
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from yellowbrick.cluster import KElbowVisualizer
from sklearn.cluster import KMeans
import ruptures as rpt
from scipy.stats import pearsonr
from src.utilities.utils import describe_df
from itables import init_notebook_mode
init_notebook_mode(all_interactive=True)
import matplotlib
matplotlib.rcParams['font.family'] = 'DejaVu Sans'  # or another available font
# %%
expected_calls_path = Path("/app/data/Call Files/Predictions/FY2022ExpectedCalls.txt")
column_names = ['CallID', 'NatureCode', 'Date', 'Time', 'Address', 'X_Coord', 'Y_Coord']
df_expected_calls = pd.read_csv(expected_calls_path, delimiter='|', skiprows=1, header=None)
df_expected_calls = df_expected_calls.dropna(axis=1, how='all')
df_expected_calls = df_expected_calls.loc[:, (df_expected_calls != False).any(axis=0)]
df_expected_calls.columns = column_names
df_expected_calls['Date'] = pd.to_datetime(df_expected_calls['Date'], format='%m/%d/%Y', errors='coerce')
df_expected_calls['Time'] = pd.to_datetime(df_expected_calls['Time'], format='%H:%M:%S.%f').dt.strftime('%H:%M:%S')
gdf_expected_calls = gpd.GeoDataFrame(
    df_expected_calls, 
    geometry=gpd.points_from_xy(df_expected_calls['X_Coord'], df_expected_calls['Y_Coord']),
    crs=CRS("EPSG:2264")# EPSG:2264 (North Carolina State Plane)
)
gdf_expected_calls = gdf_expected_calls.to_crs(CRS("EPSG:4326"))
gdf_expected_calls['Latitude'] = gdf_expected_calls.geometry.y
gdf_expected_calls['Longitude'] = gdf_expected_calls.geometry.x
gdf_expected_calls.drop(columns=['X_Coord', 'Y_Coord'], inplace=True)
gdf_expected_calls
# %%
rw_19_23_path = Path("/app/data/Call Files/Real World/FY19-23 Call Data with CAD Problem Causal Factors.csv")
df_rw = pd.read_csv(rw_19_23_path)
columns_to_keep = ['IncID', 'Code3_2022Bucket', 'TimePhonePickup', 'Address', 'Latitude', 'Longitude']
df_rw_selected = df_rw[columns_to_keep]
df_rw_selected['TimePhonePickup'] = pd.to_datetime(df_rw_selected['TimePhonePickup'], format='%m/%d/%Y %H:%M:%S')
df_rw_selected['Date'] = df_rw_selected['TimePhonePickup']
df_rw_selected['Time'] = df_rw_selected['TimePhonePickup'].dt.time
df_rw_selected.drop(columns=['TimePhonePickup'], inplace=True)
df_rw_selected.rename(columns={"IncID":"CallID", "Code3_2022Bucket":"NatureCode"}, inplace=True)
gdf_rw_selected = gpd.GeoDataFrame(
    df_rw_selected,
    geometry=gpd.points_from_xy(df_rw_selected['Longitude'], df_rw_selected['Latitude']),
    crs="EPSG:4326"  # WGS84 coordinate system
)
gdf_rw_selected
# %%
# Nature Code Mapping
nature_code_mapping = {
    'EMS-Other': 'EMS-Other',
    'Alarms N.O.S.': 'Alarms',
    'Other - No Fire N.O.S.': 'Other - No Fire',
    'EMS-Cardiac': 'EMS-Cardiac',
    'RESCUE N.O.S.': 'Rescue',
    'EMS-Resp': 'EMS-Resp',
    'EMS-Psych': 'EMS-Psych',
    'MVA': 'MVA',
    'EMS-Trauma-Gen': 'EMS-Trauma-Gen',
    'EMS-Trauma-Criminal': 'EMS-Trauma-Criminal',
    'Non-Emergency N.O.S.': 'Non-Emergency',
    'EMS-Metabolic': 'EMS-Metabolic',
    'EMS-Neuro': 'EMS-Neuro',
    'NON-STRUCTURE FIRE N.O.S.': 'Non-Structure Fire',
    'Rescue-Other N.O.S.': 'Rescue-Other',
    'HM N.O.S.': 'HazMat',
    'FIRE N.O.S.': 'Structure Fire',
    'HM - Other N.O.S.': 'HazMat - Other',
    'Other N.O.S.': 'Other - No Fire',
    'ARFF N.O.S.': 'ARFF'
}
gdf_expected_calls['NatureCode'] = gdf_expected_calls['NatureCode'].replace(nature_code_mapping)
gdf_rw_selected['NatureCode'] = gdf_rw_selected['NatureCode'].replace(nature_code_mapping)
gdf_rw_selected['NatureCode'] = gdf_rw_selected['NatureCode'].replace({'Non-Emergency Other': 'Non-Emergency'})

nature_codes_to_drop = [
    "No Previous Grouping - No Dispatch",
    "Ignore ", #white space is on purpose
    "No Previous Grouping - No NFIRS Report"
]
gdf_rw_selected = gdf_rw_selected[~gdf_rw_selected['NatureCode'].isin(nature_codes_to_drop)]
# %%
# Display the date range of the expected calls GeoDataFrame
date_min = gdf_expected_calls['Date'].min()
date_max = gdf_expected_calls['Date'].max()
gdf_rw_selected = gdf_rw_selected[(gdf_rw_selected['Date'] >= date_min) & (gdf_rw_selected['Date'] <= date_max)]
gdf_rw_selected
# %%
# Aggregate by week
gdf_expected_calls['Week'] = gdf_expected_calls['Date'].dt.isocalendar().week
gdf_rw_selected['Week'] = gdf_rw_selected['Date'].dt.isocalendar().week

weekly_expected = gdf_expected_calls.groupby('Week').size().reset_index(name='count_expected')
weekly_real = gdf_rw_selected.groupby('Week').size().reset_index(name='count_real')
weekly_comparison = pd.merge(weekly_expected, weekly_real, on='Week', how='outer').fillna(0)
# Melt the dataframe for easier plotting with Seaborn
weekly_comparison_melted = weekly_comparison.melt(id_vars=['Week'], value_vars=['count_expected', 'count_real'],
                                                  var_name='Type', value_name='Count')
plt.figure(figsize=(12, 6))
sns.lineplot(data=weekly_comparison_melted, x='Week', y='Count', hue='Type', marker='o')
plt.title('Weekly Expected vs Real Calls')
plt.xlabel('Week')
plt.ylabel('Number of Calls')
plt.legend(title='Call Type')
plt.grid(True)
plt.show()
# %%
# Daily Comparison
daily_expected = gdf_expected_calls.resample('D', on='Date').size().reset_index(name='count_expected')
daily_real = gdf_rw_selected.resample('D', on='Date').size().reset_index(name='count_real')
daily_comparison = pd.merge(daily_expected, daily_real, on='Date', how='outer').fillna(0)

daily_comparison_melted = daily_comparison.melt(id_vars=['Date'], value_vars=['count_expected', 'count_real'],
                                                var_name='Type', value_name='Count')

# Plotting
plt.figure(figsize=(12, 6))
sns.lineplot(data=daily_comparison_melted, x='Date', y='Count', hue='Type', marker='.')
plt.title('Daily Expected vs Real Calls')
plt.xlabel('Date')
plt.ylabel('Number of Calls')
plt.legend(title='Call Type')
plt.grid(True)
plt.show()

# Statistics

mean_expected = daily_comparison['count_expected'].mean()
mean_real = daily_comparison['count_real'].mean()
std_expected = daily_comparison['count_expected'].std()
std_real = daily_comparison['count_real'].std()

print(f"Mean Expected: {mean_expected}, Mean Real: {mean_real}")
print(f"Std Expected: {std_expected}, Std Real: {std_real}")
print("="*30)
correlation, p_value = pearsonr(daily_comparison['count_expected'], daily_comparison['count_real'])
print(f"Pearson Correlation: {correlation}, P-value: {p_value}")
mae = mean_absolute_error(daily_comparison['count_real'], daily_comparison['count_expected'])
rmse = np.sqrt(mean_squared_error(daily_comparison['count_real'], daily_comparison['count_expected']))

print(f"Mean Absolute Error (MAE): {mae}")
print(f"Root Mean Squared Error (RMSE): {rmse}")

# Bland-Altman Plot
plt.figure(figsize=(10, 5))
mean_values = (daily_comparison['count_expected'] + daily_comparison['count_real']) / 2
diff_values = daily_comparison['count_expected'] - daily_comparison['count_real']
mean_diff = np.mean(diff_values)
std_diff = np.std(diff_values)

plt.scatter(mean_values, diff_values, alpha=0.5)
plt.axhline(mean_diff, color='red', linestyle='--')
plt.axhline(mean_diff + 1.96 * std_diff, color='gray', linestyle='--')
plt.axhline(mean_diff - 1.96 * std_diff, color='gray', linestyle='--')
plt.title('Bland-Altman Plot')
plt.xlabel('Mean of Expected and Real Counts')
plt.ylabel('Difference between Expected and Real Counts')
plt.show()
# %%
# Daily comparison line plot

# %%
# Cross corroloation using daily comparison
daily_comparison.set_index('Date', inplace=True)
expected_counts = daily_comparison['count_expected'].values
real_counts = daily_comparison['count_real'].values

"""When you subtract the mean from each value in the series, you are centering the data. 
This step is done to focus on the fluctuations around the mean and to remove any bias due to differing 
baseline levels between the two series. This process is called detrending."""
cross_correlation = np.correlate(expected_counts - np.mean(expected_counts), 
                                 real_counts - np.mean(real_counts), 
                                 mode='full')

lags = np.arange(-len(expected_counts) + 1, len(expected_counts))

plt.figure(figsize=(12, 6))
plt.plot(lags, cross_correlation)
plt.title('Cross-Correlation between Predicted and Real Counts')
plt.xlabel('Lag')
plt.ylabel('Cross-Correlation')
plt.grid(True)
plt.show()

# Find the lag with the maximum correlation
max_corr_index = np.argmax(cross_correlation)
best_lag = lags[max_corr_index]

print(f"The best lag is {best_lag} days with a maximum cross-correlation of {cross_correlation[max_corr_index]}")
# %%
# Standardize cross correlation
expected_counts_standardized = (expected_counts - np.mean(expected_counts)) / np.std(expected_counts)
real_counts_standardized = (real_counts - np.mean(real_counts)) / np.std(real_counts)

cross_correlation_standardized = np.correlate(expected_counts_standardized, 
                                              real_counts_standardized, 
                                              mode='full')

lags_standardized = np.arange(-len(expected_counts_standardized) + 1, len(expected_counts_standardized))

plt.figure(figsize=(12, 6))
plt.plot(lags_standardized, cross_correlation_standardized)
plt.title('Cross-Correlation between Standardized Predicted and Real Counts')
plt.xlabel('Lag')
plt.ylabel('Cross-Correlation')
plt.grid(True)
plt.show()

max_corr_index_standardized = np.argmax(cross_correlation_standardized)
best_lag_standardized = lags_standardized[max_corr_index_standardized]

print(f"The best lag is {best_lag_standardized} days with a maximum cross-correlation of {cross_correlation_standardized[max_corr_index_standardized]}")
# %%
# Residual Analysis

gdf_expected_calls['Date'] = pd.to_datetime(gdf_expected_calls['Date'], format='%m/%d/%Y', errors='coerce').dt.date
gdf_rw_selected['Date'] = pd.to_datetime(gdf_rw_selected['Date'], format='%m/%d/%Y', errors='coerce').dt.date
gdf_expected_calls_agg = gdf_expected_calls.groupby(['Date', 'NatureCode']).size().reset_index(name='Count_expected')
gdf_rw_selected_agg = gdf_rw_selected.groupby(['Date', 'NatureCode']).size().reset_index(name='Count_real')

comparison_df = pd.merge(
    gdf_rw_selected_agg,
    gdf_expected_calls_agg,
    on=['Date', 'NatureCode'],
    how='outer'
).fillna(0)

comparison_df['Residual'] = comparison_df['Count_real'] - comparison_df['Count_expected']
overall_residuals = comparison_df['Residual']
residuals_by_nature_code = comparison_df.groupby('NatureCode')['Residual'].apply(list)

print("Overall Residual Analysis:")
print(f"Mean Residual: {overall_residuals.mean()}")
print(f"Std Residual: {overall_residuals.std()}")
print("\nResidual Analysis by NatureCode:")
for nature_code, residuals in residuals_by_nature_code.items():
    print(f"NatureCode: {nature_code}, Mean Residual: {pd.Series(residuals).mean()}, Std Residual: {pd.Series(residuals).std()}")

residuals_summary = comparison_df.groupby('NatureCode')['Residual'].agg(['mean', 'std']).reset_index()
residuals_summary.columns = ['NatureCode', 'Mean Residual', 'Std Residual']
plt.figure(figsize=(14, 7))
plt.bar(residuals_summary['NatureCode'], residuals_summary['Mean Residual'], yerr=residuals_summary['Std Residual'], capsize=5, color='skyblue')
plt.xticks(rotation=45, ha='right')
plt.xlabel('Nature Code')
plt.ylabel('Mean Residual')
plt.title('Residual Analysis by Nature Code')
plt.axhline(0, color='grey', linewidth=0.8, linestyle='--')
plt.text(-1, min(residuals_summary['Mean Residual']) - 5, 'Negative values indicate overprediction', fontsize=10, color='blue', verticalalignment='bottom')
plt.text(-1, max(residuals_summary['Mean Residual']) + 5, 'Positive values indicate underprediction', fontsize=10, color='blue', verticalalignment='top')

plt.tight_layout()
plt.show()

# %%
#Window Analysis against residuals to identify trends
window_size = 7  #  window size 7 days
comparison_df['Rolling_Mean_Residual'] = comparison_df['Residual'].rolling(window=window_size).mean()
comparison_df['Rolling_Std_Residual'] = comparison_df['Residual'].rolling(window=window_size).std()
# Drop NaN values that result from the rolling calculation
comparison_df.dropna(subset=['Rolling_Mean_Residual', 'Rolling_Std_Residual'], inplace=True)

X = comparison_df[['Rolling_Mean_Residual', 'Rolling_Std_Residual']].values
model = KMeans()
visualizer = KElbowVisualizer(model, k=(2, 10))
visualizer.fit(X) 
visualizer.show()  
optimal_clusters = visualizer.elbow_value_
print(f"Optimal number of clusters: {optimal_clusters}")

kmeans = KMeans(n_clusters=optimal_clusters,n_init=10, random_state=42)
comparison_df['Cluster'] = kmeans.fit_predict(X)

plt.figure(figsize=(14, 7))
for cluster in comparison_df['Cluster'].unique():
    cluster_data = comparison_df[comparison_df['Cluster'] == cluster]
    plt.scatter(cluster_data['Date'], cluster_data['Rolling_Mean_Residual'], label=f'Cluster {cluster}')

plt.axhline(0, color='grey', linewidth=0.8, linestyle='--')
plt.xlabel('Date')
plt.ylabel('Rolling Mean Residual')
plt.title('Clustering of Residuals Over Time')
plt.legend()
plt.show()

for cluster in comparison_df['Cluster'].unique():
    cluster_data = comparison_df[comparison_df['Cluster'] == cluster]
    print(f"Cluster {cluster} Summary:")
    print(cluster_data[['Date', 'NatureCode', 'Count_real', 'Count_expected', 'Residual']])
    print("\n")

# %%
# Rupture Analsis for point detection of changes
residuals = comparison_df['Residual'].values

model = "l2"  
algo = rpt.Binseg(model=model)
result = algo.fit(residuals).predict(n_bkps=4)  # n_bkps is the number of breakpoints to detect

rpt.display(residuals, result)
plt.title("Change Point Detection on Residuals")
plt.show()
# %%
describe_df(gdf_expected_calls, "Expected Calls DataFrame")
describe_df(gdf_rw_selected, "Real World Calls DataFrame")
# %%
