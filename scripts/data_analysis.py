import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

def load_data(file_path):
    # Load the CSV file
    df = pd.read_csv(file_path)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    return df

""" Summary statistics calculating mean, median, sd to understand the data """
def summary_statistics(df):
    return df.describe()

def data_quality_check(df, columns):
    # 1. Check for missing values
    missing_values = df[columns].isnull().sum()
    if missing_values.any():
        print("\nMissing Values:\n", missing_values)
    else:
        print("\nThere are no missing values.")

    # 2. Check for incorrect entries (e.g., negative values where only positive should exist)
    for col in columns:
        if df[col].dtype != 'object':
            incorrect_entries= df[df[col]< 0][col]
            if not incorrect_entries.empty:
                print(f"\nIncorrect Entries in {col}:\n", incorrect_entries)
            else:
                print(f"No incorrect entries in {col}.")

    # z_scores = np.abs((df[columns] - df[columns].mean()) / df[columns].std())
    # outliers = z_scores > 3  # Considering Z-score > 3 as an outlier threshold
    
    # for col in columns:
    #     outlier_indices = np.where(outliers[col])[0]
    #     if len(outlier_indices) > 0:
    #         print(f"Outliers in {col} (Z-score > 3):\n", df.iloc[outlier_indices][col])
    #     else:
    #         print(f"No outliers detected in {col}.")

    # 3. Check for outlier using IQR method
    for col in columns:
        if df[col].dtype != 'object':
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
            if not outliers.empty:
                print(f"\nOutliers in {col}:\n", outliers)
            else:
                print(f"No outliers in {col}.")

def plot_time_series(df, columns, date_column='Date'):
    """
    Plots time series data for the specified columns.
    
    Args:
    df (DataFrame): The dataset containing the time series data.
    columns (list): List of columns to plot.
    date_column (str): The column containing date/time information.
    
    Returns:
    None
    """
    # Convert date_column to datetime if it's not already
    df[date_column] = pd.to_datetime(df[date_column])
    
    # Plot each column in the list
    for col in columns:
        plt.figure(figsize=(10, 6))
        sns.lineplot(x=df[date_column], y=df[col])
        plt.title(f'Time Series Plot of {col} Over Time')
        plt.xlabel('Time')
        plt.ylabel(col)
        plt.grid(True)
        plt.show()


def evaluate_cleaning_impact(df, mod_columns, cleaning_column='Cleaning', date_column='Date'):
    """
    Evaluates the impact of cleaning on sensor readings (ModA, ModB) over time.
    
    Args:
    df (DataFrame): The dataset containing the sensor readings and cleaning data.
    mod_columns (list): List of sensor columns to evaluate.
    cleaning_column (str): The column indicating cleaning status.
    date_column (str): The column containing date/time information.
    
    Returns:
    None
    """
    # Convert date_column to datetime if it's not already
    df[date_column] = pd.to_datetime(df[date_column])
    
    for mod in mod_columns:
        plt.figure(figsize=(10, 6))
        
        # Plot the sensor readings over time
        sns.lineplot(x=df[date_column], y=df[mod], label=f'{mod} Reading')
        
        # Highlight the periods when cleaning was done
        sns.scatterplot(x=df[df[cleaning_column] == 1][date_column],
                        y=df[df[cleaning_column] == 1][mod], color='red', label='Cleaning')
        
        plt.title(f'Impact of Cleaning on {mod} Over Time')
        plt.xlabel('Time')
        plt.ylabel(mod)
        plt.legend()
        plt.grid(True)
        plt.show()


# Correlation Analysis
def plot_correlation_heatmap(df, columns):
    plt.figure(figsize=(10, 8))
    corr_matrix = df[columns].corr()  # Calculate the correlation matrix
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', vmin=-1, vmax=1)
    plt.title('Correlation Heatmap')
    plt.show()

def plot_pairplot(df, columns):
    sns.pairplot(df[columns])
    plt.suptitle('Pair Plot of Solar Radiation and Temperature', y=1.02)
    plt.show()

def plot_wind_scatter_matrix(df, wind_columns, irradiance_columns):
    columns_to_include = wind_columns + irradiance_columns
    sns.pairplot(df[columns_to_include])
    plt.suptitle('Scatter Matrix: Wind Conditions and Solar Irradiance', y=1.02)
    plt.show()


# Wind Analysis
def plot_wind_polar(df, speed_column, direction_column):
    # Convert wind direction from degrees to radians for polar plotting
    wd_rad = np.deg2rad(df[direction_column].dropna())

    # Set up the polar plot
    plt.figure(figsize=(8, 8))
    ax = plt.subplot(111, polar=True)

    # Scatter plot of wind direction and speed
    ax.scatter(wd_rad, df[speed_column], c=df[speed_column], cmap='viridis', alpha=0.75)

    # Set the labels and title
    ax.set_theta_zero_location('N')  # North on top
    ax.set_theta_direction(-1)  # Clockwise
    ax.set_rlabel_position(90)  # Label position
    plt.title('Wind Speed and Direction Distribution')

    # Display the plot
    plt.show()

def plot_wind_direction_variability(df, direction_column):
    # Convert wind direction to radians
    wd_rad = np.deg2rad(df[direction_column].dropna())

    # Set up the polar plot for wind direction variability
    plt.figure(figsize=(8, 8))
    ax = plt.subplot(111, polar=True)

    # Create a histogram (rose plot) for wind direction distribution
    num_bins = 36  # Number of bins (e.g., 10-degree bins)
    ax.hist(wd_rad, bins=num_bins, alpha=0.75, color='blue')

    # Set labels and title
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    plt.title('Wind Direction Variability')

    # Display the plot
    plt.show()


# Temprature Analysis
""" A scatter plot is a good way to cisualize the relationship between RH, temperature readings and solar radiation components(GHI, DNI, DHI)"""

def plot_temperature_vs_rh(df, temperature_column, rh_column, ghi_column, dni_column, dhi_column):
    plt.figure(figsize=(16, 10))

    # Scatter plot between Temperature and Relative Humidity
    plt.subplot(2, 2, 1)
    sns.scatterplot(x=df[rh_column], y=df[temperature_column])
    plt.title('Temperature vs. Relative Humidity')
    plt.xlabel('Relative Humidity (%)')
    plt.ylabel('Temperature (°C)')

    # Scatter plot between GHI and Relative Humidity
    plt.subplot(2, 2, 2)
    sns.scatterplot(x=df[rh_column], y=df[ghi_column])
    plt.title('GHI vs. Relative Humidity')
    plt.xlabel('Relative Humidity (%)')
    plt.ylabel('GHI (W/m^2)')

    # Scatter plot between DNI and Relative Humidity
    plt.subplot(2, 2, 3)
    sns.scatterplot(x=df[rh_column], y=df[dni_column])
    plt.title('DNI vs. Relative Humidity')
    plt.xlabel('Relative Humidity (%)')
    plt.ylabel('DNI (W/m^2)')

    # Scatter plot between DHI and Relative Humidity
    plt.subplot(2, 2, 4)
    sns.scatterplot(x=df[rh_column], y=df[dhi_column])
    plt.title('DHI vs. Relative Humidity')
    plt.xlabel('Relative Humidity (%)')
    plt.ylabel('DHI (W/m^2)')

    plt.tight_layout()
    plt.show()

""" Next, a correlation analysis is performed to quantify the relationships between RH, temperature, and solar radiation components. """

def correlation_analysis(df, columns):
    corr_matrix = df[columns].corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Correlation Matrix: Temperature, RH, and Solar Radiation')
    plt.show()


# Histograms
def plot_histograms(df, columns, bins=30):
    plt.figure(figsize=(16, 10))
    
    for i, column in enumerate(columns):
        plt.subplot(2, 3, i + 1)
        sns.histplot(df[column].dropna(), bins=bins, kde=False, color='skyblue')
        plt.title(f'Distribution of {column}')
        plt.xlabel(column)
        plt.ylabel('Frequency')
    
    plt.tight_layout()
    plt.show()


# Z-Score Analysis
def calculate_z_scores(df, columns, threshold=3):
    z_scores_df = pd.DataFrame()
    
    for column in columns:
        # Calculate Z-scores
        z_scores = (df[column] - df[column].mean()) / df[column].std()
        z_scores_df[column + '_z_score'] = z_scores
        
        # Flag outliers based on the threshold
        z_scores_df[column + '_outlier'] = np.where(np.abs(z_scores) > threshold, True, False)
    
    return z_scores_df


# Bubble charts
def plot_bubble_chart(df, x_column, y_column, size_column, color_column=None, title=None):
    plt.figure(figsize=(10, 6))

    # Normalize the size column to control bubble sizes
    size_normalized = (df[size_column] - df[size_column].min()) / (df[size_column].max() - df[size_column].min()) * 1000
    
    # Scatter plot with bubbles
    scatter = plt.scatter(
        df[x_column], 
        df[y_column], 
        s=size_normalized,  # Bubble size
        c=df[color_column] if color_column else 'blue',  # Bubble color
        cmap='viridis', 
        alpha=0.6,
        edgecolor='w', 
        linewidth=0.5
    )

    # Add a color bar if color_column is provided
    if color_column:
        plt.colorbar(scatter, label=color_column)
    
    plt.xlabel(x_column)
    plt.ylabel(y_column)
    plt.title(title if title else f'{x_column} vs. {y_column} with {size_column} as Bubble Size')
    plt.show()


#Data Ckeaning 
def handle_missing_values(df):
    # Drop columns where all values are null (e.g., 'Comments')
    df = df.dropna(axis=1, how='all')
    
    # Fill missing values in numeric columns with the median
    df.update(df.median())
    
    # Fill non-numeric columns with the mode
    for col in df.select_dtypes(include='object').columns:
        df[col].fillna(df[col].mode()[0], inplace=True)
    
    return df


def handle_anomalies(df, columns):
    for column in columns:
        # Replace negative values with NaN
        df[column] = df[column].apply(lambda x: np.nan if x < 0 else x)
        
        # Fill NaN values with the median
        df[column] = df[column].fillna(df[column].median())
    
    return df


def handle_outliers(df, columns):
    for column in columns:
        # Calculate the Z-score for each value
        z_scores = np.abs((df[column] - df[column].mean()) / df[column].std())
        
        # Replace outliers (e.g., Z-score > 3) with NaN
        df[column] = df[column].where(z_scores < 3)
        
        # Fill NaN values with the median
        df[column] = df[column].fillna(df[column].median())
    
    return df






      