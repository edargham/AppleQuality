import numpy as np
import pandas as pd
import os 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def read_data(path: str) -> pd.DataFrame:
    if path is None:
        raise 'Data path field in configuration was not defined or is empty.'
    else:
        dataframe = pd.read_csv(path)
        dataframe = dataframe[:-1]

        for column in dataframe.columns:
            if dataframe[column].isna().any():
                dataframe[column].dropna(inplace=True)

        dataframe['Acidity'] = dataframe['Acidity'].astype(np.float64)
        dataframe['Quality'] = (dataframe['Quality'] == 'good').astype(np.int8)
        dataframe.drop('A_id', axis=1, inplace=True)
    return dataframe


def preprocess_data(df: pd.DataFrame, target_col: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    labels = df[target_col]
    features = df.drop(target_col, axis=1, inplace=False)

    x_train, x_test, y_train, y_test = train_test_split(features, labels, train_size=0.8, random_state=42)

    transformer = StandardScaler()
    x_train_std = transformer.fit_transform(x_train, y_train)
    x_test_std = transformer.transform(x_test)

    return x_train_std, x_test_std, y_train, y_test

def plot_specialized(data, plot_type, save_path=None):
    if save_path and not os.path.exists(save_path):
        os.makedirs(save_path)
    
    if plot_type == 'histogram':
        for column in data.columns:
            plt.figure(figsize=(8, 6))
            data[column].hist()
            plt.title(f'Histogram of {column}')
            plt.xlabel(column)
            plt.ylabel('Frequency')
            if save_path:
                plt.savefig(os.path.join(save_path, f'{column}_histogram.png'))
    elif plot_type == 'boxplot':
        target_column = input("Enter the name of the target feature: ")
        data[target_column] = data[target_column].dropna()        
        unique_values = data[target_column].unique()
        for column in data.columns:
            if column != target_column:
                plt.figure(figsize=(8, 6))
                positions = np.arange(len(unique_values)) + 1
                for idx, value in enumerate(unique_values):
                    subset = data[data[target_column] == value][column]
                    jitter = np.random.normal(0, 0.1, len(subset))
                    subset_numeric = pd.to_numeric(subset, errors='coerce').dropna()
                    plt.boxplot(subset_numeric + jitter[:len(subset_numeric)], positions=[positions[idx]], widths=0.6, patch_artist=True)
                plt.title(f'Boxplot of {column} by {target_column}')
                plt.xlabel(target_column)
                plt.ylabel(column)
                plt.xticks(positions, unique_values)
                plt.grid(True)
                if save_path:
                    plt.savefig(os.path.join(save_path, f'{column}_boxplot.png'))
    else:
        print("Invalid plot type. Please choose 'histogram' or 'boxplot'.")