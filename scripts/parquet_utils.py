import os
import glob
from tqdm import tqdm
import multiprocessing
from functools import partial
import re
import pandas as pd
import numpy as np


def extract_id_from_path(file_path):
    """Extract ID from the folder path"""
    match = re.search(r"id=([^/\\]+)", file_path)
    return match[1] if match else None


def process_parquet_file(file_path):
    """Process a single parquet file and extract statistical features"""
    try:
        # Extract ID from the file path
        subject_id = extract_id_from_path(file_path)

        # Read the parquet file
        df = pd.read_parquet(file_path)

        # Initialize results dictionary with ID
        results = {"id": subject_id}

        # Columns to compute statistics on (excluding non-numeric and metadata)
        numeric_cols = ["X", "Y", "Z", "enmo", "anglez", "light", "battery_voltage"]

        # Calculate statistics for each numeric column
        for col in numeric_cols:
            if col in df.columns:
                # Skip if column has all NaN values
                if df[col].isna().all():
                    results[f"{col}_median"] = np.nan
                    results[f"{col}_q1"] = np.nan
                    results[f"{col}_q3"] = np.nan
                    results[f"{col}_std"] = np.nan
                    continue

                # Calculate statistics
                results[f"{col}_median"] = df[col].median()
                results[f"{col}_q1"] = df[col].quantile(0.25)
                results[f"{col}_q3"] = df[col].quantile(0.75)
                results[f"{col}_std"] = df[col].std()

                # Additional useful statistics
                results[f"{col}_mean"] = df[col].mean()
                results[f"{col}_min"] = df[col].min()
                results[f"{col}_max"] = df[col].max()
                results[f"{col}_iqr"] = results[f"{col}_q3"] - results[f"{col}_q1"]

        # Calculate percentage of time the device was worn
        if "non-wear_flag" in df.columns:
            results["wear_percentage"] = 100 - (df["non-wear_flag"].mean() * 100)

        # Calculate some time-based features
        if "weekday" in df.columns:
            # Distribution of data across days of the week
            for day in range(1, 8):  # 1 (Monday) to 7 (Sunday)
                results[f"day_{day}_percentage"] = (df["weekday"] == day).mean() * 100

        # Calculate activity level features
        if "enmo" in df.columns:
            # Percentage of time in different activity levels
            results["inactive_percentage"] = (
                df["enmo"] < 0.1
            ).mean() * 100  # Low activity threshold
            results["moderate_activity_percentage"] = (
                (df["enmo"] >= 0.1) & (df["enmo"] < 0.4)
            ).mean() * 100
            results["high_activity_percentage"] = (df["enmo"] >= 0.4).mean() * 100

            # Calculate activity by time of day if time information is available
            if "time_of_day" in df.columns:
                # Extract hour from time_of_day
                try:
                    df["hour"] = pd.to_datetime(df["time_of_day"]).dt.hour

                    # Calculate activity metrics by 6-hour blocks
                    for start_hour in [0, 6, 12, 18]:
                        end_hour = start_hour + 6
                        period_mask = (df["hour"] >= start_hour) & (
                            df["hour"] < end_hour
                        )
                        if period_mask.any():
                            results[f"enmo_mean_{start_hour}_{end_hour}h"] = df.loc[
                                period_mask, "enmo"
                            ].mean()
                except Exception:
                    # print the exception
                    print(f"Error processing time_of_day in {file_path}")

        # Calculate relative_date_PCIAT features if available
        if "relative_date_PCIAT" in df.columns:
            results["days_since_PCIAT_mean"] = df["relative_date_PCIAT"].mean()
            results["days_before_PCIAT"] = (df["relative_date_PCIAT"] < 0).mean() * 100
            results["days_after_PCIAT"] = (df["relative_date_PCIAT"] >= 0).mean() * 100

        return results

    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return {"id": extract_id_from_path(file_path), "error": str(e)}


def process_all_files(parent_folder, is_train=True):
    """Process all parquet files in the specified folder using parallel processing"""
    # Find all parquet files (going one level deeper into ID folders)
    all_id_folders = glob.glob(os.path.join(parent_folder, "id=*"))
    all_parquet_files = []

    for folder in all_id_folders:
        parquet_files = glob.glob(os.path.join(folder, "*.parquet"))
        all_parquet_files.extend(parquet_files)

    print(
        f"Found {len(all_parquet_files)} parquet files in {len(all_id_folders)} ID folders"
    )

    # Set up parallel processing
    num_cores = max(1, multiprocessing.cpu_count() - 1)  # Leave one core free
    print(f"Processing files using {num_cores} CPU cores")

    # Process files in parallel with progress bar
    with multiprocessing.Pool(num_cores) as pool:
        results = list(
            tqdm(
                pool.imap(process_parquet_file, all_parquet_files),
                total=len(all_parquet_files),
                desc=f"Processing {'train' if is_train else 'test'} files",
            )
        )

    return pd.DataFrame(results)
