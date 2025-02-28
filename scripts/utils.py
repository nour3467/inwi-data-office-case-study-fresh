import ppscore as pps
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import KNNImputer
from cycler import cycler
from tqdm import tqdm
import pickle
from sklearn.preprocessing import RobustScaler
import numpy as np
import os
from scipy.stats import boxcox
from datetime import datetime


# Custom style settings
def set_plot_style():
    """
    Set up a professional plot style for visualizations.
    """
    rc = {
        # "figure.figsize": (20, 7),
        "axes.facecolor": "#ffffff",
        "axes.grid": True,
        "grid.color": ".8",
        "font.family": "Candara",
        # "font.size": 14,
        "font.weight": "bold",
        "axes.prop_cycle": cycler(
            "color", ["#264653", "#2a9d8f", "#e9c46a", "#f4a261", "#e76f51"]
        ),
    }
    plt.rcParams.update(rc)


def plot_missing_data_percentage(df, fig_size=(20, 7), return_format="dict"):
    """
    Plot the percentage of missing data for each feature in the provided DataFrame.
    Parameters:
    -----------
    df : pandas.DataFrame
        The dataset to analyze for missing values
    fig_size : tuple, optional
        Size of the figure (width, height), default is (20, 7)
    return_format : str, optional
        Format of the return value. Options are "dict" or "df". Default is "dict".
    Returns:
    --------
    dict or pandas.DataFrame
        - If `return_format="dict"`, returns a dictionary where keys are percentages (rounded to two decimal places)
          and values are lists of column names with that percentage of missing values.
        - If `return_format="df"`, returns a DataFrame with two columns: 'Percentage' and 'Columns'.
    Example:
    --------
    >>> result = plot_missing_data_percentage(train_df, return_format="dict")
    >>> print(result[50.0])  # Columns with exactly 50% missing values
    """

    # Apply custom plot style
    set_plot_style()

    # Calculate the percentage of missing values for each feature
    missing_percentage = (df.isnull().sum() / len(df)) * 100

    # Create a dictionary or DataFrame based on the return_format parameter
    if return_format == "dict":
        result = {}
        for col, pct in missing_percentage.items():
            pct_rounded = round(pct, 2)  # Round percentage to two decimal places
            if pct_rounded not in result:
                result[pct_rounded] = []
            result[pct_rounded].append(col)
    elif return_format == "df":
        result = (
            missing_percentage.reset_index()
            .rename(columns={"index": "Columns", 0: "Percentage"})
            .sort_values(by="Percentage", ascending=False)
        )
    else:
        raise ValueError("Invalid return_format. Use 'dict' or 'df'.")

    # Plot the bar chart
    plt.figure(figsize=fig_size)
    ax = sns.barplot(
        x=missing_percentage.index,
        y=missing_percentage,
        palette="coolwarm",  # Use the coolwarm color palette
    )

    # Add percentage labels on each bar (rotated by 90 degrees)
    for p in ax.patches:
        if p.get_height() > 0:  # Only annotate bars with missing values
            ax.annotate(
                f"{p.get_height():.2f}%",  # Display percentage with 2 decimal places
                (
                    p.get_x() + p.get_width() / 2.0,
                    p.get_height(),
                ),  # Position of the annotation
                ha="center",  # Horizontal alignment
                va="bottom",  # Vertical alignment (adjusted for rotation)
                fontsize=14,
                color="black",
                rotation=90,  # Rotate the text by 90 degrees
                xytext=(0, 5),  # Offset from the bar top
                textcoords="offset points",  # Use offset points for positioning
            )

    # Customize the plot aesthetics
    # plt.title("Percentage of Missing Data by Feature", fontsize=20)
    plt.xlabel("Features", fontsize=16)
    plt.ylabel(r"Percentage of Missing Values (\%)", fontsize=16)
    plt.xticks(rotation=90, fontsize=14)  # Rotate x-axis labels for readability
    plt.yticks(fontsize=14)  # Adjust y-axis label font size
    sns.despine(left=True, bottom=True)  # Remove top and right spines
    plt.grid(False)  # Disable grid lines
    plt.tight_layout()
    plt.show()

    return result


def calculate_pps(df, target_col="PCIAT-PCIAT_Total", exclude_cols=None):
    """
    Calculate Predictive Power Score (PPS) for features against the target using the ppscore library.

    Parameters:
    -----------
    df : pandas.DataFrame
        The dataset to analyze.
    target_col : str, optional
        Target column name, default is 'PCIAT-PCIAT_Total'.
    exclude_cols : list, optional
        List of columns to exclude from analysis.

    Returns:
    --------
    pandas.DataFrame
        DataFrame with PPS scores and missing percentages.
    """
    if exclude_cols is None:
        exclude_cols = []

    # Ensure target column is not in exclude_cols
    if target_col in exclude_cols:
        exclude_cols.remove(target_col)

    # Drop rows with missing target values
    df_no_missing_target = df.dropna(subset=[target_col])

    # Get all feature columns
    feature_cols = [
        col for col in df.columns if col != target_col and col not in exclude_cols
    ]

    # Initialize results
    pps_results = []

    # Calculate PPS for each feature
    for feature in feature_cols:
        # Skip if feature is all missing
        if df_no_missing_target[feature].isna().all():
            pps_results.append(
                {"feature": feature, "pps": 0, "missing_percentage": 100}
            )
            continue

        # Calculate missing percentage
        missing_pct = (
            df_no_missing_target[feature].isna().sum() / len(df_no_missing_target)
        ) * 100

        # Skip if missing percentage is too high (>90%)
        if missing_pct > 90:
            pps_results.append(
                {"feature": feature, "pps": 0, "missing_percentage": missing_pct}
            )
            continue

        # Calculate PPS using ppscore
        score = pps.score(df_no_missing_target, feature, target_col)["ppscore"]
        pps_results.append(
            {"feature": feature, "pps": score, "missing_percentage": missing_pct}
        )

    # Convert results to DataFrame and sort by PPS
    pps_df = pd.DataFrame(pps_results).sort_values("pps", ascending=False)
    return pps_df


def plot_pps(pps_df, fig_size=(20, 7), top_n=20):
    """
    Plot Predictive Power Scores with a professional style.

    Parameters:
    -----------
    pps_df : pandas.DataFrame
        DataFrame with PPS scores from calculate_pps function.
    fig_size : tuple, optional
        Size of the figure, default is (20, 7).
    top_n : int, optional
        Number of top features to show, default is 20.

    Returns:
    --------
    None

    Example:
    --------
    >>> pps_results = calculate_pps(train_df)
    >>> plot_pps(pps_results, top_n=20)
    """
    # Apply custom plot style
    set_plot_style()

    # Filter to top N features by PPS score
    top_features = pps_df.head(top_n)
    if len(top_features) == 0:
        print("No PPS scores found. Check your data or calculation.")
        return

    # Create figure with specified size
    fig, ax = plt.subplots(figsize=fig_size)

    # Create bars with colors based on the default prop_cycle
    bars = ax.barh(
        top_features["feature"],
        top_features["pps"],
        color=[
            plt.rcParams["axes.prop_cycle"].by_key()["color"][i % 5]
            for i in range(len(top_features))
        ],
    )

    # Add PPS values to the bars
    for i, v in enumerate(top_features["pps"]):
        ax.text(v + 0.01, i, f"{v:.3f}", va="center", fontsize=12)

    # Add missing percentages to the y-axis labels
    labels = [
        f"{feat} ({pct:.1f}%)"
        for feat, pct in zip(
            top_features["feature"], top_features["missing_percentage"]
        )
    ]
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=14)

    # Set labels and title
    ax.set_xlabel("Predictive Power Score", fontsize=16)
    ax.set_title(f"Predictive Power Score (PPS) by Feature - Top {top_n}", fontsize=20)

    # Set x-axis limit
    ax.set_xlim(0, min(1, top_features["pps"].max() * 1.2))

    # Apply professional style
    sns.despine(ax=ax, left=True, bottom=True)

    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.show()


def robust_scale_numerical_columns(
    train_df, test_df, save_scalers=True, scalers_folder="../data/scalers"
):
    """
    Apply robust scaling to common numerical columns between train and test datasets.
    Saves each scaler separately in the specified folder.

    Parameters:
    -----------
    train_df : pandas.DataFrame
        Training dataset
    test_df : pandas.DataFrame
        Test dataset
    save_scalers : bool, optional
        Whether to save the scalers to disk, default is True
    scalers_folder : str, optional
        Path to folder where scalers will be saved, default is '../data/scalers'

    Returns:
    --------
    tuple
        (scaled_train, scaled_test, scalers_dict) - Scaled datasets and dictionary of scalers
    """
    import os

    # Create copies to avoid modifying originals
    scaled_train = train_df.copy()
    scaled_test = test_df.copy()

    # Find common numerical columns between train and test
    train_num_cols = train_df.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()
    test_num_cols = test_df.select_dtypes(include=["int64", "float64"]).columns.tolist()

    common_num_cols = [col for col in train_num_cols if col in test_num_cols]

    print(f"Scaling {len(common_num_cols)} common numerical columns")

    # Dictionary to store scalers for each column
    scalers_dict = {}

    # Create scalers directory if it doesn't exist
    if save_scalers and not os.path.exists(scalers_folder):
        os.makedirs(scalers_folder)
        print(f"Created scalers directory: {scalers_folder}")

    # Apply scaling to each column individually
    for col in tqdm(common_num_cols, desc="Scaling columns"):
        # Get non-missing values for fitting
        train_values = train_df[col].dropna().values.reshape(-1, 1)

        if len(train_values) > 0:
            # Create and fit scaler for this column
            scaler = RobustScaler()
            scaler.fit(train_values)

            # Store the scaler
            scalers_dict[col] = scaler

            # Transform non-missing values in train
            train_mask = ~train_df[col].isna()
            if train_mask.any():
                scaled_train.loc[train_mask, col] = scaler.transform(
                    train_df.loc[train_mask, col].values.reshape(-1, 1)
                )

            # Transform non-missing values in test
            test_mask = ~test_df[col].isna()
            if test_mask.any():
                scaled_test.loc[test_mask, col] = scaler.transform(
                    test_df.loc[test_mask, col].values.reshape(-1, 1)
                )

            # Save individual scaler if requested
            if save_scalers:
                # Create safe filename by replacing problematic characters
                safe_colname = (
                    col.replace("-", "_").replace("/", "_").replace("\\", "_")
                )
                scaler_path = os.path.join(scalers_folder, f"scaler_{safe_colname}.pkl")
                with open(scaler_path, "wb") as f:
                    pickle.dump(scaler, f)

    # Also save the complete dictionary for convenience
    if save_scalers and scalers_dict:
        with open(os.path.join(scalers_folder, "all_scalers.pkl"), "wb") as f:
            pickle.dump(scalers_dict, f)
        print(f"All scalers saved to {os.path.join(scalers_folder, 'all_scalers.pkl')}")
        print(f"Individual scalers saved in {scalers_folder}")

    return scaled_train, scaled_test, scalers_dict


def compare_imputation_methods(original_df, imputed_df, group_cols, fig_size=(15, 5)):
    """
    Compare the distributions of original and KNN-imputed data for each column in a group.

    Parameters:
    -----------
    original_df : pandas.DataFrame
        The dataset before imputation.
    imputed_df : pandas.DataFrame
        The dataset after imputation.
    group_cols : list
        List of numerical column names in the group to compare.
    fig_size : tuple, optional
        Size of each figure (width, height), default is (15, 5).

    Returns:
    --------
    None
    """
    set_plot_style()

    # Generate histograms for each column in the group
    for col in group_cols:
        fig, axes = plt.subplots(1, 2, figsize=fig_size)
        fig.suptitle(f"Distribution Comparison for {col}", fontsize=16)

        # Original distribution
        sns.histplot(original_df[col], bins=20, kde=True, ax=axes[0], color="#264653")
        axes[0].set_title("Original")
        axes[0].set_ylabel("Frequency")
        sns.despine(ax=axes[0], left=True, bottom=True)

        # Imputed distribution
        sns.histplot(imputed_df[col], bins=20, kde=True, ax=axes[1], color="#2a9d8f")
        axes[1].set_title("Imputed")
        axes[1].set_ylabel("Frequency")
        sns.despine(ax=axes[1], left=True, bottom=True)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()


def group_based_knn_imputation(df, group_prefix, k=5, max_k=30):
    """
    Perform KNN imputation on a group of related numerical features with the same prefix.
    If needed, dynamically increases k to find neighbors for all missing values.

    Parameters:
    -----------
    df : pandas.DataFrame
        The dataset to impute.
    group_prefix : str
        Prefix identifying the feature group (e.g., 'Physical' for Physical measurements).
    k : int, optional
        Initial number of neighbors for KNN imputation, default is 5.
    max_k : int, optional
        Maximum number of neighbors to try, default is 30.

    Returns:
    --------
    pandas.DataFrame
        DataFrame with imputed values for the specified group.
    """
    # Get all numerical columns that belong to the group
    group_cols = [
        col
        for col in df.columns
        if col.startswith(f"{group_prefix}-") and df[col].dtype in ["int64", "float64"]
    ]
    if not group_cols:
        print(f"No numerical columns found with prefix '{group_prefix}-'")
        return df

    # Create a copy of the dataframe
    df_imputed = df.copy()

    # Count initial missing values
    missing_before = df_imputed[group_cols].isna().sum().sum()
    print(f"Initial missing values in {group_prefix} group: {missing_before}")

    # Try KNN imputation with progressively larger k values until all values are imputed
    k_values = list(range(k, max_k + 1, 5))
    pbar = tqdm(k_values, desc=f"Imputing {group_prefix} group")
    for current_k in pbar:
        # Initialize the KNN imputer with current k
        imputer = KNNImputer(n_neighbors=current_k)
        # Apply imputation to the group
        try:
            df_imputed[group_cols] = imputer.fit_transform(df_imputed[group_cols])
            # Check if all values were successfully imputed
            missing_after = df_imputed[group_cols].isna().sum().sum()
            pbar.set_postfix({"k": current_k, "missing": missing_after})
            if missing_after == 0:
                pbar.set_postfix({"k": current_k, "status": "Complete"})
                print(
                    f"\nGroup-based KNN imputation completed for {group_prefix} group with k={current_k}"
                )
                print(f"All {missing_before} missing values successfully imputed")
                return df_imputed
            else:
                # Continue to next k value
                continue
        except Exception as e:
            pbar.set_postfix({"k": current_k, "error": str(e)[:20]})

    # If we've reached max_k and still have missing values, use median imputation as last resort
    missing_final = df_imputed[group_cols].isna().sum().sum()
    if missing_final > 0:
        print(
            f"\nWarning: Could not impute all values with KNN up to k={max_k}. Using median imputation as fallback."
        )
        for col in tqdm(group_cols, desc="Median imputation"):
            median_val = df_imputed[col].median()
            if pd.isna(median_val):  # If median itself is NaN (all values missing)
                median_val = 0
            df_imputed[col] = df_imputed[col].fillna(median_val)

    # Final verification
    missing_verification = df_imputed[group_cols].isna().sum().sum()
    if missing_verification > 0:
        print(
            f"Warning: {missing_verification} missing values remain after all imputation attempts"
        )
    else:
        print(f"Verification complete: All {group_prefix} values successfully imputed")

    return df_imputed


def plot_distribution(
    df, column="PCIAT-PCIAT_Total", fig_size=(22, 8), bins=30, kde=True
):
    """
    Plot variable distribution with histogram/KDE and boxplot side-by-side.

    Parameters:
    -----------
    df : pandas.DataFrame
        Dataset containing the column to analyze
    column : str, optional
        Target column to visualize (default: "PCIAT-PCIAT_Total")
    fig_size : tuple, optional
        Figure size (width, height), default (18, 6)
    bins : int, optional
        Number of histogram bins, default 30
    kde : bool, optional
        Show KDE curve, default True

    Returns:
    --------
    None
    """
    set_plot_style()

    # Validate column
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    if not np.issubdtype(df[column].dtype, np.number):
        raise TypeError(f"Column '{column}' is not numeric")

    # Create figure with gridspec
    fig = plt.figure(figsize=fig_size)
    gs = fig.add_gridspec(1, 2, width_ratios=[3, 1])

    # Distribution plot (left)
    ax1 = fig.add_subplot(gs[0])
    sns.histplot(df[column], bins=bins, kde=kde, color="#264653", ax=ax1)
    ax1.set_title(f"Distribution of {column}", fontsize=14)
    ax1.set_xlabel("Value", fontsize=12)
    ax1.set_ylabel("Frequency", fontsize=12)

    # Boxplot (right)
    ax2 = fig.add_subplot(gs[1])
    sns.boxplot(y=df[column], color="#2a9d8f", ax=ax2, width=0.5)
    ax2.set_title(f"Spread of {column}", fontsize=14)
    ax2.set_ylabel("")

    # Add statistics annotations
    stats = df[column].describe()
    ax1.text(
        0.05,
        0.95,
        f"Mean: {stats['mean']:.2f}\nMedian: {stats['50%']:.2f}\nStd: {stats['std']:.2f}",
        transform=ax1.transAxes,
        verticalalignment="top",
        fontfamily="Candara",
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="#264653"),
    )

    # Styling
    for ax in [ax1, ax2]:
        ax.grid(axis="y", linestyle="--", alpha=0.7)
        sns.despine(ax=ax, left=True, bottom=True)

    plt.tight_layout()
    plt.show()


def apply_boxcox_transform(
    df,
    column="PCIAT-PCIAT_Total",
    save_transform=True,
    fig_size=(18, 6),
    output_dir="../data/transformation",
):
    """
    Apply Box-Cox transformation, compare distributions, and save transformation parameters.

    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataset
    column : str, optional
        Column to transform (default: "PCIAT-PCIAT_Total")
    save_transform : bool, optional
        Save transformation parameters (default: True)
    fig_size : tuple, optional
        Figure size (width, height), default (18, 6)
    output_dir : str, optional
        Directory to save transformation files (default: "../data/transformation")

    Returns:
    --------
    transformed_data : array
        Box-Cox transformed data
    transform_params : dict
        Transformation parameters (lambda, shift, original_min)
    """
    set_plot_style()

    # Validate inputs
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    if not np.issubdtype(df[column].dtype, np.number):
        raise TypeError(f"Column '{column}' is not numeric")

    # Prepare data
    original_data = df[column].values
    original_min = original_data.min()

    # Handle non-positive values
    shift = 0
    if original_min <= 0:
        shift = abs(original_min) + 1
        transformed_data = original_data + shift
    else:
        transformed_data = original_data.copy()

    # Apply Box-Cox transformation
    try:
        transformed_data, lambda_ = boxcox(transformed_data)
    except ValueError as e:
        print(f"Box-Cox failed: {str(e)}. Trying Yeo-Johnson instead.")
        raise

    # Create output directory
    if save_transform:
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # filename = f"{column}_boxcox_params_{timestamp}.pkl"
        filename = f"{column}_boxcox_params.pkl"
        params = {"lambda": lambda_, "shift": shift, "original_min": original_min}
        with open(os.path.join(output_dir, filename), "wb") as f:
            pickle.dump(params, f)

    # Create comparison plot
    fig = plt.figure(figsize=fig_size)
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1])

    # Original distribution
    ax1 = fig.add_subplot(gs[0])
    sns.histplot(original_data, bins=30, kde=True, color="#264653", ax=ax1)
    ax1.set_title("Original Distribution", fontsize=14)
    ax1.set_xlabel("Value", fontsize=12)
    ax1.set_ylabel("Density", fontsize=12)

    # Transformed distribution
    ax2 = fig.add_subplot(gs[1])
    sns.histplot(transformed_data, bins=30, kde=True, color="#2a9d8f", ax=ax2)
    ax2.set_title(f"Box-Cox (λ={lambda_:.3f})", fontsize=14)
    ax2.set_xlabel("Transformed Value", fontsize=12)

    # Add statistics
    for ax, data in [(ax1, original_data), (ax2, transformed_data)]:
        stats = f"Mean: {data.mean():.2f}\nStd: {data.std():.2f}\nSkew: {pd.Series(data).skew():.2f}"
        ax.text(
            0.95,
            0.95,
            stats,
            transform=ax.transAxes,
            ha="right",
            va="top",
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="#264653"),
        )

    # Styling
    for ax in [ax1, ax2]:
        ax.grid(axis="y", linestyle="--", alpha=0.7)
        sns.despine(ax=ax, left=True, bottom=True)

    plt.tight_layout()
    plt.show()

    return transformed_data, {
        "lambda": lambda_,
        "shift": shift,
        "original_min": original_min,
    }


def plot_group_correlations_separate(
    df, target_col="PCIAT-PCIAT_Total", fig_size=(12, 10)
):
    """
    Generate separate heatmaps for each feature group showing correlations,
    displaying only the lower triangular portion of each correlation matrix.

    Parameters:
    -----------
    df : pandas.DataFrame
        Dataset containing features and target
    target_col : str, optional
        Name of target column (default: "PCIAT-PCIAT_Total")
    fig_size : tuple, optional
        Figure size (width, height) for each plot, default (12, 10)
    """
    set_plot_style()

    # Define feature groups
    feature_groups = {
        "Demographics": [col for col in df.columns if col.startswith("Basic_Demos-")],
        "CGAS": [col for col in df.columns if col.startswith("CGAS-")],
        "Physical": [col for col in df.columns if col.startswith("Physical-")],
        "Fitness": [col for col in df.columns if col.startswith("Fitness_Endurance-")],
        "FGC": [col for col in df.columns if col.startswith("FGC-")],
        "BIA": [col for col in df.columns if col.startswith("BIA-")],
        "PAQ": [col for col in df.columns if col.startswith("PAQ_")],
        "SDS": [col for col in df.columns if col.startswith("SDS-")],
        "Internet": [col for col in df.columns if col.startswith("PreInt_EduHx-")],
    }

    for group_name, features in feature_groups.items():
        # Filter numerical features
        numerical_features = (
            df[features].select_dtypes(include=[np.number]).columns.tolist()
        )

        if len(numerical_features) <= 1:
            print(f"Skipping {group_name}: Not enough numerical features")
            continue

        # Include target in the analysis
        analysis_cols = numerical_features.copy()
        if target_col not in analysis_cols and target_col in df.columns:
            analysis_cols.append(target_col)

        # Calculate correlations
        pearson_corr = df[analysis_cols].corr(method="pearson")
        spearman_corr = df[analysis_cols].corr(method="spearman")

        # Create mask for lower triangle (including diagonal)
        mask = np.triu(np.ones_like(pearson_corr, dtype=bool))

        # Plot Pearson correlation matrix (lower triangle)
        plt.figure(figsize=fig_size)
        ax1 = sns.heatmap(
            pearson_corr,
            mask=mask,
            annot=True,
            cmap="coolwarm",
            fmt=".2f",
            linewidths=0.5,
            annot_kws={"size": 10},
            vmin=-1,
            vmax=1,
            cbar_kws={"label": "Correlation Coefficient"},
        )
        plt.title(f"{group_name} - Pearson (Linear) Correlation", fontsize=16, pad=20)
        plt.tight_layout()
        plt.show()

        # Plot Spearman correlation matrix (lower triangle)
        plt.figure(figsize=fig_size)
        ax2 = sns.heatmap(
            spearman_corr,
            mask=mask,
            annot=True,
            cmap="coolwarm",
            fmt=".2f",
            linewidths=0.5,
            annot_kws={"size": 10},
            vmin=-1,
            vmax=1,
            cbar_kws={"label": "Correlation Coefficient"},
        )
        plt.title(
            f"{group_name} - Spearman (Non-Linear) Correlation", fontsize=16, pad=20
        )
        plt.tight_layout()
        plt.show()
