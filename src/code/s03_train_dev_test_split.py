"""
This script performs a stratified train-dev-test split of patient data.

Usage:
    poetry run python s03_train_dev_test_split.py
"""

import pandas as pd
from sklearn.model_selection import train_test_split


def load_data(file_path: str) -> pd.DataFrame:
    """
    Load the dataset from a CSV file.

    Args:
        file_path (str): Path to the input CSV file.

    Returns:
        pd.DataFrame: Loaded dataset.
    """
    return pd.read_csv(file_path)


def split_patients(
    data: pd.DataFrame, stratify_col: str
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Perform a stratified split into train, dev, and test sets.

    Args:
        data (pd.DataFrame): Data containing patient information.
        stratify_col (str): Column name for stratification.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
            DataFrames for train, dev, and test splits.
    """
    # Extract unique patients with their stratification labels
    patients = data.groupby("patient_number").agg({stratify_col: "first"}).reset_index()

    # Split into train+dev and test (80-20 split)
    train_dev_patients, test_patients = train_test_split(
        patients, test_size=0.2, stratify=patients[stratify_col], random_state=42
    )

    # Further split train+dev into train and dev (75-25 split of train+dev)
    train_patients, dev_patients = train_test_split(
        train_dev_patients,
        test_size=0.25,
        stratify=train_dev_patients[stratify_col],
        random_state=42,
    )

    return train_patients, dev_patients, test_patients


def assign_splits(
    data: pd.DataFrame,
    train_patients: pd.DataFrame,
    dev_patients: pd.DataFrame,
    test_patients: pd.DataFrame,
) -> pd.DataFrame:
    """
    Assign split labels (train, dev, test) to the original data based on patient splits.

    Args:
        data (pd.DataFrame): Original dataset.
        train_patients (pd.DataFrame): DataFrame for train patients.
        dev_patients (pd.DataFrame): DataFrame for dev patients.
        test_patients (pd.DataFrame): DataFrame for test patients.

    Returns:
        pd.DataFrame: Data with an added 'split' column indicating the assigned split.
    """
    data["split"] = "train"
    data.loc[data["patient_number"].isin(dev_patients["patient_number"]), "split"] = (
        "dev"
    )
    data.loc[data["patient_number"].isin(test_patients["patient_number"]), "split"] = (
        "test"
    )
    return data


def save_split_data(data: pd.DataFrame, output_path: str) -> None:
    """
    Save the split data to a CSV file.

    Args:
        data (pd.DataFrame): Data containing patient splits.
        output_path (str): Path to save the output CSV file.
    """
    data.to_csv(output_path, index=False)


def main():
    """
    Main function to perform train-dev-test split and save the results.
    """
    input_file = "../../output/audio_summary.csv"
    output_file = "../../output/split.csv"

    # Load data
    data = load_data(input_file)

    # Perform stratified split
    train_patients, dev_patients, test_patients = split_patients(
        data, stratify_col="chest_location"
    )

    # Assign splits back to the original data
    data = assign_splits(data, train_patients, dev_patients, test_patients)

    # Save split data
    save_split_data(data[["patient_number", "split"]], output_file)

    # Print split counts
    print("Split counts:")
    print(data["split"].value_counts())


if __name__ == "__main__":
    main()
