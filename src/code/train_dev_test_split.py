import pandas as pd
from sklearn.model_selection import train_test_split

# Load the CSV file
data = pd.read_csv("../../output/audio_summary.csv")

# Extract unique patients with their stratification labels
patients = data.groupby("patient_number").agg({"chest_location": "first"}).reset_index()

# Stratify by `chest_location` only
stratify_col = patients["chest_location"]

# Split patients into train+dev and test (80-20 split)
train_dev_patients, test_patients = train_test_split(
    patients, test_size=0.2, stratify=stratify_col, random_state=42
)

# Further split train+dev into train and dev (e.g., 75-25 split of train+dev)
train_patients, dev_patients = train_test_split(
    train_dev_patients,
    test_size=0.25,
    stratify=train_dev_patients["chest_location"],
    random_state=42,
)


# Assign splits back to the original data
def assign_splits(data, train_patients, dev_patients, test_patients):
    data["split"] = "train"
    data.loc[data["patient_number"].isin(dev_patients["patient_number"]), "split"] = (
        "dev"
    )
    data.loc[data["patient_number"].isin(test_patients["patient_number"]), "split"] = (
        "test"
    )
    return data


# Assign splits
data = assign_splits(data, train_patients, dev_patients, test_patients)
data = data[["patient_number", "split"]]
# Save the split data
data.to_csv("../../output/split.csv", index=False)

# Print split counts
print("Split counts:")
print(data["split"].value_counts())
