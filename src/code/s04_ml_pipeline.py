"""This file contains the code for training and testing the proposed model.

Before running this code, you may want to configure wandb and huggingface:
- wandb login --relogin
- huggingface-cli login

Usage:
    poetry run python s04_ml_pipeline.py \
    --pretrained_model_id "MIT/ast-finetuned-audioset-14-14-0.443" \
    --learning_rate 1e-4 \
    --batch_size 16 \
    --max_epochs 20 \
    --freeze_encoder \
    --save_model \
    --push_to_hub \
    --repo_id "fabiocat/icbhi_classification"
"""

import os
import json
import time
from typing import List, Tuple, Union
import seaborn as sns

import torch
import torchaudio
import torch.nn as nn
import pytorch_lightning as pl
import pandas as pd
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
)
import matplotlib.pyplot as plt
from torch_audiomentations import Compose, Gain, Shift
from transformers import AutoModel, AutoProcessor
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
import glob
import argparse
from huggingface_hub import PyTorchModelHubMixin
from collections import Counter
import random


class AudioDataset(torch.utils.data.Dataset):
    """
    Custom dataset for loading, processing, and balancing audio files for multi-label classification.

    Attributes:
        file_paths (list): List of file paths to audio files.
        labels (list): Corresponding binary labels for crackle_flag and wheeze_flag.
        processor: Transformer processor for feature extraction.
        augment (optional): Audio augmentation pipeline.
        win_length (int): Minimum window length in milliseconds for audio samples.
        balance_classes (bool): Whether to balance classes during dataset initialization.
    """

    def __init__(
        self,
        file_paths: List[Union[Tuple[str, str], str]],
        labels: List[List[int]],
        processor,
        augment=None,
        win_length: int = 25,
        concatenate_samples: bool = False,
        increase_factor: int = 1,
    ):
        self.file_paths: List[Union[Tuple[str, str], str]] = file_paths
        self.labels = labels
        self.processor = processor
        self.augment = augment
        self.concatenate_samples = concatenate_samples

        self.min_length = (
            win_length / 1000 * processor.sampling_rate
            if hasattr(processor, "sampling_rate")
            else win_length / 1000 * processor.feature_extractor.sampling_rate
        )

        if self.concatenate_samples:
            print("Balancing classes...")
            self._concatenate_samples(increase_factor)

    def _concatenate_samples(self, increase_factor: int = 1):
        """
        Generating synthetic samples by concatenating two random samples
        where the label is the logical union (max) of two selected labels.

        Args:
            increase_factor (int): The min factor by which to increase the number of samples for each label.
        """
        label_counts = [Counter(label[i] for label in self.labels) for i in range(2)]
        max_counts = [
            max(counter.values()) * increase_factor for counter in label_counts
        ]

        print("Initial counts:")
        for idx, counter in enumerate(label_counts):
            print(f"Label[{idx}] counts: {dict(counter)}")

        print(f"Tentative target counts for balancing: {max_counts}")

        augmented_file_paths = []
        augmented_labels = []

        for label_idx, max_count in enumerate(max_counts):
            for value in [0, 1]:
                while label_counts[label_idx][value] < max_count:
                    idx1 = random.choice(
                        [
                            i
                            for i, lbl in enumerate(self.labels)
                            if lbl[label_idx] == value
                        ]
                    )
                    idx2 = random.randint(0, len(self.file_paths) - 1)

                    new_label = [
                        max(self.labels[idx1][0], self.labels[idx2][0]),
                        max(self.labels[idx1][1], self.labels[idx2][1]),
                    ]

                    augmented_file_paths.append(
                        (self.file_paths[idx1], self.file_paths[idx2])
                    )
                    augmented_labels.append(new_label)
                    label_counts[label_idx][value] += 1

        # Extend the dataset with augmented data after loops
        self.file_paths.extend(augmented_file_paths)  # type: ignore[arg-type]
        self.labels.extend(augmented_labels)  # type: ignore[arg-type]

        # Final label counts
        final_label_counts = [
            Counter(label[i] for label in self.labels) for i in range(2)
        ]
        print("Final counts after concatenation:")
        for idx, counter in enumerate(final_label_counts):
            print(f"Label[{idx}] counts: {dict(counter)}")

    def __getitem__(self, idx: int):
        """
        Returns the processed waveform and corresponding label for a given index.

        Args:
            idx (int): Index of the sample.

        Returns:
            tuple: Processed input values and corresponding label tensor.
        """
        if isinstance(self.file_paths[idx], tuple):  # Augmented pair
            audio_path1, audio_path2 = self.file_paths[idx]  # type: ignore
            waveform1, sr1 = torchaudio.load(audio_path1)
            waveform2, sr2 = torchaudio.load(audio_path2)

            if sr1 != sr2:
                raise ValueError("Sampling rates of paired audio files must match.")

            waveform = torch.cat(
                [waveform1.mean(0), waveform2.mean(0)], dim=-1
            ).unsqueeze(0)
            sr = sr1
        else:
            audio_path = self.file_paths[idx]
            waveform, sr = torchaudio.load(audio_path)
            waveform = waveform.mean(0).unsqueeze(0)

        # Ensure waveform meets the minimum length
        while waveform.size(-1) < self.min_length:
            waveform = torch.cat([waveform, waveform], dim=-1)

        # Apply augmentation if specified
        if self.augment:
            waveform = self.augment(
                samples=waveform.unsqueeze(0), sample_rate=sr
            ).squeeze(0)

        # Process the waveform using the processor
        inputs = self.processor(
            waveform.squeeze(),
            sampling_rate=sr,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )

        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return inputs.input_values.squeeze(0), label

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.file_paths)


class FabModel(pl.LightningModule, PyTorchModelHubMixin):
    """
    PyTorch Model for audio classification using a pre-trained Transformer model.

    Attributes:
        encoder_id (str): ID of the pre-trained transformer encoder.
        num_labels (int): Number of output labels.
        learning_rate (float): Learning rate for the optimizer.
        frozen (bool): Whether to freeze the encoder layers.
    """

    def __init__(
        self,
        encoder_id: str,
        num_labels: int,
        learning_rate: float = 1e-4,
        frozen: bool = True,
    ) -> None:
        """
        Initializes the FabModel.

        Args:
            encoder_id (str): ID of the pre-trained transformer encoder.
            num_labels (int): Number of output labels.
            learning_rate (float, optional): Learning rate for the optimizer. Defaults to 1e-4.
            frozen (bool, optional): Whether to freeze the encoder layers. Defaults to True.
        """
        super().__init__()
        self.model = AutoModel.from_pretrained(encoder_id, trust_remote_code=True)
        self.classifier = nn.Linear(self.model.config.hidden_size, num_labels)
        self.relu = nn.ReLU()
        self.criterion = nn.BCEWithLogitsLoss()
        self.dropout = nn.Dropout(0.5)
        self.learning_rate = learning_rate

        if frozen:
            for param in self.model.parameters():
                param.requires_grad = False

        self.test_results: list[dict] = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        outputs = self.model(x).last_hidden_state
        pooled_output = outputs.mean(dim=1)
        pooled_output = self.relu(pooled_output)
        pooled_output = self.dropout(pooled_output)
        return self.classifier(pooled_output)

    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        """
        Training step for the model.

        Args:
            batch (tuple): Batch of input data and labels.
            batch_idx (int): Index of the current batch.

        Returns:
            torch.Tensor: Loss tensor.
        """
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        """
        Validation step for the model.

        Args:
            batch (tuple): Batch of input data and labels.
            batch_idx (int): Index of the current batch.

        Returns:
            torch.Tensor: Loss tensor.
        """
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        self.log("val_loss", loss)
        return loss

    def test_step(self, batch: tuple, batch_idx: int) -> dict:
        """
        Test step for the model.

        Args:
            batch (tuple): Batch of input data and labels.
            batch_idx (int): Index of the current batch.

        Returns:
            dict: Dictionary containing logits and labels.
        """
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        self.log("test_loss", loss)

        self.test_results.append({"logits": logits.detach(), "labels": y.detach()})
        return {"logits": logits, "labels": y}

    def on_test_epoch_end(self) -> None:
        """
        Aggregates outputs from all test_step calls at the end of the test phase.
        """
        # Collect all logits and labels
        all_logits = torch.cat(
            [result["logits"] for result in self.test_results], dim=0
        )
        all_labels = torch.cat(
            [result["labels"] for result in self.test_results], dim=0
        )

        # Store for later use or saving
        self.test_results = {"logits": all_logits, "labels": all_labels}  # type: ignore

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """
        Configures the optimizer for the model.

        Returns:
            torch.optim.Optimizer: Optimizer object.
        """
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)


# Utility functions
def compute_metrics(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    y_pred_binary: torch.Tensor,
    labels: list,
) -> dict:
    """
    Computes various metrics for audio classification.

    Args:
        y_true (torch.Tensor): True labels.
        y_pred (torch.Tensor): Predicted labels.
        y_pred_binary (torch.Tensor): Predicted binary labels.
        labels (list): The list of labels

    Returns:
        dict: Dictionary containing various metrics.
    """
    num_labels = y_true.shape[1]
    metrics = {
        "label_specific": {
            f"label_{labels[i]}": {
                "f1_score": f1_score(y_true[:, i], y_pred_binary[:, i]),
                "precision": precision_score(y_true[:, i], y_pred_binary[:, i]),
                "recall": recall_score(y_true[:, i], y_pred_binary[:, i]),
                "roc_auc": roc_auc_score(y_true[:, i], y_pred[:, i]),
            }
            for i in range(num_labels)
        },
        "aggregated": {
            "f1_macro": f1_score(y_true, y_pred_binary, average="macro"),
            "precision_macro": precision_score(y_true, y_pred_binary, average="macro"),
            "recall_macro": recall_score(y_true, y_pred_binary, average="macro"),
            "f1_micro": f1_score(y_true, y_pred_binary, average="micro"),
            "precision_micro": precision_score(y_true, y_pred_binary, average="micro"),
            "recall_micro": recall_score(y_true, y_pred_binary, average="micro"),
            "auc_macro": roc_auc_score(y_true, y_pred, average="macro"),
            "auc_micro": roc_auc_score(y_true, y_pred, average="micro"),
        },
    }
    return metrics


def save_metrics(metrics: dict, output_dir: str):
    """
    Save metrics to a JSON file.

    Args:
        metrics (dict): Dictionary containing metrics.
        output_dir (str): Directory to save the metrics file.
    """
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f)


def save_confusion_matrices(
    y_true: torch.Tensor, y_pred_binary: torch.Tensor, output_dir: str, labels: list
):
    """
    Save confusion matrices for each label.

    Args:
        y_true (torch.Tensor): True labels.
        y_pred_binary (torch.Tensor): Predicted binary labels.
        output_dir (str): Directory to save the confusion matrices.
        labels (list): The list of labels
    """
    os.makedirs(output_dir, exist_ok=True)
    num_labels = y_true.shape[1]

    fig, axs = plt.subplots(1, num_labels, figsize=(12, 6))
    for i in range(num_labels):
        cm = confusion_matrix(y_true[:, i], y_pred_binary[:, i])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=axs[i])
        axs[i].set_title(f"Confusion Matrix for Label {labels[i]}")
        axs[i].set_xlabel("Predicted")
        axs[i].set_ylabel("True")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confusion_matrices.png"))
    plt.close(fig)


def save_files(file_dict: dict, output_dir: str):
    """
    Save file lists to JSON files.

    Args:
        file_dict (dict): Dictionary containing file lists.
        output_dir (str): Directory to save the JSON files.
    """
    os.makedirs(output_dir, exist_ok=True)
    for file_type, file_list in file_dict.items():
        with open(os.path.join(output_dir, f"{file_type}_files.json"), "w") as f:
            json.dump(file_list, f)


def save_model_to_disk(model: torch.nn.Module, training_results_dir: str):
    """
    Save the model to disk.

    Args:
        model (torch.nn.Module): Model to save.
        training_results_dir (str): Directory to save the model.
    """
    model_dir = os.path.join(training_results_dir, "model")
    os.makedirs(model_dir, exist_ok=True)
    try:
        model.save_pretrained(model_dir)
        _ = FabModel.from_pretrained(model_dir)
        print("Model saved to disk.")
    except Exception as e:
        print(f"Error saving model to disk: {e}")


def push_to_hub(model: torch.nn.Module, repo_id: str):
    """
    Push the model to the Hugging Face Hub.

    Args:
        model (torch.nn.Module): Model to push.
        repo_id (str): URL of the Hugging Face repository.
    """
    try:
        model.push_to_hub(repo_id)
        _ = FabModel.from_pretrained(repo_id)
        print("Model pushed to the hub successfully.")
    except Exception as e:
        print(f"Error pushing model to the hub: {e}")


def save_experiment_metadata(
    training_results_dir: str,
    training_time: float,
    batch_size: int,
    learning_rate: float,
    max_epochs: int,
    encoder_id: str,
    train_files: list,
    dev_files: list,
    test_files: list,
):
    """
    Save experiment metadata to a JSON file.

    Args:
        training_results_dir (str): Directory to save the metadata file.
        training_time (float): Training time in seconds.
        batch_size (int): Batch size.
        learning_rate (float): Learning rate.
        max_epochs (int): Maximum number of epochs.
        encoder_id (str): ID of the pre-trained transformer encoder.
        train_files (list): List of training files.
        dev_files (list): List of development files.
        test_files (list): List of test files.
    """
    metadata = {
        "training_time": training_time,
        "hardware": {
            "device": "GPU" if torch.cuda.is_available() else "CPU",
            "gpu_model": torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else None,
            "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
            "gpu_memory_total": f"{torch.cuda.get_device_properties(0).total_memory / (1024 ** 3):.2f} GB"
            if torch.cuda.is_available()
            else None,
        },
        "training_params": {
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "optimizer": "Adam",
            "loss_function": "BCEWithLogitsLoss",
            "max_epochs": max_epochs,
            "pretrained_model_id": encoder_id,
        },
        "data_split": {
            "train_files": train_files,
            "dev_files": dev_files,
            "test_files": test_files,
        },
    }

    metadata_dir = os.path.join(training_results_dir, "metadata")
    os.makedirs(metadata_dir, exist_ok=True)
    with open(os.path.join(metadata_dir, "experiment_metadata.json"), "w") as f:
        json.dump(metadata, f)
    print("Experiment metadata saved.")


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Read parameters from command line arguments."
    )

    # Add arguments in the same order as in the main function
    parser.add_argument(
        "--pretrained_model_id",
        type=str,
        default="MIT/ast-finetuned-audioset-14-14-0.443",
        help="ID of the pretrained model.",
    )
    parser.add_argument(
        "--freeze_encoder",
        action="store_true",
        default=False,
        help="Freeze encoder weights.",
    )
    parser.add_argument(
        "--max_epochs", type=int, default=0, help="Maximum number of training epochs."
    )
    parser.add_argument(
        "--batch_size", type=int, default=16, help="Batch size for training."
    )
    parser.add_argument(
        "--learning_rate", type=float, default=1e-4, help="Learning rate for training."
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=16,
        help="Number of workers for data loading.",
    )
    parser.add_argument(
        "--augmentation_flag",
        action="store_true",
        default=False,
        help="Enable data augmentation.",
    )
    parser.add_argument(
        "--high_pass_cutoff",
        type=int,
        default=0,
        help="High-pass filter cutoff frequency.",
    )
    parser.add_argument(
        "--low_pass_cutoff",
        type=int,
        default=4000,
        help="Low-pass filter cutoff frequency.",
    )
    parser.add_argument(
        "--order",
        type=int,
        default=4,
        help="Order of the Butterworth filter.",
    )
    parser.add_argument(
        "--save_model",
        action="store_true",
        default=False,
        help="Save model checkpoint.",
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        default=False,
        help="Push model to Hugging Face Hub.",
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        help="Hugging Face Hub repository ID.",
    )

    return parser.parse_args()


def main():
    """
    Main function for training and evaluating the model.
    """
    # Parse arguments
    args = parse_args()

    # Print argument details for verification
    print("Model Training Configuration:")
    print(f"  Pretrained encoder ID: {args.pretrained_model_id}")
    print(f"  Freeze encoder: {args.freeze_encoder}")
    print(f"  Maximum epochs: {args.max_epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Number of workers: {args.num_workers}")
    print(f"  Data augmentation flag: {args.augmentation_flag}")
    print(f"  High-pass cutoff frequency: {args.high_pass_cutoff}")
    print(f"  Low-pass cutoff frequency: {args.low_pass_cutoff}")
    print(f"  Order of the Butterworth filter: {args.order}")
    print(f"  Save model: {args.save_model}")
    print(f"  Push to Hub: {args.push_to_hub}")
    print(f"  Hugging Face Hub repository ID: {args.repo_id}")

    # Define experiment name and directories
    experiment_name = (
        f"audio_classification_experiment__16000__{args.high_pass_cutoff}__"
        f"{args.low_pass_cutoff}__{args.order}__{args.pretrained_model_id.replace('/', '_')}"
    )
    training_results_dir = os.path.join("../../training_results", experiment_name)
    dataset_path = os.path.join(
        "../../output",
        f"16000__{args.high_pass_cutoff}__{args.low_pass_cutoff}__{args.order}",
    )
    split_file = "../../output/split.csv"

    os.makedirs(training_results_dir, exist_ok=True)

    # Initialize logging
    print("Initializing wandb logging...")
    wandb_logger = WandbLogger(
        project="audio_classification", name=experiment_name, save_dir="../../wandb"
    )

    # Load dataset split information
    print("Loading dataset split information...")
    split_df = pd.read_csv(split_file)

    # Prepare file paths and labels for splits
    splits = {"train": ([], []), "dev": ([], []), "test": ([], [])}
    for _, row in split_df.iterrows():
        label_files = glob.glob(
            os.path.join(dataset_path, f"{row['patient_number']}_*.wav")
        )
        for file in label_files:
            try:
                crackle, wheeze = map(int, file.replace(".wav", "").split("__")[-2:])
                label = [crackle, wheeze]
                splits[row["split"]][0].append(file)
                splits[row["split"]][1].append(label)
            except ValueError:
                print(f"Invalid file format: {file}")

    train_files, train_labels = splits["train"]
    dev_files, dev_labels = splits["dev"]
    test_files, test_labels = splits["test"]

    # Initialize processor
    try:
        print(f"Initializing processor for {args.pretrained_model_id}...")
        processor = AutoProcessor.from_pretrained(
            args.pretrained_model_id, trust_remote_code=True
        )
    except Exception as e:
        print(f"Error initializing processor for {args.pretrained_model_id}: {e}")
        exit(1)

    # Define augmentation transforms
    print("Defining augmentation transforms...")
    augment = (
        Compose(
            transforms=[
                Gain(min_gain_in_db=-15, max_gain_in_db=15, p=0.5),
                Shift(
                    min_shift=0.1,
                    max_shift=0.9,
                    shift_unit="fraction",
                    rollover=True,
                    p=0.5,
                ),
            ]
        )
        if args.augmentation_flag
        else None
    )

    # Initialize datasets
    print("Initializing datasets...")
    train_dataset = AudioDataset(
        train_files,
        train_labels,
        processor,
        augment=augment,
        concatenate_samples=True,
        increase_factor=2,
    )
    dev_dataset = AudioDataset(dev_files, dev_labels, processor)
    test_dataset = AudioDataset(test_files, test_labels, processor)

    # Initialize data loaders
    print("Initializing data loaders...")
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=min(args.num_workers, os.cpu_count()),
    )
    dev_loader = torch.utils.data.DataLoader(
        dev_dataset,
        batch_size=args.batch_size,
        num_workers=min(args.num_workers, os.cpu_count()),
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        num_workers=min(args.num_workers, os.cpu_count()),
    )

    # Training
    print("Starting training...")
    model = FabModel(
        encoder_id=args.pretrained_model_id,
        num_labels=2,
        learning_rate=args.learning_rate,
        frozen=args.freeze_encoder,
    )
    early_stopping = EarlyStopping(
        monitor="val_loss", patience=3, min_delta=0.01, mode="min", verbose=True
    )
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        precision="16-mixed",
        log_every_n_steps=10,
        accelerator="auto",
        devices="auto",
        logger=wandb_logger,
        callbacks=[early_stopping],
    )
    start_time = time.time()
    trainer.fit(model, train_loader, dev_loader)
    training_time = time.time() - start_time

    # Saving model
    print("Saving model...")
    if args.save_model:
        save_model_to_disk(model, training_results_dir)

    # Save experiment metadata
    print("Saving experiment metadata...")
    save_experiment_metadata(
        training_results_dir=training_results_dir,
        training_time=training_time,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_epochs=args.max_epochs,
        encoder_id=args.pretrained_model_id,
        train_files=train_files,
        dev_files=dev_files,
        test_files=test_files,
    )

    # Evaluation
    print("Starting evaluation...")
    trainer.test(model, test_loader)
    all_logits = model.test_results["logits"].cpu().numpy()
    all_labels = model.test_results["labels"].cpu().numpy()
    all_predictions_binary = (all_logits > 0).astype(int)

    # Save metrics
    print("Saving evaluation metrics...")
    label_names = ["Crackle", "Wheeze"]
    metrics = compute_metrics(
        y_true=torch.tensor(all_labels),
        y_pred=torch.tensor(all_logits),
        y_pred_binary=torch.tensor(all_predictions_binary),
    )
    save_metrics(metrics, os.path.join(training_results_dir, "evaluation"))

    # Save confusion matrices
    print("Saving confusion matrices...")
    save_confusion_matrices(
        y_true=torch.tensor(all_labels),
        y_pred_binary=torch.tensor(all_predictions_binary),
        output_dir=os.path.join(training_results_dir, "evaluation"),
        labels=label_names,
    )

    # Push to hub
    if args.push_to_hub:
        print("Pushing model to Hugging Face Hub...")
        push_to_hub(model, args.repo_id)


if __name__ == "__main__":
    main()
