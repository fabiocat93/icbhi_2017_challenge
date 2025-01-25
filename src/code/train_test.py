"""This file contains the code for training and testing the proposed model."""

# !wandb login --relogin
# huggingface-cli login

# %%
import os
import json
import time
from typing import Tuple, Dict, List, Any

import torch
import torchaudio
import torch.nn as nn
import pytorch_lightning as pl
import pandas as pd
import numpy as np
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
from optimum.onnxruntime import ORTModelForAudioClassification


# %%
class AudioDataset(torch.utils.data.Dataset):
    """
    Custom dataset for loading and processing audio files for multi-label classification.

    Attributes:
        file_paths (list): List of file paths to audio files.
        labels (list): Corresponding labels for crackle_flag and wheeze_flag.
        processor: Transformer processor for feature extraction.
        augment: Optional audio augmentation pipeline.
    """

    def __init__(
        self, file_paths: list, labels: list, processor, augment=None, win_length=25
    ):
        self.file_paths = file_paths
        self.labels = labels
        self.processor = processor
        self.augment = augment
        try:
            self.min_length = win_length / 1000 * processor.sampling_rate
        except Exception:
            self.min_length = (
                win_length / 1000 * processor.feature_extractor.sampling_rate
            )

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        audio_path = self.file_paths[idx]
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        waveform, sr = torchaudio.load(audio_path)
        waveform = waveform.mean(dim=0).unsqueeze(0)  # Mono conversion

        while waveform.size(-1) < self.min_length:
            print(f"Extending {audio_path} to {self.min_length} samples")
            waveform = torch.cat([waveform, waveform], dim=-1)

        if self.augment:
            waveform = waveform.unsqueeze(0)
            waveform = self.augment(samples=waveform, sample_rate=sr)

        try:
            inputs = self.processor(
                waveform.squeeze(),
                sampling_rate=sr,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
        except Exception as e:
            print(f"Error processing {audio_path}: {e}")
            print(waveform.shape, sr)
        return inputs.input_values.squeeze(0), label


class AudioClassificationModel(pl.LightningModule, PyTorchModelHubMixin):
    """
    # TODO: add metadata
    LightningModule for audio classification using a pre-trained Transformer model.

    Attributes:
        model_name (str): Name of the pre-trained transformer model.
        num_labels (int): Number of output labels.
    """

    def __init__(
        self,
        model_name: str = "MIT/ast-finetuned-audioset-14-14-0.443",
        num_labels: int = 2,
        training_results_dir: str = "./training_results_dir",
        frozen=True,
        learning_rate: float = 1e-4,
        repo_url="https://github.com/fabiocat93/icbhi_2017_challenge",
        pipeline_tag="audio-classification",
        license="mit",
        tags=["audio", "classification"],
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        self.frozen_encoder = frozen
        print("Frozen model layers:")
        for name, param in self.model.named_parameters():
            if "embeddings" in name or frozen:
                param.requires_grad = False
                print(name)
        self.classifier = nn.Linear(self.model.config.hidden_size, num_labels)
        self.relu = nn.ReLU()
        self.criterion = nn.BCEWithLogitsLoss()
        self.training_results_dir = training_results_dir
        self.test_outputs: List[Dict[str, Any]] = []
        self.training_start_time = None  # Track training start time
        self.pretrained_model_id = model_name  # Store the pretrained model ID
        self.learning_rate = learning_rate

    def on_train_start(self):
        # Record the start time of training
        self.training_start_time = time.time()

    def on_train_end(self):
        try:
            # Calculate training time
            if self.training_start_time:
                training_time = time.time() - self.training_start_time
            else:
                training_time = None

            # TODO: add more info (e.g., the actual number of epochs to the metadata)
            # Collect metadata
            metadata = {
                "training_time": training_time,
                "hardware": {
                    "device": "GPU" if torch.cuda.is_available() else "CPU",
                    "gpu_model": torch.cuda.get_device_name(0)
                    if torch.cuda.is_available()
                    else None,
                    "cuda_version": torch.version.cuda
                    if torch.cuda.is_available()
                    else None,
                    "gpu_memory_total": f"{torch.cuda.get_device_properties(0).total_memory / (1024 ** 3):.2f} GB"
                    if torch.cuda.is_available()
                    else None,
                },
                "training_params": {
                    "batch_size": self.trainer.train_dataloader.loaders.batch_size,
                    "learning_rate": self.learning_rate,
                    "optimizer": "Adam",
                    "loss_function": "BCEWithLogitsLoss",
                    "max_epochs": self.trainer.max_epochs,
                    "dataset_path": getattr(
                        self, "dataset_path", "Unknown"
                    ),  # Optionally set dataset_path as an attribute
                    "pretrained_model_id": self.pretrained_model_id,
                },
                "data_split": {
                    "train_files": getattr(self, "train_files", []),
                    "dev_files": getattr(self, "dev_files", []),
                    "test_files": getattr(self, "test_files", []),
                },
            }

            # Save metadata
            os.makedirs(self.training_results_dir, exist_ok=True)
            with open(f"{self.training_results_dir}/metadata.json", "w") as f:
                json.dump(metadata, f)
        except Exception as e:
            print(f"Error saving metadata: {e}")

        if self.save_model:
            # Export the model to ONNX
            onnx_export_dir = os.path.join(self.training_results_dir, "onnx_model")
            os.makedirs(onnx_export_dir, exist_ok=True)

            # Save the model locally
            self.save_pretrained(os.path.join(self.training_results_dir, "model"))

            # Initialize and push the ONNX model repository
            repo_id = "fabiocat/icbhi_classification-onnx"
            try:
                model = ORTModelForAudioClassification.from_pretrained(
                    self.pretrained_model_id,
                    export=True,
                )
                model.save_pretrained(
                    onnx_export_dir, push_to_hub=self.push_to_hub, repository_id=repo_id
                )
                print("Model exported to ONNX format successfully.")
            except Exception as e:
                print("Error exporting model to ONNX format:", e)

        if self.push_to_hub:
            # Push to the hub
            try:
                self.push_to_hub("fabiocat/icbhi_classification")
                # TODO: add model card
                _ = AudioClassificationModel.from_pretrained(
                    "fabiocat/icbhi_classification"
                )
                print("Model pushed to the hub successfully.")
            except Exception as e:
                print("Error pushing the model to the hub:", e)

    def forward(self, x):
        outputs = self.model(x).last_hidden_state
        pooled_output = outputs.mean(dim=1)
        if self.frozen_encoder:
            pooled_output = self.relu(pooled_output)
        return self.classifier(pooled_output)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        self.log("val_loss", loss)
        return {"val_loss": loss, "logits": logits, "labels": y}

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        self.log("test_loss", loss)
        # Save outputs for later processing
        self.test_outputs.append({"logits": logits, "labels": y})
        return {"test_loss": loss, "logits": logits, "labels": y}

    def on_test_epoch_end(self):
        y_true, y_pred = [], []
        for output in self.test_outputs:
            logits = output["logits"]
            labels = output["labels"]
            preds = torch.sigmoid(logits).cpu().numpy()
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds > 0.5)

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        # Metrics
        f1 = f1_score(y_true, y_pred, average="macro")
        precision = precision_score(y_true, y_pred, average="macro")
        recall = recall_score(y_true, y_pred, average="macro")
        try:
            auc = roc_auc_score(y_true, y_pred, average="macro")
        except ValueError as e:
            print(f"Unable to compute AUC: {e}")
            auc = 0.0

        # Logging metrics
        self.log("test_f1", f1)
        self.log("test_precision", precision)
        self.log("test_recall", recall)
        self.log("test_auc", auc)

        # Save metrics to JSON
        metrics = {
            "f1_score": f1,
            "precision": precision,
            "recall": recall,
            "roc_auc": auc,
        }
        os.makedirs(
            os.path.join(self.training_results_dir, "evaluation"), exist_ok=True
        )
        metrics_path = os.path.join(
            self.training_results_dir, "evaluation", "test_metrics.json"
        )
        with open(metrics_path, "w") as f:
            json.dump(metrics, f)

        # Save predictions to JSON
        predictions = {
            "true_labels": y_true.tolist(),
            "predicted_labels": y_pred.tolist(),
        }
        predictions_path = os.path.join(
            self.training_results_dir, "evaluation", "test_predictions.json"
        )
        with open(predictions_path, "w") as f:
            json.dump(predictions, f)

        # Confusion Matrix
        cm = confusion_matrix(y_true.argmax(axis=1), y_pred.argmax(axis=1))
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
        ax.set_title("Confusion Matrix")
        fig.colorbar(im, ax=ax)  # Associate the colorbar with the image
        plt.xlabel("Predicted Labels")
        plt.ylabel("True Labels")

        # Save confusion matrix to specified directory
        os.makedirs(self.training_results_dir, exist_ok=True)
        save_path = os.path.join(
            self.training_results_dir, "evaluation", "confusion_matrix.png"
        )
        plt.savefig(save_path)
        plt.close(fig)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)


# %%
def parse_args():
    parser = argparse.ArgumentParser(
        description="Read model parameters from command line arguments."
    )

    # Add arguments
    parser.add_argument(
        "--high_pass_cutoff",
        type=int,
        default=4000,
        help="High-pass filter cutoff frequency.",
    )
    parser.add_argument(
        "--low_pass_cutoff",
        type=int,
        default=0,
        help="Low-pass filter cutoff frequency.",
    )
    parser.add_argument(
        "--order",
        type=int,
        default=4,
        help="Order of the Butterworth filter.",
    )
    parser.add_argument(
        "--pretrained_model_id",
        type=str,
        default="MIT/ast-finetuned-audioset-14-14-0.443",
        help="ID of the pretrained model.",
    )
    parser.add_argument(
        "--max_epochs", type=int, default=0, help="Maximum number of training epochs."
    )
    parser.add_argument(
        "--augmentation_flag",
        action="store_true",
        default=False,
        help="Enable data augmentation.",
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
        "--freeze_encoder",
        action="store_true",
        default=False,
        help="Freeze encoder weights.",
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        default=False,
        help="Push model to Hugging Face Hub.",
    )
    parser.add_argument(
        "--save_model",
        action="store_true",
        default=False,
        help="Save model checkpoint.",
    )

    return parser.parse_args()


# %%
def main():
    """
    Main function for training and evaluating the model.
    """
    # Parse arguments
    args = parse_args()
    high_pass_cutoff = args.high_pass_cutoff
    low_pass_cutoff = args.low_pass_cutoff
    order = args.order
    pretrained_model_id = args.pretrained_model_id
    max_epochs = args.max_epochs
    augmentation_flag = args.augmentation_flag
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    num_workers = args.num_workers
    freeze_encoder = args.freeze_encoder
    push_to_hub = args.push_to_hub
    save_model = args.save_model

    print("High-pass cutoff frequency:", high_pass_cutoff)
    print("Low-pass cutoff frequency:", low_pass_cutoff)
    print("Order of the Butterworth filter:", order)
    print("Pretrained model ID:", pretrained_model_id)
    print("Maximum epochs:", max_epochs)
    print("Data augmentation flag:", augmentation_flag)
    print("Batch size:", batch_size)
    print("Learning rate:", learning_rate)
    print("Number of workers:", num_workers)
    print("Freeze encoder:", freeze_encoder)
    print("Push to Hub:", push_to_hub)
    print("Save model:", save_model)

    # Initialize all variables
    experiment_name = f"audio_classification_experiment__16000__{low_pass_cutoff}__{high_pass_cutoff}__{order}__{pretrained_model_id.replace('/', '_')}"
    wandb_logger = WandbLogger(
        project="audio_classification", name=experiment_name, save_dir="../../wandb"
    )

    training_results_dir = f"../../training_results/{experiment_name}"
    os.makedirs(training_results_dir, exist_ok=True)

    # File paths and splits
    dataset_path = f"../../output/preprocessed__16000__{low_pass_cutoff}__{high_pass_cutoff}__{order}"
    split_file = "../../output/split.csv"
    split_df = pd.read_csv(split_file)

    train_files, train_labels, dev_files, dev_labels, test_files, test_labels = (
        [],
        [],
        [],
        [],
        [],
        [],
    )
    for _, row in split_df.iterrows():
        patient_path = os.path.join(dataset_path, f"{row['patient_number']}_*.wav")
        label_files = glob.glob(patient_path)
        for file in label_files:
            crackle, wheeze = map(int, file.replace(".wav", "").split("__")[-2:])
            label = [crackle, wheeze]
            if row["split"] == "train":
                train_files.append(file)
                train_labels.append(label)
            elif row["split"] == "dev":
                dev_files.append(file)
                dev_labels.append(label)
            else:
                test_files.append(file)
                test_labels.append(label)

    try:
        processor = AutoProcessor.from_pretrained(
            pretrained_model_id, trust_remote_code=True
        )
    except Exception as e:
        print(f"Error initializing processor for {pretrained_model_id}: {e}")
        exit(1)

    if augmentation_flag:
        augment = Compose(
            transforms=[
                Gain(min_gain_in_db=-15, max_gain_in_db=15, p=0.5),
                Shift(
                    min_shift=0.1,
                    max_shift=0.5,
                    shift_unit="fraction",
                    rollover=True,
                    p=0.5,
                ),
            ]
        )
    else:
        augment = None

    train_dataset = AudioDataset(train_files, train_labels, processor, augment=augment)
    dev_dataset = AudioDataset(dev_files, dev_labels, processor)
    test_dataset = AudioDataset(test_files, test_labels, processor)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=min(num_workers, os.cpu_count()),
    )
    dev_loader = torch.utils.data.DataLoader(
        dev_dataset, batch_size=batch_size, num_workers=min(num_workers, os.cpu_count())
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=min(num_workers, os.cpu_count()),
    )

    model = AudioClassificationModel(
        pretrained_model_id,
        num_labels=2,
        training_results_dir=training_results_dir,
        frozen=freeze_encoder,
        learning_rate=learning_rate,
        push_to_hub=push_to_hub,
        save_model=save_model,
    )
    early_stopping = EarlyStopping(
        monitor="val_loss", patience=3, min_delta=0.001, mode="min", verbose=True
    )

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        precision=16,
        log_every_n_steps=10,
        accelerator="auto",
        devices=None,
        strategy="ddp",
        logger=wandb_logger,
        callbacks=[early_stopping],
    )

    # Train
    trainer.fit(model, train_loader, dev_loader)
    # Evaluate
    trainer.test(model, test_loader)


if __name__ == "__main__":
    main()
