"""
This file contains the code for inference using the trained model.
Usage:
    poetry run python s05_inference.py --file_path ../../output/16000__0__4000__4/130_1p2_Ar_mc_AKGC417L__4__1__1.wav"""

import json
import torch
import torchaudio
from transformers import AutoProcessor
from typing import Union
from pathlib import Path
import torch.nn as nn
from transformers import AutoModel
import argparse
import time

from huggingface_hub import PyTorchModelHubMixin
import pytorch_lightning as pl


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
        encoder_id: str = "MIT/ast-finetuned-audioset-14-14-0.443",
        num_labels: int = 2,
        learning_rate: float = 1e-4,
        frozen: bool = True,
    ) -> None:
        """
        Initializes the FabModel.

        Args:
            encoder_id (str): ID of the pre-trained transformer encoder.
                Default is "MIT/ast-finetuned-audioset-14-14-0.443".
            num_labels (int): Number of output labels.
                Default is 2.
            learning_rate (float, optional): Learning rate for the optimizer. Defaults to 1e-4.
            frozen (bool, optional): Whether to freeze the encoder layers. Defaults to True.
        """
        super().__init__()
        self.encoder_id = encoder_id
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


def validate_wav_file(file_path: Union[str, Path]) -> bool:
    """
    Validates the input .wav file. It must be mono and have a sampling rate of 16kHz.

    Args:
        file_path (Union[str, Path]): Path to the .wav file.

    Returns:
        bool: True if the file is valid, False otherwise.
    """
    try:
        waveform, sr = torchaudio.load(file_path)
        if waveform.size(0) != 1:
            print("The file must be mono.")
            return False
        if sr != 16000:
            print("The file must have a sampling rate of 16kHz.")
            return False
        return True
    except Exception as e:
        print(f"Error loading .wav file: {e}")
        return False


def predict(file_path: str, model_path: str = "fabiocat/icbhi_classification") -> dict:
    """
    Predicts the output using the pre-trained model for a given .wav file.

    Args:
        file_path (str): Path to the .wav file.
        model_path (str): Path or URI to the saved model directory.

    Returns:
        dict: Prediction results.
    """
    # Measure start time
    total_start_time = time.time()

    # Load the processor and model
    print("Loading processor and model...")
    load_start_time = time.time()
    model = FabModel.from_pretrained(model_path)
    model.eval()
    load_end_time = time.time()
    print(f"Processor and model loaded in {load_end_time - load_start_time:.4f} seconds.")

    # Load and preprocess the audio file
    print("Processing audio file...")
    preprocess_start_time = time.time()
    waveform, sr = torchaudio.load(file_path)
    waveform = waveform.mean(0).unsqueeze(0)  # Convert to mono
    processor = AutoProcessor.from_pretrained(model.encoder_id, trust_remote_code=True)
    inputs = processor(
        waveform.squeeze(),
        sampling_rate=sr,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )
    preprocess_end_time = time.time()
    print(f"Audio file processed in {preprocess_end_time - preprocess_start_time:.4f} seconds.")

    # Make predictions
    print("Making predictions...")
    prediction_start_time = time.time()
    with torch.no_grad():
        logits = model(inputs.input_values)
        probabilities = torch.sigmoid(logits).squeeze().tolist()
    prediction_end_time = time.time()
    print(f"Predictions made in {prediction_end_time - prediction_start_time:.4f} seconds.")

    # Prepare output
    labels = ["Crackle", "Wheeze"]  # Update based on your model's labels
    predictions = {label: prob for label, prob in zip(labels, probabilities)}

    # Measure total time
    total_end_time = time.time()
    print(f"Total time taken: {total_end_time - total_start_time:.4f} seconds.")

    return predictions
    

def main():
    parser = argparse.ArgumentParser(
        description="Predict audio labels using the trained model."
    )
    parser.add_argument(
        "--file_path", type=str, required=True, help="Path to the .wav file."
    )

    args = parser.parse_args()

    # Validate .wav file
    if not validate_wav_file(args.file_path):
        print("Invalid .wav file. Please provide a valid mono .wav file.")
        return

    if (
        args.file_path
        == "../../output/16000__0__4000__4/130_1p2_Ar_mc_AKGC417L__4__1__1.wav"
    ):
        print(
            "For the given audio example, the ideal output would be: {'Crackle': 1.0, 'Wheeze': 1.0}"
        )
        print()

    # Predict and print results
    predictions = predict(args.file_path)
    print("Scores:", json.dumps(predictions, indent=4))


if __name__ == "__main__":
    main()
