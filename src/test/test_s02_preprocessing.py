"""This module contains tests for the preprocessing module.

This is only a simple example to show how tests would be written if I had more time.
"""

import torch
from src.code.s02_preprocessing import normalize_audio


def test_normalize_audio():
    """Test the normalize_audio function."""
    # Create a sample waveform tensor
    waveform = torch.tensor([[-0.5, 0.0, 0.5, 1.0, -1.0, 0.8]], dtype=torch.float32)

    # Call the function
    normalized_waveform = normalize_audio(waveform)

    # Check if the max absolute value is 1
    assert torch.allclose(
        normalized_waveform.abs().max(), torch.tensor(1.0)
    ), "Normalization failed: Max value is not 1"

    # Check if values are within the range [-1, 1]
    assert (
        normalized_waveform.min() >= -1.0 and normalized_waveform.max() <= 1.0
    ), "Values are out of range [-1, 1]"
