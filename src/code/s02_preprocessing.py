"""
This script processes audio files in a specified folder by resampling, normalizing,
filtering, and segmenting them based on provided segmentation files. It saves
the processed audio segments in the output folder.

Usage:
    poetry run python s02_preprocessing.py --input_folder ../../data/ICBHI_final_database/labels/ \
    --output_folder ../../output/ --resample_rate 16000 --highcut 4000 --order 4
"""

import os
import torch
from tqdm import tqdm
from typing import Optional, List, Tuple
from scipy import signal
from speechbrain.augment.time_domain import Resample
import torchaudio
import argparse


def resample_audios(
    waveform: torch.Tensor,
    sampling_rate: int,
    resample_rate: int,
    lowcut: Optional[float] = None,
    order: int = 4,
) -> torch.Tensor:
    """
    Resamples an audio waveform to a target sampling rate with optional low-pass filtering.

    Args:
        waveform (torch.Tensor): The audio waveform as a PyTorch tensor.
        sampling_rate (int): Original sampling rate of the audio.
        resample_rate (int): Target sampling rate.
        lowcut (Optional[float]): Low-pass filter cutoff frequency. Defaults to resample_rate / 2 - 100.
        order (int): Butterworth filter order.

    Returns:
        torch.Tensor: Resampled waveform.
    """
    if lowcut is None:
        lowcut = resample_rate / 2 - 100
    sos = signal.butter(order, lowcut, btype="low", output="sos", fs=resample_rate)

    channels = []
    for channel in waveform:
        filtered_channel = torch.from_numpy(
            signal.sosfiltfilt(sos, channel.numpy().copy()).copy()
        ).float()
        resampler = Resample(orig_freq=sampling_rate, new_freq=resample_rate)
        resampled_channel = resampler(filtered_channel.unsqueeze(0)).squeeze(0)
        channels.append(resampled_channel)

    return torch.stack(channels)


def normalize_audio(waveform: torch.Tensor) -> torch.Tensor:
    """
    Normalizes an audio waveform to the range [-1, 1].

    Args:
        waveform (torch.Tensor): The audio waveform as a PyTorch tensor.

    Returns:
        torch.Tensor: Normalized waveform.
    """
    return waveform / waveform.abs().max()


def segment_audio(
    waveform: torch.Tensor,
    sampling_rate: int,
    segments: List[Tuple[float, float, int, int]],
    output_path: str,
    base_filename: str,
) -> None:
    """
    Segments audio based on start and end times and saves them as separate files.

    Args:
        waveform (torch.Tensor): The audio waveform as a PyTorch tensor.
        sampling_rate (int): Sampling rate of the audio.
        segments (List[Tuple[float, float, int, int]]): List of segments (start, end, anomaly1, anomaly2).
        output_path (str): Path to save the segmented audio files.
        base_filename (str): Base name for the segmented files.
    """
    for i, (start, end, anomaly1, anomaly2) in enumerate(segments):
        start_sample = int(start * sampling_rate)
        end_sample = int(end * sampling_rate)
        segment = waveform[:, start_sample:end_sample].clone()
        segment_filename = os.path.join(
            output_path, f"{base_filename}__{i}__{anomaly1}__{anomaly2}.wav"
        )
        torchaudio.save(segment_filename, segment, sampling_rate)


def process_audio_file(
    filepath: str,
    output_folder: str,
    resample_rate: int,
    lowcut: float,
    highcut: float,
    order: int,
) -> None:
    """
    Processes a single audio file: resamples, normalizes, and segments it.

    Args:
        filepath (str): Path to the audio file.
        output_folder (str): Directory to save the processed audio files.
        resample_rate (int): Target sampling rate.
        lowcut (float): Low-pass filter cutoff frequency.
        highcut (float): High-pass filter cutoff frequency.
        order (int): Butterworth filter order.
    """
    try:
        waveform, sampling_rate = torchaudio.load(filepath)
        base_filename = os.path.splitext(os.path.basename(filepath))[0]
        segmentation_file = os.path.join(
            os.path.dirname(filepath), f"{base_filename}.txt"
        )
        if not os.path.exists(segmentation_file):
            print(f"Segmentation file not found for {filepath}. Skipping.")
            return

        with open(segmentation_file, "r") as f:
            segments = [
                (
                    float(line.split()[0]),
                    float(line.split()[1]),
                    int(line.split()[2]),
                    int(line.split()[3]),
                )
                for line in f.readlines()
            ]

        waveform = resample_audios(waveform, sampling_rate, resample_rate)

        if lowcut > 0:
            sos = signal.butter(
                order, lowcut, btype="high", output="sos", fs=resample_rate
            )
            waveform = torch.from_numpy(
                signal.sosfiltfilt(sos, waveform.numpy().copy()).copy()
            ).float()

        if highcut < resample_rate:
            sos = signal.butter(
                order, highcut, btype="low", output="sos", fs=resample_rate
            )
            waveform = torch.from_numpy(
                signal.sosfiltfilt(sos, waveform.numpy().copy()).copy()
            ).float()

        waveform = normalize_audio(waveform)
        os.makedirs(output_folder, exist_ok=True)
        segment_audio(waveform, resample_rate, segments, output_folder, base_filename)

    except Exception as e:
        print(f"Error processing file {filepath}: {e}")


def process_audio_folder(
    input_folder: str,
    output_folder: str,
    resample_rate: int,
    lowcut: float,
    highcut: float,
    order: int,
) -> None:
    """
    Processes all .wav files in a folder and its subfolders.

    Args:
        input_folder (str): Path to the input folder containing .wav files.
        output_folder (str): Path to the folder to save processed audio files.
        resample_rate (int): Target sampling rate.
        lowcut (float): Low-pass filter cutoff frequency.
        highcut (float): High-pass filter cutoff frequency.
        order (int): Butterworth filter order.
    """
    file_paths = [
        os.path.join(root, file)
        for root, _, files in os.walk(input_folder)
        for file in files
        if file.lower().endswith(".wav")
    ]

    for filepath in tqdm(file_paths, desc="Processing audio files"):
        process_audio_file(
            filepath, output_folder, resample_rate, lowcut, highcut, order
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audio preprocessing pipeline.")
    parser.add_argument(
        "--input_folder",
        type=str,
        required=True,
        help="Path to the input folder containing .wav files.",
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        required=True,
        help="Path to the folder to save processed audio files.",
    )
    parser.add_argument(
        "--resample_rate",
        type=int,
        default=16000,
        help="Target sampling rate for resampling.",
    )
    parser.add_argument(
        "--lowcut", type=int, default=0, help="Low-pass filter cutoff frequency."
    )
    parser.add_argument(
        "--highcut", type=int, default=4000, help="High-pass filter cutoff frequency."
    )
    parser.add_argument(
        "--order", type=int, default=4, help="Butterworth filter order."
    )

    args = parser.parse_args()

    output_folder = os.path.join(
        args.output_folder,
        f"{args.resample_rate}__{args.lowcut}__{args.highcut}__{args.order}",
    )
    process_audio_folder(
        args.input_folder,
        output_folder,
        args.resample_rate,
        args.lowcut,
        args.highcut,
        args.order,
    )
