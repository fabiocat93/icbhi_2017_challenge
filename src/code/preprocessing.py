import os
import torch
from tqdm import tqdm
from typing import Optional
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
    """Resamples an audio waveform to the target sampling rate with optional low-pass filtering."""
    if lowcut is None:
        lowcut = resample_rate / 2 - 100
    sos = signal.butter(order, lowcut, btype="low", output="sos", fs=resample_rate)

    channels = []
    for channel in waveform:
        filtered_channel = torch.from_numpy(
            signal.sosfiltfilt(sos, channel.numpy()).copy()
        ).float()
        resampler = Resample(orig_freq=sampling_rate, new_freq=resample_rate)
        resampled_channel = resampler(filtered_channel.unsqueeze(0)).squeeze(0)
        channels.append(resampled_channel)

    resampled_waveform = torch.stack(channels)
    return resampled_waveform


def normalize_audio(waveform: torch.Tensor) -> torch.Tensor:
    """Normalizes an audio waveform to the range [-1, 1]."""
    return waveform / waveform.abs().max()


def segment_audio(
    waveform: torch.Tensor,
    sampling_rate: int,
    segments: list,
    output_path: str,
    base_filename: str,
) -> None:
    """Segments audio based on start and end times from a segmentation list."""
    for i, (start, end, abnomaly1, abnomaly2) in enumerate(segments):
        start_sample = int(start * sampling_rate)
        end_sample = int(end * sampling_rate)
        segment = waveform[:, start_sample:end_sample]
        segment_filename = os.path.join(
            output_path, f"{base_filename}__{i}__{abnomaly1}__{abnomaly2}.wav"
        )
        torchaudio.save(segment_filename, segment, sampling_rate)


def process_audio_file(
    filepath: str,
    output_folder: str,
    resample_rate: int,
) -> None:
    """Processes a single audio file: resamples, normalizes, and segments it."""
    try:
        # Load audio and get corresponding segmentation file
        waveform, sampling_rate = torchaudio.load(filepath)
        base_filename = os.path.splitext(os.path.basename(filepath))[0]
        segmentation_file = os.path.join(
            os.path.dirname(filepath), f"{base_filename}.txt"
        )
        if not os.path.exists(segmentation_file):
            print(f"Segmentation file not found for {filepath}. Skipping.")
            return

        # Parse segmentation file
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

        # Resample audio
        waveform = resample_audios(waveform, sampling_rate, resample_rate)

        # Normalize audio
        waveform = normalize_audio(waveform)

        # Create output directory
        os.makedirs(output_folder, exist_ok=True)

        # Segment and save
        segment_audio(waveform, resample_rate, segments, output_folder, base_filename)

    except Exception as e:
        print(f"Error processing file {filepath}: {e}")


def process_audio_folder(
    input_folder: str, output_folder: str, resample_rate: int
) -> None:
    """Processes all .wav files in the folder and its subfolders."""
    # Collect all .wav files
    file_paths = []
    for root, _, files in os.walk(input_folder):
        for file in files:
            if file.lower().endswith(".wav"):
                file_paths.append(os.path.join(root, file))

    # Process files with progress bar
    for filepath in tqdm(file_paths, desc="Processing audio files"):
        process_audio_file(filepath, output_folder, resample_rate)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract audio metadata from .wav files."
    )
    parser.add_argument(
        "--input_folder",
        type=str,
        help="Path to the input folder containing .wav files.",
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        help="Path to the folder to save the output CSV file.",
    )
    parser.add_argument(
        "--resample_rate",
        type=int,
        default=16000,
        help="Target sampling rate for resampling.",
    )

    args = parser.parse_args()

    input_folder = args.input_folder
    output_folder = args.output_folder
    resample_rate = args.resample_rate

    process_audio_folder(input_folder, output_folder, resample_rate)
