# poetry run python audio_summary.py --input_folder ../../data/ICBHI_final_database/labels/ --output_folder ../../output

import os
import torch
import torchaudio
from typing import Dict, List
import pandas as pd
from tqdm import tqdm
import argparse


def extract_audio_info(filepath: str) -> Dict[str, object]:
    """Extracts audio metadata from a .wav file.

    Args:
        filepath (str): Path to the .wav file.

    Returns:
        Dict[str, object]: Dictionary containing audio metadata including duration,
                           channels, sampling rate, and encoding details.
    """
    waveform, sampling_rate = torchaudio.load(filepath, format="wav")
    num_frames = waveform.size(-1)
    duration = num_frames / sampling_rate  # in seconds

    # Infer bit depth and encoding from waveform's dtype
    dtype_to_bits = {
        torch.float32: (32, "PCM_F"),
        torch.float16: (16, "PCM_F"),
        torch.int16: (16, "PCM_S"),
        torch.int8: (8, "PCM_S"),
        torch.uint8: (8, "PCM_U"),
    }
    bits_per_sample, encoding = dtype_to_bits.get(waveform.dtype, (-1, "UNKNOWN"))

    channels = waveform.size(0)
    if channels == 1:
        mono_stereo = "mono"
    elif channels == 2:
        similarity = torch.corrcoef(waveform.view(2, -1))[0, 1].item()
        mono_stereo = "mono" if similarity > 0.99 else "stereo"
    else:
        mono_stereo = f"{channels}_channel_audio"

    return {
        "filepath": filepath,
        "num_frames": num_frames,
        "sampling_rate": sampling_rate,
        "duration": duration,
        "encoding": encoding,
        "bits_per_sample": bits_per_sample,
        "channels": channels,
        "mono_stereo_estimation": mono_stereo,
    }


def extract_metadata_from_filename(filename: str) -> Dict[str, str]:
    """Extracts metadata from the filename based on predefined conventions.

    Args:
        filename (str): The filename to extract metadata from.

    Returns:
        Dict[str, str]: Dictionary containing extracted metadata elements.
    """
    parts = filename.split("_")
    if len(parts) < 5:
        raise ValueError(
            f"Filename {filename} does not conform to the expected format."
        )

    return {
        "patient_number": parts[0],
        "recording_index": parts[1],
        "chest_location": parts[2],
        "acquisition_mode": parts[3],
        "recording_equipment": parts[4].split(".")[0],
    }


def process_audio_folder(input_folder: str, output_folder: str) -> None:
    """Processes all .wav files in a folder and its subfolders, extracting metadata.

    Args:
        input_folder (str): Path to the folder containing .wav files.
        output_folder (str): Path to the output folder to save the CSV file.
    """
    audio_metadata: List[Dict[str, object]] = []

    # Collect all .wav files
    file_paths = []
    for root, _, files in os.walk(input_folder):
        for file in files:
            if file.lower().endswith(".wav"):
                file_paths.append(os.path.join(root, file))

    # Process files with progress bar
    for filepath in tqdm(file_paths, desc="Processing audio files"):
        try:
            audio_info = extract_audio_info(filepath)
            filename_metadata = extract_metadata_from_filename(
                os.path.basename(filepath)
            )
            combined_metadata = {**audio_info, **filename_metadata}
            audio_metadata.append(combined_metadata)
        except Exception as e:
            print(f"Error processing {filepath}: {e}")

    if len(audio_metadata) == 0:
        print("No .wav files found in the input folder.")
        return

    # Save to CSV
    os.makedirs(output_folder, exist_ok=True)
    output_csv = os.path.join(output_folder, "audio_summary.csv")
    df = pd.DataFrame(audio_metadata)
    df.to_csv(output_csv, index=False)
    print(f"Audio summary saved to {output_csv}")


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

    args = parser.parse_args()

    input_folder = args.input_folder
    output_folder = args.output_folder

    process_audio_folder(input_folder, output_folder)
