
import os
import glob
import h5py
import numpy as np
import json
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt
from tqdm import tqdm
from loguru import logger
import argparse
import warnings
import mne
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class FIFToHDF5ConverterOneModality:
    """
    Converter for FIF files to HDF5 format for single modality (EEG only) with 256 channels.
    Designed to work with the pretrained SleepFM model using config_one_modality.json.
    Each epoch is exactly 5 seconds at 200 Hz. 
    Path structure: /data/project/eeg_foundation/data/DoC_rs_resampled_5s/sub-{ID}/04_clean_eeg/clean/EEG-epo.fif
    """
    
    def __init__(self, root_dir, target_dir, config_path, resample_rate=200, num_threads=1, num_files=-1):
        self.resample_rate = resample_rate 
        self.root_dir = root_dir
        self.target_dir = target_dir
        self.num_threads = num_threads
        self.num_files = num_files
        self.config = self.load_config(config_path)
        self.file_locations = self.get_files()
        
        # Get channel count from config
        self.num_channels = self.config.get("BAS_CHANNELS", 256)
        self.epoch_duration = 5  # Each epoch is 5 seconds
        logger.info(f"Configured for {self.num_channels} channels (single modality: BAS)")
        logger.info(f"Expected epoch duration: {self.epoch_duration}s at {self.resample_rate} Hz")

    def load_config(self, config_path):
        """Load configuration from JSON file."""
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config

    def get_files(self):
        """
        Search for FIF files using the new path structure:
        /data/project/eeg_foundation/data/DoC_rs_resampled_5s/sub-{ID}/04_clean_eeg/clean/EEG-epo.fif
        """
        # Search for EEG-epo.fif files in the expected structure
        pattern = os.path.join(self.root_dir, 'sub-*/04_clean_eeg/clean/EEG-epo.fif')
        file_paths = glob.glob(pattern)
        
        # Fallback to recursive search if no files found with expected structure
        if not file_paths:
            logger.warning("No files found with expected structure, falling back to recursive search")
            file_paths = glob.glob(os.path.join(self.root_dir, '**/*-epo.[fF][iI][fF]'), recursive=True)
        
        if not file_paths:
            file_paths = glob.glob(os.path.join(self.root_dir, '**/*.[fF][iI][fF]'), recursive=True)
        
        file_names = [os.path.basename(path) for path in file_paths]
        logger.info(f"Found {len(file_paths)} FIF files")
        return file_paths, file_names

    def read_fif(self, file_path):
        """
        Read FIF file and extract signals.
        New format: (n_epochs, 256 channels, timepoints) where each epoch is 5s at 200 Hz.
        No concatenation needed - epochs are already the correct length.
        """
        logger.info(f'Reading FIF file: {file_path}')
        epochs = mne.read_epochs(file_path, verbose='WARNING')
        signals = epochs.get_data()  # Shape: (n_epochs, n_channels, n_times)
        
        sample_rate = epochs.info['sfreq']
        channel_names = np.array(epochs.ch_names)
        
        n_epochs, n_channels, n_times = signals.shape
        epoch_duration = n_times / sample_rate
        
        logger.info(f'Epochs shape: {signals.shape} (n_epochs, n_channels, n_times)')
        logger.info(f'Number of epochs: {n_epochs}')
        logger.info(f'Number of channels: {n_channels}')
        logger.info(f'Sample rate: {sample_rate} Hz')
        logger.info(f'Epoch duration: {epoch_duration:.2f}s')
        
        # Verify epoch duration is ~5 seconds
        if not np.isclose(epoch_duration, self.epoch_duration, atol=0.1):
            logger.warning(f'Epoch duration {epoch_duration:.2f}s differs from expected {self.epoch_duration}s')
        
        return signals, sample_rate, channel_names

    def safe_standardize(self, signal):
        """Standardize signal with safe handling for zero std."""
        mean = np.mean(signal)
        std = np.std(signal)
        
        if std == 0:
            standardized_signal = (signal - mean)
        else:
            standardized_signal = (signal - mean) / std
        
        return standardized_signal
        
    def filter_signal(self, signal, sample_rate):
        """Apply low-pass filter before downsampling."""
        nyquist_freq = sample_rate / 2
        cutoff = min(self.resample_rate / 2, nyquist_freq)
        normalized_cutoff = cutoff / nyquist_freq
        b, a = butter(4, normalized_cutoff, btype='low', analog=False)
        filtered_signal = filtfilt(b, a, signal)
        return filtered_signal

    def process_epochs(self, signals, sample_rate):
        """
        Process epochs: standardize each epoch per channel.
        Input shape: (n_epochs, n_channels, n_times)
        Output shape: (n_epochs, n_channels, n_times)
        
        Since data is already at 200 Hz and epochs are 5s, no resampling needed.
        """
        n_epochs, n_channels, n_times = signals.shape
        processed_signals = np.zeros_like(signals)
        
        # Check if resampling is needed (data should be at 200 Hz)
        if not np.isclose(sample_rate, self.resample_rate, atol=1):
            logger.warning(f'Sample rate {sample_rate} Hz differs from expected {self.resample_rate} Hz')
            # Resample if needed
            new_n_times = int(n_times * self.resample_rate / sample_rate)
            processed_signals = np.zeros((n_epochs, n_channels, new_n_times))
            
            for epoch_idx in range(n_epochs):
                for ch_idx in range(n_channels):
                    signal = signals[epoch_idx, ch_idx, :]
                    duration = n_times / sample_rate
                    
                    original_time_points = np.linspace(0, duration, num=n_times, endpoint=False)
                    new_time_points = np.linspace(0, duration, num=new_n_times, endpoint=False)
                    
                    if sample_rate > self.resample_rate:
                        signal = self.filter_signal(signal, sample_rate)
                    
                    resampled_signal = np.interp(new_time_points, original_time_points, signal)
                    processed_signals[epoch_idx, ch_idx, :] = self.safe_standardize(resampled_signal)
        else:
            # No resampling needed, just standardize
            for epoch_idx in range(n_epochs):
                for ch_idx in range(n_channels):
                    processed_signals[epoch_idx, ch_idx, :] = self.safe_standardize(signals[epoch_idx, ch_idx, :])
        
        # Check for NaN
        nan_epochs = np.any(np.isnan(processed_signals), axis=(1, 2))
        if nan_epochs.any():
            logger.warning(f'Found {nan_epochs.sum()} epochs with NaN values, removing them')
            processed_signals = processed_signals[~nan_epochs]
        
        logger.info(f'Processed {processed_signals.shape[0]} epochs')
        return processed_signals

    def save_to_hdf5(self, signals, channel_names, file_path):
        """
        Save signals to HDF5 file for single modality (BAS only).
        Input shape: (n_epochs, n_channels, n_times)
        
        Each channel is saved as E1, E2, ..., E256.
        Data is stored as (n_epochs, n_times) per channel for easy epoch access.
        """
        logger.info(f'Saving to HDF5: {file_path}')
        
        n_epochs, n_channels, n_times = signals.shape
        num_channels_to_save = min(n_channels, self.num_channels)
        
        logger.info(f'Saving {num_channels_to_save} channels, {n_epochs} epochs as BAS modality')
        
        with h5py.File(file_path, 'w') as hdf:
            # Save epochs data: shape (n_epochs, n_channels, n_times)
            hdf.create_dataset(
                'epochs',
                data=signals[:, :num_channels_to_save, :],
                dtype='float32',
                chunks=(1, num_channels_to_save, n_times),
                compression="gzip"
            )
            
            # Also save per-channel for backward compatibility
            for i in range(num_channels_to_save):
                channel_name = f"E{i+1}"
                # Shape: (n_epochs, n_times)
                channel_data = signals[:, i, :]
                
                hdf.create_dataset(
                    channel_name, 
                    data=channel_data,
                    dtype='float32', 
                    chunks=(1, n_times), 
                    compression="gzip"
                )
            
            # Store metadata
            hdf.attrs['num_channels'] = num_channels_to_save
            hdf.attrs['num_epochs'] = n_epochs
            hdf.attrs['n_times'] = n_times
            hdf.attrs['sample_rate'] = self.resample_rate
            hdf.attrs['epoch_duration'] = self.epoch_duration
            hdf.attrs['modality'] = 'BAS'
            
            if num_channels_to_save < self.num_channels:
                logger.warning(f'Only {num_channels_to_save} channels available, config expects {self.num_channels}')
        
        logger.info(f'Saved {num_channels_to_save} channels x {n_epochs} epochs to {file_path}')

    def convert(self, fif_path, hdf5_path):
        """Convert a single FIF file to HDF5."""
        signals, sample_rate, channel_names = self.read_fif(fif_path)
        processed_signals = self.process_epochs(signals, sample_rate)
        self.save_to_hdf5(processed_signals, channel_names, hdf5_path)

    def get_subject_id(self, fif_path):
        """
        Extract subject ID from path.
        Expected path: .../sub-{ID}/04_clean_eeg/clean/EEG-epo.fif
        """
        parts = fif_path.split(os.sep)
        for part in parts:
            if part.startswith('sub-'):
                return part.replace('sub-', '')
        # Fallback: use parent directory name
        return os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(fif_path))))

    def convert_all(self):
        """Convert all FIF files sequentially."""
        fif_files, fif_names = self.get_files()
        
        if self.num_files != -1:
            fif_files = fif_files[:self.num_files]
        
        logger.info(f"Processing {len(fif_files)} files")
        
        for fif_file in tqdm(fif_files, desc="Converting FIF files"):
            if not fif_file.lower().endswith(".fif"):
                continue
            
            # Extract subject ID and create output filename
            subject_id = self.get_subject_id(fif_file)
            hdf5_file = os.path.join(self.target_dir, f"{subject_id}.hdf5")

            try:
                self.convert(fif_file, hdf5_file)
                logger.info(f"Successfully converted: {fif_file} -> {hdf5_file}")
            except Exception as e:
                warnings.warn(f"Could not process {fif_file}. Error: {str(e)}")
                continue

    def convert_all_multiprocessing(self):
        """
        Convert all FIF files (sequential processing for stability).
        Named for compatibility with original preprocessing.py.
        """
        logger.info("Starting conversion (sequential processing)")
        fif_files, fif_names = self.get_files()

        if self.num_files != -1:
            fif_files = fif_files[:self.num_files]

        logger.info(f"Processing {len(fif_files)} files")
        
        for i, fif_file in enumerate(fif_files):
            logger.info(f"\nProcessing file {i+1}/{len(fif_files)}: {fif_file}")
            
            if not fif_file.lower().endswith(".fif"):
                continue
            
            # Extract subject ID and create output filename
            subject_id = self.get_subject_id(fif_file)
            hdf5_file = os.path.join(self.target_dir, f"{subject_id}.hdf5")
                
            try:
                self.convert(fif_file, hdf5_file)
                
                # Verify the file was created
                if os.path.exists(hdf5_file):
                    with h5py.File(hdf5_file, 'r') as f:
                        n_epochs = f.attrs.get('num_epochs', 'N/A')
                        n_channels = f.attrs.get('num_channels', len(f.keys()) - 1)
                        logger.info(f"Created HDF5 with {n_channels} channels, {n_epochs} epochs")
                else:
                    logger.error(f"File was not created: {hdf5_file}")
                    
            except Exception as e:
                logger.error(f"Failed to process {fif_file}: {str(e)}")
                import traceback
                traceback.print_exc()
                continue
        
        logger.info("Conversion complete")

    def verify_output(self, hdf5_path):
        """Verify the output HDF5 file structure."""
        logger.info(f"Verifying: {hdf5_path}")
        
        with h5py.File(hdf5_path, 'r') as f:
            logger.info(f"Number of datasets: {len(f.keys())}")
            logger.info(f"Datasets: {list(f.keys())[:10]}...")  # Show first 10
            
            # Check epochs dataset
            if 'epochs' in f:
                dataset = f['epochs']
                logger.info(f"epochs shape: {dataset.shape}, dtype: {dataset.dtype}")
            
            # Check first channel
            if 'E1' in f:
                dataset = f['E1']
                logger.info(f"E1 shape: {dataset.shape}, dtype: {dataset.dtype}")
                
            # Check metadata
            for attr in ['num_channels', 'num_epochs', 'n_times', 'sample_rate', 'epoch_duration']:
                if attr in f.attrs:
                    logger.info(f"Stored {attr}: {f.attrs[attr]}")

    def plot_results(self, resampled_signals, channel_names, num_channels_to_plot=5):
        """Plot a subset of resampled signals for verification."""
        num_signals = min(len(resampled_signals), num_channels_to_plot)
        fig, axs = plt.subplots(num_signals, 1, figsize=(15, 3*num_signals), sharex=True)
        
        samples_to_plot = 10 * self.resample_rate
        sample_to_start = 10 * self.resample_rate
        
        for i in range(num_signals):
            signal = resampled_signals[i]
            name = channel_names[i] if i < len(channel_names) else f"E{i+1}"
            signal_chunk = signal[sample_to_start:sample_to_start+samples_to_plot]
            
            if num_signals > 1:
                axs[i].plot(signal_chunk)
                axs[i].set_title(name)
                axs[i].set_ylabel('Amplitude')
            else:
                axs.plot(signal_chunk)
                axs.set_title(name)
                axs.set_ylabel('Amplitude')
        
        if num_signals > 1:
            axs[-1].set_xlabel('Samples')
        else:
            axs.set_xlabel('Samples')
            
        plt.tight_layout()
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess FIF files to HDF5 for single modality (256 EEG channels)"
    )
    parser.add_argument(
        "--config_path", 
        type=str, 
        default='/home/triniborrell/home/projects/sleepfm-clinical/sleepfm/pretrained_model/config_one_modality.json',
        help='Path to config JSON file'
    )
    parser.add_argument(
        "--root_dir",
        type=str,
        default='/data/project/eeg_foundation/data/DoC_rs_resampled_5s',
        help='Path to FIF files (containing sub-{ID}/04_clean_eeg/clean/EEG-epo.fif)'
    )
    parser.add_argument(
        "--target_dir",
        type=str,
        default='/home/triniborrell/home/projects/sleepfm-clinical/output_one_modality_new',
        help='Path to save HDF5 files'
    )
    parser.add_argument(
        "--num_threads", 
        type=int, 
        default=1, 
        help="Number of threads (currently sequential only)"
    )
    parser.add_argument(
        "--num_files", 
        type=int, 
        default=-1, 
        help="Number of files to process. -1 for all"
    )
    parser.add_argument(
        "--resample_rate", 
        type=int, 
        default=200, 
        help="Target sampling rate (data should already be at 200 Hz)"
    )
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.target_dir, exist_ok=True)
    
    logger.info("=" * 60)
    logger.info("Single Modality Preprocessing (256 EEG Channels)")
    logger.info("=" * 60)
    logger.info(f"Config: {args.config_path}")
    logger.info(f"Input: {args.root_dir}")
    logger.info(f"Output: {args.target_dir}")
    logger.info(f"Resample rate: {args.resample_rate} Hz")
    logger.info("=" * 60)

    converter = FIFToHDF5ConverterOneModality(
        root_dir=args.root_dir,
        target_dir=args.target_dir,
        config_path=args.config_path,
        num_threads=args.num_threads,
        num_files=args.num_files,
        resample_rate=args.resample_rate
    )

    converter.convert_all_multiprocessing()
    
    # Verify first output file
    output_files = glob.glob(os.path.join(args.target_dir, '*.hdf5'))
    if output_files:
        converter.verify_output(output_files[0])


if __name__ == "__main__":
    main()
