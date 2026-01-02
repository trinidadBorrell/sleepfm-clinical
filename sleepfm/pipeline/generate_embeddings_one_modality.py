"""
Generate embeddings for single modality (256 EEG channels as BAS).
Uses the pretrained SleepFM model with config_one_modality.json.

FIF files contain epochs of shape (n_epochs, 256 channels, timepoints).
Each epoch is exactly 5 seconds at 200 Hz. 
Path structure: /data/project/eeg_foundation/data/DoC_rs_resampled_5s/sub-{ID}/04_clean_eeg/clean/EEG-epo.fif
"""

import yaml
import torch
from torch import nn
from loguru import logger
import os
import sys
import glob
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import load_config, load_data, count_parameters
from models.dataset import SetTransformerDataset, collate_fn
from models.models import SetTransformer, PositionalEncoding
import click
import time
import math
import datetime
import numpy as np
import tqdm
import h5py
import mne


def get_subject_id(fif_path):
    """Extract subject ID from path."""
    parts = fif_path.split(os.sep)
    for part in parts:
        if part.startswith('sub-'):
            return part.replace('sub-', '')
    return os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(fif_path))))


def load_and_preprocess_fif(fif_path, target_sample_rate=128):
    """
    Load all epochs from a FIF file and preprocess them.
    Returns: (data, subject_id) where data shape is (n_epochs, n_channels, n_times)
    """
    epochs = mne.read_epochs(fif_path, verbose='ERROR', preload=True)
    data = epochs.get_data()  # Shape: (n_epochs, n_channels, n_times)
    sample_rate = epochs.info['sfreq']
    subject_id = get_subject_id(fif_path)
    
    n_epochs, n_channels, n_times = data.shape
    
    # Resample if needed (200 Hz -> 128 Hz for SleepFM)
    if not np.isclose(sample_rate, target_sample_rate, atol=1):
        new_n_times = int(n_times * target_sample_rate / sample_rate)
        resampled = np.zeros((n_epochs, n_channels, new_n_times))
        
        duration = n_times / sample_rate
        orig_t = np.linspace(0, duration, n_times, endpoint=False)
        new_t = np.linspace(0, duration, new_n_times, endpoint=False)
        
        for ep in range(n_epochs):
            for ch in range(n_channels):
                resampled[ep, ch] = np.interp(new_t, orig_t, data[ep, ch])
        data = resampled
    
    # Standardize per epoch per channel
    for ep in range(data.shape[0]):
        for ch in range(data.shape[1]):
            mean = data[ep, ch].mean()
            std = data[ep, ch].std()
            if std > 0:
                data[ep, ch] = (data[ep, ch] - mean) / std
            else:
                data[ep, ch] = data[ep, ch] - mean
    
    return data, subject_id


@click.command("generate_embeddings_one_modality")
@click.option("--model_path", type=str, default='/home/triniborrell/home/projects/sleepfm-clinical/sleepfm/pretrained_model/')
@click.option("--config_path", type=str, default='/home/triniborrell/home/projects/sleepfm-clinical/sleepfm/pretrained_model/config_one_modality.json')
@click.option("--data_path", type=str, default='/data/project/eeg_foundation/data/DoC_rs_resampled_5s', help="Path to FIF files")
@click.option("--num_workers", type=int, default=4)
@click.option("--batch_size", type=int, default=32)
@click.option("--max_seq_length", type=int, default=128, help="Max sequence length for positional encoding")
@click.option("--target_sample_rate", type=int, default=128, help="Target sample rate for SleepFM model")
@click.option("--epochs_per_sequence", type=int, default=6, help="Number of 5-second epochs per sequence (6 = 30 seconds)")
def generate_embeddings_one_modality(
    model_path,
    config_path,
    data_path,
    num_workers, 
    batch_size,
    max_seq_length,
    target_sample_rate,
    epochs_per_sequence
):
    # Load config for single modality
    config = load_config(config_path)

    current_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    output = os.path.join(model_path, "doc_embeddings")
    output_per_epoch = os.path.join(model_path, "doc_embeddings_per_epoch")
    os.makedirs(output, exist_ok=True)
    os.makedirs(output_per_epoch, exist_ok=True)

    # Model parameters from config
    modality_types = config.get("modality_types", ["BAS"])
    in_channels = config["in_channels"]
    patch_size = config["patch_size"]
    embed_dim = config["embed_dim"]
    num_heads = config["num_heads"]
    num_layers = config["num_layers"]
    pooling_head = config["pooling_head"]
    dropout = 0.0  # No dropout during inference

    logger.info("=" * 60)
    logger.info("Single Modality Embedding Generation (DoC FIF files)")
    logger.info("=" * 60)
    logger.info(f"Data Path: {data_path}")
    logger.info(f"Output Path: {output}")
    logger.info(f"Output per-epoch Path: {output_per_epoch}")
    logger.info(f"Target sample rate: {target_sample_rate} Hz")
    logger.info(f"Batch Size: {batch_size}; Number of Workers: {num_workers}")
    logger.info(f"Max sequence length: {max_seq_length}")
    logger.info(f"Epochs per sequence: {epochs_per_sequence} ({epochs_per_sequence * 5} seconds)")
    logger.info("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # Find FIF files
    start = time.time()
    pattern = os.path.join(data_path, 'sub-*/04_clean_eeg/clean/EEG-epo.fif')
    fif_paths = glob.glob(pattern)
    
    if not fif_paths:
        logger.warning("No files found with expected structure, trying recursive search")
        fif_paths = glob.glob(os.path.join(data_path, '**/*-epo.fif'), recursive=True)
    
    if not fif_paths:
        fif_paths = glob.glob(os.path.join(data_path, '**/*.fif'), recursive=True)
    
    logger.info(f"Found {len(fif_paths)} FIF files")
    
    if len(fif_paths) == 0:
        logger.error("No FIF files found to process.")
        return

    logger.info(f"Files found in {time.time() - start:.1f} seconds")

    # Load model
    logger.info(f"Loading model: {config['model']}")
    model = SetTransformer(
        in_channels, 
        patch_size, 
        embed_dim, 
        num_heads, 
        num_layers, 
        pooling_head=pooling_head, 
        dropout=dropout,
        max_seq_length=max_seq_length
    )
    
    if device.type == "cuda":
        model = torch.nn.DataParallel(model)
    model.to(device)
    
    total_layers, total_params = count_parameters(model)
    logger.info(f'Trainable parameters: {total_params / 1e6:.2f} million')
    logger.info(f'Number of layers: {total_layers}')

    # Load pretrained weights
    checkpoint_path = os.path.join(model_path, "best.pt")
    logger.info(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Remove 'module.' prefix from checkpoint keys (saved with DataParallel)
    state_dict = checkpoint["state_dict"]
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    # Load weights (may need to handle max_seq_length mismatch)
    if device.type == "cuda":
        # Model is wrapped in DataParallel
        model.module.load_state_dict(state_dict, strict=False)
        # Reinitialize positional encoding if needed
        if max_seq_length != 128:
            logger.info(f"Reinitializing positional encoding for max_seq_length={max_seq_length}")
            model.module.positional_encoding = PositionalEncoding(max_seq_length, embed_dim).to(device)
    else:
        model.load_state_dict(state_dict, strict=False)
        if max_seq_length != 128:
            logger.info(f"Reinitializing positional encoding for max_seq_length={max_seq_length}")
            model.positional_encoding = PositionalEncoding(max_seq_length, embed_dim).to(device)
    
    model.eval()
    logger.info("Model loaded and set to eval mode")

    # Number of samples per 5-second epoch
    samples_per_epoch = int(5 * target_sample_rate)  # 5 seconds * 128 Hz = 640

    # Process each FIF file and generate embeddings
    with torch.no_grad():
        for fif_path in tqdm.tqdm(fif_paths, desc="Processing FIF files"):
            try:
                # Load all epochs from this file at once
                data, subject_id = load_and_preprocess_fif(fif_path, target_sample_rate)
                n_epochs = data.shape[0]
                n_channels = data.shape[1]
                
                # Group epochs into sequences (discard incomplete sequences)
                # Each sequence: concatenate epochs_per_sequence epochs along time axis
                n_sequences = n_epochs // epochs_per_sequence
                
                if n_sequences == 0:
                    logger.warning(f"Subject {subject_id} has only {n_epochs} epochs, need at least {epochs_per_sequence}. Skipping.")
                    continue
                
                discarded_epochs = n_epochs % epochs_per_sequence
                if discarded_epochs > 0:
                    logger.info(f"Subject {subject_id}: Using {n_sequences * epochs_per_sequence}/{n_epochs} epochs (discarding last {discarded_epochs})")
                
                all_pooled = []
                all_sequence = []
                
                # Process sequences in batches
                sequences_processed = 0
                batch_sequences = []
                
                for seq_idx in range(n_sequences):
                    start_epoch = seq_idx * epochs_per_sequence
                    end_epoch = min(start_epoch + epochs_per_sequence, n_epochs)
                    actual_epochs = end_epoch - start_epoch
                    
                    # Get epochs for this sequence: (epochs_per_sequence, n_channels, samples_per_epoch)
                    epoch_group = data[start_epoch:end_epoch]
                    
                    # Concatenate along time axis: (n_channels, epochs_per_sequence * samples_per_epoch)
                    # Transpose to (n_channels, epochs_per_sequence, samples) then reshape
                    sequence_data = epoch_group.transpose(1, 0, 2).reshape(n_channels, -1)
                    
                    batch_sequences.append(sequence_data)
                    
                    # Process batch when full or at the end
                    if len(batch_sequences) == batch_size or seq_idx == n_sequences - 1:
                        # Stack into batch: (batch, n_channels, time)
                        batch_data = torch.tensor(np.stack(batch_sequences), dtype=torch.float32, device=device)
                        
                        # Create mask for channels (all False = all channels valid)
                        mask_bas = torch.zeros(batch_data.shape[0], batch_data.shape[1], dtype=torch.bool, device=device)
                        
                        # Generate embeddings
                        emb_pooled, emb_sequence = model(batch_data, mask_bas)
                        
                        all_pooled.append(emb_pooled.cpu().numpy())
                        if emb_sequence is not None:
                            all_sequence.append(emb_sequence.cpu().numpy())
                        
                        batch_sequences = []
                
                # Concatenate all batches
                pooled_embs = np.concatenate(all_pooled, axis=0)  # (n_sequences, embed_dim)
                mean_emb = pooled_embs.mean(axis=0)
                
                # Save aggregated embedding
                output_path = os.path.join(output, f"{subject_id}.hdf5")
                with h5py.File(output_path, 'w') as f:
                    f.create_dataset('embedding', data=mean_emb, dtype='float32')
                    f.create_dataset('embedding_per_sequence', data=pooled_embs, dtype='float32')
                    f.attrs['num_epochs'] = n_epochs
                    f.attrs['num_sequences'] = n_sequences
                    f.attrs['epochs_per_sequence'] = epochs_per_sequence
                    f.attrs['embed_dim'] = mean_emb.shape[0]
                
                # Save per-sequence embeddings
                output_path_epochs = os.path.join(output_per_epoch, f"{subject_id}.hdf5")
                with h5py.File(output_path_epochs, 'w') as f:
                    f.create_dataset('pooled', data=pooled_embs, dtype='float32')
                    if all_sequence:
                        seq_embs = np.concatenate(all_sequence, axis=0)
                        f.create_dataset('sequence', data=seq_embs, dtype='float32')
                    f.attrs['num_sequences'] = n_sequences
                    f.attrs['epochs_per_sequence'] = epochs_per_sequence
                
            except Exception as e:
                logger.error(f"Failed to process {fif_path}: {e}")
                continue

    logger.info("Embedding generation complete!")
    logger.info(f"Aggregated embeddings saved to: {output}")
    logger.info(f"Per-epoch embeddings saved to: {output_per_epoch}")


if __name__ == '__main__':
    generate_embeddings_one_modality()
