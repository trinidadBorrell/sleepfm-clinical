"""
Preprocessing for per-token embeddings.
Uses doc_embeddings_per_epoch with 'sequence' (one 128-dim embedding per 5-second token).
Shape in HDF5: (n_sequences, tokens_per_sequence, 128) -> flattened to (n_sequences * tokens_per_sequence, 128)
"""
import numpy as np
import os
import h5py
import argparse
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def main():
    parser = argparse.ArgumentParser(description='Preprocess per-token embeddings')
    parser.add_argument('--main_path', type=str, 
                        default='/home/triniborrell/home/projects/sleepfm-clinical/sleepfm/pretrained_model/doc_embeddings_per_epoch',
                        help='Path to doc_embeddings_per_epoch folder')
    parser.add_argument('--output_path', type=str, default='./data_per_token', 
                        help='Path to save preprocessed data')
    parser.add_argument('--metadata', type=str, 
                        default='/data/project/eeg_foundation/data/metadata/patient_labels_with_controls.csv',
                        help='Path to metadata CSV file')
    args = parser.parse_args()

    os.makedirs(args.output_path, exist_ok=True)

    # Load metadata
    metadata = None
    label_encoder = None
    if args.metadata and os.path.exists(args.metadata):
        metadata = pd.read_csv(args.metadata)
        print(f"Loaded metadata with shape: {metadata.shape}")
        
        label_encoder = LabelEncoder()
        metadata['diagnostic_encoded'] = label_encoder.fit_transform(metadata['diagnostic_crs_final'])
        print(f"Label mapping: {dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))}")
        
    embeddings_list = []
    labels_list = []
    patient_ids = []
    sequence_ids = []
    token_ids = []
        
    for file in os.listdir(args.main_path):
        if file.endswith('.hdf5'):
            print(f"Processing {file}")
            file_path = os.path.join(args.main_path, file)
            with h5py.File(file_path, 'r') as f:
                patient_id = file.replace('.hdf5', '')
                
                # Get label for this patient
                patient_label = np.nan
                if metadata is not None:
                    patient_row = metadata[metadata['subject'] == patient_id]
                    if not patient_row.empty:
                        patient_label = patient_row['diagnostic_encoded'].values[0]
                    else:
                        print(f"Warning: No metadata found for patient {patient_id}")
                
                if 'sequence' not in f:
                    print(f"Warning: 'sequence' not found in {file}, skipping...")
                    continue
                
                data = f['sequence'][:]  # Shape: (n_sequences, tokens_per_sequence, 128)
                n_sequences, tokens_per_seq, embed_dim = data.shape
                print(f"  Shape: {data.shape} ({n_sequences} sequences × {tokens_per_seq} tokens)")
                
                # Flatten to per-token: each token becomes a separate row
                for seq_idx in range(n_sequences):
                    for token_idx in range(tokens_per_seq):
                        patient_ids.append(patient_id)
                        labels_list.append(patient_label)
                        sequence_ids.append(seq_idx)
                        token_ids.append(token_idx)
                        embeddings_list.append(data[seq_idx, token_idx])
    
    # Create DataFrame
    csv_data = pd.DataFrame(embeddings_list)
    csv_data.insert(0, 'patient_id', patient_ids)
    csv_data.insert(1, 'sequence_id', sequence_ids)
    csv_data.insert(2, 'token_id', token_ids)
    csv_data.insert(3, 'diagnostic_label', labels_list)
    
    output_file = os.path.join(args.output_path, 'embeddings_per_token.csv')
    csv_data.to_csv(output_file, index=False)
    
    # Save label encoder mapping
    if label_encoder is not None:
        mapping_file = os.path.join(args.output_path, 'label_mapping.txt')
        with open(mapping_file, 'w') as f:
            for label, encoded in zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)):
                f.write(f"{label}: {encoded}\n")
        print(f"Label mapping saved to {mapping_file}")
    
    print("Preprocessing complete!")
    print(f"Embeddings saved to {output_file}")
    print(f"Final shape: {csv_data.shape}")
    print(f"Unique patients: {len(set(patient_ids))}")
    print(f"Total tokens: {len(csv_data)}")
    
if __name__ == "__main__":
    main()
