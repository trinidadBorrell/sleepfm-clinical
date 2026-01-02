"""
Prediction pipeline for per-sequence embeddings.
Loads a trained model and makes predictions on new data.
"""
import numpy as np
import pandas as pd
import argparse
import joblib
import os


def load_model(model_path):
    """Load a trained model from disk."""
    model = joblib.load(model_path)
    return model


def preprocess_input(data, id_col='patient_id'):
    """
    Preprocess input data for prediction.
    Removes ID columns and diagnostic_label if present.
    """
    sequence_ids = None
    ids = None
    
    if id_col in data.columns:
        ids = data[id_col]
        data = data.drop(columns=[id_col])
    
    if 'sequence_id' in data.columns:
        sequence_ids = data['sequence_id']
        data = data.drop(columns=['sequence_id'])
    
    if 'diagnostic_label' in data.columns:
        data = data.drop(columns=['diagnostic_label'])
    
    return data, ids, sequence_ids


def predict(model, X):
    """Make predictions using the trained model."""
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1]
    return predictions, probabilities


def decode_predictions(predictions):
    """Convert binary predictions to human-readable labels."""
    label_map = {0: 'MCS+/MCS-', 1: 'UWS'}
    return [label_map[p] for p in predictions]


def aggregate_subject_predictions(results, id_col='patient_id', method='majority'):
    """
    Aggregate sequence-level predictions to subject-level.
    """
    if method == 'majority':
        subject_results = results.groupby(id_col).agg({
            'prediction': lambda x: x.mode().iloc[0],
            'probability_UWS': 'mean',
            'sequence_id': 'count'
        }).rename(columns={'sequence_id': 'n_sequences'})
    else:  # mean probability
        subject_results = results.groupby(id_col).agg({
            'probability_UWS': 'mean',
            'sequence_id': 'count'
        }).rename(columns={'sequence_id': 'n_sequences'})
        subject_results['prediction'] = (subject_results['probability_UWS'] >= 0.5).astype(int)
    
    subject_results['label'] = decode_predictions(subject_results['prediction'].values)
    subject_results = subject_results.reset_index()
    
    return subject_results


def main():
    """Main prediction pipeline."""
    parser = argparse.ArgumentParser(description='Make predictions using per-sequence model')
    parser.add_argument('--model_path', type=str, 
                        default='/home/triniborrell/home/projects/sleepfm-clinical/sleepfm/ml_model/trained_model_per_sequence/random_forest_per_sequence.joblib',
                        help='Path to trained model')
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to input data (CSV with embeddings)')
    parser.add_argument('--output_path', type=str, default=None,
                        help='Path to save predictions (optional)')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Classification threshold (default: 0.5)')
    parser.add_argument('--aggregate', type=str, default='majority', 
                        choices=['majority', 'mean', 'none'],
                        help='Aggregation method: majority (vote), mean (probability), or none')
    
    args = parser.parse_args()
    
    # Load model
    print(f"Loading model from {args.model_path}")
    model = load_model(args.model_path)
    
    # Load input data
    print(f"Loading data from {args.data_path}")
    data = pd.read_csv(args.data_path)
    print(f"Loaded {len(data)} sequences")
    
    # Preprocess
    id_col = 'patient_id' if 'patient_id' in data.columns else 'subject_id'
    X, ids, sequence_ids = preprocess_input(data, id_col=id_col)
    
    has_sequences = sequence_ids is not None
    if has_sequences:
        print(f"Detected {len(data[id_col].unique())} subjects, {len(data)} sequences")
    
    # Make predictions
    print("Making predictions...")
    predictions, probabilities = predict(model, X)
    
    # Apply custom threshold
    if args.threshold != 0.5:
        print(f"Applying custom threshold: {args.threshold}")
        predictions = (probabilities >= args.threshold).astype(int)
    
    # Create results DataFrame
    results = pd.DataFrame({
        'prediction': predictions,
        'probability_UWS': probabilities,
        'label': decode_predictions(predictions)
    })
    
    if ids is not None:
        results.insert(0, id_col, ids.values)
    
    if sequence_ids is not None:
        results.insert(1, 'sequence_id', sequence_ids.values)
    
    # Display results
    print("\n" + "="*50)
    print("SEQUENCE-LEVEL PREDICTIONS")
    print("="*50)
    if len(results) > 20:
        print(results.head(20).to_string(index=False))
        print(f"... ({len(results) - 20} more rows)")
    else:
        print(results.to_string(index=False))
    
    print(f"\nSequence-level Summary:")
    print(f"  - Total sequences: {len(results)}")
    print(f"  - Predicted MCS+/MCS- (0): {(predictions == 0).sum()}")
    print(f"  - Predicted UWS (1): {(predictions == 1).sum()}")
    
    # Aggregate to subject level
    subject_results = None
    if has_sequences and args.aggregate != 'none':
        print(f"\n" + "="*50)
        print(f"SUBJECT-LEVEL PREDICTIONS (aggregation: {args.aggregate})")
        print("="*50)
        subject_results = aggregate_subject_predictions(results, id_col=id_col, method=args.aggregate)
        print(subject_results.to_string(index=False))
        
        print(f"\nSubject-level Summary:")
        print(f"  - Total subjects: {len(subject_results)}")
        print(f"  - Predicted MCS+/MCS- (0): {(subject_results['prediction'] == 0).sum()}")
        print(f"  - Predicted UWS (1): {(subject_results['prediction'] == 1).sum()}")
    
    # Save results
    if args.output_path:
        results.to_csv(args.output_path, index=False)
        print(f"\nSequence-level predictions saved to {args.output_path}")
        
        if subject_results is not None:
            subject_output = args.output_path.replace('.csv', '_subject_level.csv')
            subject_results.to_csv(subject_output, index=False)
            print(f"Subject-level predictions saved to {subject_output}")
    
    return results, subject_results


if __name__ == "__main__":
    main()
