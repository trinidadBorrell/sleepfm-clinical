"""
Prediction pipeline for per-token embeddings.
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
    token_ids = None
    ids = None
    
    if id_col in data.columns:
        ids = data[id_col]
        data = data.drop(columns=[id_col])
    
    if 'sequence_id' in data.columns:
        sequence_ids = data['sequence_id']
        data = data.drop(columns=['sequence_id'])
    
    if 'token_id' in data.columns:
        token_ids = data['token_id']
        data = data.drop(columns=['token_id'])
    
    if 'diagnostic_label' in data.columns:
        data = data.drop(columns=['diagnostic_label'])
    
    return data, ids, sequence_ids, token_ids


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
    Aggregate token-level predictions to subject-level.
    """
    if method == 'majority':
        subject_results = results.groupby(id_col).agg({
            'prediction': lambda x: x.mode().iloc[0],
            'probability_UWS': 'mean',
            'token_id': 'count'
        }).rename(columns={'token_id': 'n_tokens'})
    else:  # mean probability
        subject_results = results.groupby(id_col).agg({
            'probability_UWS': 'mean',
            'token_id': 'count'
        }).rename(columns={'token_id': 'n_tokens'})
        subject_results['prediction'] = (subject_results['probability_UWS'] >= 0.5).astype(int)
    
    subject_results['label'] = decode_predictions(subject_results['prediction'].values)
    subject_results = subject_results.reset_index()
    
    return subject_results


def aggregate_sequence_predictions(results, id_col='patient_id', method='majority'):
    """
    Aggregate token-level predictions to sequence-level (intermediate aggregation).
    """
    group_cols = [id_col, 'sequence_id']
    
    if method == 'majority':
        seq_results = results.groupby(group_cols).agg({
            'prediction': lambda x: x.mode().iloc[0],
            'probability_UWS': 'mean',
            'token_id': 'count'
        }).rename(columns={'token_id': 'n_tokens'})
    else:  # mean probability
        seq_results = results.groupby(group_cols).agg({
            'probability_UWS': 'mean',
            'token_id': 'count'
        }).rename(columns={'token_id': 'n_tokens'})
        seq_results['prediction'] = (seq_results['probability_UWS'] >= 0.5).astype(int)
    
    seq_results['label'] = decode_predictions(seq_results['prediction'].values)
    seq_results = seq_results.reset_index()
    
    return seq_results


def main():
    """Main prediction pipeline."""
    parser = argparse.ArgumentParser(description='Make predictions using per-token model')
    parser.add_argument('--model_path', type=str, 
                        default='/home/triniborrell/home/projects/sleepfm-clinical/sleepfm/ml_model/trained_model_per_token/random_forest_per_token.joblib',
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
    print(f"Loaded {len(data)} tokens")
    
    # Preprocess
    id_col = 'patient_id' if 'patient_id' in data.columns else 'subject_id'
    X, ids, sequence_ids, token_ids = preprocess_input(data, id_col=id_col)
    
    has_tokens = token_ids is not None
    if has_tokens:
        print(f"Detected {len(data[id_col].unique())} subjects, {len(data)} tokens")
    
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
    
    if token_ids is not None:
        results.insert(2, 'token_id', token_ids.values)
    
    # Display token-level results
    print("\n" + "="*50)
    print("TOKEN-LEVEL PREDICTIONS")
    print("="*50)
    if len(results) > 20:
        print(results.head(20).to_string(index=False))
        print(f"... ({len(results) - 20} more rows)")
    else:
        print(results.to_string(index=False))
    
    print(f"\nToken-level Summary:")
    print(f"  - Total tokens: {len(results)}")
    print(f"  - Predicted MCS+/MCS- (0): {(predictions == 0).sum()}")
    print(f"  - Predicted UWS (1): {(predictions == 1).sum()}")
    
    # Aggregate to sequence level
    sequence_results = None
    if has_tokens and sequence_ids is not None and args.aggregate != 'none':
        print(f"\n" + "="*50)
        print(f"SEQUENCE-LEVEL PREDICTIONS (aggregation: {args.aggregate})")
        print("="*50)
        sequence_results = aggregate_sequence_predictions(results, id_col=id_col, method=args.aggregate)
        if len(sequence_results) > 20:
            print(sequence_results.head(20).to_string(index=False))
            print(f"... ({len(sequence_results) - 20} more rows)")
        else:
            print(sequence_results.to_string(index=False))
    
    # Aggregate to subject level
    subject_results = None
    if has_tokens and args.aggregate != 'none':
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
        print(f"\nToken-level predictions saved to {args.output_path}")
        
        if sequence_results is not None:
            seq_output = args.output_path.replace('.csv', '_sequence_level.csv')
            sequence_results.to_csv(seq_output, index=False)
            print(f"Sequence-level predictions saved to {seq_output}")
        
        if subject_results is not None:
            subject_output = args.output_path.replace('.csv', '_subject_level.csv')
            subject_results.to_csv(subject_output, index=False)
            print(f"Subject-level predictions saved to {subject_output}")
    
    return results, sequence_results, subject_results


if __name__ == "__main__":
    main()
