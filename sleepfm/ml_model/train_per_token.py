"""
Training pipeline for per-token embeddings.
Uses GroupKFold with patient_id as groups to prevent data leakage.
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    precision_score, recall_score, f1_score, roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score
)
import pandas as pd
import os
import argparse
import joblib
import seaborn as sns


def load_data(data_path):
    """Load and preprocess data for training."""
    data = pd.read_csv(data_path)
    return data


def convert_to_binary(labels, valid_labels=[2, 3, 4]):
    """
    Convert multiclass labels to binary.
    Labels 2 (MCS+) and 3 (MCS-) -> 0
    Label 4 (UWS) -> 1
    """
    return labels.apply(lambda x: 0 if x in [2, 3] else 1)


def filter_valid_labels(data, label_col='diagnostic_label', valid_labels=[2, 3, 4]):
    """Filter data to only include valid diagnostic labels."""
    mask = data[label_col].isin(valid_labels)
    return data[mask].copy()


def evaluate_model(model, X_test, y_test, groups_test, output_path):
    """Evaluate the trained model and generate metrics and plots."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_test, y_prob) if len(np.unique(y_test)) > 1 else 0,
        'average_precision': average_precision_score(y_test, y_prob) if len(np.unique(y_test)) > 1 else 0
    }
    
    print("\n" + "="*50)
    print("TEST SET EVALUATION METRICS")
    print("="*50)
    for metric, value in metrics.items():
        print(f"{metric.upper()}: {value:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['MCS+/MCS- (0)', 'UWS (1)'], zero_division=0))
    
    # Generate plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 1. Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0],
                xticklabels=['MCS+/MCS- (0)', 'UWS (1)'],
                yticklabels=['MCS+/MCS- (0)', 'UWS (1)'])
    axes[0, 0].set_title('Confusion Matrix')
    axes[0, 0].set_xlabel('Predicted')
    axes[0, 0].set_ylabel('Actual')
    
    # 2. ROC Curve
    if len(np.unique(y_test)) > 1:
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        axes[0, 1].plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC (AUC = {metrics["roc_auc"]:.3f})')
    axes[0, 1].plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    axes[0, 1].set_xlabel('False Positive Rate')
    axes[0, 1].set_ylabel('True Positive Rate')
    axes[0, 1].set_title('ROC Curve')
    axes[0, 1].legend(loc='lower right')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Precision-Recall Curve
    if len(np.unique(y_test)) > 1:
        precision, recall, _ = precision_recall_curve(y_test, y_prob)
        axes[1, 0].plot(recall, precision, 'g-', linewidth=2, 
                        label=f'PR (AP = {metrics["average_precision"]:.3f})')
    axes[1, 0].set_xlabel('Recall')
    axes[1, 0].set_ylabel('Precision')
    axes[1, 0].set_title('Precision-Recall Curve')
    axes[1, 0].legend(loc='lower left')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Feature Importance (top 20)
    feature_importance = pd.DataFrame({
        'feature': X_test.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False).head(20)
    
    sns.barplot(data=feature_importance, x='importance', y='feature', ax=axes[1, 1], palette='viridis')
    axes[1, 1].set_title('Top 20 Feature Importances')
    axes[1, 1].set_xlabel('Importance')
    axes[1, 1].set_ylabel('Feature')
    
    plt.tight_layout()
    plot_path = os.path.join(output_path, 'evaluation_plots_per_token.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nPlots saved to {plot_path}")
    
    # Save metrics to file
    metrics_path = os.path.join(output_path, 'metrics_per_token.txt')
    with open(metrics_path, 'w') as f:
        f.write("TEST SET EVALUATION METRICS (Per-Token Model)\n")
        f.write("="*50 + "\n")
        for metric, value in metrics.items():
            f.write(f"{metric.upper()}: {value:.4f}\n")
        f.write("\nClassification Report:\n")
        f.write(classification_report(y_test, y_pred, target_names=['MCS+/MCS- (0)', 'UWS (1)'], zero_division=0))
        f.write(f"\nTest subjects: {len(np.unique(groups_test))}\n")
        f.write(f"Test tokens: {len(y_test)}\n")
    print(f"Metrics saved to {metrics_path}")
    
    return metrics


def main():
    """Main training pipeline with GroupKFold cross-validation."""
    parser = argparse.ArgumentParser(description='Train per-token ML model')
    parser.add_argument('--data_path', type=str, 
                        default='/home/triniborrell/home/projects/sleepfm-clinical/sleepfm/ml_model/data_per_token/embeddings_per_token.csv',
                        help='Path to preprocessed embeddings CSV')
    parser.add_argument('--output_path', type=str, 
                        default='/home/triniborrell/home/projects/sleepfm-clinical/sleepfm/ml_model/trained_model_per_token',
                        help='Path to save trained model')
    parser.add_argument('--n_splits', type=int, default=5,
                        help='Number of cross-validation folds')
    parser.add_argument('--test_fold', type=int, default=0,
                        help='Which fold to use as held-out test set (0 to n_splits-1)')
    parser.add_argument('--random_state', type=int, default=42,
                        help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_path, exist_ok=True)
    
    # Load data
    print(f"Loading data from {args.data_path}")
    data = load_data(args.data_path)
    print(f"Loaded data shape: {data.shape}")
    
    # Filter to valid labels
    valid_labels = [2, 3, 4]
    data = filter_valid_labels(data, valid_labels=valid_labels)
    print(f"After filtering to valid labels {valid_labels}: {len(data)} tokens")
    
    # Extract features, labels, and groups
    id_col = 'patient_id'
    feature_cols = [c for c in data.columns if c not in [id_col, 'sequence_id', 'token_id', 'diagnostic_label']]
    
    X = data[feature_cols]
    y = convert_to_binary(data['diagnostic_label'])
    groups = data[id_col]
    
    print(f"\nFeatures shape: {X.shape}")
    print(f"Unique patients: {len(groups.unique())}")
    print(f"Class distribution: {y.value_counts().to_dict()}")
    
    # GroupKFold splitting
    gkf = GroupKFold(n_splits=args.n_splits)
    splits = list(gkf.split(X, y, groups))
    
    # Use one fold as held-out test set
    test_idx = splits[args.test_fold][1]
    train_val_idx = np.concatenate([splits[i][1] for i in range(args.n_splits) if i != args.test_fold])
    
    X_trainval = X.iloc[train_val_idx]
    y_trainval = y.iloc[train_val_idx]
    groups_trainval = groups.iloc[train_val_idx]
    
    X_test = X.iloc[test_idx]
    y_test = y.iloc[test_idx]
    groups_test = groups.iloc[test_idx]
    
    print(f"\nSplit using GroupKFold (test_fold={args.test_fold}):")
    print(f"  Train+Val: {len(X_trainval)} tokens from {len(groups_trainval.unique())} patients")
    print(f"  Test: {len(X_test)} tokens from {len(groups_test.unique())} patients")
    print(f"  Test patients: {sorted(groups_test.unique().tolist())}")
    
    # Verify no overlap
    train_patients = set(groups_trainval.unique())
    test_patients = set(groups_test.unique())
    overlap = train_patients & test_patients
    if overlap:
        raise ValueError(f"Data leakage detected! Overlapping patients: {overlap}")
    print(f"  ✓ No patient overlap between train and test (no data leakage)")
    
    # Grid search with GroupKFold on train+val
    print("\n" + "="*50)
    print("TRAINING MODEL WITH GRID SEARCH (GroupKFold CV)")
    print("="*50)
    
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [5, 10, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', None]
    }
    
    rf = RandomForestClassifier(random_state=args.random_state, class_weight='balanced')
    
    # Inner GroupKFold for hyperparameter tuning
    inner_gkf = GroupKFold(n_splits=args.n_splits - 1)
    
    grid_search = GridSearchCV(
        rf, param_grid, cv=inner_gkf, scoring='f1',
        n_jobs=-1, verbose=2, return_train_score=True
    )
    
    grid_search.fit(X_trainval, y_trainval, groups=groups_trainval)
    
    print(f"\nBest parameters: {grid_search.best_params_}")
    print(f"Best CV F1 score: {grid_search.best_score_:.4f}")
    
    best_model = grid_search.best_estimator_
    
    # Evaluate on held-out test set
    print("\n" + "="*50)
    print("EVALUATING ON HELD-OUT TEST SET")
    print("="*50)
    metrics = evaluate_model(best_model, X_test, y_test, groups_test, args.output_path)
    
    # Save model
    model_path = os.path.join(args.output_path, 'random_forest_per_token.joblib')
    joblib.dump(best_model, model_path)
    print(f"\nModel saved to {model_path}")
    
    # Save grid search results
    cv_results = pd.DataFrame(grid_search.cv_results_)
    cv_results_path = os.path.join(args.output_path, 'grid_search_results_per_token.csv')
    cv_results.to_csv(cv_results_path, index=False)
    print(f"Grid search results saved to {cv_results_path}")
    
    # Save best parameters
    params_path = os.path.join(args.output_path, 'best_params_per_token.txt')
    with open(params_path, 'w') as f:
        f.write("Best Parameters from Grid Search (Per-Token Model):\n")
        f.write("="*50 + "\n")
        for param, value in grid_search.best_params_.items():
            f.write(f"  {param}: {value}\n")
        f.write(f"\nBest CV F1 Score: {grid_search.best_score_:.4f}\n")
        f.write(f"\nGroupKFold Configuration:\n")
        f.write(f"  n_splits: {args.n_splits}\n")
        f.write(f"  test_fold: {args.test_fold}\n")
        f.write(f"  Groups: patient_id\n")
    print(f"Best parameters saved to {params_path}")
    
    # Save test set patient IDs for reference
    test_ids_path = os.path.join(args.output_path, 'test_patient_ids.txt')
    with open(test_ids_path, 'w') as f:
        f.write("Test Set Patient IDs:\n")
        for pid in sorted(groups_test.unique()):
            f.write(f"  {pid}\n")
    
    print("\n" + "="*50)
    print("TRAINING COMPLETE!")
    print("="*50)


if __name__ == "__main__":
    main()
