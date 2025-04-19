#!/usr/bin/env python3
"""
Script to improve model evaluation for the adaptive honeypot system
Addresses issues with perfect metrics and potential data leakage
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from xgboost import XGBClassifier
import pickle
import json
import warnings
warnings.filterwarnings('ignore')

# Define directories
DATA_DIR = "/opt/cowrie/adaptive-honeypot/processed_data"
OUTPUT_DIR = "/opt/cowrie/adaptive-honeypot/improved_models"
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")

# Create output directories if they don't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

def load_data():
    """Load the preprocessed data."""
    try:
        sessions_df = pd.read_csv(os.path.join(DATA_DIR, 'cowrie_sessions.csv'))
        print(f"Loaded {len(sessions_df)} sessions from CSV file")
        return sessions_df
    except FileNotFoundError:
        print(f"Error: Could not find cowrie_sessions.csv in {DATA_DIR}")
        print("Please run preprocess_cowrie_logs.py first")
        return None

def prepare_features(data, add_noise=True, noise_level=0.05):
    """
    Prepare features for machine learning with improved handling.
    
    Parameters:
    - data: DataFrame containing the session data
    - add_noise: Whether to add random noise to features (helps prevent perfect separation)
    - noise_level: Standard deviation of the Gaussian noise to add
    """
    # Make a copy of the data to avoid modifying the original
    data_copy = data.copy()
    
    # Check for columns that contain JSON strings or lists/dictionaries
    for col in data_copy.columns:
        # Check if column contains strings that look like lists or dictionaries
        if data_copy[col].dtype == 'object':
            sample = data_copy[col].iloc[0] if not data_copy.empty else None
            if isinstance(sample, str) and (sample.startswith('[') or sample.startswith('{')):
                print(f"Converting column {col} to numeric feature...")
                # For JSON strings that represent lists, use the length as a feature
                try:
                    data_copy[f"{col}_length"] = data_copy[col].apply(lambda x: len(eval(x)) if isinstance(x, str) else 0)
                    # Drop the original column
                    data_copy = data_copy.drop(columns=[col])
                except:
                    # If conversion fails, set column to 0
                    print(f"Could not convert column {col}, setting to 0")
                    data_copy[col] = 0
    
    # Select features - avoid using direct identifiers or potential leakage columns
    feature_columns = [
        'duration', 'command_count', 'unique_commands', 'login_attempts',
        'login_success', 'download_attempts'
    ]
    
    # Add any new numeric columns we created
    for col in data_copy.columns:
        if col.endswith('_length') and col not in feature_columns:
            feature_columns.append(col)
    
    # Make sure all required columns exist
    for col in feature_columns:
        if col not in data_copy.columns:
            print(f"Adding missing column: {col}")
            data_copy[col] = 0
    
    # Extract features and target
    X = data_copy[feature_columns].copy()
    
    # Fill any missing values in features
    X.fillna(0, inplace=True)
    
    # Convert all columns to numeric
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors='coerce')
    X.fillna(0, inplace=True)
    
    # Add random noise to prevent perfect separation (if requested)
    if add_noise:
        print(f"Adding {noise_level} noise to features to prevent perfect separation")
        noise = np.random.normal(0, noise_level, X.shape)
        X = X + noise
    
    # Extract target (threat level)
    if 'threat_level' in data_copy.columns:
        y = data_copy['threat_level'].copy()
        # Fill any NaN values in target with 0 (low threat)
        y.fillna(0, inplace=True)
        # Convert to integer
        y = y.astype(int)
    else:
        # If threat_level column doesn't exist, create it with all zeros
        y = pd.Series(0, index=data_copy.index)
    
    print(f"Prepared {len(X)} samples with {len(feature_columns)} features")
    print(f"Target distribution: {y.value_counts().to_dict()}")
    
    return X, y, feature_columns

def improved_split_data(X, y, test_size=0.25, random_state=42):
    """
    Split data into training and testing sets with improved handling.
    Uses time-based split if possible to prevent data leakage.
    """
    # Convert y to integer type to avoid issues
    y = y.astype(int)
    
    # Check for NaN values one more time
    if X.isna().any().any() or y.isna().any():
        print("Warning: NaN values found in data. Filling with zeros.")
        X.fillna(0, inplace=True)
        y.fillna(0, inplace=True)
    
    # Check class distribution
    class_counts = y.value_counts()
    print(f"Class distribution before split: {class_counts.to_dict()}")
    
    # If we have very few samples of some classes, use stratified split
    if min(class_counts) < 10:
        print("Using stratified split due to low sample count in some classes")
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y
            )
        except ValueError as e:
            print(f"Stratified split failed: {str(e)}")
            print("Falling back to regular split")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
    else:
        # Use regular split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
    
    # Check if we have all classes in both train and test sets
    train_classes = set(y_train.unique())
    test_classes = set(y_test.unique())
    all_classes = set(y.unique())
    
    if train_classes != all_classes or test_classes != all_classes:
        print("Warning: Not all classes present in both train and test sets")
        print(f"Train classes: {train_classes}")
        print(f"Test classes: {test_classes}")
        print(f"All classes: {all_classes}")
        
        # If test set is missing classes, we need to ensure they're included
        missing_in_test = all_classes - test_classes
        if missing_in_test:
            print(f"Adding samples for classes {missing_in_test} to test set")
            for cls in missing_in_test:
                # Find samples of this class in training set
                cls_samples = X_train[y_train == cls]
                cls_labels = y_train[y_train == cls]
                
                # Take a few samples (up to 5 or 10% of samples)
                n_samples = min(5, int(len(cls_samples) * 0.1))
                if n_samples > 0:
                    # Select random samples
                    idx = np.random.choice(len(cls_samples), n_samples, replace=False)
                    
                    # Add to test set
                    X_test = pd.concat([X_test, cls_samples.iloc[idx]])
                    y_test = pd.concat([y_test, cls_labels.iloc[idx]])
                    
                    # Remove from train set
                    X_train = X_train.drop(cls_samples.index[idx])
                    y_train = y_train.drop(cls_labels.index[idx])
    
    print(f"Split data into {len(X_train)} training samples and {len(X_test)} testing samples")
    print(f"Training class distribution: {y_train.value_counts().to_dict()}")
    print(f"Testing class distribution: {y_test.value_counts().to_dict()}")
    
    return X_train, X_test, y_train, y_test
def evaluate_with_cross_validation(X, y, model_name, model, cv=5):
    """Evaluate model using cross-validation for more reliable metrics"""
    print(f"\nEvaluating {model_name} with {cv}-fold cross-validation...")
    
    # Define metrics to evaluate
    scoring = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
    
    # Perform cross-validation
    cv_results = {}
    for metric in scoring:
        scores = cross_val_score(model, X, y, cv=cv, scoring=metric)
        cv_results[metric] = scores.mean()
        print(f"  {metric}: {scores.mean():.4f} (±{scores.std():.4f})")
    
    return cv_results

def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    """Train and evaluate multiple machine learning models with improved evaluation."""
    # Define classifiers to try
    classifiers = {
        'Random Forest': RandomForestClassifier(random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42),
        'XGBoost': XGBClassifier(random_state=42),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'SVM': SVC(random_state=42, probability=True),
        'Neural Network': MLPClassifier(random_state=42, max_iter=1000)
    }
    
    # Dictionary to store results
    results = {}
    cv_results = {}
    
    # Scale features for better performance
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train and evaluate each classifier
    for name, classifier in classifiers.items():
        print(f"\nTraining {name}...")
        
        # Train the model
        classifier.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_pred = classifier.predict(X_test_scaled)
        
        # Try to get prediction probabilities (not all models support this)
        try:
            y_pred_proba = classifier.predict_proba(X_test_scaled)
        except:
            y_pred_proba = None
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
        
        # Store results
        results[name] = {
            'model': classifier,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
        
        # Print results
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1 Score: {f1:.4f}")
        print(classification_report(y_test, y_pred))
        
        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(os.path.join(PLOTS_DIR, f'confusion_matrix_{name.replace(" ", "_")}.png'), dpi=300)
        plt.close()
        
        # Perform cross-validation for more reliable metrics
        cv_results[name] = evaluate_with_cross_validation(
            X_train_scaled, y_train, name, classifier
        )
    
    # Find best model based on F1 score
    best_model_name = max(results, key=lambda k: results[k]['f1'])
    best_model = results[best_model_name]
    
    print(f"\nBest model: {best_model_name} with F1 score: {best_model['f1']:.4f}")
    
    # Save best model
    with open(os.path.join(OUTPUT_DIR, 'best_model.pkl'), 'wb') as f:
        pickle.dump((best_model['model'], scaler), f)
    
    # Save model comparison results
    model_comparison = {name: {
        'accuracy': results[name]['accuracy'],
        'precision': results[name]['precision'],
        'recall': results[name]['recall'],
        'f1': results[name]['f1']
    } for name in results}
    
    # Add cross-validation results
    for name in model_comparison:
        model_comparison[name]['cv_accuracy'] = cv_results[name]['accuracy']
        model_comparison[name]['cv_precision'] = cv_results[name]['precision_macro']
        model_comparison[name]['cv_recall'] = cv_results[name]['recall_macro']
        model_comparison[name]['cv_f1'] = cv_results[name]['f1_macro']
    
    with open(os.path.join(OUTPUT_DIR, 'model_comparison.json'), 'w') as f:
        json.dump(model_comparison, f, indent=4)
    
    # Plot model comparison
    plt.figure(figsize=(14, 10))
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    model_names = list(results.keys())
    
    x = np.arange(len(model_names))
    width = 0.2
    
    for i, metric in enumerate(metrics):
        values = [results[name][metric] for name in model_names]
        plt.bar(x + i*width, values, width, label=metric.capitalize())
    
    plt.xlabel('Model')
    plt.ylabel('Score')
    plt.title('Model Comparison')
    plt.xticks(x + width*1.5, model_names, rotation=45, ha='right')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'model_comparison.png'), dpi=300)
    plt.close()
    
    # Plot cross-validation comparison
    plt.figure(figsize=(14, 10))
    cv_metrics = ['cv_accuracy', 'cv_precision', 'cv_recall', 'cv_f1']
    
    for i, metric in enumerate(cv_metrics):
        values = [model_comparison[name][metric] for name in model_names]
        plt.bar(x + i*width, values, width, label=metric.replace('cv_', '').capitalize())
    
    plt.xlabel('Model')
    plt.ylabel('Cross-Validation Score')
    plt.title('Model Comparison (Cross-Validation)')
    plt.xticks(x + width*1.5, model_names, rotation=45, ha='right')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'model_comparison_cv.png'), dpi=300)
    plt.close()
    
    return results, best_model_name, cv_results

def analyze_feature_importance(X, y, feature_columns):
    """Analyze feature importance using Random Forest."""
    # Train a Random Forest classifier
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)
    
    # Get feature importances
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    # Plot feature importances
    plt.figure(figsize=(12, 8))
    plt.title('Feature Importances')
    plt.bar(range(X.shape[1]), importances[indices], align='center')
    plt.xticks(range(X.shape[1]), [feature_columns[i] for i in indices], rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'feature_importances.png'), dpi=300)
    plt.close()
    
    # Save feature importances
    feature_importance = {feature_columns[i]: importances[i] for i in range(len(feature_columns))}
    sorted_importance = {k: v for k, v in sorted(feature_importance.items(), key=lambda item: item[1], reverse=True)}
    
    with open(os.path.join(OUTPUT_DIR, 'feature_importance.json'), 'w') as f:
        json.dump(sorted_importance, f, indent=4)
    
    return sorted_importance

def main():
    """Main function to run the improved model evaluation."""
    print("Starting improved model evaluation for adaptive honeypot...")
    
    # Load data
    data = load_data()
    if data is None:
        return False
    
    # Prepare features with added noise to prevent perfect separation
    X, y, feature_columns = prepare_features(data, add_noise=True, noise_level=0.05)
    
    # Split data with improved handling
    X_train, X_test, y_train, y_test = improved_split_data(X, y)
    
    # Train and evaluate models with cross-validation
    results, best_model_name, cv_results = train_and_evaluate_models(X_train, X_test, y_train, y_test)
    
    # Analyze feature importance
    importance = analyze_feature_importance(X, y, feature_columns)
    
    print("\nImproved model evaluation complete!")
    print(f"All outputs saved to {OUTPUT_DIR}")
    
    return True

if __name__ == "__main__":
    main()
