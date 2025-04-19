#!/usr/bin/env python3
"""
Machine Learning Models for Adaptive Honeypot System (Part 1)
This script implements various ML models for threat detection and adaptive behavior.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import pickle
import json

# ML libraries
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support, silhouette_score
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from xgboost import XGBClassifier

# Set plot style
plt.style.use('ggplot')
sns.set(style="darkgrid")

# Define directories
DATA_DIR = "/opt/cowrie/adaptive-honeypot/processed_data"
OUTPUT_DIR = "/opt/cowrie/adaptive-honeypot/ml_models"
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")

# Create output directories if they don't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

def load_data():
    """Load the preprocessed data."""
    sessions_df = pd.read_csv(os.path.join(DATA_DIR, 'cowrie_sessions.csv'))
    
    # Convert timestamp to datetime if needed
    if 'start_time' in sessions_df.columns:
        sessions_df['start_datetime'] = pd.to_datetime(sessions_df['start_time'], errors='coerce')
    
    if 'end_time' in sessions_df.columns:
        sessions_df['end_datetime'] = pd.to_datetime(sessions_df['end_time'], errors='coerce')
    
    return sessions_df



def prepare_features(data):
    """Prepare features for machine learning."""
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
    
    # Select features
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
    
    return X, y, feature_columns


def split_data(X, y, test_size=0.25, random_state=42):
    """Split data into training and testing sets."""
    # Convert y to integer type to avoid issues
    y = y.astype(int)
    
    # Check for NaN values one more time
    if X.isna().any().any() or y.isna().any():
        print("Warning: NaN values found in data. Filling with zeros.")
        X.fillna(0, inplace=True)
        y.fillna(0, inplace=True)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    return X_train, X_test, y_train, y_test


def build_preprocessing_pipeline(feature_columns):
    """Build preprocessing pipeline for numerical and categorical features."""
    # Identify numeric and categorical columns
    numeric_features = [col for col in feature_columns 
                        if col not in ['session_id', 'src_ip', 'username', 'password', 'commands']]
    
    # Create preprocessing pipelines
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features)
        ]
    )
    
    return preprocessor
def train_threat_classifier(X_train, y_train, X_test, y_test, preprocessor):
    """Train and evaluate multiple classifiers for threat detection."""
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
    
    # Train and evaluate each classifier
    for name, classifier in classifiers.items():
        print(f"Training {name}...")
        
        # Create pipeline with preprocessing and classifier
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', classifier)
        ])
        
        # Train the model
        pipeline.fit(X_train, y_train)
        
        # Make predictions
        y_pred = pipeline.predict(X_test)
        y_pred_proba = pipeline.predict_proba(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
        
        # Store results
        results[name] = {
            'pipeline': pipeline,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
        
        # Print results
        print(f"{name} Results:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1 Score: {f1:.4f}")
        print(classification_report(y_test, y_pred))
        print("\n")
        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(os.path.join(PLOTS_DIR, f'confusion_matrix_{name.replace(" ", "_")}.png'), dpi=300)
        
        # Plot ROC curve for multi-class
        plt.figure(figsize=(10, 8))
        n_classes = len(np.unique(y_test))
        
        if n_classes > 2:
            # One-vs-Rest approach for multiclass
            for i in range(n_classes):
                fpr, tpr, _ = roc_curve(
                    (y_test == i).astype(int), 
                    y_pred_proba[:, i]
                )
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, lw=2,
                        label=f'Class {i} (AUC = {roc_auc:.2f})')
        else:
            # Binary classification
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba[:, 1])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=2,
                    label=f'ROC curve (AUC = {roc_auc:.2f})')
        
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {name}')
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(PLOTS_DIR, f'roc_curve_{name.replace(" ", "_")}.png'), dpi=300)
    
    # Find best model based on F1 score
    best_model_name = max(results, key=lambda k: results[k]['f1'])
    best_model = results[best_model_name]
    
    print(f"Best model: {best_model_name} with F1 score: {best_model['f1']:.4f}")
    
    # Save best model
    with open(os.path.join(OUTPUT_DIR, 'best_threat_classifier.pkl'), 'wb') as f:
        pickle.dump(best_model['pipeline'], f)
    
    # Save model comparison results
    model_comparison = {name: {
        'accuracy': results[name]['accuracy'],
        'precision': results[name]['precision'],
        'recall': results[name]['recall'],
        'f1': results[name]['f1']
    } for name in results}
    
    with open(os.path.join(OUTPUT_DIR, 'model_comparison.json'), 'w') as f:
        json.dump(model_comparison, f, indent=4)
    # Plot model comparison
    plt.figure(figsize=(12, 8))
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
    
    return results, best_model_name
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
    
    # Save feature importances
    feature_importance = {feature_columns[i]: importances[i] for i in range(len(feature_columns))}
    sorted_importance = {k: v for k, v in sorted(feature_importance.items(), key=lambda item: item[1], reverse=True)}
    
    with open(os.path.join(OUTPUT_DIR, 'feature_importance.json'), 'w') as f:
        json.dump(sorted_importance, f, indent=4)
    
    return sorted_importance
def train_clustering_model(X, preprocessor):
    """Train a clustering model to identify attack patterns."""
    # Preprocess the data
    X_processed = preprocessor.fit_transform(X)
    
    # Determine optimal number of clusters using the elbow method
    inertia = []
    k_range = range(2, 11)
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X_processed)
        inertia.append(kmeans.inertia_)
    
    # Plot elbow curve
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, inertia, 'o-')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Inertia')
    plt.title('Elbow Method for Optimal k')
    plt.grid(True)
    plt.savefig(os.path.join(PLOTS_DIR, 'kmeans_elbow.png'), dpi=300)
    
    # Choose optimal k (this is a simple heuristic, could be improved)
    # Find the point of maximum curvature
    k_diff = np.diff(inertia)
    k_diff2 = np.diff(k_diff)
    optimal_k = k_range[np.argmax(np.abs(k_diff2)) + 1]
    
    print(f"Optimal number of clusters: {optimal_k}")
    
    # Train KMeans with optimal k
    kmeans = KMeans(n_clusters=optimal_k, random_state=42)
    kmeans.fit(X_processed)
    
    # Save the model
    with open(os.path.join(OUTPUT_DIR, 'kmeans_clustering.pkl'), 'wb') as f:
        pickle.dump((kmeans, preprocessor), f)
    
    # Visualize clusters using PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_processed)
    
    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=kmeans.labels_, cmap='viridis', alpha=0.7)
    plt.colorbar(scatter, label='Cluster')
    plt.title('Attack Pattern Clusters (PCA Visualization)')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.savefig(os.path.join(PLOTS_DIR, 'kmeans_clusters.png'), dpi=300)
    
    return kmeans, optimal_k
def build_adaptive_response_model(X, y, kmeans_labels):
    """Build a model for adaptive honeypot responses based on threat level and attack patterns."""
    # Combine threat level and cluster information
    X_adaptive = X.copy()
    X_adaptive['cluster'] = kmeans_labels
    
    # Define response strategies based on threat level and cluster
    # This is a simplified approach - in a real system, this would be more sophisticated
    response_strategies = {
        'low_interaction': 0,  # Minimal interaction, basic responses
        'medium_interaction': 1,  # More interactive, simulated vulnerabilities
        'high_interaction': 2   # Full system emulation, extensive logging
    }
    
    # Create a mapping from (threat_level, cluster) to response_strategy
    # This is a rule-based approach for demonstration
    # In a real system, this could be learned from data or defined by experts
    
    # Initialize with default low interaction
    response_mapping = {}
    
    # For each unique combination of threat level and cluster
    for threat in np.unique(y):
        for cluster in np.unique(kmeans_labels):
            # Default strategy
            strategy = 'low_interaction'
            
            # High threat level gets high interaction
            if threat == 2:  # high threat
                strategy = 'high_interaction'
            # Medium threat level gets medium interaction
            elif threat == 1:  # medium threat
                strategy = 'medium_interaction'
            # For low threat, use cluster information
            else:
                # If it's in a cluster associated with potential reconnaissance
                # This is just an example - in reality, you'd analyze cluster characteristics
                if cluster in [0, 2]:  # Assuming clusters 0 and 2 show reconnaissance patterns
                    strategy = 'medium_interaction'
            
            # Store the mapping
            response_mapping[(int(threat), int(cluster))] = response_strategies[strategy]
    
    # Save the response mapping
    with open(os.path.join(OUTPUT_DIR, 'adaptive_response_mapping.json'), 'w') as f:
        json.dump({str(k): int(v) for k, v in response_mapping.items()}, f, indent=4)
    
    # Create a simple function to determine response based on threat level and cluster
    def get_adaptive_response(threat_level, cluster):
        key = (int(threat_level), int(cluster))
        return response_mapping.get(key, response_strategies['low_interaction'])
    
    # Save the function as part of a simple model
    adaptive_model = {
        'response_mapping': response_mapping,
        'response_strategies': {v: k for k, v in response_strategies.items()}
    }
    
    with open(os.path.join(OUTPUT_DIR, 'adaptive_response_model.pkl'), 'wb') as f:
        pickle.dump(adaptive_model, f)
    
    return adaptive_model
def generate_model_summary():
    """Generate a summary of the machine learning models."""
    summary = []
    
    summary.append("# Machine Learning Models for Adaptive Honeypot")
    summary.append("\n## Overview")
    summary.append("This document summarizes the machine learning models developed for the adaptive honeypot system.")
    
    # Threat Classification Model
    summary.append("\n## Threat Classification Model")
    
    try:
        with open(os.path.join(OUTPUT_DIR, 'model_comparison.json'), 'r') as f:
            model_comparison = json.load(f)
        
        summary.append("\n### Model Comparison")
        summary.append("| Model | Accuracy | Precision | Recall | F1 Score |")
        summary.append("|-------|----------|-----------|--------|----------|")
        
        for model_name, metrics in model_comparison.items():
            summary.append(f"| {model_name} | {metrics['accuracy']:.4f} | {metrics['precision']:.4f} | {metrics['recall']:.4f} | {metrics['f1']:.4f} |")
        
        # Find best model
        best_model = max(model_comparison.items(), key=lambda x: x[1]['f1'])
        summary.append(f"\nThe best performing model is **{best_model[0]}** with an F1 score of {best_model[1]['f1']:.4f}.")
    except:
        summary.append("\nModel comparison data not available.")
    
    # Feature Importance
    summary.append("\n## Feature Importance")
    
    try:
        with open(os.path.join(OUTPUT_DIR, 'feature_importance.json'), 'r') as f:
            feature_importance = json.load(f)
        
        summary.append("\n### Top Features")
        summary.append("| Feature | Importance |")
        summary.append("|---------|------------|")
        
        for feature, importance in list(feature_importance.items())[:10]:  # Top 10 features
            summary.append(f"| {feature} | {importance:.4f} |")
        
        summary.append("\nThese features are the most predictive for threat classification.")
    except:
        summary.append("\nFeature importance data not available.")
    # Clustering Model
    summary.append("\n## Attack Pattern Clustering")
    summary.append("\nA K-means clustering model was used to identify distinct attack patterns in the data.")
    
    # Adaptive Response Model
    summary.append("\n## Adaptive Response Model")
    summary.append("\nThe adaptive response model combines threat classification and attack pattern clustering to determine the appropriate honeypot response strategy.")
    summary.append("\n### Response Strategies")
    summary.append("1. **Low Interaction**: Minimal interaction, basic responses")
    summary.append("2. **Medium Interaction**: More interactive, simulated vulnerabilities")
    summary.append("3. **High Interaction**: Full system emulation, extensive logging")
    
    # Implementation
    summary.append("\n## Implementation")
    summary.append("\nThe models are implemented in a pipeline that:")
    summary.append("1. Classifies the threat level of an incoming connection")
    summary.append("2. Identifies the attack pattern cluster")
    summary.append("3. Determines the appropriate response strategy")
    summary.append("4. Adapts the honeypot behavior accordingly")
    
    # Write summary to file
    with open(os.path.join(OUTPUT_DIR, 'model_summary.md'), 'w') as f:
        f.write('\n'.join(summary))
    
    return '\n'.join(summary)

def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    """Train and evaluate multiple machine learning models."""
    models = {}
    
    # Check if we have enough classes for classification
    unique_classes = np.unique(y_train)
    if len(unique_classes) < 2:
        print(f"Warning: Only {len(unique_classes)} class found in the data. Some models may not work.")
        print(f"Creating synthetic data for training...")
        
        # Create some synthetic data with multiple classes
        from sklearn.utils import resample
        
        # If we have only one class (e.g., all 0s), create some samples with class 1
        if len(unique_classes) == 1:
            # Get the existing class
            existing_class = unique_classes[0]
            # Create a new class
            new_class = 1 if existing_class == 0 else 0
            
            # Create synthetic samples for the new class
            X_synthetic = resample(X_train, n_samples=max(5, int(len(X_train) * 0.1)))
            y_synthetic = np.full(len(X_synthetic), new_class)
            
            # Combine with original data
            X_train_augmented = pd.concat([X_train, pd.DataFrame(X_synthetic, columns=X_train.columns)])
            y_train_augmented = pd.concat([y_train, pd.Series(y_synthetic)])
            
            print(f"Added {len(X_synthetic)} synthetic samples with class {new_class}")
            
            # Use the augmented data for training
            X_train = X_train_augmented
            y_train = y_train_augmented
    
    # Define models to train
    model_definitions = {
        'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'gradient_boosting': GradientBoostingClassifier(random_state=42),
        'xgboost': XGBClassifier(random_state=42),
        'logistic_regression': LogisticRegression(max_iter=1000, random_state=42),
        'svm': SVC(probability=True, random_state=42),
        'neural_network': MLPClassifier(max_iter=1000, random_state=42)
    }
    
    # Train and evaluate each model
    for name, model in model_definitions.items():
        print(f"Training {name}...")
        
        try:
            # Train the model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_test, y_pred, average='weighted'
            )
            
            # Store model and metrics
            models[name] = {
                'model': model,
                'metrics': {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1
                },
                'predictions': y_pred
            }
            
            print(f"  Accuracy: {accuracy:.4f}")
            print(f"  F1 Score: {f1:.4f}")
        except Exception as e:
            print(f"  Error training {name}: {str(e)}")
            print(f"  Skipping this model...")
    
    if models:
        # Determine best model based on F1 score
        best_model = max(models.items(), key=lambda x: x[1]['metrics']['f1'])
        print(f"\nBest model: {best_model[0]} (F1: {best_model[1]['metrics']['f1']:.4f})")
    else:
        print("No models were successfully trained.")
    
    return models
def save_models(models, X_train, feature_columns):
    """Save trained models and related data."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Save the best model
    if models:
        best_model_name = max(models.items(), key=lambda x: x[1]['metrics']['f1'])[0]
        best_model = models[best_model_name]['model']
        
        with open(os.path.join(OUTPUT_DIR, 'best_model.pkl'), 'wb') as f:
            pickle.dump(best_model, f)
        
        print(f"Saved best model ({best_model_name}) to {os.path.join(OUTPUT_DIR, 'best_model.pkl')}")
    
    # Save feature columns
    with open(os.path.join(OUTPUT_DIR, 'feature_columns.json'), 'w') as f:
        json.dump(feature_columns, f)
    
    # Save a sample of the training data for reference
    sample_data = X_train.head(5)
    sample_data.to_csv(os.path.join(OUTPUT_DIR, 'sample_data.csv'), index=False)
    
    return True

def perform_clustering(X, y):
    """Perform clustering to identify attack patterns."""
    # Scale the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Determine optimal number of clusters using silhouette score
    silhouette_scores = []
    K_range = range(2, min(6, len(X) // 10 + 1))  # Limit based on data size
    
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        cluster_labels = kmeans.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, cluster_labels) if len(np.unique(cluster_labels)) > 1 else 0
        silhouette_scores.append(score)
    
    # If we couldn't calculate scores (e.g., not enough data), default to 2 clusters
    if not silhouette_scores:
        optimal_k = 2
    else:
        optimal_k = K_range[np.argmax(silhouette_scores)]
    
    print(f"Optimal number of clusters: {optimal_k}")
    
    # Train final clustering model
    kmeans = KMeans(n_clusters=optimal_k, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)
    
    # Save the clustering model
    with open(os.path.join(OUTPUT_DIR, 'kmeans_model.pkl'), 'wb') as f:
        pickle.dump(kmeans, f)
    
    # Save the scaler
    with open(os.path.join(OUTPUT_DIR, 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)
    
    return kmeans, clusters

def create_adaptive_response(y, clusters):
    """Create mapping between threat levels, clusters, and response strategies."""
    # Define response strategies
    response_strategies = {
        0: "low",      # Low interaction
        1: "medium",   # Medium interaction
        2: "high"      # High interaction
    }
    
    # Create mapping
    response_mapping = {}
    
    # Get unique threat levels and clusters
    threat_levels = np.unique(y)
    cluster_ids = np.unique(clusters)
    
    # For each combination of threat level and cluster
    for threat in threat_levels:
        for cluster in cluster_ids:
            # Determine appropriate response strategy
            if threat == 2:  # High threat
                strategy = 2  # High interaction
            elif threat == 1:  # Medium threat
                strategy = 1  # Medium interaction
            else:  # Low threat
                strategy = 0  # Low interaction
            
            # Store in mapping
            response_mapping[(int(threat), int(cluster))] = response_strategies[strategy]
    
    # Save mapping
    with open(os.path.join(OUTPUT_DIR, 'adaptive_response_mapping.json'), 'w') as f:
        json.dump({f"{k[0]},{k[1]}": v for k, v in response_mapping.items()}, f, indent=4)
    
    return response_mapping

def generate_visualizations(X, y, models, X_test, y_test, clusters):
    """Generate visualizations for the models and data."""
    os.makedirs(PLOTS_DIR, exist_ok=True)
    
    # 1. Feature importance for the best model
    if models and hasattr(list(models.values())[0]['model'], 'feature_importances_'):
        best_model_name = max(models.items(), key=lambda x: x[1]['metrics']['f1'])[0]
        best_model = models[best_model_name]['model']
        
        if hasattr(best_model, 'feature_importances_'):
            plt.figure(figsize=(10, 6))
            feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': best_model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            sns.barplot(x='importance', y='feature', data=feature_importance)
            plt.title(f'Feature Importance ({best_model_name})')
            plt.tight_layout()
            plt.savefig(os.path.join(PLOTS_DIR, 'feature_importance.png'))
    
    # 2. Confusion matrix for the best model
    if models:
        best_model_name = max(models.items(), key=lambda x: x[1]['metrics']['f1'])[0]
        y_pred = models[best_model_name]['predictions']
        
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix ({best_model_name})')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, 'confusion_matrix.png'))
    
    # 3. Cluster visualization (if we have enough features)
    if X.shape[1] >= 2:
        # Use PCA to reduce to 2D if needed
        if X.shape[1] > 2:
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X)
        else:
            X_pca = X.values
        
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis', alpha=0.6)
        plt.colorbar(scatter, label='Cluster')
        plt.title('Attack Pattern Clusters')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, 'clusters.png'))
    
    return True

def generate_model_summary(models, X_test, y_test, feature_columns):
    """Generate a summary of model performance."""
    if not models:
        print("No models to summarize.")
        return
    
    # Create summary markdown
    summary = "# Machine Learning Model Summary\n\n"
    
    # Model performance comparison
    summary += "## Model Performance Comparison\n\n"
    summary += "| Model | Accuracy | Precision | Recall | F1 Score |\n"
    summary += "|-------|----------|-----------|--------|----------|\n"
    
    for name, model_data in models.items():
        metrics = model_data['metrics']
        summary += f"| {name} | {metrics['accuracy']:.4f} | {metrics['precision']:.4f} | "
        summary += f"{metrics['recall']:.4f} | {metrics['f1']:.4f} |\n"
    
    # Best model
    best_model_name = max(models.items(), key=lambda x: x[1]['metrics']['f1'])[0]
    summary += f"\nBest model: **{best_model_name}** (F1: {models[best_model_name]['metrics']['f1']:.4f})\n\n"
    
    # Feature importance
    summary += "## Feature Importance\n\n"
    best_model = models[best_model_name]['model']
    
    if hasattr(best_model, 'feature_importances_'):
        feature_importance = pd.DataFrame({
            'feature': feature_columns,
            'importance': best_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        summary += "| Feature | Importance |\n"
        summary += "|---------|------------|\n"
        
        for _, row in feature_importance.iterrows():
            summary += f"| {row['feature']} | {row['importance']:.4f} |\n"
    else:
        summary += "Feature importance not available for this model type.\n"
    
    # Save summary
    with open(os.path.join(OUTPUT_DIR, 'model_summary.md'), 'w') as f:
        f.write(summary)
    
    print(f"Model summary saved to {os.path.join(OUTPUT_DIR, 'model_summary.md')}")
    return True




def main():
    """Main function."""
    try:
        print("Loading preprocessed data...")
        data = load_data()
        
        if data.empty:
            print("Error: No data found. Please run the preprocessing script first.")
            return
        
        print("Preparing features...")
        X, y, feature_columns = prepare_features(data)
        
        print("Splitting data into training and testing sets...")
        X_train, X_test, y_train, y_test = split_data(X, y)
        
        print("Training and evaluating models...")
        models = train_and_evaluate_models(X_train, X_test, y_train, y_test)
        
        print("Saving models...")
        save_models(models, X_train, feature_columns)
        
        print("Performing clustering analysis...")
        kmeans, clusters = perform_clustering(X, y)
        
        print("Creating adaptive response mapping...")
        response_mapping = create_adaptive_response(y, clusters)
        
        print("Generating visualizations...")
        generate_visualizations(X, y, models, X_test, y_test, clusters)
        
        print("Generating model summary...")
        generate_model_summary(models, X_test, y_test, feature_columns)
        
        print(f"Machine learning models trained and saved to {OUTPUT_DIR}")
        print(f"Visualizations saved to {PLOTS_DIR}")
        
        # Print some basic statistics
        class_counts = pd.Series(y).value_counts().sort_index()
        print("\nThreat level distribution:")
        for level, count in enumerate(class_counts):
            level_name = "Low" if level == 0 else "Medium" if level == 1 else "High"
            print(f"  {level_name}: {count} sessions")
        
        cluster_counts = pd.Series(clusters).value_counts().sort_index()
        print("\nCluster distribution:")
        for cluster, count in enumerate(cluster_counts):
            print(f"  Cluster {cluster}: {count} sessions")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        print("Please check your data and try again.")
        import traceback
        traceback.print_exc()
        return


if __name__ == "__main__":
    main()
