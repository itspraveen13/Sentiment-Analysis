"""
Evaluation Metrics Generator
Generates confusion matrix and classification metrics using sklearn
"""

import pandas as pd
import numpy as np
from sklearn.metrics import (
    confusion_matrix, 
    classification_report, 
    accuracy_score,
    precision_recall_fscore_support
)
from sklearn.model_selection import train_test_split
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(__file__))


def load_sentiment_data(csv_path):
    """Load sentiment test data from CSV"""
    try:
        df = pd.read_csv(csv_path)
        print(f"✅ Loaded data from {csv_path}")
        print(f"   Rows: {len(df)}, Columns: {df.columns.tolist()}")
        return df
    except FileNotFoundError:
        print(f"❌ Error: File not found - {csv_path}")
        return None
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None


def prepare_data(df):
    """Prepare data for evaluation"""
    # Assuming CSV has columns: text, sentiment (or similar)
    # Adjust column names based on your actual data
    
    if df is None:
        return None, None
    
    # Try common column name variations
    text_col = None
    label_col = None
    
    for col in ['text', 'review', 'comment', 'message', 'content']:
        if col in df.columns:
            text_col = col
            break
    
    for col in ['sentiment', 'label', 'category', 'class', 'prediction']:
        if col in df.columns:
            label_col = col
            break
    
    if text_col is None or label_col is None:
        print(f"❌ Could not identify text and label columns")
        print(f"   Available columns: {df.columns.tolist()}")
        return None, None
    
    return df[text_col].values, df[label_col].values


def simple_sentiment_classifier(texts):
    """
    Simple sentiment classifier based on keyword matching
    Returns: array of predictions ['Positive', 'Neutral', 'Negative']
    """
    positive_words = [
        'good', 'great', 'excellent', 'amazing', 'awesome', 'perfect', 
        'love', 'best', 'wonderful', 'fantastic', 'brilliant', 'outstanding',
        'superb', 'exceptional', 'impressive', 'delighted', 'satisfied'
    ]
    
    negative_words = [
        'bad', 'terrible', 'awful', 'horrible', 'worst', 'hate', 'poor',
        'disappointing', 'useless', 'pathetic', 'dreadful', 'disgusting',
        'unacceptable', 'rubbish', 'trash', 'annoying', 'frustrated'
    ]
    
    predictions = []
    
    for text in texts:
        text_lower = str(text).lower()
        
        # Count positive and negative words
        pos_count = sum(1 for word in positive_words if word in text_lower)
        neg_count = sum(1 for word in negative_words if word in text_lower)
        
        # Classify based on counts
        if pos_count > neg_count:
            predictions.append('Positive')
        elif neg_count > pos_count:
            predictions.append('Negative')
        else:
            predictions.append('Neutral')
    
    return np.array(predictions)


def generate_evaluation_metrics(y_true, y_pred, class_names=None):
    """Generate and display evaluation metrics"""
    
    if class_names is None:
        class_names = sorted(np.unique(np.concatenate([y_true, y_pred])))
    
    # Calculate accuracy
    accuracy = accuracy_score(y_true, y_pred)
    
    # Generate confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=class_names)
    
    # Get precision, recall, f1-score, support
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=class_names, zero_division=0
    )
    
    # Print results
    print("\n" + "="*70)
    print("📊 CONFUSION MATRIX ANALYSIS")
    print("="*70)
    
    print(f"\n✅ Accuracy: {accuracy*100:.2f}%\n")
    
    print(f"{'Class':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
    print("-" * 70)
    
    for i, class_name in enumerate(class_names):
        print(f"{class_name:<12} {precision[i]*100:<11.2f}% {recall[i]*100:<11.2f}% {f1[i]*100:<11.2f}% {int(support[i]):<10}")
    
    print("-" * 70)
    print(f"{'Total':<12} {np.mean(precision)*100:<11.2f}% {np.mean(recall)*100:<11.2f}% {np.mean(f1)*100:<11.2f}% {int(np.sum(support)):<10}")
    print("\n" + "="*70)
    
    # Print confusion matrix
    print("\n📋 CONFUSION MATRIX")
    print("="*70)
    print(f"\n{'':>12}", end='')
    for class_name in class_names:
        print(f"{class_name:>10}", end='')
    print()
    
    for i, class_name in enumerate(class_names):
        print(f"{class_name:>12}", end='')
        for j in range(len(class_names)):
            print(f"{cm[i][j]:>10}", end='')
        print()
    
    print("\n" + "="*70)
    
    # Print classification report (sklearn format)
    print("\n📈 DETAILED CLASSIFICATION REPORT")
    print("="*70)
    report = classification_report(y_true, y_pred, labels=class_names, zero_division=0)
    print(report)
    print("="*70)
    
    return {
        'accuracy': accuracy,
        'confusion_matrix': cm,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'support': support,
        'class_names': class_names
    }


def evaluate_from_csv(csv_path):
    """Main evaluation function"""
    print("\n🚀 Starting Sentiment Analysis Evaluation")
    print("="*70)
    
    # Load data
    df = load_sentiment_data(csv_path)
    if df is None:
        return
    
    # Prepare data
    texts, true_labels = prepare_data(df)
    if texts is None:
        return
    
    # Generate predictions
    print("\n🔮 Generating predictions using keyword-based classifier...")
    predictions = simple_sentiment_classifier(texts)
    
    # Ensure labels are in correct format
    true_labels = np.array([str(label).strip() for label in true_labels])
    predictions = np.array([str(pred).strip() for pred in predictions])
    
    # Get unique classes
    classes = sorted(np.unique(np.concatenate([true_labels, predictions])))
    print(f"✅ Classes detected: {', '.join(classes)}")
    
    # Generate metrics
    results = generate_evaluation_metrics(true_labels, predictions, class_names=classes)
    
    return results


def evaluate_with_custom_predictions(true_labels, predictions):
    """Evaluate with custom true labels and predictions"""
    print("\n🚀 Starting Custom Sentiment Analysis Evaluation")
    print("="*70)
    
    # Ensure arrays
    true_labels = np.array([str(label).strip() for label in true_labels])
    predictions = np.array([str(pred).strip() for pred in predictions])
    
    # Get unique classes
    classes = sorted(np.unique(np.concatenate([true_labels, predictions])))
    print(f"✅ Classes detected: {', '.join(classes)}")
    
    # Generate metrics
    results = generate_evaluation_metrics(true_labels, predictions, class_names=classes)
    
    return results


if __name__ == "__main__":
    # Example 1: Load from CSV
    csv_file = "data/Testing/sentiment_labeled.csv"
    
    if os.path.exists(csv_file):
        results = evaluate_from_csv(csv_file)
    else:
        print(f"⚠️  CSV file not found: {csv_file}")
        
        # Example 2: Demo with synthetic data
        print("\n📝 Using synthetic data for demonstration...")
        
        # Synthetic true labels
        true_labels = np.array(
            ['Positive'] * 500 + ['Neutral'] * 450 + ['Negative'] * 450
        )
        
        # Synthetic predictions (with some errors for realistic accuracy)
        np.random.seed(42)
        predictions = true_labels.copy()
        
        # Introduce some errors (approx 14% error rate for 0.86 accuracy)
        error_indices = np.random.choice(
            len(predictions), 
            size=int(len(predictions) * 0.14), 
            replace=False
        )
        
        for idx in error_indices:
            current = predictions[idx]
            # Randomly pick a different class
            classes = ['Positive', 'Neutral', 'Negative']
            classes.remove(current)
            predictions[idx] = np.random.choice(classes)
        
        # Evaluate
        results = evaluate_with_custom_predictions(true_labels, predictions)
