"""
Evaluation Metrics Generator
Generates confusion matrix and classification metrics using sklearn
"""

import sys
import io

# Fix encoding for Windows console (UTF-8)
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

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
    # Handle both absolute and relative paths
    if not os.path.isabs(csv_path):
        # Try relative to Backend directory first
        backend_path = os.path.join(os.path.dirname(__file__), csv_path)
        if os.path.exists(backend_path):
            csv_path = backend_path
        # Then try relative to project root
        else:
            root_path = os.path.join(os.path.dirname(__file__), '..', csv_path)
            if os.path.exists(root_path):
                csv_path = root_path
    
    try:
        df = pd.read_csv(csv_path)
        print(f"[OK] Loaded data from {csv_path}")
        print(f"   Rows: {len(df)}, Columns: {df.columns.tolist()}")
        return df
    except FileNotFoundError:
        print(f"[WARN] CSV file not found: {csv_path}")
        return None
    except Exception as e:
        print(f"[ERROR] Error loading data: {e}")
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
    Optimized sentiment classifier with enhanced word detection
    Returns: array of predictions ['Positive', 'Neutral', 'Negative']
    """
    positive_words = {
        # Strong positives (weight 2.0)
        'love': 2, 'amazing': 2, 'awesome': 2, 'excellent': 2, 'perfect': 2,
        'fantastic': 2, 'brilliant': 2, 'outstanding': 2, 'wonderful': 2, 'best': 2,
        'superb': 2, 'exceptional': 2, 'phenomenal': 2, 'incredible': 2, 'magnificent': 2,
        'glorious': 2, 'marvelous': 2, 'stellar': 2, 'thrilled': 2, 'delighted': 2,
        'exquisite': 2, 'heavenly': 2, 'extraordinary': 2, 'rapturous': 2, 'elated': 2,
        'ecstatic': 2, 'blissful': 2, 'joyful': 2, 'exultant': 2, 'splendid': 2,
        'enchanting': 2, 'delightful': 2, 'sublime': 2, 'radiant': 2, 'remarkable': 2,
        # Medium positives (weight 1.5)
        'great': 1.5, 'impressive': 1.5, 'satisfied': 1.5, 'happy': 1.5,
        'lovely': 1.5, 'beautiful': 1.5, 'charming': 1.5, 'adorable': 1.5,
        'terrific': 1.5, 'grand': 1.5, 'pleased': 1.5, 'enjoyed': 1.5,
        'impressed': 1.5, 'thrilled': 1.5, 'recommend': 1.5, 'ideal': 1.5,
        'enchanted': 1.5, 'gorgeous': 1.5, 'valued': 1.5,
        # Light positives (weight 1.0)
        'good': 1, 'keen': 1, 'sharp': 1, 'smart': 1, 'nice': 0.7, 'pleasant': 1,
        'worth': 0.8, 'value': 0.8, 'benefit': 1, 'advantage': 1,
        'satisfactory': 0.5, 'fair': 0.5, 'decent': 0.5, 'okay': 0.3, 'alright': 0.3,
        'fine': 0.3, 'capable': 0.8, 'adequate': 0.5, 'passable': 0.5, 'acceptable': 0.7,
        'serviceable': 0.5, 'functional': 0.5, 'well': 1, 'super': 1.2, 'top': 1
    }
    
    negative_words = {
        # Strong negatives (weight 2.0)
        'hate': 2, 'terrible': 2, 'awful': 2, 'horrible': 2, 'worst': 2,
        'disappointing': 2, 'useless': 2, 'pathetic': 2, 'dreadful': 2, 'disgusting': 2,
        'unacceptable': 2, 'rubbish': 2, 'trash': 2, 'regret': 2, 'broken': 2,
        'failed': 2, 'failure': 2, 'appalling': 2, 'atrocious': 2, 'deplorable': 2,
        'detestable': 2, 'diabolical': 2, 'revolting': 2, 'shameful': 2, 'contemptible': 2,
        'nightmare': 2, 'disaster': 2, 'catastrophe': 2, 'abominable': 2, 'heinous': 2,
        'vile': 2, 'odious': 2, 'nefarious': 2, 'sinful': 2, 'wicked': 2,
        'sinister': 2, 'treacherous': 2, 'nightmarish': 2, 'calamitous': 2, 'insufferable': 2,
        'repulsive': 2, 'loathsome': 2, 'abhorrent': 2, 'atrocious': 2, 'appalling': 2,
        'filthy': 2, 'noxious': 2, 'nauseating': 2, 'bleak': 2, 'dire': 2,
        # Medium negatives (weight 1.5)
        'poor': 1.5, 'annoying': 1.5, 'frustrated': 1.5, 'angry': 1.5, 'upset': 1.5,
        'disappointed': 1.5, 'mistake': 1.5, 'defect': 1.5, 'inferior': 1.5,
        'mediocre': 1.5, 'subpar': 1.5, 'inadequate': 1.5, 'insufficient': 1.5,
        'unwanted': 1.5, 'offensive': 1.5, 'alarming': 1.5, 'unpleasant': 1.5,
        'unsuitable': 1.5, 'disastrous': 1.5, 'disheartening': 1.5, 'dismal': 1.5,
        'ghastly': 1.5, 'horrendous': 1.5, 'despicable': 1.5, 'dreadfully': 1.5,
        'contemptuous': 1.5, 'sordid': 1.5, 'deplorable': 1.5, 'damaging': 1.5,
        # Light negatives (weight 1.0)
        'bad': 1, 'weak': 1, 'issue': 1, 'problem': 1, 'bug': 1, 'sad': 1,
        'evil': 1, 'fraud': 1, 'unfit': 1, 'harm': 1, 'loss': 1,
        'dark': 0.8, 'dirty': 0.8, 'boring': 0.8
    }
    
    negations = ['not', 'no', 'never', 'neither', 'nobody', 'nothing', 'nowhere', 
                 'cannot', 'can\'t', 'don\'t', 'didn\'t', 'won\'t', 'wouldn\'t', 
                 'hardly', 'barely', 'isn\'t', 'aren\'t', 'wasn\'t', 'weren\'t']
    
    intensifiers = ['very', 'extremely', 'absolutely', 'definitely', 'surely', 
                    'certainly', 'really', 'quite', 'so', 'such', 'too', 'truly', 
                    'incredibly', 'truly', 'deeply', 'highly', 'utterly']
    
    weak_indicators = ['okay', 'alright', 'fair', 'decent', 'fine', 'average', 'standard']
    
    predictions = []
    
    for text in texts:
        text_lower = str(text).lower()
        words = text_lower.split()
        
        pos_score = 0.0
        neg_score = 0.0
        exclamation_count = text.count('!')
        weak_words_count = 0
        
        for i, word in enumerate(words):
            # Clean punctuation from word
            clean_word = word.strip('.,!?;:()').lower()
            
            # Check for negation (look back 2 words)
            has_negation = False
            for j in range(max(0, i - 2), i):
                if words[j].strip('.,!?;:()').lower() in negations:
                    has_negation = True
                    break
            
            # Check for intensifier (look back 1 word)
            has_intensifier = False
            for j in range(max(0, i - 1), i):
                if words[j].strip('.,!?;:()').lower() in intensifiers:
                    has_intensifier = True
                    break
            
            intensifier_boost = 1.6 if has_intensifier else 1.0
            
            if clean_word in positive_words:
                word_score = positive_words[clean_word]
                if has_negation:
                    neg_score += word_score * intensifier_boost
                else:
                    pos_score += word_score * intensifier_boost
            elif clean_word in negative_words:
                word_score = negative_words[clean_word]
                if has_negation:
                    pos_score += word_score * intensifier_boost
                else:
                    neg_score += word_score * intensifier_boost
            elif clean_word in weak_indicators:
                weak_words_count += 1
        
        # Punctuation indicators
        if exclamation_count >= 2:
            pos_score += 1.0
        
        # Classify with tuned threshold (0.7)
        threshold = 0.7
        diff = abs(pos_score - neg_score)
        
        if pos_score == 0 and neg_score == 0:
            predictions.append('Neutral')
        elif diff <= threshold:
            predictions.append('Neutral')
        elif pos_score > neg_score:
            predictions.append('Positive')
        else:
            predictions.append('Negative')
    
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
    print("\n[EVAL] Starting Sentiment Analysis Evaluation")
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
    print("\n[INFO] Generating predictions using keyword-based classifier...")
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
    
    # Try to find the CSV file
    csv_path = csv_file
    if not os.path.isabs(csv_path):
        # Try relative to Backend directory first
        backend_path = os.path.join(os.path.dirname(__file__), csv_path)
        if os.path.exists(backend_path):
            csv_path = backend_path
        # Then try relative to current working directory
        elif os.path.exists(csv_path):
            csv_path = csv_path
        # Then try parent directory
        else:
            root_path = os.path.join(os.path.dirname(__file__), '..', csv_path)
            if os.path.exists(root_path):
                csv_path = root_path
    
    if os.path.exists(csv_path):
        results = evaluate_from_csv(csv_path)
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
