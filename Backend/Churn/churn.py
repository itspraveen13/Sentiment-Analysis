from flask import request, jsonify
import pandas as pd
from collections import Counter
from io import BytesIO
import json
import random
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
with open(BASE_DIR / 'churn_recommendation.json', 'r', encoding='utf-8') as recommendations_file:
    recommendations_data = json.load(recommendations_file)

REQUIRED_COLUMNS = {
    'sentiment': 'Sentiment',
    'churnreason': 'Churn Reason',
    'totalcharges': 'Total Charges',
    'contract': 'Contract',
}

DEFAULT_CONTRACT_TYPES = ['Month-to-month', 'One year', 'Two year']

def _normalize_column(name):
    return ''.join(ch for ch in str(name).lower() if ch.isalnum())

def _read_upload(file, file_name):
    ext = Path(file_name or '').suffix.lower()
    content = BytesIO(file.read())
    if ext in ['.xlsx', '.xls']:
        return pd.read_excel(content)
    content.seek(0)
    try:
        return pd.read_csv(content, encoding='utf-8')
    except UnicodeDecodeError:
        content.seek(0)
        try:
            return pd.read_csv(content, encoding='utf-8-sig')
        except UnicodeDecodeError:
            content.seek(0)
            return pd.read_csv(content, encoding='latin-1')

def process_file():
    try:
        file = request.files.get('file')
        if file is None or file.filename == '':
            return jsonify({'error': 'No file provided'}), 400
        try:
            data = _read_upload(file, file.filename)
        except Exception as e:
            return jsonify({'error': f'Could not read file. Please upload a valid CSV or XLSX. Details: {e}'}), 400

        if data.empty:
            return jsonify({'error': 'Uploaded file is empty'}), 400

        normalized_map = {_normalize_column(col): col for col in data.columns}
        missing = [label for key, label in REQUIRED_COLUMNS.items() if key not in normalized_map]
        if missing:
            return jsonify({'error': f'Missing required columns: {", ".join(missing)}'}), 400

        sentiment_col = normalized_map['sentiment']
        churn_reason_col = normalized_map['churnreason']
        total_charges_col = normalized_map['totalcharges']
        contract_col = normalized_map['contract']

        neglist = []
        poslist = []
        total_charges_yes = 0.0
        total_charges_no = 0.0

        contract_counts_pos = {}  # Dictionary for churned customers
        contract_counts_neg = {}  # Dictionary for unchurned customers
        pos_count = 0.0
        neg_count = 0.0

        data[total_charges_col] = pd.to_numeric(data[total_charges_col], errors='coerce').fillna(0.0)

        for _, row in data.iterrows():
            churn_reason = row.get(churn_reason_col)
            contract_type = row.get(contract_col)
            total_charges = float(row.get(total_charges_col, 0.0) or 0.0)

            sentiment_value = str(row.get(sentiment_col, '')).strip().lower()
            if sentiment_value == 'negative':
                if pd.notna(churn_reason):
                    neglist.append(str(churn_reason))
                total_charges_yes += total_charges
                pos_count += 1
                if contract_type in contract_counts_neg:
                    contract_counts_neg[contract_type] += 1
                else:
                    contract_counts_neg[contract_type] = 1
            elif sentiment_value == 'positive':
                if pd.notna(churn_reason):
                    poslist.append(str(churn_reason))
                total_charges_no += total_charges
                neg_count += 1
                if contract_type in contract_counts_pos:
                    contract_counts_pos[contract_type] += 1
                else:
                    contract_counts_pos[contract_type] = 1


        neg_word_freq = Counter(neglist)
        unique_dict = dict(neg_word_freq)
        total_count = pos_count + neg_count
        num_negative = len(neglist)
        num_positive = len(poslist)
        top_five = neg_word_freq.most_common(8)
        result = {}
        if num_negative > 0:
            for word, freq in top_five:
                percent_freq = (freq / num_negative) * 100
                result[word] = percent_freq
        recommendation_texts = []
        top_four = [item[0] for item in top_five[:5]]
        for title in top_four:
            title_lower = title.lower()  # Convert title to lowercase
            for key in recommendations_data:
                if key.lower() == title_lower:
                    selected_recommendation = random.choice(recommendations_data[key])
                    recommendation_texts.append(selected_recommendation)


        for contract_type in DEFAULT_CONTRACT_TYPES:
            contract_counts_pos.setdefault(contract_type, 0)
            contract_counts_neg.setdefault(contract_type, 0)
        contract_counts_pos = dict(list(contract_counts_pos.items())[:3])
        contract_counts_neg = dict(list(contract_counts_neg.items())[:3])
        result_dict = {
            'word_frequency': top_five,
            'num_negative': num_negative,
            'num_positive': num_positive,
            'total': total_count,
            'reasons': unique_dict,
            'charges_churn': total_charges_yes,
            'charges_not_churn': total_charges_no,
            'total_charges': total_charges_no + total_charges_yes,
            'contract_counts_pos': contract_counts_pos,
            'contract_counts_neg': contract_counts_neg,
            'recommendation': recommendation_texts,
            'churn_rate': (num_negative / total_count) * 100 if total_count else 0
        }
        return jsonify(result_dict)
    except Exception as e:
        print("Error: ", e)
        return jsonify({'error': str(e)}), 500
