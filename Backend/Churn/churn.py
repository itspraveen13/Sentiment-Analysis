from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import spacy
from collections import Counter
from io import BytesIO
from collections import Counter
import math
import json
import random

with open('churn_recommendation.json', 'r') as recommendations_file:
    recommendations_data = json.load(recommendations_file)

def process_file():
    try:
        file = request.files['file']
        if file is None:
            return jsonify({'error': 'No file provided'}), 400

        file_content = file.read()
        data = pd.read_csv(BytesIO(file_content))

        neglist = []
        poslist = []
        total_charges_yes = 0
        total_charges_no = 0

        contract_counts_pos = {}  # Dictionary for churned customers
        contract_counts_neg = {}  # Dictionary for unchurned customers
        pos_count = 0.0
        neg_count = 0.0

        for index, row in data.iterrows():
            churn_reason = row['Churn Reason']
            total_charges = row['Total Charges']
            contract_type = row['Contract']

            if not math.isnan(total_charges):
                total_charges = float(total_charges)
            else:
                total_charges = 0

            if row['Sentiment'] == 'Negative':
                neglist.append(churn_reason)
                total_charges_yes += total_charges
                pos_count += 1
                if contract_type in contract_counts_neg:
                    contract_counts_neg[contract_type] += 1
                else:
                    contract_counts_neg[contract_type] = 1
            elif row['Sentiment'] == 'Positive':
                poslist.append(churn_reason)
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
        neg_word_freq = Counter(neglist)
        top_five = neg_word_freq.most_common(8)
        result = {}
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
            'churn_rate': (num_negative / total_count) * 100
        }
        return jsonify(result_dict)
    except Exception as e:
        print("Error: ", e)
        return jsonify({'error': str(e)}), 500