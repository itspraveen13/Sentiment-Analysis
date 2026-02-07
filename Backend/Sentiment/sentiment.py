from flask import *
import pandas as pd
from transformers import BertTokenizer
from transformers import TFBertForSequenceClassification
import tensorflow as tf
import json
import speech_recognition as sr
import numpy as np
import tempfile
import os
import collections
import random
from pydub import AudioSegment
import io

negative_sentiment_words = []
with open('neg_words.txt', 'r') as file:
        for line in file:
            line = line.strip().lower()
            negative_sentiment_words.append(line)
positive_sentiment_words = []
with open('pos_words.txt', 'r') as file:
        for line in file:
            line = line.strip().lower()
            positive_sentiment_words.append(line)
reason_words = []
with open('reason_words.txt', 'r') as file:
        for line in file:
            line = line.strip().lower()
            reason_words.append(line)

with open('country.json', 'r') as json_file:
    data = json.load(json_file)

with open('topic_wise_bow.json', 'r') as json_file:
    titles_data = json.load(json_file)

with open('insight_dataset.json', 'r') as recommendations_file:
    recommendations_data = json.load(recommendations_file)


from langdetect import detect, LangDetectException
from translate import Translator as SimpleTranslator
translator = SimpleTranslator(to_lang="en")
def convert_to_english(text):
    try:
        lang_info = detect(text)
        if lang_info != 'en':
            text = translator.translate(text)
        return text
    except (LangDetectException, Exception) as e:
        print(f"Error translating to English: {e}")
        return text


# Create a dictionary to store sentiment-related keywords for different titles
title_keywords = {}
for title, keywords in titles_data.items():
    title_keywords[title] = keywords

def convert_audio_to_wav(audio_data):
    try:
        audio = AudioSegment.from_file(io.BytesIO(audio_data))
        # Convert the audio to WAV format
        audio = audio.set_channels(1)  # Set to mono channel if needed
        audio = audio.set_frame_rate(16000)  # Adjust the sample rate if needed
        output_buffer = io.BytesIO()
        audio.export(output_buffer, format="wav")
        return output_buffer.getvalue()
    except Exception as e:
        print(f"Error converting audio: {e}")
        return None

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
model = TFBertForSequenceClassification.from_pretrained('tf_model')
labels = ['Negative', 'Positive']  # (0:negative, 1:positive)
 
def get_coordinates(country_name):
    for country_info in data:
        if country_info['name'] == country_name:
            return country_info['latlng']  
    return [21.7679,78.8718]

def predict_text():
    if request.method == 'POST':
        data = request.get_json()
        if data is None:
            return jsonify({'error': 'No data provided.'}), 400

        text = data.get('text')
        if text is None:
            return jsonify({'error': 'No text provided.'}), 400

        # Detect the language of the input text
        text = convert_to_english(text)
        predict_input = tokenizer.encode(text, truncation=True, padding=True, return_tensors="tf")
        tf_output = model.predict(predict_input)[0]
        tf_prediction = tf.nn.softmax(tf_output, axis=1)
        label = tf.argmax(tf_prediction, axis=1)
        label = label.numpy()
        confidence = tf.reduce_max(tf_prediction, axis=1).numpy()[0]
        confidence = round(float(confidence) * 100, 2)
        unique_positive_titles = collections.Counter()
        unique_negative_titles = collections.Counter()
        words = text.split()
        words_corresponding_to_title = []
        if labels[label[0]] == 'Negative':
            reasons = []
            for word in negative_sentiment_words:
                if word in text.lower():
                    reasons.append(word)
                mapped_titles = set()
                keywords_of_title = set()
                for keyword in words:
                    for title, keywords in title_keywords.items():
                        for title_keyword in keywords:
                            if keyword.lower() == title_keyword.lower():
                                mapped_titles.add(title)  # Use a set to store unique titles
                                keywords_of_title.add(keyword)
                unique_negative_titles.update(mapped_titles)
                words_corresponding_to_title = list(keywords_of_title)
                final_mapped_titles = list(mapped_titles) if mapped_titles else []
            return jsonify({'predictions': [{'text': text, 'sentiment': labels[label[0]], 'reasons': reasons, 'titles' : final_mapped_titles, 'confidence': confidence, 'title_words': words_corresponding_to_title}]})
        elif labels[label[0]] == 'Positive':
            reasons = []
            for word in positive_sentiment_words:
                if word in text.lower():
                    reasons.append(word)
                mapped_titles = set()
                keywords_of_title = set()
                for keyword in words:
                    for title, keywords in title_keywords.items():
                        for title_keyword in keywords:
                            if keyword.lower() == title_keyword.lower():
                                mapped_titles.add(title)
                                keywords_of_title.add(keyword)
                unique_positive_titles.update(mapped_titles)
                words_corresponding_to_title = list(keywords_of_title)
                final_mapped_titles = list(mapped_titles) if mapped_titles else []
            return jsonify({'predictions': [{'text': text, 'sentiment': labels[label[0]], 'reasons': reasons, 'titles' : final_mapped_titles, 'confidence': confidence, 'title_words': words_corresponding_to_title}]})
        else:
            return jsonify({'predictions': [{'text': text, 'sentiment': labels[label[0]], 'confidence': confidence}]})
        
def predict_file():
    if request.method == 'POST':
        file = request.files['file']
        if not file:
            return jsonify({'error': 'No file provided.'}), 400

        file_extension = os.path.splitext(file.filename)[1].lower()

        if file_extension == '.csv':
            df = pd.read_csv(file)
        elif file_extension == '.xlsx':
            df = pd.read_excel(file, engine='openpyxl')
        elif file_extension == '.txt':
            # Create a temporary CSV file from the text data
            with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as temp_csv:
                temp_csv.write(file.read())
                temp_csv.seek(0)
                df = pd.read_csv(temp_csv.name, encoding='utf-8')
        else:
            return jsonify({'error': 'Unsupported file format.'}), 400
        
        columns_to_process = []
        df.columns = df.columns.str.lower()

        for column in df.columns:
            if column in ['text', 'age','gender', 'country']:
                columns_to_process.append(column)

        predictions = []

        coordinates = {}

        # Iterate through the rows in the DataFrame
        if 'country' in columns_to_process:
            for index, row in df.iterrows():
                country = row['country']
                if country not in coordinates:
                    # Fetch coordinates for the country (replace this with your actual function)
                    country_coordinates = get_coordinates(country)
                    if country_coordinates is not None:
                        coordinates[country] = country_coordinates

        positive_counts = {}  # Dictionary to store positive sentiment counts per country
        negative_counts = {}  # Dictionary to store negative sentiment counts per country
        total_positive = 0
        total_negative = 0
        unique_positive_reasons = collections.Counter()
        unique_negative_reasons = collections.Counter()
        unique_positive_titles = collections.Counter()
        unique_negative_titles = collections.Counter()

        for index, row in df.iterrows():
            if 'age' in columns_to_process:
                age = row['age']
            else:
                age = 0
            if 'text' in columns_to_process:
                text = row['text']
            else:
                text = 'None'
            if 'country' in columns_to_process:
                country = row['country']
            else:
                country = 'None'
                
            if 'gender' in columns_to_process:
                gender = row['gender']
            else:
                gender = 'none'
            words_corresponding_to_title = []
            text = convert_to_english(text)
            predict_input = tokenizer.encode(text, truncation=True, padding=True, return_tensors="tf")
            tf_output = model.predict(predict_input)[0]
            tf_prediction = tf.nn.softmax(tf_output, axis=1)
            label = tf.argmax(tf_prediction, axis=1)
            label = label.numpy()
            confidence = tf.reduce_max(tf_prediction, axis=1).numpy()[0]
            confidence = round(float(confidence) * 100, 2)
            result_dict = {
                'text': text,
                'sentiment': labels[label[0]],
                'confidence': confidence,
                'country': country,
                'coordinates': coordinates.get(country),
                'gender': gender,
                'age': age
            }
            words = text.split()
            if labels[label[0]] == 'Negative':
                reasons = []
                for word in negative_sentiment_words:
                    if word in text.lower():
                        reasons.append(word)
                unique_negative_reasons.update(reasons)
                mapped_titles = set()
                keywords_of_title = set()
                for keyword in words:
                    for title, keywords in title_keywords.items():
                        for title_keyword in keywords:
                            if keyword.lower() == title_keyword.lower():
                                mapped_titles.add(title)  # Use a set to store unique titles
                                keywords_of_title.add(keyword)
                words_corresponding_to_title = list(keywords_of_title)                
                unique_negative_titles.update(mapped_titles)
                final_mapped_titles = list(mapped_titles) if mapped_titles else []
                result_dict['titles'] = final_mapped_titles
                result_dict['reasons'] = list(reasons)
                result_dict["words_corresponding_to_title"] = words_corresponding_to_title
                predictions.append(result_dict)
                current_negative_count = negative_counts.get(country, 0) + 1
                negative_counts[country] = current_negative_count
                total_negative += 1
            elif labels[label[0]] == 'Positive':
                reasons = []
                for word in positive_sentiment_words:
                    if word in text.lower():
                        reasons.append(word)
                unique_positive_reasons.update(reasons)
                mapped_titles = set()
                keywords_of_title = set()
                for keyword in words:
                    for title, keywords in title_keywords.items():
                        for title_keyword in keywords:
                            if keyword.lower() == title_keyword.lower():
                                mapped_titles.add(title)  # Use a set to store unique titles
                                keywords_of_title.add(keyword)
                words_corresponding_to_title = list(keywords_of_title)
                unique_positive_titles.update(mapped_titles)
                final_mapped_titles = list(mapped_titles) if mapped_titles else []
                result_dict['titles'] = final_mapped_titles
                result_dict['reasons'] = list(reasons)
                result_dict["words_corresponding_to_title"] = words_corresponding_to_title
                predictions.append(result_dict)
                current_positive_count = positive_counts.get(country, 0) + 1
                positive_counts[country] = current_positive_count
                total_positive += 1
            else:
                print("Else ", country, " ", coordinates.get(country))
                predictions.append(result_dict)

        net_sentiment_score = (abs(total_positive - total_negative) / (total_positive + total_negative)) * 100
        prevailing_sentiment = "positive" if net_sentiment_score > 0 else "negative" if net_sentiment_score < 0 else "neutral"
        unique_positive_reasons = [reason for reason, count in unique_positive_reasons.most_common(5)]
        unique_negative_reasons = [reason for reason, count in unique_negative_reasons.most_common(5)]
        unique_positive_titles = [title for title, count in unique_positive_titles.most_common(3)]
        unique_negative_titles = [title for title, count in unique_negative_titles.most_common(3)]
        insight_text = f"The Net Sentiment score indicating a {abs(net_sentiment_score)}% difference between " \
                      f"Positive and Negative, highlights a prevailing {prevailing_sentiment} sentiment. " \
                      f"The overall sentiment leans towards the {prevailing_sentiment} side, influenced by " \
                      f"the factors such as {', '.join(unique_positive_titles)} (positive) and " \
                      f"{', '.join(unique_negative_titles)} (negative)."
        recommendation_texts = []
        for title in unique_negative_titles:
            if title in recommendations_data:
                selected_recommendation = random.choice(recommendations_data[title])
                recommendation_texts.append(selected_recommendation)

        return jsonify({
            'predictions': predictions,
            'positive_counts': positive_counts,
            'negative_counts': negative_counts,
            'total_positive': total_positive,
            'total_negative': total_negative,
            'insight': {
                'insight_text': insight_text,
                'net_sentiment_score': net_sentiment_score,
                'prevailing_sentiment': prevailing_sentiment,
                'negative_sentiment_count': total_negative,
                'positive_sentiment_count': total_positive,
                'top_reasons_for_negative': unique_negative_reasons[0:5],
                'top_titles_for_negative': unique_negative_titles,
                'top_reasons_for_positive': unique_positive_reasons[0:5],
                'top_titles_for_positive': unique_positive_titles,
                'recommendations' : recommendation_texts
            }
        })
    else:
        return "error"

def predict_audio():
    file = request.files['file']
    recognizer = sr.Recognizer()
    with sr.AudioFile(file) as source:
        try:
            audio_data = recognizer.record(source)
            text = recognizer.recognize_google(audio_data)
            print("text: ", text)
            detected_language = detect(text)
        except Exception as e:
            # Handle translation error
            print(f"Translation error: {e}")
    if detected_language != "en":
        try:
            text = translator.translate(text)
        except Exception as e:
            # Handle translation error
            print(f"Translation error: {e}")
    predict_input = tokenizer.encode(text, truncation=True, padding=True, return_tensors="tf")
    tf_output = model.predict(predict_input)[0]
    tf_prediction = tf.nn.softmax(tf_output, axis=1)
    label = tf.argmax(tf_prediction, axis=1)
    label = label.numpy()
    confidence = tf.reduce_max(tf_prediction, axis=1).numpy()[0]
    confidence = round(float(confidence) * 100, 2)
    unique_positive_titles = collections.Counter()
    unique_negative_titles = collections.Counter()
    words = text.split()
    words_corresponding_to_title = []
    if labels[label[0]] == 'Negative':
        reasons = []
        for word in negative_sentiment_words:
            if word in text.lower():
                reasons.append(word)
            mapped_titles = set()
            keywords_of_title = set()
            for keyword in words:
                for title, keywords in title_keywords.items():
                    for title_keyword in keywords:
                        if keyword.lower() == title_keyword.lower():
                            mapped_titles.add(title)  # Use a set to store unique titles
                            keywords_of_title.add(keyword)
            unique_negative_titles.update(mapped_titles)
            words_corresponding_to_title = list(keywords_of_title)
            final_mapped_titles = list(mapped_titles) if mapped_titles else []
        return jsonify({'predictions': [{'text': text, 'sentiment': labels[label[0]], 'reasons': reasons, 'titles' : final_mapped_titles, 'confidence': confidence, 'title_words': words_corresponding_to_title}]})
    elif labels[label[0]] == 'Positive':
        reasons = []
        for word in positive_sentiment_words:
            if word in text.lower():
                reasons.append(word)
            mapped_titles = set()
            keywords_of_title = set()
            for keyword in words:
                for title, keywords in title_keywords.items():
                    for title_keyword in keywords:
                        if keyword.lower() == title_keyword.lower():
                            mapped_titles.add(title)
                            keywords_of_title.add(keyword)
            unique_positive_titles.update(mapped_titles)
            words_corresponding_to_title = list(keywords_of_title)
            final_mapped_titles = list(mapped_titles) if mapped_titles else []
        return jsonify({'predictions': [{'text': text, 'sentiment': labels[label[0]], 'reasons': reasons, 'titles' : final_mapped_titles, 'confidence': confidence, 'title_words': words_corresponding_to_title}]})
    else:
        return jsonify({'predictions': [{'text': text, 'sentiment': labels[label[0]], 'confidence': confidence}]})
    
