from flask import Flask, request, jsonify
from googleapiclient.discovery import build
from datetime import datetime
from flask_cors import CORS
import praw
global sp_src_path,sp_trg_path,english,device
from flask_cors import CORS
from transformers import BertTokenizer
from transformers import TFBertForSequenceClassification
import tensorflow as tf
import numpy as np
from langdetect import detect
import json
import collections
import random
from googleapiclient.errors import HttpError
from googletrans import Translator

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Load BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
model = TFBertForSequenceClassification.from_pretrained('tf_model')
labels = ['Negative', 'Positive']  # (0:negative, 1:positive)
api_key = 'AIzaSyDvOvhzBGEHLnDpuOBpJu0L1ALVUATl-HI'

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
with open('insight_dataset.json', 'r') as recommendations_file:
    recommendations_data = json.load(recommendations_file)

translator = Translator()

def convert_to_english(text):
    try:
        lang_info = translator.detect(text)
        if lang_info.lang != 'en':
            print("diff")
            print("lang = ", lang_info.lang) 
            text = translator.translate(text, dest='en').text
        return text
    except Exception as e:
        print(f"Error translating to English: {e}")
        return text

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

with open('topic_wise_bow.json', 'r') as json_file:
    titles_data = json.load(json_file)

title_keywords = {}
for title, keywords in titles_data.items():
    title_keywords[title] = keywords

# Initialize PRAW with your Reddit API credentials
reddit = praw.Reddit(
    client_id='LchMksVUmRUeyg',
    client_secret='gb1XyXX-r0ycV9KKFM-ujFVNOogO_w',
    user_agent='Data Scraping'
)

def search_and_fetch_comments(keyword, count):
    # Search for posts containing the keyword
    results = reddit.subreddit('all').search(keyword, sort='relevance', time_filter='all', limit=count)
    
    # List to store comments
    all_comments = []

    for submission in results:
        submission.comments.replace_more(limit=None)
        for comment in submission.comments.list():
            if isinstance(comment, praw.models.Comment):
                all_comments.append(comment.body)
    
    return all_comments

def video_comments(video_id, count):
    youtube = build('youtube', 'v3', developerKey=api_key)
    all_comments = []
    comment_counter = 0

    try:
        # Retrieve YouTube video comments
        video_response = youtube.commentThreads().list(
            part='snippet,replies',
            videoId=video_id
        ).execute()

        # Iterate over video response
        while video_response and comment_counter < count:
            for item in video_response['items']:
                comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
                all_comments.append(comment)
                comment_counter += 1

                if comment_counter >= count:
                    break  # Stop fetching comments if count limit is reached

                replycount = item['snippet']['totalReplyCount']
                if replycount > 0:
                    replies = [reply['snippet']['textDisplay'] for reply in item['replies']['comments']]
                    all_comments.extend(replies)
                    comment_counter += len(replies)

            if comment_counter < count and 'nextPageToken' in video_response:
                video_response = youtube.commentThreads().list(
                    part='snippet,replies',
                    videoId=video_id,
                    pageToken=video_response['nextPageToken']
                ).execute()
            else:
                break

    except HttpError as e:
        error_details = e.error_details
        if 'commentsDisabled' in str(error_details):
            return []
        else:
            print(f"HttpError: {error_details}")
            raise e

    return all_comments

def get_comments():    # Set the API key
    # Get the keyword, start date, and end date from the request
    keyword = request.json.get('keyword')
    start_date_str = request.json.get('start_date')
    end_date_str = request.json.get('end_date')
    count = request.json.get('count')
    max_count = request.json.get('count')
    # Convert start date and end date strings to datetime objects
    start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
    end_date = datetime.strptime(end_date_str, '%Y-%m-%d')

    # Set the search parameters
    max_results = count
    published_after = start_date.strftime('%Y-%m-%dT%H:%M:%SZ')
    published_before = end_date.strftime('%Y-%m-%dT%H:%M:%SZ')

    # Create a YouTube Data API client
    youtube = build('youtube', 'v3', developerKey=api_key)

    # Call the search.list method to retrieve video results
    search_response = youtube.search().list(
        q=keyword,
        type='video',
        part='id',
        maxResults=max_results,
        publishedAfter=published_after,
        publishedBefore=published_before
    ).execute()

    # Extract the video IDs from the search results
    video_ids = [search_result['id']['videoId'] for search_result in search_response.get('items', [])]

    # Ensure video_ids list is not empty
    video_comments_list = []
    counter = 0

    if video_ids:
        # Choose the first video ID from the list
        for video_id in video_ids:
            # Call the function to retrieve video comments
            if counter <= count:
                counter += 1
                video_comments_list.extend(video_comments(video_id, max_count))
            else:
                break

    reddit_comments_list = search_and_fetch_comments(keyword, max_count)
    # Perform sentiment analysis on YouTube comments
    total_positive = 0
    total_negative = 0
    unique_positive_reasons = collections.Counter()
    unique_negative_reasons = collections.Counter()
    unique_positive_titles = collections.Counter()
    unique_negative_titles = collections.Counter()
    youtube_texts = video_comments_list
    reddit_texts = reddit_comments_list
    combined_text =  reddit_texts[0:(int(max_count/2))] + youtube_texts[0:(int(max_count/2))] 
    predictions = []
    youtube_result = []
    reddit_result = []
    youtube_total_negative = 0
    youtube_total_positive = 0
    reddit_total_negative = 0
    reddit_total_positive = 0
    counts = 0
    for text in combined_text:
        if counts >= max_count:
            break
        else:
            counts += 1
            text = convert_to_english(text)
            predict_input = tokenizer.encode(text, truncation=True, padding=True, return_tensors="tf")
            tf_output = model.predict(predict_input)[0]
            tf_prediction = tf.nn.softmax(tf_output, axis=1)
            probabilities = tf_prediction.numpy()[0]
            label = np.argmax(probabilities)
            confidence = round(float(probabilities[label]) * 100, 2)
            
            result_dict = {
                'text': text,
                'sentiment': labels[label],
                'confidence': confidence,
            }

            words = text.split()
            
            if labels[label] == 'Negative':
                reasons = [word for word in negative_sentiment_words if word in text.lower()]
                unique_negative_reasons.update(reasons)
                
                mapped_titles = set()
                for keyword in words:
                    for title, keywords in title_keywords.items():
                        for title_keyword in keywords:
                            if keyword.lower() == title_keyword.lower():
                                mapped_titles.add(title)
                unique_negative_titles.update(mapped_titles)
                final_mapped_titles = list(mapped_titles) if mapped_titles else []
                result_dict['titles'] = final_mapped_titles
                result_dict['reasons'] = list(reasons)
                predictions.append(result_dict)
                total_negative += 1

                if text in youtube_texts:
                    youtube_result.append(result_dict)
                    youtube_total_negative += 1
                else:
                    reddit_result.append(result_dict)
                    reddit_total_negative += 1

            elif labels[label] == 'Positive':
                reasons = [word for word in positive_sentiment_words if word in text.lower()]
                unique_positive_reasons.update(reasons)

                mapped_titles = set()
                for keyword in words:
                    for title, keywords in title_keywords.items():
                        for title_keyword in keywords:
                            if keyword.lower() == title_keyword.lower():
                                mapped_titles.add(title)
                unique_positive_titles.update(mapped_titles)
                final_mapped_titles = list(mapped_titles) if mapped_titles else []
                result_dict['titles'] = final_mapped_titles
                result_dict['reasons'] = list(reasons)
                predictions.append(result_dict)
                total_positive += 1

                if text in youtube_texts:
                    youtube_result.append(result_dict)
                    youtube_total_positive += 1
                else:
                    reddit_result.append(result_dict)
                    reddit_total_positive += 1

            else:
                print("i wonder how!")

    net_sentiment_score = (abs(total_positive - total_negative) / (total_positive + total_negative)) * 100

        # Find the prevailing sentiment
    prevailing_sentiment = "positive" if net_sentiment_score > 0 else "negative" if net_sentiment_score < 0 else "neutral"

        # Convert sets to lists to maintain unique values
    unique_positive_reasons = [reason for reason, count in unique_positive_reasons.most_common(5)]
    unique_negative_reasons = [reason for reason, count in unique_negative_reasons.most_common(5)]
    unique_positive_titles = [title for title, count in unique_positive_titles.most_common(3)]
    unique_negative_titles = [title for title, count in unique_negative_titles.most_common(3)]

        # Construct the insight text
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
            'youtube' : youtube_result,
            'reddit' : reddit_result,
            'reddit_positive' : reddit_total_positive,
            'reddit_negative' : reddit_total_negative,
            'youtube_positive' : youtube_total_positive,
            'youtube_negative' : youtube_total_negative,
            'predictions': predictions,
            'total_positive': total_positive,
            'total_negative': total_negative,
            'insight': {
                'insight_text': insight_text,
                'net_sentiment_score': net_sentiment_score,
                'prevailing_sentiment': prevailing_sentiment,
                'negative_sentiment_count': total_negative,
                'positive_sentiment_count': total_positive,
                'top_reasons_for_negative': unique_negative_reasons[0:5],
                'top_titles_for_negative': unique_negative_titles[0:5],
                'top_reasons_for_positive': unique_positive_reasons[0:5],
                'top_titles_for_positive': unique_positive_titles[0:5],
                'recommendations': recommendation_texts
            }
    })
