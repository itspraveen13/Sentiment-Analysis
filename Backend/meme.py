import clip
from PIL import Image
from flask import *
app = Flask(__name__)

clipmodel, preprocess = clip.load("ViT-B/32", "cpu")


@app.route('/meme', methods=['POST'])
def analyze_meme():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:

        # Analyze the sentiment of the meme
        sentiment = analyze_meme_sentiment(file)

        return jsonify({'sentiment': sentiment})
    except Exception as e:
        print(e)
        return jsonify({'error': str(e)}), 500
if __name__ == '__main__':
    app.run(host='localhost', port=5000, debug=False)