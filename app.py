from flask import Flask, render_template, request
import re
from nltk.sentiment.vader import SentimentIntensityAnalyzer

app = Flask(__name__)
sia = SentimentIntensityAnalyzer()

def remove_special_characters_from_file(file_content):
    pattern = r'[^a-zA-Z\s]'
    cleaned_content = re.sub(pattern, '', file_content)
    return cleaned_content

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/sentiment', methods=['GET', 'POST'])
def sentiment():
    if request.method == 'POST':
        original_text = None

        # Check if text input is provided
        if 'text' in request.form and request.form['text'].strip() != '':
            original_text = request.form['text']
            cleaned_content = remove_special_characters_from_file(original_text)
            sentiment = sia.polarity_scores(cleaned_content)
            return render_template('sentiment.html', sentiment=sentiment, input_type="text", original_text=original_text)
    return render_template('sentiment.html')

if __name__ == '__main__':
    app.run(debug=True)
