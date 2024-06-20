from flask import Flask, render_template, request
import pickle
from text_preprocessing import preprocess_text
from text_preprocessing import to_lower, remove_email, remove_url, remove_punctuation, lemmatize_word
app = Flask(__name__)
with open('Sentiment_Analysis.pkl', 'rb') as f:
    model = pickle.load(f)
    
@app.route('/', methods=['GET', 'POST'])
def index():
    sentiment = None
    if request.method == 'POST':
            # Get the tweet text from the form
            tweet_text = request.form['tweet_text']
            print(tweet_text)
            # Preprocess the tweet text
            processed_text = preprocess_text(tweet_text)
            # Make predictions using the loaded model
            prediction = model.predict([processed_text])[0]
            # Determine the sentiment based on the prediction
            def sentiment(x:float):
                if x < -0.05 : return 'negative'
                if x > 0.35 : return 'positive'
                return 'neutral'
            
            print(sentiment)
    return render_template('index.html', sentiment=sentiment)   



if __name__ == '__main__':
    app.run(debug=True)        