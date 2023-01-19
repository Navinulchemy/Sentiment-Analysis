#required libraries
from flask import Flask, render_template, request
import pandas as pd
import pickle
import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model

#vocabulary,input size
vocab_size=10000
sent_len=250

texts=pickle.load(open("texts.pkl","rb"))
# Pre-processing the  new text data
tokenizer = Tokenizer(num_words=vocab_size)
tokenizer.fit_on_texts(texts)
lemmatizer = WordNetLemmatizer()
app = Flask(__name__)

# Loading the pre-trained LSTM model
model=load_model("sentiment_model.h5")

#web app code
@app.route('/', methods=['GET', 'POST'])
def sentiment_analysis():
    if request.method == 'POST':
        text = request.form['text']

        text = re.sub(r'[^a-zA-Z\s]', '', text)

        # Removing stopwords
        lower=[word.lower().strip() for word in text.split()]
        stop_words = set(stopwords.words("english"))
        words = [word for word in lower if word not in stop_words]
   
        #Removing short words
        words = " ".join([lemmatizer.lemmatize(word) for word in words if len(word) > 2])
        words=pd.Series(words).values
        sequence1=tokenizer.texts_to_sequences((words))
        seq = pad_sequences(sequence1,padding="post",maxlen=sent_len)
        
         # Using the pre-trained LSTM model to classify sentiment
        p=model.predict(seq)
        
        if p[0] <= 0.5:
            result="positive"
        else:
            result="negative"
       
        return render_template('result.html', sentiment=result)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
