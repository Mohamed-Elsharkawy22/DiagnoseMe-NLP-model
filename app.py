import numpy as np
from flask import Flask, request, jsonify
import pickle
from nltk import PorterStemmer
from keras import models


app = Flask(__name__)


model = models.load_model('NLP_model.h5')

with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

with open('le.pkl', 'rb') as f:
    le = pickle.load(f)


ps = PorterStemmer()

def normalize(text):
    if text == text:
        text = text.lower()
        text = text.replace('_', ' ')
        text = ' '.join([ps.stem(word) for word in text.split()])
    else:
        text = ''
    return text



@app.route('/predict',methods=['POST'])
def predict():

    data = request.json

    try:
        text = data['text']
    except KeyError:
        return jsonify({'Error': "No text sent!"})
    
    text = normalize(text)
    text = vectorizer.transform([text])
    prediction = model.predict(text.toarray())
    lst =  [(prob, idx)  for (idx, prob) in enumerate(prediction[0])]
    import heapq as hq
    hq.heapify(lst)
    top_three = hq.nlargest(3, lst)
    idxes = []
    idxes = [idx for (val, idx) in top_three ]    
    prediction = le.inverse_transform(idxes)

    try:
        result = jsonify(", ".join(prediction))
    except TypeError as e:
        result = jsonify({'Error', str(e)})   

    return result


if __name__ == '__main__':
    app.run(debug=True)
