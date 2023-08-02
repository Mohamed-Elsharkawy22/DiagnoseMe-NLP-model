import numpy as np
from flask import Flask, request, jsonify
import pickle
from nltk import PorterStemmer
from keras import models
from csv import reader


def csv_to_array(file_path):
    array_2d = []
    with open(file_path, 'r') as csv_file:
        csv_reader = reader(csv_file)
        for row in csv_reader:
            array_2d.append(row)
    return array_2d


app = Flask(__name__)

file_path = 'symptom_precaution.csv' 
result = csv_to_array(file_path)
result = result[1:]
DiseaseToPrecutions = {}

for lst in result:
  DiseaseToPrecutions[lst[0]] = ", ".join(lst[1:])



model = models.load_model('NLP_model.h5')

with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

with open('le.pkl', 'rb') as f:
    le = pickle.load(f)


ps = PorterStemmer()

def normalize(text):
    if text == text:
        text = text.lower()
        text = text.replace('_', ' ').replace(",", " ")
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
    DiseaseIdx = prediction.argmax()   
    prediction = le.inverse_transform([DiseaseIdx])
    precutions = DiseaseToPrecutions[prediction[0]]
    
    data = {
    'Disease': prediction[0],
    'precautions': precutions
    }

    try:
        result = jsonify(data)
    except TypeError as e:
        result = jsonify({'Error', str(e)})   

    return result


if __name__ == '__main__':
    app.run(port=8000, debug=True)
