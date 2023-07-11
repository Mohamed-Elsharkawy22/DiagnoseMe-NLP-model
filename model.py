import pandas as pd
import numpy as np
import xgboost as xgb
import nltk
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

df = pd.read_csv('dataset.csv')

ps = PorterStemmer()

def normalize(text):
    if text == text:
      text = text.lower()
      text = text.replace('_', ' ')
      text = ' '.join([ps.stem(word) for word in text.split()])
    else:
      text = ''
    return text



for i in range(1, len(df.columns)):
	df[df.columns[i]] = df[df.columns[i]].apply(lambda x: normalize(x))




cat_symptom = []
for symptom in df[df.columns[1:]].values:
    cat_symptom.append(' '.join(symptom).strip())
df['cat_symptom'] = cat_symptom






vectorizer = TfidfVectorizer()
le = LabelEncoder()
X = df['cat_symptom'].values
y = le.fit_transform(df[df.columns[0]].values)
y = to_categorical(y)



X_train, X_test, y_train, y_test = train_test_split(X, y,
													random_state=88,
													stratify=y)



X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)








from keras import layers, models

model = models.Sequential()
model.add(layers.Dense(64, activation = 'relu', input_shape= (198,)))
model.add(layers.Dense(64, activation = 'relu'))
model.add(layers.Dense(41, activation = 'softmax'))



from keras import optimizers, losses, metrics
model.compile(optimizer=optimizers.RMSprop(learning_rate=.001),
              loss = losses.categorical_crossentropy,
              metrics=[metrics.categorical_accuracy])


x_val = X_train[:1000]
partial_x_train = X_train[1000:]

y_val = y_train[:1000]
partial_y_train = y_train[1000:]



history = model.fit(partial_x_train.toarray(),
                    partial_y_train,
                    epochs= 15,
                    batch_size =128,
                    validation_data=(x_val.toarray(), y_val))





history_dict = history.history
print(history_dict.keys())


model.save('NLP_model.h5')

model = models.load_model('NLP_model.h5')

# Make predictions
y_pred = model.predict(X_test.toarray())

for lst in y_pred:
    mx = lst[0]
    idxmx = 0
    for i, val in enumerate(lst):
        if val > mx:
            mx = val
            idxmx = i
    for i, val in enumerate(lst):
        if i == idxmx:
            lst[i] = 1.
        else:
            lst[i] = 0.
            
      
pickle.dump(vectorizer, open('vectorizer.pkl', 'wb'))
pickle.dump(le, open('le.pkl', 'wb'))
