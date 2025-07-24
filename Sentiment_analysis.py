import pandas as pd

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string
from sklearn.feature_extraction.text import TfidfVectorizer


df = pd.read_excel("sentiment_analysis_data_10000.xlsx")

df

import nltk
nltk.download("stopwords")
nltk.download("punkt_tab")
stop_words = stopwords.words("english")
ps = PorterStemmer()

def preprocessing(text):
  text = text.lower()
  token = nltk.word_tokenize(text)
  new = []
  for i in token:
    if i not in stop_words and i not in string.punctuation:
      new.append(ps.stem(i))
  return " ".join(new)

df["clean"] = df["Sentence"].apply(preprocessing)
df["clean"]

vector = TfidfVectorizer()
X = vector.fit_transform(df["clean"])
y = df["Sentiment"]

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=55)

log = LogisticRegression()
log.fit(X_train, y_train)

train_score = log.score(X_train, y_train)
test_score = log.score(X_test, y_test)
print(train_score)
print(test_score)

def pred_text(text):
  text = text.lower()
  text = preprocessing(text)
  vec = vector.transform([text])
  predict = log.predict(vec)[0]
  return "Positive.😘" if predict == 1 else "Negative.🤬"


c = "I am bad"
pred_text(c)

d = "You are handsome"
pred_text(d)

e = "You are terrible"
pred_text(e)

f = "Terrible performance"
pred_text(f)

g = "What a disappointed experience"
pred_text(g)

h = "Such a delightful and feature product"
pred_text(h)

i = "I absolutely love how incredible the code is"
pred_text(i)