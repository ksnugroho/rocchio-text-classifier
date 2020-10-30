import pickle as pickle
import pandas as pd
import re
import os
import string
import unicodedata
import nltk
from bs4 import BeautifulSoup
from emo_unicode import UNICODE_EMO, EMOTICONS
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

from flask import Flask, request, render_template

THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))

# tfidf path
tfidf_path = os.path.join(
    THIS_FOLDER, "../03_feature_engineering/03_pickle/03_tfidf.pickle"
)
with open(tfidf_path, "rb") as tfidf_path:
    tfidf = pickle.load(tfidf_path)

# roc model path
model_path = os.path.join(
    THIS_FOLDER, "../04_model_training/04_pickle/04_model_rocchio.pickle"
)
with open(model_path, "rb") as model_path:
    roc = pickle.load(model_path)

app = Flask(__name__)

category_codes = {
    "DINAS KEPENDUDUKAN DAN PENCATATAN SIPIL KOTA MALANG": 0,
    "DINAS PEKERJAAN UMUM DAN PENATAAN RUANG KOTA MALANG": 1,
    "DINAS LINGKUNGAN HIDUP KOTA MALANG": 2,
    "DINAS PENDIDIKAN KOTA MALANG": 3,
    "DINAS PERUMAHAN DAN KAWASAN PERMUKIMAN KOTA MALANG": 4,
    "SATUAN POLISI PAMONG PRAJA KOTA MALANG": 5,
    "DINAS PERHUBUNGAN KOTA MALANG": 6,
}


def text_preprocessing(text):
    text = text.strip().lower()
    text = re.sub(r"https?://\S+|www\.\S+", "", text)  # remove url
    text = re.sub("\S*@\S*\s?", "", text)  # remove email
    text = re.sub("\[[^]]*\]", "", text)  # remove beetwen square brackets []
    text = re.sub("[-+]?[0-9]+", "", text)  # remove number
    emoticon_pattern = re.compile(u"(" + u"|".join(k for k in EMOTICONS) + u")")
    emoji_pattern = re.compile(
        "["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002500-\U00002BEF"  # chinese char
        "]+",
        flags=re.UNICODE,
    )
    text = emoji_pattern.sub(r"", text)  # remove emoji
    text = emoticon_pattern.sub(r"", text)  # remove emoticon
    text = re.sub(r"[^\w\s]", "", text)  # remove punctuation
    text = (
        unicodedata.normalize("NFKD", text)
        .encode("ascii", "ignore")
        .decode("utf-8", "ignore")
    )  # remove non-ascii
    normal_word_path = os.path.join(THIS_FOLDER, "../00_data/key_norm.csv")
    normal_word = pd.read_csv(normal_word_path)
    text = " ".join(
        [
            normal_word[normal_word["singkat"] == word]["hasil"].values[0]
            if (normal_word["singkat"] == word).any()
            else word
            for word in text.split()
        ]
    )

    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    text = stemmer.stem(text)

    words = nltk.word_tokenize(text)  # tokonize

    stopword = stopwords.words("indonesian")
    more_stopword = ["daring", "online", "pd"]  # add more stopword to default corpus
    stop_factory = stopword + more_stopword

    clean_words = []
    for word in words:
        if word not in stop_factory:
            clean_words.append(word)
    words = clean_words
    words = " ".join(words)  # join
    return words


def create_features(text):
    df = pd.DataFrame(columns=["text"])
    df.loc[0] = text
    df["text"] = df["text"].apply(text_preprocessing)

    features = tfidf.transform(df["text"]).toarray()
    return features


def get_category_name(category_id):
    for category, id_ in category_codes.items():
        if id_ == category_id:
            return category


def predict_from_text(text):
    # Predict using the input model
    roc_prediction = roc.predict(create_features(text))[0]

    # Return result
    roc_category = get_category_name(roc_prediction)
    return roc_category


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    name = request.form["name"]
    message = request.form["ticket"]

    result = predict_from_text(message)
    return render_template("result.html", name=name, message=message, result=result)


if __name__ == "__main__":
    app.run(debug=True, port=5000)