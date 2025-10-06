import re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.corpus import stopwords
import nltk

# Pastikan stopwords tersedia
try:
    stop_words = set(stopwords.words('indonesian'))
except LookupError:
    nltk.download('stopwords')
    stop_words = set(stopwords.words('indonesian'))

factory = StemmerFactory()
stemmer = factory.create_stemmer()

def preprocessing(text):
    if not isinstance(text, str):
        return ""
    text = text.lower() #case folding
    text = re.sub(r'http\S+|www\S+|[^a-z\s]', ' ', text)  #cleansing
    tokens = text.split()   #tokenizing
    tokens = [word for word in tokens if word not in stop_words and len(word) > 2]  #stopword removal
    stemmed = [stemmer.stem(word) for word in tokens]   #stemming
    return ' '.join(stemmed)
