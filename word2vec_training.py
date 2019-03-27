from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn
import re
import pandas as pd
import nltk

# upload files
data = pd.read_csv("DataAll.csv",usecols=[0,1,2], names=["id","text","token_text"], skiprows=1, encoding="iso-8859-1")
train_data = data.sample(frac=0.7, random_state=27)
df = pd.concat([data, train_data])
test_data = df.drop_duplicates(keep=False)

cleanedTokens = []

for tk in train_data["token_text"]:
    token_text = tk.split(" ")
    cleanedTokens.append(token_text)

print(cleanedTokens)

from gensim.models import Word2Vec

model = Word2Vec(
        cleanedTokens,
        size=15,
        window=10,
        min_count=2,
        workers=10)
model.train(cleanedTokens, total_examples=len(cleanedTokens), epochs=10)

model.save("word2vec.model")