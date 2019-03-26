import pandas as pd
from gensim.models import Word2Vec
import csv
import os

# Data initialization
data = pd.read_csv("DataAll.csv",usecols=[0,1,2], names=["id","text","token_text"], skiprows=1, encoding="iso-8859-1")
train_data = data.sample(frac=0.7, random_state=27)
df = pd.concat([data, train_data])
test_data = df.drop_duplicates(keep=False)

# test_data.to_csv("test_data.csv")

# Token identification
def get_common_tokens(top_n):
    neighbour_hood = pd.read_csv("NeighbourhoodCounts.csv", usecols=[0], nrows=top_n, names=["tokens"])
    tokens = neighbour_hood["tokens"].tolist()
    return tokens

common_tokens = get_common_tokens(25)

# Model loading
model = Word2Vec.load("word2vec.model")

# Ensure file doesn't exist
os.remove("med2vec_results.csv")

# Calulating drugs
def process_doc(note_set, offset, threshold):
    count = 0
    for note in note_set["token_text"]:
        meds = find_meds(note, offset, threshold)
        meds_to_string = ",".join(meds)
        with open('med2vec_results.csv', mode='a') as med2vec_output:
            writer = csv.writer(med2vec_output, delimiter=',',
                                quotechar='"', quoting=csv.QUOTE_MINIMAL)
            writer.writerow([str(count), meds_to_string])
        count += 1


def find_meds(text, offset, threshold):
    tokens = text.split(" ")
    index = 0

    all_meds = list()

    for tk in tokens:
        if tk in common_tokens:
            meds = calculate_similar_meds(tokens, index, offset, threshold)
            all_meds.extend(meds)
        index += 1

        all_meds = list(dict.fromkeys(all_meds))
    return all_meds


def calculate_similar_meds(tokens, current_index, offset, threshold):
    search_start = min_start(current_index, offset)
    search_end = max_end(current_index, offset, len(tokens))

    meds = []

    for i in range(search_start, search_end):
        try:
            similarity_dist = model.similarity(
                tokens[i], tokens[current_index])
            if similarity_dist > threshold:
                meds.append(tokens[i])
        except Exception:
            continue

    meds = [x for x in meds if x not in common_tokens]
    return meds


def min_start(index, offset):
    if index - offset < 0:
        return 0
    else:
        return index - offset


def max_end(index, offset, length):
    if index + offset >= length:
        return length - 1
    else:
        return index + offset


if __name__ == "__main__":
    process_doc(data, 2, 0.8)
