import pandas as pd
from gensim.models import Word2Vec

data = pd.read_csv("TokenizedAll.csv", usecols=[
                   1], names=["id", "text"], skiprows=1)

common_tokens = ["mg", "on", "daily", "day", "for",
                 "qd", "regular", "medication", "prn", "with", "bid", "relief", "ongoing", "oral", "give", "of", "at", "one", "strict", "fluid", "iv", "take", "diet", "therapy"]

model = Word2Vec.load("word2vec.model")


def process_doc(note_set, offset, threshold):
    final = list()

    for note in note_set["text"]:
        meds = find_meds(note, offset, threshold)
        final.append(meds)

    return final


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

    identified_meds = []

    for i in range(search_start, search_end):
        try:
            similarity_dist = model.similarity(
                tokens[i], tokens[current_index])
            if similarity_dist > threshold:
                identified_meds.append(tokens[i])
        except Exception:
            continue

    identified_meds = [x for x in identified_meds if x not in common_tokens]
    return identified_meds


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
    print(process_doc(data, 3, 0.8))
