from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn
import re
import pandas as pd
import nltk

# upload files
data1 = pd.read_csv("Test1+2Formatted.csv")

data3 = pd.read_csv("mt_chart.csv")
train_data3 = data3.sample(frac=0.7, random_state=27)
df = pd.concat([data3, train_data3])
test_data3 = df.drop_duplicates(keep=False)

data4 = pd.read_csv("mt_discharge.csv")
train_data4 = data4.sample(frac=0.7, random_state=27)
df = pd.concat([data4, train_data4])
test_data3 = df.drop_duplicates(keep=False)


# start the tokenization process

tokens = []
for paragraph in data1["FreeText"]:
    tk = re.split(" |,|-|:", paragraph.replace(".", ""))
    tk = [i.lower() for i in tk if len(i) > 0]
    tokens += [tk]

for paragraph in train_data3["transcription"]:
    tk = re.split(" |,|-|:", paragraph.replace(".",""))
    tk = [i.lower() for i in tk if len(i) > 0]
    tokens += [tk]
    
for paragraph in train_data4["transcription"]:
    tk = re.split(" |,|-|:", paragraph.replace(".",""))
    tk = [i.lower() for i in tk if len(i) > 0]
    tokens += [tk]

appos = {
"aren't": "are not",
"can't": "cannot",
"couldn't": "could not",
"didn't": "did not",
"doesn't": "does not",
"don't": "do not",
"hadn't": "had not",
"hasn't": "has not",
"haven't": "have not",
"he'd": "he would",
"he'll": "he will",
"he's": "he is",
"i'd": "I would",
"i'd": "I had",
"i'll": "I will",
"i'm": "I am",
"isn't": "is not",
"it's": "it is",
"it'll": "it will",
"i've": "I have",
"let's": "let us",
"mightn't": "might not",
"mustn't": "must not",
"shan't": "shall not",
"she'd": "she would",
"she'll": "she will",
"she's": "she is",
"shouldn't": "should not",
"that's": "that is",
"there's": "there is",
"they'd": "they would",
"they'll": "they will",
"they're": "they are",
"they've": "they have",
"we'd": "we would",
"we're": "we are",
"weren't": "were not",
"we've": "we have",
"what'll": "what will",
"what're": "what are",
"what's": "what is",
"what've": "what have",
"where's": "where is",
"who'd": "who would",
"who'll": "who will",
"who're": "who are",
"who's": "who is",
"who've": "who have",
"won't": "will not",
"wouldn't": "would not",
"you'd": "you would",
"you'll": "you will",
"you're": "you are",
"you've": "you have",
"'re": " are",
"wasn't": "was not",
"we'll": " will",
"didn't": "did not"
}

words = []
for paragraph in tokens:
    wrd = []
    for i in paragraph:
        if i in appos:
            wrd += appos[i].split(" ")
        else:
            wrd += [i]
    words += [wrd]

# getting to see if the words are nouns, adverbs, adjectives

# WordNet POS tags are: NOUN = 'n', ADJ = 's', VERB = 'v', ADV = 'r', ADJ_SAT = 'a'
# Descriptions (c) https://web.stanford.edu/~jurafsky/slp3/10.pdf
tag_map = {
        'CC': None,  # coordin. conjunction (and, but, or)
        'CD': wn.NOUN,  # cardinal number (one, two)
        'DT': None,  # determiner (a, the)
        'EX': wn.ADV,  # existential ‘there’ (there)
        'FW': None,  # foreign word (mea culpa)
        'IN': wn.ADV,  # preposition/sub-conj (of, in, by)
        'JJ': wn.ADJ,  # adjective (yellow)
        'JJR': wn.ADJ,  # adj., comparative (bigger)
        'JJS': wn.ADJ,  # adj., superlative (wildest)
        'LS': None,  # list item marker (1, 2, One)
        'MD': None,  # modal (can, should)
        'NN': wn.NOUN,  # noun, sing. or mass (llama)
        'NNS': wn.NOUN,  # noun, plural (llamas)
        'NNP': wn.NOUN,  # proper noun, sing. (IBM)
        'NNPS': wn.NOUN,  # proper noun, plural (Carolinas)
        'PDT': wn.ADJ,  # predeterminer (all, both)
        'POS': None,  # possessive ending (’s )
        'PRP': None,  # personal pronoun (I, you, he)
        'PRP$': None,  # possessive pronoun (your, one’s)
        'RB': wn.ADV,  # adverb (quickly, never)
        'RBR': wn.ADV,  # adverb, comparative (faster)
        'RBS': wn.ADV,  # adverb, superlative (fastest)
        'RP': wn.ADJ,  # particle (up, off)
        'SYM': None,  # symbol (+,%, &)
        'TO': None,  # “to” (to)
        'UH': None,  # interjection (ah, oops)
        'VB': wn.VERB,  # verb base form (eat)
        'VBD': wn.VERB,  # verb past tense (ate)
        'VBG': wn.VERB,  # verb gerund (eating)
        'VBN': wn.VERB,  # verb past participle (eaten)
        'VBP': wn.VERB,  # verb non-3sg pres (eat)
        'VBZ': wn.VERB,  # verb 3sg pres (eats)
        'WDT': None,  # wh-determiner (which, that)
        'WP': None,  # wh-pronoun (what, who)
        'WP$': None,  # possessive (wh- whose)
        'WRB': None,  # wh-adverb (how, where)
}


def get_wordnet_pos(tag):
    try:
        if tag_map[tag] != None:
            return tag_map[tag]
    except:
        return wn.NOUN
    return wn.NOUN


def get_lemmated_toekens(tokens):
    stop_words = stopwords.words("english")
    lem = WordNetLemmatizer()

    tokens = nltk.pos_tag(tokens)

    outputTokens = []
    for i in range(len(tokens)):
        if (tokens[i][1] == "IN"):  # intentially leaves injunctions in the tokens list
            outputTokens += [tokens[i][0]]
        elif tokens[i][0] not in stop_words and tokens[i][0].isalpha():
            # converts all other words into the base form
            outputTokens += [lem.lemmatize(tokens[i]
                                           [0], get_wordnet_pos(tokens[i][1]))]
    return outputTokens


# gets every word into an array, lowercase, stemming and lemmitization
cleanedTokens = []
for paragraph in words:
    cleanedTokens += [get_lemmated_toekens(paragraph)]

from gensim.models import Word2Vec

model = Word2Vec(
        cleanedTokens,
        size=15,
        window=10,
        min_count=2,
        workers=10)
model.train(cleanedTokens, total_examples=len(cleanedTokens), epochs=10)

model.save("word2vec.model")