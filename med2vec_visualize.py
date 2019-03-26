from sklearn.decomposition import PCA
from matplotlib import pyplot
from gensim.models import Word2Vec

model = Word2Vec.load("word2vec.model")

pca = PCA(n_components = 2)
result = pca.fit_transform(model[model.wv.vocab])
print(result)

pyplot.scatter(result[:,0],result[:,1])

words = list(model.wv.vocab)
for i, word in enumerate(words):
    pyplot.annotate(word, xy=(result[i, 0], result[i, 1]))

pyplot.show()