# coding: UTF-8

import nltk.stem
import scipy as sp
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

def dist_norm(v1, v2):
    v1_norm = v1/sp.linalg.norm(v1.toarray())
    v2_norm = v2/sp.linalg.norm(v2.toarray())
    delta = v1_norm - v2_norm
    return sp.linalg.norm(delta.toarray())

english_stemmer = nltk.stem.SnowballStemmer('english')

class StemmedTfidfVectorizer(TfidfVectorizer):
    def build_analyzer(self):
        analyzer = super(TfidfVectorizer, self).build_analyzer()
        return lambda doc: (english_stemmer.stem(w) for w in analyzer(doc))

posts = [open(os.path.join('txt/', f)).read() for f in os.listdir('txt/')]
vec = StemmedTfidfVectorizer(min_df=1, stop_words='english')
X_train = vec.fit_transform(posts)
num_samples, num_features = X_train.shape

new_post = "imaging databases"
new_post_vec = vec.transform([new_post])

dist = []
for i in range(0, num_samples):
    dist.append(dist_norm(X_train.getrow(i), new_post_vec))
    print("post{} : dist={:.3f}, {}".format(i+1, dist[i], posts[i]))

print("Best post : {} , dist = {:.3f}".format(dist.index(min(dist))+1, min(dist)))
