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
つるちゃん (id:tsuruchan_0827) 1年前

« 第1章 準備運動 １章　Pythonで始める機械学習 »
Featured Articles
chainerの使い方
chainerの畳み込みニューラルネットワークで10種類の画像を識別（CIFAR-10）
中間層はどのぐらいが良いのか...？
Profile
Written by tsuruchan.

人工知能の研究をしている大学生。
機械学習とDeepLearningの勉強中。
対話システムを作りたいな。
読者になる 11
Category
実践　機械学習システム (6)
chainer (4)
DeepLearning (4)
Kaggle (1)
Linux (1)
人工知能セミナー (1)
前処理 (1)
言語処理１００ (1)
Search

記事を検索
