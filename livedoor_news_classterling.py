# coding: UTF-8
import tensorflow
import sys # モジュール属性 argv を取得するため
import MeCab
import gensim
from makeDicForFiles import makedicDocVec

mecab = MeCab.Tagger('mecabrc')

model = Word2Vec.load("livedoor_wordmodel.model")

# テキストファイルがあるディレクトリ
dir_dic = [
    "dokujo-tsushin",
    "it-life-hack",
    "kaden-channel",
    "livedoor-homme",
    "movie-enter",
    "peachy",
    "smax",
    "sports-watch",
    "topic-news",
]

# 文書ごとのベクトルを作成
dicDocVec = {}


# 辞書作成
def makeDicForFiles(files,dicname):
    for file in files:
        f = open(dicname+'/'+file)
        line = f.readline() # 1行を文字列として読み込む(改行文字も含まれる)

        while line:
            print line
            result = mecab.parse(line)
            line = f.readline()

        f.close

# 配列作成
def makeAryWord(line):

    for token in tokenize(line):
        if token != None:
            # ベクトルが必要や
            aryWord.append(model.wv[token])


def tokenize(text):
    '''
    とりあえず形態素解析して名詞だけ取り出す感じにしてる
    '''
    node = mecab.parseToNode(text)
    while node:
        node = node.next
