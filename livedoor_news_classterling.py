# coding: UTF-8
import tensorflow
import sys # モジュール属性 argv を取得するため
import MeCab
import gensim

mecab = MeCab.Tagger('mecabrc')

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
def makedicDocVec():

    for dicname in dir_dic:
        files = os.listdir(dicname)
        makeDicForFiles(files,dicname)

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

def makeDic(line):
    result = mecab.parse(line)
    word = result[1:]
