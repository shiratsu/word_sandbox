# coding: UTF-8
#
# このファイルはまずモデルを作る
# このモデルを元に本データがどっちに分類されるのか予測する
#
import gensim
import MeCab
import os

mecab = MeCab.Tagger('-d /usr/local/lib/mecab/dic/mecab-ipadic-neologd')
aryLine = []
aryWord = []
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

# 辞書作成
def makedicDocVec():

    for dicname in dir_dic:
        files = os.listdir('formakemodel/livedoor/'+dicname)
        makeDicForFiles(files,dicname)

# 辞書作成
def makeDicForFiles(files,dicname):
    for file in files:
        if not file.startswith('.'):
            f = open('formakemodel/livedoor/'+dicname+'/'+file,encoding='utf-8', errors='ignore')
            line = f.readline() # 1行を文字列として読み込む(改行文字も含まれる)

            while line:
                makeAryWord(line)
                # aryLine.append(aryWord)
                line = f.readline()
                # aryWord = []

            f.close

# 配列作成
def makeAryWord(line):
    # tokenize(line)
    for token in tokenize(line):
        print(token)
        # ベクトルが必要や
        aryWord.append(token)


def tokenize(text):
    '''
    とりあえず形態素解析して名詞だけ取り出す感じにしてる
    '''
    # print(text)
    mecab.parse('') # <= 空文字列をparseする
    node = mecab.parseToNode(text)
    # # print(node)
    # i = 0
    while node:
        yield node.surface
        node = node.next



makedicDocVec()

# モデルを作成して保存
model = Word2Vec(aryWord)
model.save("livedoor_wordmodel.model")
