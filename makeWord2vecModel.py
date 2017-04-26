# coding: UTF-8
#
# このファイルはまずモデルを作る
# このモデルを元に本データがどっちに分類されるのか予測する
#
import gensim
import MeCab

mecab = MeCab.Tagger('mecabrc')
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
        files = os.listdir(dicname)
        makeDicForFiles(files,dicname)

# 辞書作成
def makeDicForFiles(files,dicname):
    for file in files:
        f = open(dicname+'/'+file)
        line = f.readline() # 1行を文字列として読み込む(改行文字も含まれる)

        while line:
            makeAryWord(line)
            line = f.readline()

        f.close

# 配列作成
def makeAryWord(line):
    result = mecab.parse(line)
    word = result[1:]
    aryWord.append(word)


# モデルを作成して保存
model = Word2Vec(aryWord)
model.save("livedoor_wordmodel.model")
