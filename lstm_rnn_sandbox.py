# coding: UTF-8

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
import numpy as np
import random, sys
import MeCab
import sys # モジュール属性 argv を取得するため
mecab = MeCab.Tagger('mecabrc')

step = 3
sentences = []
nextWords = []

word2id = {}
id2word = {}

def get_words(strFile):
    '''
    記事群のdictについて、形態素解析してリストにして返す
    '''

    '''
    記事群のdictについて、形態素解析してリストにして返す
    '''
    lineNum = 1
    # p = re.compile(r"https?://[\w/:%#\$&\?\(\)~\.=\+\-]+")

    for line in open('./txt/'+strFile, 'r'):
        # # print(p.match(line))
        # if lineNum != 1 and lineNum != 2 and p.search(line) is None:
        #     # print(line)
        #     get_words_main(line)
        # lineNum+=1
        # aryWord.append(get_words_main(line))
        get_words_main(line)


def get_words_main(content):
    '''
    一つの記事を形態素解析して返す
    '''

    localWord = []
    localNextWord = []

    for token in tokenize(content):
        aryWord.append(token)

def tokenize(text):
    '''
    とりあえず形態素解析して名詞だけ取り出す感じにしてる
    '''
    node = mecab.parseToNode(text)
    while node:
        if node.feature.split(',')[0]:
            yield node.surface.lower()
        node = node.next

# 時系列になるようにワードを作成
def makeHistoryWords():
    print(aryWord)

if __name__ == '__main__':
    argvs = sys.argv  # コマンドライン引数を格納したリストの取得
    get_words(argvs[1])
