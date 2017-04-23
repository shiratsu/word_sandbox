# coding: UTF-8

import numpy as np
from chainer import cuda, Variable, FunctionSet, optimizers
import chainer.functions as F
from CharRNN import CharRNN, make_initial_state
import numpy as np
import random, sys
import MeCab
import sys # モジュール属性 argv を取得するため
mecab = MeCab.Tagger('mecabrc')

step = 3
maxlen = 20
sentences = []
nextWords = []
aryWord = []

word2id = {}
id2word = {}

model = Sequential()

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

# input data
def makeDataSet():

    words = list(aryWord)
    dataset = np.ndarray((len(words),), dtype=np.int32)
    for i, word in enumerate(words):
        if word not in vocab:
            vocab[word] = len(vocab)
        dataset[i] = vocab[word]
    # print("-----------word--------")
    # print(words)
    print("-----------vocab--------")
    print(vocab)
    print('corpus length:', len(words))
    print('vocab size:', len(vocab))
    return dataset, words, vocab

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
    # print(aryWord)

    for i in range(0, len(aryWord)):
        if i+1 < len(aryWord):
            sentences.append(aryWord[i])
            nextWords.append(aryWord[i+1])
    print(nextWords)
    print("-----------------------------")
    print(sentences[4])
    print(nextWords[4])


if __name__ == '__main__':
    argvs = sys.argv  # コマンドライン引数を格納したリストの取得
    get_words(argvs[1])
