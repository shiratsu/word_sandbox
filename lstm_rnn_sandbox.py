# coding: UTF-8
import argparse
import pickle
import os
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
aryWord = []


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

    vocab = {}
    words = aryWord
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
    get_words("kokoro_utf8.txt")
    makeDataSet()

    # arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir',                   type=str,   default='data/tinyshakespeare')
    parser.add_argument('--checkpoint_dir',             type=str,   default='cv')
    parser.add_argument('--gpu',                        type=int,   default=-1)
    parser.add_argument('--rnn_size',                   type=int,   default=128)
    parser.add_argument('--learning_rate',              type=float, default=2e-3)
    parser.add_argument('--learning_rate_decay',        type=float, default=0.97)
    parser.add_argument('--learning_rate_decay_after',  type=int,   default=10)
    parser.add_argument('--decay_rate',                 type=float, default=0.95)
    parser.add_argument('--dropout',                    type=float, default=0.0)
    parser.add_argument('--seq_length',                 type=int,   default=50)
    parser.add_argument('--batchsize',                  type=int,   default=50)
    parser.add_argument('--epochs',                     type=int,   default=50)
    parser.add_argument('--grad_clip',                  type=int,   default=5)
    parser.add_argument('--init_from',                  type=str,   default='')

    args = parser.parse_args()

    if not os.path.exists(args.checkpoint_dir):
        os.mkdir(args.checkpoint_dir)

    n_epochs    = args.epochs
    n_units     = args.rnn_size
    batchsize   = args.batchsize
    bprop_len   = args.seq_length
    grad_clip   = args.grad_clip

    train_data, words, vocab = makeDataSet()
    pickle.dump(vocab, open('data/vocab.bin', 'wb'))
    if args.gpu >= 0:
        cuda.get_device(args.gpu).use()
        model.to_gpu()

    optimizer = optimizers.RMSprop(lr=args.learning_rate, alpha=args.decay_rate, eps=1e-8)
    optimizer.setup(model)

    if len(args.init_from) > 0:
        model = pickle.load(open(args.init_from, 'rb'))
    else:
        model = CharRNN(len(vocab), n_units)

    if args.gpu >= 0:
        cuda.get_device(args.gpu).use()
        model.to_gpu()

    optimizer = optimizers.RMSprop(lr=args.learning_rate, alpha=args.decay_rate, eps=1e-8)
    optimizer.setup(model)

    whole_len    = train_data.shape[0]
    jump         = whole_len / batchsize
    epoch        = 0
    start_at     = time.time()
    cur_at       = start_at
    state        = make_initial_state(n_units, batchsize=batchsize)
    if args.gpu >= 0:
        accum_loss   = Variable(cuda.zeros(()))
        for key, value in state.items():
            value.data = cuda.to_gpu(value.data)
    else:
        accum_loss   = Variable(np.zeros((), dtype=np.float32))

    print('going to train {} iterations'.format(jump * n_epochs))
    # print(train_data)
    # print(batchsize)
    for i in range(int(jump * n_epochs)):

        # for j in range(batchsize):
        #     print(train_data[int((jump * j + i) % whole_len]))

        # ある言葉の次に何が来るかを推測するために、yは次のindexを指定
        x_batch = np.array([train_data[int((jump * j + i) % whole_len)]
                            for j in range(batchsize)])
        y_batch = np.array([train_data[int((jump * j + i + 1) % whole_len)]
                            for j in range(batchsize)])

        if args.gpu >=0:
            x_batch = cuda.to_gpu(x_batch)
            y_batch = cuda.to_gpu(y_batch)

        state, loss_i = model.forward_one_step(x_batch, y_batch, state, dropout_ratio=args.dropout)
        accum_loss   += loss_i

        if (i + 1) % bprop_len == 0:  # Run truncated BPTT
            now = time.time()
            print('{}/{}, train_loss = {}, time = {:.2f}'.format((i+1)/bprop_len, jump, accum_loss.data / bprop_len, now-cur_at))
            cur_at = now

            optimizer.zero_grads()
            accum_loss.backward()
            accum_loss.unchain_backward()  # truncate
            if args.gpu >= 0:
                accum_loss = Variable(cuda.zeros(()))
            else:
                accum_loss = Variable(np.zeros((), dtype=np.float32))

            optimizer.clip_grads(grad_clip)
            optimizer.update()

        if (i + 1) % 10000 == 0:
            # モデルの情報をファイルに書き込んでいる
            # トレーニング結果を別で利用するため
            # これをすれば毎回トレーニングさせてから、文章作らせる必要ないね
            fn = ('%s/charrnn_epoch_%.2f.chainermodel' % (args.checkpoint_dir, float(i)/jump))
            pickle.dump(copy.deepcopy(model).to_cpu(), open(fn, 'wb'))
            pickle.dump(copy.deepcopy(model).to_cpu(), open('%s/latest.chainermodel'%(args.checkpoint_dir), 'wb'))

        if (i + 1) % jump == 0:
            epoch += 1

            if epoch >= args.learning_rate_decay_after:
                optimizer.lr *= args.learning_rate_decay
                print('decayed learning rate by a factor {} to {}'.format(args.learning_rate_decay, optimizer.lr))

        sys.stdout.flush()
