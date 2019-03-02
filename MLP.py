#!/user/bin/env python
# coding:UTF-8

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions
from chainer.datasets import TupleDataset

import pickle
import numpy as np
from sklearn.model_selection import train_test_split

MODEL_FEATURE = '579_MLP'
MODEL_VERSION = 4
MODEL_RESULT = MODEL_FEATURE+'_'+str(MODEL_VERSION)+'_result'
MODEL_NAME = 'File_{}_test_{}.npz'.format(MODEL_FEATURE, MODEL_VERSION)

FILE_NUM = 579
VERSION = 1
PICKLE_SPLIT_NUM = 10
FEATURE = 'sankou'
EXTENSION = '.pickle'
PICKLE_FILES = ['train_{0}_{1}_{2}_v{3}{4}'.format(FILE_NUM,i,FEATURE,VERSION,EXTENSION) for i in range(PICKLE_SPLIT_NUM)]

n_epoch = 100
#ネットワークモデル化
class MLP(chainer.Chain):
    #各層の定義
    def __init__(self, class_labels=4):
        super(MLP, self).__init__()
        with self.init_scope():
            #入力層の定義は省略できる
            #データから自動で判断してくれる
            #self.bnorm = L.BatchNormalization(1)
            """
            self.l1 = L.Linear(None, 1000)
            self.l2 = L.Linear(None, 500)
            self.l3 = L.Linear(None, 100)
            self.l4 = L.Linear(None, class_labels)
            """
            self.l1 = L.Linear(None, 80)
            self.l2 = L.Linear(None, 40)
            self.l3 = L.Linear(None, class_labels)
    #活性化関数の定義
    #今回はrelu関数を用いている
    def __call__(self, x):
        #x = self.bnorm(x)
        h = F.relu(self.l1(x))
        h = F.relu(self.l2(h))
        h = F.relu(self.l3(h))
        return h

if __name__ == '__main__':
    #データの準備
    #chainer.datasets.TupleDataset型
    #xtrain,xtest : 訓練データ
    #ytrain_label,ytest_label : ラベルデータ
    data_train = []
    data_label = []
    for inp in PICKLE_FILES:
        with open(inp, 'rb') as f:
            temp = pickle.load(f)
            data_train += temp[0]
            data_label += temp[1]
    train = np.array([[d] for d in data_train], dtype=np.float32)
    label = np.array(data_label, dtype=np.int32)
    xtrain, xvalid, ytrain_label, yvalid_label = train_test_split(
            train, label, test_size=0.2)
    train = TupleDataset(xtrain, ytrain_label)
    valid = TupleDataset(xvalid, yvalid_label)

    #モデルのインスタンスの作成
    model = L.Classifier(MLP())

    #最適化手法の定義
    #今回はAdamを用いている
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)
    #正則化項の追加
    #optimizer.add_hook(chainer.optimizer.WeightDecay(0.0005))

    #訓練データとテストデータを32個のバッチにわける(1つのバッチを学習する→1イテレータ)
    train_iter = chainer.iterators.SerialIterator(train, 32)
    test_iter = chainer.iterators.SerialIterator(valid, 32, repeat=False, shuffle=False)

    #学習単位の定義(イテレータ毎にoptimizerによって学習を行う)
    updater = training.StandardUpdater(train_iter, optimizer, device=-1)
    #10000epoch学習を行う、結果を出力するresultディレクトリを指定
    trainer = training.Trainer(updater, (n_epoch, 'epoch'), out=MODEL_RESULT)

    #1epoch学習するとtest_iterを用いて評価を行う
    trainer.extend(extensions.Evaluator(test_iter, model))

    trainer.extend(extensions.LogReport())
    #epoch毎に損失関数と正答率を表示
    trainer.extend(extensions.PrintReport(['epoch', 'main/loss', 'validation/main/loss','main/accuracy', 'validation/main/accuracy', 'elapsed_time']))

    #損失関数と正答率のepoch毎の変換をグラフで出力
    trainer.extend(extensions.PlotReport(['main/loss', 'validation/main/loss'],'epoch', file_name='loss.png'))
    trainer.extend(extensions.PlotReport(['main/accuracy', 'validation/main/accuracy'],'epoch', file_name='accuracy.png'))

    #学習の進捗状況を表示
    trainer.extend(extensions.ProgressBar())

    #学習の開始
    trainer.run()
    #学習後のモデルデータの保存
    chainer.serializers.save_npz(MODEL_NAME, model)
