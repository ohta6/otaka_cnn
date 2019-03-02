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

MODEL_FEATURE = '579_CNN'
MODEL_VERSION = 3
MODEL_RESULT = MODEL_FEATURE+'_'+str(MODEL_VERSION)+'_result'
MODEL_NAME = 'File_{}_test_{}.npz'.format(MODEL_FEATURE, MODEL_VERSION)

FILE_NUM = 579
VERSION = 2
PICKLE_SPLIT_NUM = 10
FEATURE = 'gakkai'
EXTENSION = '.pickle'
PICKLE_FILES = ['train_{0}_{1}_{2}_v{3}{4}'.format(FILE_NUM,i,FEATURE,VERSION,EXTENSION) for i in range(PICKLE_SPLIT_NUM)]

TEST_DATA = 'test_{}_{}_v{}.pickle'.format(FILE_NUM, FEATURE, VERSION)
n_epoch = 100
#ネットワークモデル化

class CNN(chainer.Chain):
    def __init__(self, class_labels=4):
        super(CNN, self).__init__()
        with self.init_scope():
            self.conv1_1 = L.Convolution2D(None, 16, ksize=5, pad=2, nobias=True)
            self.conv1_2 = L.Convolution2D(None, 16, ksize=5, pad=2, nobias=True)
            self.conv2_1 = L.Convolution2D(None, 32, ksize=3, pad=1, nobias=True)
            self.conv2_2 = L.Convolution2D(None, 32, ksize=3, pad=1, nobias=True)
            self.conv3_1 = L.Convolution2D(None, 64, ksize=3, pad=1, nobias=True)
            self.conv3_2 = L.Convolution2D(None, 64, ksize=3, pad=1, nobias=True)
            self.conv3_3 = L.Convolution2D(None, 64, ksize=3, pad=1, nobias=True)
            self.fc1 = L.Linear(None, 512, nobias=True)
            self.fc2 = L.Linear(None, 512, nobias=True)
            self.fc3 = L.Linear(None, class_labels, nobias=True)
            self.bnorm = L.BatchNormalization(1)
            self.ratio = 0.5

    def __call__(self, x):
        x = self.bnorm(x)
        conv1_1 = self.conv1_1(x)
        conv1_1 = F.relu(conv1_1)
        conv1_2 = self.conv1_2(conv1_1)
        conv1_2 = F.relu(conv1_2)
        pool1 = F.max_pooling_2d(conv1_2, ksize=2, stride=2)
        pool1 = F.dropout(pool1, ratio=self.ratio)
        conv2_1 = self.conv2_1(pool1)
        conv2_1 = F.relu(conv2_1)
        conv2_2 = self.conv2_2(conv2_1)
        conv2_2 = F.relu(conv2_2)
        pool2 = F.max_pooling_2d(conv2_2, ksize=2, stride=2)
        pool2 = F.dropout(pool2, ratio=self.ratio)
        conv3_1 = self.conv3_1(pool2)
        conv3_1 = F.relu(conv3_1)
        conv3_2 = self.conv3_2(conv3_1)
        conv3_2 = F.relu(conv3_2)
        conv3_3 = self.conv3_3(conv3_2)
        conv3_3 = F.relu(conv3_3)
        pool3 = F.max_pooling_2d(conv3_3, ksize=2, stride=2)
        pool3 = F.dropout(pool3, ratio=self.ratio)
        fc1 = self.fc1(pool3)
        fc1 = F.relu(fc1)
        fc2 = self.fc2(fc1)
        fc2 = F.relu(fc2)
        fc3 = self.fc3(fc2)
        return fc3

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
    #データの形式を変えたので直す
    train = np.array([[d] for d in data_train], dtype=np.float32)
    #train = np.array([[d[0].reshape(2000)] for d in data], dtype=np.float32)
    label = np.array(data_label, dtype=np.int32)
    xtrain, xvalid, ytrain_label, yvalid_label = train_test_split(
            train, label, test_size=0.2)
    train = TupleDataset(xtrain, ytrain_label)
    valid = TupleDataset(xvalid, yvalid_label)

    #モデルのインスタンスの作成
    model = L.Classifier(CNN())

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
