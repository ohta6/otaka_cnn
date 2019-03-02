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
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import CNN
import MLP
model_name = 'File_579_CNN_test_3.npz'
TEST_PICKLE = 'test_579_gakkai_v2.pickle'

#ラベルごとに何個ずつ分類されているか
def my_metrics(model, train, label):
    chainer.serializers.load_npz(model_name, model)
    total_result = []
    for tr, la in zip(train, label):
        result = model.predictor(chainer.Variable(np.array([tr])))
        result = result.data
        pred = np.argmax(result[0])
        total_result.append(pred)
    met = confusion_matrix(label, total_result)
    report = classification_report(label, total_result)
    return met, report

#オオタカの鳴き声と分類されているものはどこか
def when_otaka(model, train, label):
    chainer.serializers.load_npz(model_name, model)
    result = model.predictor(chainer.Variable(train))
    result = result.data
    pred = [r.argmax() for r in result]
    otaka_pred = [1 if p == 9 else 0 for p in pred]
    otaka_label = [1 if p == 9 else 0 for p in label]
    x = np.arange(len(otaka_pred))
    plt.subplot(2, 1, 1)
    plt.plot(x, otaka_label)
    plt.title('label')
    
    plt.subplot(2, 1, 2)
    plt.plot(x, otaka_pred)
    plt.title('pred')
    plt.show()

if __name__ == '__main__':
    #データの準備
    #chainer.datasets.TupleDataset型
    #xtrain,xtest : 訓練データ
    #ytrain_label,ytest_label : ラベルデータ
    with open(TEST_PICKLE, 'rb') as f:
        data = pickle.load(f)
    train = np.array([[d] for d in data[0]], dtype=np.float32)
    label = np.array(data[1], dtype=np.int32)

    #モデルのインスタンスの作成
    model = L.Classifier(CNN.CNN())
    mat, report = my_metrics(model, train, label)
    print(mat)
    print(report)

