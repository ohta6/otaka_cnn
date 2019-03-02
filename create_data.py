# encode:utf-8

import csv
import datetime as dt
import time
import pickle
import librosa
import numpy as np
from scipy.stats import zscore
from numpy import random

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
"""
正解csvファイルとwavファイルから
学習データを出力する
"""
FILE_NUM = 579
VERSION = 2
PICKLE_SPLIT_NUM = 10
FEATURE = 'gakkai'
EXTENSION = '.pickle'
PICKELE_FILES = ['train_{0}_{1}_{2}_v{3}{4}'.format(FILE_NUM,i,FEATURE,VERSION,EXTENSION) for i in range(PICKLE_SPLIT_NUM)]
TEST_PICKLE = 'test_{0}_{1}_v{2}{3}'.format(FILE_NUM,FEATURE,VERSION,EXTENSION)
ADDITIONAL = ['./../data/20171122_otaka/trimed{}.wav'.format(str(i)) for i in range(1,14)]

CSV_FILE1 = './../data/5.csv'
WAV_FILES1 = ['./../data/5_20150212_1.wav',
            './../data/5_20150212_2.wav',
            './../data/5_20150212_3.wav',
            './../data/5_20150212_4.wav']
CSV_FILE2 = './../data/7_mod.csv'
WAV_FILES2 = ['./../data/7_20150211_1.wav',
            './../data/7_20150211_2.wav',
            './../data/7_20150211_3.wav',
            './../data/7_20150211_4.wav']
CSV_FILE3 = './../data/9M.csv'
WAV_FILES3 = ['./../data/9M_1.wav',
            './../data/9M_2.wav',
            './../data/9M_3.wav',
            './../data/9M_4.wav']
def sec2time(sec):
    return dt.time(hour=sec//3600, minute=(sec%3600)//60, second=sec%60)
def str2time(string):
    a = dt.datetime.strptime(string, '%H:%M:%S')
    return dt.time(hour=a.hour, minute=a.minute, second=a.second)
def time2sec(time):
    return time.hour*3600 + time.minute*60 + time.second
def str2sec(string):
    return time2sec(str2time(string))
def compare_time(time1, time2):
    return time2[1] >= time1[0] and time1[1] >= time2[0]

def is_in(sec, elist):
    for e in elist:
        start = time2sec(e[0])
        end = time2sec(e[1])
        if start <= sec and sec <= end:
            return True
    return False

def import_csv_data(csvfile):
    events = []
    with open(csvfile, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        header = next(reader)
        for row in reader:
            event_start = row[1]
            event_end = row[2]
            event_num = False
            # 最初の3つのイベントは起きていないので削る
            # 1:その他
            # 2:動物の鳴き声
            # 3:オオタカの鳴き声
            for i, r in enumerate(row[6:12]):
                if r != '':
                    if i == 0 or i == 1 or i == 2 or i == 4:
                        event_num = 1
                    elif i == 3:
                        event_num = 2
                    elif i == 5:
                        event_num = 3
                    else:
                        raise
            events.append([str2sec(event_start), str2sec(event_end), event_num])
    return events

def convert_wav(wavfiles, sampling_rate=44100, hop_length=441):
    data = []
    for wavfile in wavfiles:
        x, fs = librosa.load(wavfile, sr=sampling_rate)
        mfccs = librosa.feature.mfcc(x, sr=fs, hop_length=hop_length)
        num_sec = sampling_rate // hop_length
        for i in range(len(mfccs.T) // num_sec):
            temp = mfccs.T[i*num_sec:(i+1)*num_sec,:]
            data.append(temp)
    return data

def generate_train_data(wavfiles,
                        csv_file=False,
                        sampling_rate=44100,
                        hop_length=441):
    if csv_file:
        events = import_csv_data(csv_file)
        data = convert_wav(wavfiles)
        data_label = []
        for sec, datum in enumerate(data):
            flag = True
            for e in events:
                if e[0] <= sec and sec <= e[1]:
                    data_label.append([datum, e[2]])
                    flag = False
            if flag:
                data_label.append([datum, 0])
        
        #データ水増しのため、オオタカの場合は秒間もとる
        for e in events:
            if e[2] == 3:
                for sec in range(e[0], e[1]):
                    mae = data[sec][50:]
                    usiro = data[sec+1][:50]
                    data_label.append([np.concatenate((mae, usiro), axis=0), 3])
    else:
        data = convert_wav(wavfiles)
        data_label = []
        for datum in data:
            data_label.append([datum, 3])

    return data_label

if __name__ == '__main__':
    data_label = generate_train_data(WAV_FILES1, CSV_FILE1)
    data_label = data_label + generate_train_data(WAV_FILES2, CSV_FILE2)
    data_label = data_label + generate_train_data(WAV_FILES3, CSV_FILE3)
    data_label = data_label + generate_train_data(ADDITIONAL)
    noise = []
    separated = []
    for dl in data_label:
        if dl[1] == 0:
            noise.append(dl)
        else:
            separated.append(dl)

    filter_noise = list(range(len(noise)))
    undersampled = random.choice(filter_noise, 8406, replace=False)
    undersampled = [noise[f] for f in undersampled]
    data_label = separated + undersampled
    counter = [0]*4
    for d in data_label:
        counter[d[1]] += 1
    print(counter)
    train = [d[0] for d in data_label]
    label = [d[1] for d in data_label]
    xtrain, xtest, ytrain_label, ytest_label = train_test_split(
            train, label, test_size=0.2)
    counter = [0]*4
    for d in ytest_label:
        counter[int(d)] += 1
    print(counter)
    #オオタカにノイズを乗せて水増しするならここでやる
    len_data = len(xtrain)
    b = len_data // PICKLE_SPLIT_NUM
    """
    for i, p in enumerate(PICKELE_FILES):
        with open(p, 'wb') as f:
            pickle.dump([xtrain[i*b:(i+1)*b], ytrain_label[i*b:(i+1)*b]], f)
    with open(TEST_PICKLE, 'wb') as f:
        pickle.dump([xtest, ytest_label], f)
    """
