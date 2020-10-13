import os
import pandas as pd
import numpy as np


# 1. 데이터 파일 가져오기
# total_train, total_test.csv
def load_emg_data(emg_path, train_file_name, test_file_name):
    print("data loading..")
    csv_path      = os.path.join('./' + emg_path + '/' + train_file_name)
    csv_test_path = os.path.join('./' + emg_path + '/' + test_file_name)
    return pd.read_csv(csv_path), pd.read_csv(csv_test_path)


# 2. 데이터 정돈하기 - 3차원 배열로 변경(데이터 번호, 샘플링 수, 센서 채널수)
# 메모리 사용 조절을 어떻게 할까?
def make_train_set(train_data, test_data, sampling_freq, num_classes):
    print("data making..")
    data_samples = sampling_freq
    data_chanels = (np.shape(train_data))[1]

    train_data_numbers = int((np.shape(train_data))[0]/sampling_freq)
    train_each_data_numbers = int(train_data_numbers / num_classes)

    test_data_numbers = int((np.shape(test_data))[0]/sampling_freq)
    test_each_data_numbers = int(test_data_numbers / num_classes)

    train_data = np.float32(train_data)
    test_data = np.float32(test_data)

    traindata = np.zeros((train_data_numbers, data_samples, data_chanels)   # (1) train 데이터 만들기
                          , np.float32)
    for j in range (train_data_numbers):
        for i in range (data_samples):
            for k in range(data_chanels):
                traindata[j][i][k] = train_data[(j * data_samples) + i][k]  # 2차원 csv를 3차원 배열로

    trainlabel = np.zeros((train_data_numbers,num_classes),int)             # (2) train 라벨 만들기
    for i in range(num_classes):
        for k in range(train_each_data_numbers):
            trainlabel[k + (train_each_data_numbers * i), i] = 1            # 각 동작별로 구분해 정답데이터 만들기

    testdata = np.zeros((test_data_numbers, data_samples, data_chanels),    # (3) test 데이터 만들기
                        np.float32)
    for j in range(test_data_numbers):
        for i in range(data_samples):
            for k in range(data_chanels):
                testdata[j][i][k] = test_data[(j * data_samples) + i][k]

    testlabel = np.zeros((test_data_numbers, num_classes),int)               # (4) test 라벨 만들기
    for i in range(num_classes):
        for k in range(test_each_data_numbers):
            testlabel[k + (test_each_data_numbers * i), i] = 1
    traindata = np.abs(traindata)
    testdata = np.abs(testdata)
    return traindata, trainlabel, testdata, testlabel


if __name__ == "__main__":
    print("Hello")