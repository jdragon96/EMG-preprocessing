import loading
import preprocessing
import model
import numpy as np

"""
# load_emg_data(데이터 폴더, 학습 데이터 이름, 테스트 데이터 이름)
# csv파일 읽기
"""

traindata, testdata = loading.load_emg_data("sEMG_data", "total_train4.csv", "total_test4.csv")

"""
# make_train_set(데이터 폴더, 학습 데이터 이름, 테스트 데이터 이름)
# 데이터 labeling
"""
traindata, trainlabel, testdata, testlabel = loading.make_train_set(traindata, testdata, 100, 4)


"""
# moving_avg(학습 데이터, 테스트 데이터, window크기, overlap, ch당 높이, window당 길이)
# moving_avg(train_data, test_data, window, overlap, ch_height = 6, avg_width = 1)

# fft(train_data, test_data, ch_height = 6, fre_width = 1)

# stft(train_data, test_data, sampling_freq ,window, overlap, freq_height = 1, time_width = 1)

골라서 사용
"""
train_img, test_img, height, width = preprocessing.moving_avg(traindata, testdata, 10, 8, 6, 1)
#train_img, test_img, height, width = preprocessing.stft(traindata, testdata, 100, 10, 8)

"""
# cnn_model(input_shape, learning_rate = 0.00001, dropout = 0.5, num_classes = 5)
# model_fit(cnn_model, traindata, trainlabel, model_file_name, epoch = 200, batch_size = 50):
# model_pred(model_file_name, testdata, testlabel, num_classes = 4)
"""
save_name = 'NAV10_8'
train_model = model.cnn_model((height, width, 1))
train_model = model.model_fit(train_model, train_img, trainlabel, save_name, 'save')
model.model_pred(save_name, test_img, testlabel)