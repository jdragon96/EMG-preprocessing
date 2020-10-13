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
# make_train_set(train_data, test_data, sampling_freq, num_classes):
# 데이터 labeling
"""
traindata, train_label, testdata, test_label = loading.make_train_set(traindata, testdata, 100, 4)




"""
# moving_avg(학습 데이터, 테스트 데이터, window크기, overlap, ch당 높이, window당 길이)
# moving_avg(train_data, test_data, window, overlap, ch_height = 6, avg_width = 1)

# fft(train_data, test_data, ch_height = 6, fre_width = 1)

# stft(train_data, test_data, sampling_freq ,window, overlap, freq_height = 1, time_width = 1)

# validation(train_data, train_label, num_classes = 4, val_percent = 0.9):

골라서 사용
"""
#train_image, test_image, height, width = preprocessing.moving_avg(traindata, testdata, 12, 6, 1, 1)
train_image, test_image, height, width = preprocessing.stft(traindata, testdata, 100, 6, 2, 1)

#train_image, train_label, val_img, val_label = preprocessing.validation(train_image, trainlabel, train_percent = 0.9)



"""
# cnn_model(input_shape, learning_rate = 0.00001, dropout = 0.5, num_classes = 5)
def model_fit(cnn_model, traindata, trainlabel, model_file_name, val_img , val_label, mode, epoch = 200, batch_size = 50)

model_pred는 save된 모델을 load하여 테스트
# model_pred(model_file_name, testdata, testlabel, num_classes = 4)
"""
save_name = 'MAV10_8_500_323264_6_1'
train_model = model.cnn_model((height, width, 1))
train_model = model.model_fit(train_model, train_image, train_label, save_name,mode  = 'save',  epoch = 300)
model.model_pred(save_name, test_image, test_label)