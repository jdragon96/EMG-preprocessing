"""
Author    : jaeyong Seong
title     : sEMG signal preprocessing

moving average, FFT, STFT
"""

import numpy as np

from scipy.fftpack import fft
from scipy import signal

import matplotlib.pyplot as plt




def moving_avg(train_data, test_data, window, overlap, ch_height = 6, avg_width = 1):
    """
    :param train_data: 학습데이터
    :param test_data:  테스트 데이터
    :param window:     window 크기
    :param overlap:    window overlap 크기
    :param ch_height:  채널당 y축 픽셀 수
    :param avg_width:  window당 x축 픽셀 수

    moving average를 이용한 평균값 데이터 구축
    """
    print("moving average preprocessing..")
    train_data_numbers = (np.shape(train_data))[0]                          # 10400
    test_data_numbers  = (np.shape(test_data))[0]
    data_chanels       = (np.shape(train_data))[2]                          # 8
    data_per_set       = (np.shape(train_data))[1]                          # 100
    train_total_windows= int((data_per_set - window)/(window - overlap))    # must win > ovl
    test_total_windows = int((data_per_set - window) / (window - overlap))  # must win > ovl
    avg_seg       = []
    train_all_avg = []
    test_all_avg = []
    element_sum = 0
    average = 0
    avg_len = 0

    image_height = int(data_chanels * ch_height)
    image_width  = int(train_total_windows * avg_width)

    train_image = np.zeros((train_data_numbers, image_height, image_width), np.float32)
    test_image = np.zeros((test_data_numbers, image_height, image_width), np.float32)

    for i in range(train_data_numbers):
        for j in range(data_chanels):
            for k in range(train_total_windows):                            # 전체 윈도우 수
                for x in range(window):
                    element_sum += train_data[i][k * (window - overlap) + x][j]
                average = element_sum / window

                train_all_avg.append(average)
                avg_seg.clear()
                element_sum = 0

    percent = 0
    for num in range(train_data_numbers):
        for ch in range(data_chanels):
            for len in range(train_total_windows):
                train_image[num,
                            image_height - ch_height - (ch * ch_height) : image_height - (ch * ch_height),
                            avg_width * len : avg_width * len + avg_width] = train_all_avg[avg_len]
                avg_len += 1
        if(num%(train_data_numbers*0.1) == 0):
            print("train : " ,percent,"%")
            percent += 10
    avg_len = 0



    for i in range(test_data_numbers):
        for j in range(data_chanels):
            for k in range(test_total_windows):                        # 전체 윈도우 수
                for x in range(window):
                    element_sum += test_data[i][k * (window - overlap) + x][j]
                average = element_sum / window

                test_all_avg.append(average)
                avg_seg.clear()
                element_sum = 0

    percent = 0
    for num in range(test_data_numbers):
        for ch in range(data_chanels):
            for len in range(test_total_windows):
                test_image[num,
                            image_height - ch_height - (ch * ch_height) : image_height - (ch * ch_height),
                            avg_width * len : avg_width * len + avg_width] = test_all_avg[avg_len]
                avg_len += 1
        if (num % (test_data_numbers * 0.1) == 0):
            print("test : ", percent, "%")
            percent += 10

    plt.imshow(train_image[0,:,:])
    plt.show()
    train_image = train_image.reshape(train_data_numbers, image_height, image_width, 1)
    test_image = test_image.reshape(test_data_numbers, image_height, image_width, 1)

    del avg_len, element_sum, average, train_all_avg, avg_seg, train_data, test_data
    return train_image, test_image, image_height, image_width




def fft(train_data, test_data, ch_height = 6, fre_width = 1):
    """
    :param train_data:학습 데이터
    :param test_data: 테스트 데이터
    :param ch_height: 채널당 y축 크기
    :param fre_width: 1Hz간격의 주파수 대역 폭

    FFT를 이용한 주파수 분석
    """
    train_data_numbers = (np.shape(train_data))[0]
    test_data_numbers = (np.shape(test_data))[0]
    data_chanels = (np.shape(train_data))[2]  # 8
    data_per_set = (np.shape(train_data))[1]  # 100
    image_height = int(ch_height * data_chanels)
    image_width  = int(data_per_set/2)
    train_fft    = []
    test_fft     = []

    train_image  = np.zeros((train_data_numbers, image_height, image_width), np.float32)
    test_image   = np.zeros((test_data_numbers, image_height, image_width), np.float32)

    for num in range(train_data_numbers):
        for ch in range(data_chanels):
            train_fft = fft(train_data[num, :, ch])
            fft_norm  = np.abs(train_fft)
            for feq in range(int(data_per_set/2)):
                train_image[num,
                            image_height - ch_height - (ch * ch_height): image_height - (ch * ch_height),
                            feq * fre_width: feq * fre_width + fre_width] = fft_norm[feq]


    for num in range(test_data_numbers):
        for ch in range(data_chanels):
            test_fft = fft(test_data[num, :, ch])
            fft_norm = np.abs(test_fft)
            for f in range(int(data_per_set/2)):
                test_image[num,
                           image_height - ch_height - (ch * ch_height): image_height - (ch * ch_height),
                           feq * fre_width : feq * fre_width + fre_width] = fft_norm[feq]

    plt.imshow(train_image[0, :, :])
    plt.show()

    train_image = train_image.reshape(train_data_numbers, image_height, image_width, 1)
    test_image = test_image.reshape(test_data_numbers, image_height, image_width, 1)
    input_shape = (image_height, image_width)

    del test_fft, train_fft, fft_norm, train_data, test_data
    return train_image, test_image, image_height, image_width




def stft(train_data, test_data, sampling_freq ,window, overlap, freq_height = 1, time_width = 1):
    """
    :param train_data: 학습 데이터
    :param test_data: 테스트 데이터
    :param sampling_freq: 주파수 샘플링주파수
    :param window: STFT를 계산할 window 사이즈
    :param overlap: window overlap 크기
    :param freq_height: 주파수당 y축 픽셀크기
    :param time_width: 시간당 x축 픽셀 크기

    STFT연산을 통해 time-frequency-domain sEMG분석
    """
    # 이미지 형성을 위한 Hyper parameter
    print("STFT preprocessing..")
    train_data_numbers = (np.shape(train_data))[0]  # 학습 데이터 총 개수
    test_data_numbers = (np.shape(test_data))[0]    # 테스트 데이터 총 개수
    data_chanels = (np.shape(train_data))[2]        # 데이터의 채널 수
    f, t, Zxx = signal.stft(train_data[0, :, 0], sampling_freq, nperseg=window, noverlap=overlap)
    freq_resolution = (np.shape(f))[0]              # 주파수 해상도
    time_resolution = (np.shape(t))[0]              # 시간 해상도
    print(time_resolution)
    print(freq_resolution)

    # 이미지 크기 설정
    image_height    = int(freq_resolution * data_chanels * freq_height)       # 6X8X1 = 48
    image_width     = int(time_resolution * time_width)                     # 11X1 = 11

    # STFT값을 입력하기 위한 image frame
    train_image = np.zeros((train_data_numbers, image_height, image_width), np.float32)
    test_image   = np.zeros((test_data_numbers, image_height, image_width), np.float32)

    percent = 0
    for num in range(train_data_numbers):
        for ch in range(data_chanels):
            f, t, Zxx = signal.stft(train_data[num,:, ch], sampling_freq, nperseg = window, noverlap=overlap)
            for freq in range(freq_resolution):
                for time in range(time_resolution):
                    train_image[num,
                    image_height - ch * freq_resolution - freq_height * freq - freq_height:
                    image_height - ch * freq_resolution - freq_height * freq,
                    time_width * time: time_width * time + time_width] = np.abs(Zxx[freq][time])
        if (num % (train_data_numbers * 0.1) == 0):
            print("train : " , percent, "%")
            percent += 10

    percent = 0
    for num in range(test_data_numbers):
        for ch in range(data_chanels):
            f, t, Zxx = signal.stft(test_data[num,:, ch], sampling_freq, nperseg=window, noverlap=overlap)
            for freq in range(freq_resolution):
                for time in range(time_resolution):
                    test_image[num,
                    image_height - ch * freq_resolution - freq_height * freq - freq_height:
                    image_height - ch * freq_resolution - freq_height * freq,
                    time_width * time: time_width * time + time_width] = np.abs(Zxx[freq][time])
        if (num % (test_data_numbers * 0.1) == 0):
            print("train : ", percent, "%")
            percent += 10

    plt.imshow(train_image[0, :, :])
    plt.show()

    # CNN학습을 위한 4-Dimension 데이터 형성
    train_image = train_image.reshape(train_data_numbers, image_height, image_width, 1)
    test_image = test_image.reshape(test_data_numbers, image_height, image_width, 1)
    input_shape = (image_height, image_width)

    return train_image, test_image, image_height, image_width