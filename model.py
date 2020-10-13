import matplotlib.pyplot as plt

import numpy as np

from keras.optimizers import Adam, sgd, adadelta
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten,Conv2D, MaxPooling2D
from keras.callbacks import EarlyStopping
from keras.backend import clear_session
from keras.models import model_from_json
import keras.backend as K




def cnn_model(input_shape, learning_rate = 0.00001, dropout = 0.5, num_classes = 4):
    cnn_model = Sequential()
    cnn_model.add(Conv2D(64, kernel_size=(2, 2), input_shape=input_shape, padding="same"))
    cnn_model.add(MaxPooling2D(pool_size=(2, 2), padding="same"))
    cnn_model.add(Conv2D(128, kernel_size=(2, 2), input_shape=input_shape, padding="same"))
    cnn_model.add(MaxPooling2D(pool_size=(2, 2), padding="same"))
    cnn_model.add(Conv2D(128, kernel_size=(2, 2), input_shape=input_shape, padding="same"))
    cnn_model.add(MaxPooling2D(pool_size=(2, 2), padding="same"))
    cnn_model.add(Flatten())
    cnn_model.add(Dense(1024, activation='relu'))
    cnn_model.add(Dropout(dropout))
    cnn_model.add(Dense(num_classes, activation='softmax'))
    cnn_model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=learning_rate), metrics=['accuracy'])

    return cnn_model




def model_fit(cnn_model, traindata, trainlabel, model_file_name,  mode, val_img =0, val_label=0, epoch = 500, batch_size = 50):
    print("training..")
    early_stop = EarlyStopping(monitor='loss', patience=10, verbose=1)
    history = cnn_model.fit(traindata, trainlabel, epochs = epoch, verbose=2, callbacks=[early_stop],  batch_size = batch_size)
    #history = cnn_model.fit(traindata, trainlabel, epochs=epoch, verbose=2, batch_size=batch_size,
    #                        validation_data=(val_img, val_label))
    cnn_model.summary()

    predict = cnn_model.predict(traindata)
    print('Train loss:', history.history["loss"][-1])
    print('Train acc:', history.history["accuracy"][-1])

    fig, loss_ax = plt.subplots()
    acc_ax = loss_ax.twinx()

    loss_ax.plot(history.history['loss'], 'y', label='train loss')
    #loss_ax.plot(history.history['val_loss'], 'r', label='val loss')
    loss_ax.set_xlabel('epoch')
    loss_ax.set_ylabel('loss')
    loss_ax.legend(loc='upper left')

    acc_ax.plot(history.history['accuracy'], 'b', label='train acc')
    #acc_ax.plot(history.history['val_accuracy'], 'g', label='val acc')
    acc_ax.set_ylabel('accuracy')
    acc_ax.legend(loc='upper right')

    plt.show()

    if(mode == 'save'):
        model_json = cnn_model.to_json()
        with open("./model_data/" +  model_file_name + ".json", "w") as json_file:
            json_file.write(model_json)
        cnn_model.save_weights("./model_data/" + model_file_name + ".h5")
        print("Saved model to disk")

    return cnn_model




def model_pred(model_file_name, testdata, testlabel, num_classes = 4):
    json_file = open('./model_data/' + model_file_name + ".json", "r")
    loaded_model_json = json_file.read()
    json_file.close()
    cnn_model_test = model_from_json(loaded_model_json)
    cnn_model_test.load_weights('./model_data/' + model_file_name + ".h5")
    print("Loaded model from disk")
    print(cnn_model_test)

    test_each_data_numbers  = int((np.shape(testdata))[0]/num_classes)
    print(test_each_data_numbers)
    test_pred = cnn_model_test.predict(testdata)

    print('##############', test_pred.shape)
    print(test_pred)
    argmax_pred = np.argmax(test_pred, axis = 1)   # argmax로
    all_pred_result = []
    all_correct_num = []

    for j in range(num_classes):
        pred_frame = []
        label_frame = []
        correct_num = []
        for i in range(test_each_data_numbers):
            pred_frame.append(np.argmax(test_pred[i + (test_each_data_numbers * j)]))
            label_frame.append(np.argmax(testlabel[i + (test_each_data_numbers * j)]))
            correct_num.append(label_frame[i] == pred_frame[i])
        print("predict : ", pred_frame)  # , "correct num : ",correct_num.count(True))
        all_pred_result.append(pred_frame)
        all_correct_num.append(correct_num.count(True))

    print("correct number : ", all_correct_num)

    plt.plot(argmax_pred, label="pred")
    cnn_model_test.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.00001), metrics=['accuracy'])
    score = cnn_model_test.evaluate(testdata, testlabel, verbose=0, steps=50)
    print('Test loss:', score[0])
    print('Test acc:', score[1])

    clear_session()
    K.clear_session()
    del cnn_model_test

    plt.legend()
    plt.show()
