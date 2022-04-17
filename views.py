def classifier():
    model = Sequential()

    #trainData = tensorflow.keras.preprocessing.image_dataset_from_directory(
    #    directory='TrainingData/',
    #    labels='inferred',
    #    label_mode='int',
    #    color_mode="grayscale",
    #    batch_size=5,
    #    image_size=(64, 64))
    #testData = tensorflow.keras.preprocessing.image_dataset_from_directory(
    #    directory='TrainingData/',
    #    labels='inferred',
    #    label_mode='int',
    #    color_mode="grayscale",
    #    batch_size=5,
    #    image_size=(64, 64))


    #trainDataImage = np.concatenate([ x for x, y in trainData ], axis=0)
    #trainDataLabel = np.concatenate([ y for x, y in trainData ], axis=0)

    #testDataImage  = np.concatenate([ x for x, y in testData  ], axis=0)
    #testDataLabel  = np.concatenate([ y for x, y in testData  ], axis=0)


    #model.add(Conv2D(60, kernel_size = 1, activation='relu', input_shape=(64, 64, 1), padding='same'))
    #model.add(Dropout(0.2))

    #model.add(Conv2D(35, kernel_size = 1, activation='relu', padding='same'))
    #model.add(Dropout(0.2))

    model.add(MaxPool2D(2, 2))

    model.add(Conv2D(20, kernel_size = 1, activation='relu', padding='same'))
    model.add(Dropout(0.2))

    model.add(MaxPool2D(2, 2))

    model.add(Conv2D(10, kernel_size = 1, activation='relu', padding='same'))
    model.add(Dropout(0.2))

    model.add(Flatten())

    model.add(Dense(2, activation='softmax'))

    model.summary()

    def Reshaper(var, imNumber, isImage):
        if(isImage==True):
            np.reshape(var, (imNumber, 64, 64, 1))
        else:
            np.reshape(var, (imNumber, 1))
        return var

    trainDataImage = Reshaper(trainDataImage, 500, True)
    trainDataLabel = Reshaper(trainDataLabel, 500, False)

    def OneHotEncode(DataLabel, labelNum):
        OneHot = np.zeros((labelNum, 2))
        count=0

        while (count < labelNum):
            OneHot[count][DataLabel[count].astype(int)] = 1
            count=count+1

        return OneHot

    trainDataLabel = OneHotEncode(trainDataLabel, 500)

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    model.fit(trainDataImage, trainDataLabel, epochs=2, batch_size=25)

    testDataImage = Reshaper(testDataImage, 500, True)
    testDataLabel = Reshaper(testDataLabel, 500, False)
    testDataLabel = OneHotEncode(testDataLabel, 500)

    output = model.evaluate(testDataImage, testDataLabel, verbose=True, batch_size=5)

    print("The model loss and accuracy respectively:", output)

    model.save('PneumoniaModel.h5')



from flask import Flask, redirect, url_for, render_template, request, session, flash, send_file
from datetime import timedelta
from flask_wtf import FlaskForm
from wtforms import FileField, SubmitField
from flask import Flask, redirect, url_for, render_template, request, session, flash, send_file
from datetime import timedelta
from flask_wtf import FlaskForm
from wtforms import FileField, SubmitField
from werkzeug.utils import secure_filename
import os 
import tensorflow
import keras
from keras.layers import Dense
from keras.layers import Input
from keras.layers import Dropout
from keras.layers import Conv2D
from keras.layers import Flatten
from keras.layers import MaxPool2D
from keras.models import Sequential
import numpy as np
import image_dataset_loader
import csv