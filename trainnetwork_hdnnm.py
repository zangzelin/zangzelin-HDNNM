import time as timetool

import keras
import numpy as np
import tensorflow as tf
from keras import optimizers
from keras.layers import (Conv2D, Dense, Dropout, Flatten, Input, MaxPooling2D,
                          concatenate)
from keras.models import Model
from keras.utils import np_utils, plot_model


def importdata(dataset):
    # This function import the data from the database(csv file)
    # And return the feature and the label of the problem
    # 'dataset' is the name of the file
    # 'train_feature, train_label, test_feature, test_label'
    # is training data
    # 'inputnum' is number of input data
    # 'nb_classes' is the number of class

    # load the data
    data = np.loadtxt('./data/'+dataset,
                      dtype=float, delimiter=',')
    features = data[:, 1:-2]
    labels1 = data[:, -2]
    labels2 = data[:, -1]
    num_of_jobs = int(max(labels1))+1
    num_of_machine = int(max(labels2))+1
    inputnum = features.shape[1]

    # divide the input data, divide it to train data, test_data and left data
    datashape = data.shape
    num_train = int(datashape[0]*0.6)
    num_test = int(datashape[0]*0.2)

    independent_input = 8
    twodimensional_input = 8*8
    m = 8
    n = 8

    train_feature1 = features[:num_train, :independent_input]*0.999
    train_feature2 = features[:num_train,
                              independent_input:independent_input+twodimensional_input]*0.999
    train_feature3 = features[:num_train, independent_input +
                              twodimensional_input:independent_input+twodimensional_input*2]*0.999
    train_label1 = labels1[:num_train]
    train_label2 = labels2[:num_train]
    test_feature1 = features[num_train:num_test +
                             num_train, :independent_input]*0.999
    test_feature2 = features[num_train:num_test+num_train,
                             independent_input:independent_input+twodimensional_input]*0.999
    test_feature3 = features[num_train:num_test+num_train, independent_input +
                             twodimensional_input:independent_input+twodimensional_input*2]*0.999
    test_label1 = labels1[num_train:num_test+num_train]
    test_label2 = labels2[num_train:num_test+num_train]

    # reshape the data
    train_feature1 = train_feature1.reshape(
        train_feature1.shape[0], independent_input, 1)
    train_feature2 = train_feature2.reshape(train_feature2.shape[0], m, n, 1)
    train_feature3 = train_feature3.reshape(train_feature3.shape[0], m, n, 1)

    test_feature1 = test_feature1.reshape(
        test_feature1.shape[0], independent_input, 1)
    test_feature2 = test_feature2.reshape(test_feature2.shape[0],  m, n, 1)
    test_feature3 = test_feature3.reshape(test_feature3.shape[0],  m, n, 1)

    train_label1 = train_label1.reshape(train_label1.shape[0], 1)
    train_label2 = train_label2.reshape(train_label2.shape[0], 1)
    test_label1 = test_label1.reshape(test_label1.shape[0], 1)
    test_label2 = test_label2.reshape(test_label2.shape[0], 1)

    # translate the label data into onehot shape
    train_label1 = np_utils.to_categorical(train_label1, num_of_jobs)
    train_label2 = np_utils.to_categorical(train_label2, num_of_machine)
    test_label1 = np_utils.to_categorical(test_label1, num_of_jobs)
    test_label2 = np_utils.to_categorical(test_label2, num_of_machine)

    train_feature = [train_feature1, train_feature2, train_feature3]
    test_feature = [test_feature1, test_feature2, test_feature3]
    train_label = train_label1
    test_label = test_label1

    return train_feature, train_label, test_feature, test_label, inputnum, num_of_jobs


def CreateModelAnn(inputnum, numOfJobs, layer=15):
    # This function is used to create the ann model in keras
    # 'inputnum' is number of input data
    # 'numOfJobs' is the number of class
    # Layer is the number of the hidden ann layers
    # The output model is the created model used to train
    # See http://keras-cn.readthedocs.io/en/latest/models/model/

    sess = tf.InteractiveSession()
    # 
    input1 = Input(shape=(8, 1), name='in1')
    input2 = Input(shape=(numOfJobs, numOfJobs, 1), name='in2')
    input3 = Input(shape=(numOfJobs, numOfJobs, 1), name='in3')

    # ???
    mid1 = Dense(64, use_bias=True)(input1)
    mid1 = Dense(64, use_bias=True)(mid1)
    mid1 = Dense(64, use_bias=True)(mid1)
    out1 = Dense(64, use_bias=True)(mid1)
    out1 = Flatten()(out1)

    # ???
    mid2 = Conv2D(64, (3, 3), padding='same')(input2)
    mid2 = Conv2D(64, (3, 3), padding='same')(mid2)
    mid2 = Conv2D(64, (3, 3), padding='same')(mid2)
    out2 = Conv2D(64, (3, 3), padding='same')(mid2)
    out2 = Flatten()(out2)

    # ???
    mid3 = Conv2D(64, (3, 3), padding='same')(input3)
    mid3 = Conv2D(64, (3, 3), padding='same')(mid3)
    mid3 = Conv2D(64, (3, 3), padding='same')(mid3)
    out3 = Conv2D(64, (3, 3), padding='same')(mid3)
    out3 = Flatten()(out3)

    # ??
    concatenated = concatenate([out1, out2, out3])
    mid = Dense(100, activation='relu', use_bias=True)(concatenated)
    mid = Dense(100, activation='relu', use_bias=True)(mid)
    mid = Dense(100, activation='relu', use_bias=True)(mid)
    mid = Dense(100, activation='relu', use_bias=True)(mid)
    mid = Dropout(0.25)(mid)
    mid = Dense(49, activation='relu', use_bias=True)(mid)
    mid = Dropout(0.25)(mid)
    mid = Dense(100, activation='relu', use_bias=True)(mid)
    out = Dense(numOfJobs, activation='softmax',
                name='out', use_bias=True)(mid)

    model = Model([input1, input2, input3], out)
    sgd = optimizers.SGD(lr=0.05)

    model.compile(
        optimizer='sgd',
        loss='mean_squared_error',
        metrics=['accuracy'],

    )
    return model


def TrainNetwork(model, train_feature, train_label, test_feature, test_label,
                 batch_size, epochs):
    # network training function
    # see http://keras-cn.readthedocs.io/en/latest/models/model/

    model.fit(
        train_feature, train_label, batch_size=batch_size, epochs=epochs,
        verbose=1, validation_data=(test_feature, test_label)
    )
    return model


def savenetwork(model, name):
    # This function is used to save the the medol trained above

    time_now = int(timetool.time())
    time_local = timetool.localtime(time_now)
    time = timetool.strftime("%Y_%m_%d::%H_%M_%S", time_local)
    print('current time is :', time)
    savename = './model/' + 'ann_schedule_' + time + name+'.h5'
    model.save(savename)
    return savename


def main(dataset='featureandlable_traindata_m=8_n=8_timelow=6_timehight=30_numofloop=1000.csv'):

    # ann parmater
    layer_of_cnn = 15

    # import the the data
    # dataset = 'featureandlable_traindata_m=8_n=8_timelow=6_timehight=30_numofloop=1000.csv'
    train_feature, train_label, test_feature, test_label, inputnum, num_of_jobs = importdata(
        dataset)
    print('successfully import the dataset :', dataset)
    print('the shape of train_feature is :',
          train_feature[0].shape, train_feature[1].shape, train_feature[2].shape)
    print('the shape of train_label is :', train_label[0].shape)
    print('the shape of test_feature is :',
          test_feature[0].shape, test_feature[1].shape, test_feature[2].shape)
    print('the shape of test_label is :', test_label[0].shape)

    # create the nn model
    model = CreateModelAnn(inputnum, num_of_jobs, layer=layer_of_cnn)
    plot_model(model, to_file='model.png')  # draw the figure of this model

    # training parmater
    batch_size = 49
    epochs = 100

    # train the network
    model = TrainNetwork(model, train_feature, train_label,
                         test_feature, test_label, batch_size, epochs)

    # save the model
    savename = "hdnnm_layer"+str(layer_of_cnn) + '_' + dataset
    savename = savenetwork(model, savename)
    return test_feature, test_label, savename


if __name__ == '__main__':
    test_feature, test_label, svaename = main()
