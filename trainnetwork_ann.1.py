import numpy as np
from keras.utils import np_utils
import tensorflow as tf
from keras.layers import Dropout, Conv2D, MaxPooling2D, Input, Dense, Flatten, concatenate
from keras.models import Model
import keras
from keras.utils import plot_model
import time as tiimmee


def importdata(dataset):
    # Ｔhis function import the data from the database(csv file)
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
    numofjobs = int(max(labels1))+1
    numofmashine = int(max(labels2))+1
    inputnum = features.shape[1]

    # devide the input data, devide it to train data, test_data and left data
    datashape = data.shape
    num_train = int(datashape[0]*0.6)
    num_test = int(datashape[0]*0.2)

    train_feature = features[:num_train, :]*0.999
    train_label1 = labels1[:num_train]
    train_label2= labels2[:num_train]
    test_feature = features[num_train:num_test+num_train, :]*0.999
    test_label1 = labels1[num_train:num_test+num_train]
    test_label2 = labels2[num_train:num_test+num_train]

    # reshape the data
    train_feature = train_feature.reshape(train_feature.shape[0], inputnum)
    test_feature = test_feature.reshape(test_feature.shape[0], inputnum)
    train_label1 = train_label1.reshape(train_label1.shape[0], 1)
    train_label2 = train_label2.reshape(train_label2.shape[0], 1)
    test_label1 = test_label1.reshape(test_label1.shape[0], 1)
    test_label2 = test_label2.reshape(test_label2.shape[0], 1)

    # translate the label data into onehot shape
    train_label1 = np_utils.to_categorical(train_label1, numofjobs)
    train_label2 = np_utils.to_categorical(train_label2, numofmashine)
    test_label1 = np_utils.to_categorical(test_label1, numofjobs)
    test_label2 = np_utils.to_categorical(test_label2, numofmashine)

    train_label = [train_label1, train_label2]
    test_label = [test_label1, test_label2]

    return train_feature, train_label, test_feature, test_label, inputnum, numofjobs


def creatmodel_ann(inputnum, numOfJobs, layer=15):
    # This function is used to create the ann model in keras
    # 'inputnum' is number of input data
    # 'numOfJobs' is the number of class
    # Layer is the number of the hidden ann layers
    # The output model is the created model used to train
    # See http://keras-cn.readthedocs.io/en/latest/models/model/

    sess = tf.InteractiveSession()
    input1 = Input(shape=(inputnum, ), name='i1')
    mid = Dense(200, activation='relu', use_bias=True)(input1)
    for i in range(layer):
        mid = Dense(150, activation='relu', use_bias=True)(mid)
    mid = Dense(150, activation='relu', use_bias=True)(mid)
    mid = Dense(150, activation='relu', use_bias=True)(mid)
    mid = Dense(100, activation='relu', use_bias=True)(mid)
    mid = Dense(80, activation='relu', use_bias=True)(mid)
    mid = Dropout(0.25)(mid)
    mid = Dense(50, activation='relu', use_bias=True)(mid)
    mid = Dropout(0.25)(mid)
    mid = Dense(20, activation='relu', use_bias=True)(mid)
    out1 = Dense(numOfJobs, activation='softmax',
                name='out1', use_bias=True)(mid)
    out2 = Dense(numOfJobs, activation='softmax',
                name='out2', use_bias=True)(mid)

    model = Model(input1, [out1, out2])
    sgd = keras.optimizers.SGD(
        lr=0.11, momentum=0.0, decay=0.0, nesterov=False)

    model.compile(
        optimizer=sgd,
        loss='mean_squared_error',
        metrics=['accuracy'],
    )

    return model


def triannetwork(model, train_feature, train_label, test_feature, test_label,
                 batch_size, epochs):
    # network training function
    # see http://keras-cn.readthedocs.io/en/latest/models/model/

    model.fit(
        [train_feature], train_label, batch_size=batch_size, epochs=epochs,
        verbose=1, validation_data=([test_feature], test_label)
    )
    return model


def savenetwork(model, name):
    # This function is used to save the the medol trained above

    time_now = int(tiimmee.time())
    time_local = tiimmee.localtime(time_now)
    time = tiimmee.strftime("%Y_%m_%d::%H_%M_%S", time_local)
    print('current time is :', time)
    savename = './model/' + 'ann_schedual_' + time + name+'.h5'
    model.save(savename)
    return savename


def main(dataset = 'featureandlable_traindata_m=8_n=8_timelow=6_timehight=30_numofloop=1000.csv'):

    # ann parmater
    layerofann = 15


    # import the the data
    # dataset = 'featureandlable_traindata_m=8_n=8_timelow=6_timehight=30_numofloop=1000.csv'
    train_feature, train_label, test_feature, test_label, inputnum, numofjobs = importdata(
        dataset)
    print('successfully import the dataset :', dataset)
    print('the shape of train_feature is :', train_feature.shape)
    print('the shape of train_label is :', train_label[0].shape)
    print('the shape of test_feature is :', test_feature.shape)
    print('the shape of test_label is :', test_label[0].shape)

    # create the nn model
    model = creatmodel_ann(inputnum, numofjobs, layer=layerofann)
    plot_model(model, to_file='model.png')  # draw the figure of this model

    # training parmater
    batch_size = 49
    epochs = 100
    
    # train the network
    model = triannetwork(model, train_feature, train_label,
                         test_feature, test_label, batch_size, epochs)

    # save the model 
    savename = "ann_layer"+str(layerofann) + '_' + dataset
    savename = savenetwork(model, savename)
    return test_feature, test_label, savename

if __name__ == '__main__':
    test_feature, test_label, svaename = main()
