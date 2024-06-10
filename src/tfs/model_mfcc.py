import librosa
import librosa.display
import numpy as np

import tensorflow as tf
from keras import (layers, optimizers, metrics, Model, losses)
from keras.callbacks import (ModelCheckpoint, TensorBoard)
from keras.applications import InceptionV3

import os
import pylab
import PIL.Image
from glob import glob
import shutil

# directory
mfcc_dir = 'mfcc_datas/mfcc_fig' # the relative directory where music graph datas are stored
model_dir = 'models/6_classes_model' # the relative directory to output the model


music_dir = 'raw_musics/musc_country' # the file path to get raw music
train_options = ["2 classes", "6 classes", "subclasses"]
select_option = train_options[1]


class GLOB:
    if select_option == train_options[0]:
        categories = ['edm', 'acoustic']
        output_dim = 2
    elif select_option == train_options[1]:
        categories = ['edm', 'lovesong', 'country', 'r&b', 'rock', 'hiphop']
        output_dim = 6
    else:
        categories = None
        output_dim = None
        print("please select a correct train option to have appropriate categories and output dim")
        raise ValueError

    model_weights_dir = None
    best_model_weight_path = None
    log_dir = None
    model = None
    

def model_selector(model_dir=''):
    '''select which model in the file system you want to choose'''
    if not model_dir:
        model_dir = input('please input your model model_dir')
    GLOB.model_weights_dir = f"{model_dir}/model_weights"
    GLOB.best_model_weight_path = f"{GLOB.model_weights_dir}/best.h5"
    GLOB.log_dir = f"{model_dir}/model_log"
    
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

# inception_V3
def __inception_V3():
    input_shape = (480, 640, 3)
    model = InceptionV3(include_top=True, input_shape=input_shape, weights=None, classes=GLOB.output_dim)
    # construct lr schedule
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=1e-3,
        decay_steps=1000,
        decay_rate=0.9
        )

    model.compile(optimizer=optimizers.Adam(learning_rate=lr_schedule), loss=losses.CategoricalCrossentropy(), metrics=[metrics.CategoricalAccuracy()])

    return model

# model compile
def cnn_model():
    input_shape = (480, 640, 3)
    input = layers.Input(input_shape)

    X = layers.Conv2D(8,kernel_size=(3,3),strides=(1,1))(input)
    X = layers.BatchNormalization(axis=3)(X)
    X = layers.Activation('relu')(X)
    X = layers.MaxPooling2D((2,2))(X)
    
    X = layers.Conv2D(16,kernel_size=(3,3),strides = (1,1))(X)
    X = layers.BatchNormalization(axis=3)(X)
    X = layers.Activation('relu')(X)
    X = layers.MaxPooling2D((2,2))(X)

    X = layers.Conv2D(64,kernel_size=(3,3),strides=(1,1))(X)
    X = layers.BatchNormalization(axis=-1)(X)
    X = layers.Activation('relu')(X)
    X = layers.MaxPooling2D((2,2))(X)
    
    X = layers.Conv2D(128,kernel_size=(3,3),strides=(1,1))(X)
    X = layers.BatchNormalization(axis=-1)(X)
    X = layers.Activation('relu')(X)
    X = layers.MaxPooling2D((2,2))(X)

    
    X = layers.Flatten()(X)
    
    X = layers.Dropout(rate=0.3)(X)

    output = layers.Dense(GLOB.output_dim, activation='softmax')(X)

    # compile model
    model = Model(input, output, name='music_type_model')

    # construct lr schedule
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=1e-3,
        decay_steps=1000,
        decay_rate=0.9
        )

    model.compile(optimizer=optimizers.Adam(learning_rate=lr_schedule), loss=losses.CategoricalCrossentropy(), metrics=[metrics.CategoricalAccuracy()])

    return model


def train(train_input, train_output, val_input, val_output, epochs):
    model = __inception_V3()
    # make file path
    best_model_file = os.path.join(GLOB.best_model_weight_path)
    if not os.path.exists(GLOB.model_weights_dir):
        os.mkdir(GLOB.model_weights_dir)
    
    if not os.path.exists(GLOB.log_dir):
        os.mkdir(GLOB.log_dir)
    
    # make the callback list
    checkpoint = ModelCheckpoint(
        filepath=best_model_file,
        monitor="val_categorical_accuracy",
        save_best_only=True,
        mode='max'
    )

    tensorboard = TensorBoard(
        log_dir=GLOB.log_dir
    )
    cbk_list = [checkpoint, tensorboard]

    # train the model
    history = model.fit(
        train_input, train_output,
        validation_data=[val_input, val_output],
        batch_size=32,
        epochs=epochs,
        callbacks=cbk_list
    )
    return history


def train_network(model_dir=model_dir, epochs=200):
    model_selector(model_dir) # selet the output dir
    t_i, t_o, v_i, v_o = prepare_data()
    train(t_i, t_o, v_i, v_o, epochs=epochs)
    

def two_classes_predict(song, model=''):   
    # select model and load weight
    model_selector(model_dir=model)
    model = GLOB.model
    model.load_weights(GLOB.best_model_weight_path)

    # get mfcc image
    img_path = __get_mfcc(audio_filename=song, mfcc_name=song.split('/')[-1].split('.')[0], label='Test', output_dir="test_fig")
    print(img_path)

    # iterate through each data
    output = 0
    for img in img_path:
        img_array = __read_image(img) # !! might directly read the whole package and make one ndarray
        input = np.reshape(img_array, (1, 480, 640, 3))
        output += model.predict(input)[0][0] # two dim model [first_data_result, second_data_result]
        os.remove(img)

    avg_output = output / len(img_path)    
    print("The average output is", avg_output)
    if avg_output > 0.5:
        print(f"\n{song} is edm\n")
        return "edm"
    else:
        print(f"\n{song} is not edm\n")
        return "acoustic"

train_network(epochs=200)


