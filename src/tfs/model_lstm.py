print("importing module......")
import numpy as np
from sys import getsizeof
import os
import shutil

import json
import librosa
from glob import glob

import pandas as pd
import tensorflow as tf
from keras.layers import (LSTM, Dense, Dropout, BatchNormalization as BatchNorm)
from keras import (Input, Model, optimizers, losses, metrics)
print("done")

class GLOB:
    categories = ['blackskirt', 'funky', 'jazz', 'rock', 'hiphop', 'edm', 'latin', 'acoustic', 'chinese', 'musical', 'country', 'r&b', 'chill']
    segment_len = 862
    spec_len = 1025
    model_weights_dir = 'model/model_weights'
    best_model_weight_path = model_weights_dir + "/best_model.h5"
    log_dir = 'model/model_log'


def analyse_audio_file(audio_filename):
    '''Get the audio file input and return a numpy array about its fft'''

    # get the fft data for the given audio file
    print("getting wave data of the audio......")
    song_wave = librosa.load(audio_filename)[0]
    print("done")

    print("doing the fft for the audio input......")
    song_fft = librosa.stft(song_wave)
    print("--> song_fft shape: ", song_fft.shape)
    print("done")

    print("processing numpy array......")
    song_fft = song_fft.astype(np.float32)
    song_fft = np.swapaxes(song_fft, 0, 1)
    print("--> song_fft transposed shape", song_fft.shape)
    print("done")

    print("slicing segment......")
    rows_num = song_fft.shape[0]
    if (rows_num > GLOB.segment_len):
        song_fft = song_fft[:GLOB.segment_len]
    if (rows_num < GLOB.segment_len):
        song_fft = np.vstack([song_fft, np.array([[0] * GLOB.spec_len] * (GLOB.segment_len -  rows_num))])
    print("--> song_fft trimmed shape", song_fft.shape)
    print("done")

    print("casting to list array......")
    song_fft = song_fft.tolist()
    print("--> song_fft converted type", type(song_fft))
    print("done")

    return song_fft


def arrange_music_files(label):
    '''Move the files in workplace to music file folder in correct format'''

    print("preparing music files......")
    if label not in GLOB.categories:
        print("The label is not defined")
        raise ValueError

    ans = input(f"the current label is {label}, are you sure this is the right label for these songs and want to continue? (y/n)")
    while(True):
        if ans == 'y' or ans == 'Y':
            song_files = glob("*.mp3")
            for song in song_files:
                # get dir name
                dir = os.path.join("music_file", song.split(".")[0])
                # create directory
                if os.path.exists(dir):
                    shutil.rmtree(dir)
                os.mkdir(dir)
                # move file to the directory
                shutil.move(song, os.path.join(dir, f"{label}.wav"))
            break
        elif ans == 'n':
            break
        else:
            ans = input("please enter (y/n)")
            continue



def grab_segment():
    '''Get the spectrum stream of the whole song from dataset, and return [[spectrum_0, spectrum_1, ...], ...] list'''
    
    # import module
    print("importing grab segment module......")
    from pychorus import find_and_output_chorus
    print("done")

    for dir in glob("music_file/*"):
        print(f"=================getting dir: {dir}==================")
        # locate the audio file
        try:
            print("searching for audio and label......")
            audio_filename = glob(f"{dir}/*")[0]
            label = audio_filename.split("/")[2].split(".")[0]
            print("--> song label", label)
            print("done")
        except IndexError as e:
            print(e)
            continue

        # extract the chorus segment
        print("extrating the chorus segment......")
        start = find_and_output_chorus(audio_filename, f"{dir}/chorus_segment.wav", 20)
        if not start:
            print("WARNING: can't find chorus")
            print("deleting the directory......")
            shutil.rmtree(dir)
            print("done")
            continue
        audio_filename = f"{dir}/chorus_segment.wav"
        print("--> audio file path", audio_filename)
        print("done")

        # get the fft conversion of the given audio file
        song_fft = analyse_audio_file(audio_filename)

        # label the song
        '''
        seeking for 
        ------------------------------------
        songtye 0 | 1 | 2 | 3 | ..... | n |
        song    0 | 0 | 0 | 1 | ......| 0 |
        ------------------------------------
        pd.get_dummies() is for a list of categories, for only one categrory, 
        it's hard to use the function for one hot coding
        '''
        print("labeling the song......")
        one_hot_dict = dict([(x, 0) for x in GLOB.categories])
        if label in GLOB.categories:
            one_hot_dict[label] += 1
        else:
            print("label is not defined in the categories")
            raise ValueError
        print("done")

        print("converting to list array......")
        one_hot_list = list(one_hot_dict.values())
        print("--> one hot encoding for the song", one_hot_list)
        print("done")

        # store it in a json file
        print("constructing the json file")
        json_dict = {
            'input': song_fft,
            'output': one_hot_list
        }
        json_file_name = f'dataset/{audio_filename.split("/")[1]}.json'
        with open(json_file_name, 'w') as json_file:
            json.dump(json_dict, json_file)
        print("done")

        # remove the directory
        print("deleting the directory......")
        shutil.rmtree(dir)
        print("done")


def get_light_effects():
    '''Get the corresponding RGB output and return [[RGB, RGB, ...], ...]'''
    pass


def prepare_data():
    '''Process the spectrum stream of the whole song and return the sequence list and output list'''
    dataset_files = glob("dataset/*.json")
    input = []
    output = []
    for data_file_name in dataset_files:
        print(f"preparing sequence for {data_file_name}.......")
        json_file = open(data_file_name, 'r')
        data = json.load(json_file)
        input.append(np.array(data['input']))
        output.append(np.array(data['output']))
        print("--> input shape", np.array(data['input']).shape)
        print("--> output shape", np.array(data['output']).shape)
        print("done")
    
    # prepare input and output
    print("preparing datasets......")
    input = np.array(input, dtype=np.float64)
    output = np.array(output, dtype=np.float64)
    print("--> input shape", input.shape)
    print("--> input type", input.dtype)
    print("--> output shape", output.shape)
    print("--> output type", output.dtype)
    print("done")

    print("slicing datas......")
    slice_point = round(input.shape[0] * 0.75)
    train_input = input[:slice_point]
    train_output = output[:slice_point]
    val_input = input[slice_point:]
    val_output = output[slice_point:]
    print("done")
    return train_input, train_output, val_input, val_output

def check_type():
    '''check the number of songs in each music genre'''
    dataset_files = glob("dataset/*")
    output_list = []
    for data_file_name in dataset_files:
        print("getting output......")
        json_file = open(data_file_name, 'r')
        data = json.load(json_file)
        output = data['output']
        print("--> output list", output)
        for i in range(len(GLOB.categories)):
          if output[i] == 1:
            output = i
            break
        print("--> output index", output)
        output = GLOB.categories[output]
        print("--> output category", output)
        output_list.append(output)
        print("done")
    
    # checking each type
    cat = dict((x, 0) for x in GLOB.categories)
    for song_type in output_list:
      cat[song_type] += 1

    print(cat)


def create_network():
    '''Create the neuron network'''

    print("constructing model......")
    input = Input(shape=(GLOB.segment_len, GLOB.spec_len))
    x = LSTM(256, return_sequences=True, recurrent_dropout=0.3)(input)
    x = LSTM(128)(x)
    x = BatchNorm()(x)
    x = Dropout(0.5)(x)
    x = Dense(64, activation='tanh')(x)
    x = BatchNorm()(x)
    x = Dropout(0.5)(x)
    x = Dense(16, activation='tanh')(x)
    x = Dropout(0.3)(x)
    output = Dense(len(GLOB.categories), activation='softmax')(x)

    model = Model(input, output, name="music_type_model")
    model.summary()
    print("done")
    # set learning rate schedule
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=1e-3,
        decay_steps=1000,
        decay_rate=0.9
    )
    # compile mode
    print("compiling model......")
    model.compile(optimizer=optimizers.Adam(learning_rate=lr_schedule), loss=losses.CategoricalCrossentropy(), metrics=[metrics.CategoricalAccuracy()])
    print("done")
    return model


def train(train_input, train_output, val_input, val_output, epochs=1500):
    '''Train the model'''
    print("importing model training module")
    from keras.callbacks import ModelCheckpoint
    from keras.callbacks import TensorBoard
    print("done")

    model = create_network()
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
        batch_size = 16,
        epochs = epochs,
        validation_data = (val_input, val_output),
        callbacks = cbk_list
    )
    return history



def train_network():
    '''The function will execute the whole training steps'''
    t_i, t_o, v_i, v_o = prepare_data()
    create_network()
    train(t_i, t_o, v_i, v_o)


def predict_type(audio_file_path):
    '''Use trained model to get the type of music'''
    print("predicting model......")
    print("==============================importing weights==============================")
    model = create_network()
    model.load_weights(GLOB.best_model_weight_path)
    print("====================================done=====================================")

    print("==============================preparing audio data===========================")
    audio_input = analyse_audio_file(audio_file_path)
    print("converting datatype to numpy......")
    audio_input = np.array(audio_input)
    print("done")
    print("reshaping numpy array......")
    audio_input = audio_input.reshape(1, 862, 1025)
    print("--> reshaped data shape", audio_input.shape)
    print("done")
    print("====================================done=====================================")

    print("predicting data......")
    cat = model.predict(x=audio_input)
    print("--> one hot categary of the audio file", cat)
    print("done")

    print("preparing output......")
    print("reverting to one dimensional array......")
    cat = cat.reshape(-1)
    print("--> converted shape", cat.shape)
    print("--> converted one hot code", cat)
    print('done')
    print("converting to pandas series......")
    cat = pd.Series(cat)
    print("done")
    print("finding the final music type......")
    music_type = cat.idxmax()
    print("done")
    audio_name = audio_file_path.split("/")[-1]
    print(f"The predicted music type of {audio_name} is ------------> \"{GLOB.categories[music_type]}\"!!")
    return music_type

train_network()