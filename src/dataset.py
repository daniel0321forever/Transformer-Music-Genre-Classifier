import os
import argparse
import logging
import sys
import numpy as np
import librosa
from glob import glob
import PIL.Image
from tqdm import tqdm
import pickle

from torch.utils.data import Dataset
from torch import save, Tensor
from torch.nn import functional as F

import whisper

# script_directory = os.path.dirname(os.path.abspath(sys.argv[0]))
# print(script_directory)


class MEL10(Dataset):

    def __init__(self) -> None:
        super().__init__()
        self.genre_map = {
            "blues": 0,
            "classical": 1,
            "country": 2,
            "disco": 3,
            "hiphop": 4,
            "jazz": 5,
            "metal": 6,
            "pop": 7,
            "reggae": 8,
            "rock": 9,
        }

        self.mfcc_data_list = []
        self.coarse_genre_list = []
        self.genre_list = []

    def to_mel(self, y: np.ndarray, sr: int):
        '''
        Get the processed mfcc data from sound wave

        @Returns:
        The mfcc data in (frames, bin) shape
        '''
        S = librosa.feature.mfcc(y=y, sr=sr)  # (bin, frames)
        S = (S - S.mean()) * 0.9 / S.std()

        S = S.transpose(0, 1)  # (frames, bin)
        return S

    def add_song(self, audio_dir):
        '''
        Convert the music data in a directory to target format (mfcc format)

        @Params
        audio_dir: The directory the contains the audio files that should be processed
        '''

        genre_dirs = os.listdir(audio_dir)

        for g_dir in genre_dirs:
            genre = os.path.basename(g_dir)
            print("Processing music genre", genre)

            song_files = os.listdir(os.path.join(audio_dir, g_dir))

            if song_files:
                for song in tqdm(song_files):
                    # iterate through the segments

                    # try:
                    song = os.path.join(audio_dir, g_dir, song)
                    y, sr = librosa.load(path=song)

                    for start in range(9):
                        for i in range(start, 100, 10):
                            # slice segments
                            if sr * (i + 10) > y.shape[0]:
                                break
                            segment = y[i * sr:(i+10) * sr]
                            self.mfcc_data_list.append(
                                self.to_mel(segment, sr))
                            self.genre_list.append(self.genre_map[genre])

                    # except:
                    #     logging.error(f"Errors occurs when processing audio file {song.split('/')[-1]}(segment {i})")
                    #     raise EOFError

        self.genre_list = F.one_hot(Tensor(self.genre_list).to(dtype=int))
        # print("One hot genre size", self.genre_list.shape)

    def __len__(self):
        return len(self.genre_list)

    def __getitem__(self, index):
        return self.mfcc_data_list[index], self.genre_list[index]


class WhisperMel(Dataset):

    def __init__(self) -> None:
        super().__init__()
        self.genre_map = {
            "blues": 0,
            "classical": 1,
            "country": 2,
            "disco": 3,
            "hiphop": 4,
            "jazz": 5,
            "metal": 6,
            "pop": 7,
            "reggae": 8,
            "rock": 9,
        }

        self.mfcc_data_list = []
        self.coarse_genre_list = []
        self.genre_list = []

    def to_mel(self, y: np.ndarray, sr: int):
        '''
        Get the processed mfcc data from sound wave

        @Returns:
        The mfcc data in (frames, bin) shape
        '''
        S = whisper.log_mel_spectrogram(y)

        return S

    def add_song(self, audio_dir):
        '''
        Convert the music data in a directory to target format (mfcc format)

        @Params
        audio_dir: The directory the contains the audio files that should be processed
        '''

        genre_dirs = os.listdir(audio_dir)
        window_len = 10

        for g_dir in genre_dirs:
            genre = os.path.basename(g_dir)
            print("Processing music genre", genre)

            song_files = os.listdir(os.path.join(audio_dir, g_dir))

            if song_files:
                for song in tqdm(song_files):
                    # iterate through the segments

                    # try:
                    song = os.path.join(audio_dir, g_dir, song)
                    y, sr = librosa.load(path=song, sr=16000)

                    pos = 0

                    for start in range(window_len - 1):
                        pos = start
                        while sr * (pos + window_len) < y.shape[0]:
                            segment = y[pos * sr:(pos+window_len) * sr]
                            self.mfcc_data_list.append(
                                self.to_mel(segment, sr))
                            self.genre_list.append(self.genre_map[genre])

                    # except:
                    #     logging.error(f"Errors occurs when processing audio file {song.split('/')[-1]}(segment {i})")
                    #     raise EOFError

        self.genre_list = F.one_hot(Tensor(self.genre_list).to(dtype=int))
        # print("One hot genre size", self.genre_list.shape)

    def __len__(self):
        return len(self.genre_list)

    def __getitem__(self, index):
        return self.mfcc_data_list[index], self.genre_list[index]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--dataset_dir',
                        help='specify the stored directory of the result dataset', required=True)
    parser.add_argument('-i', '--audio_dir', help='specify the input of the raw music data',
                        required=False, default='../dataset/GTZAN/genres_original')

    args = parser.parse_args()
    audio_dir = args.audio_dir
    output_path = args.dataset_dir

    if output_path[-4:] != ".pkl":
        output_path += ".pkl"

    dataset = WhisperMel()
    dataset.add_song(audio_dir)

    print("dataset length", len(dataset))
    print("MFCC data shape", dataset[0][0].shape)

    with open(output_path, 'wb') as f:
        pickle.dump(dataset, f)

    print("Finish file dumping")
