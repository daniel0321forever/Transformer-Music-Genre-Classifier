import logging
import argparse
import os

import pickle
from tqdm import tqdm, trange
import librosa

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

from model.transformer_model import WhisperClassifier
from model.conv1d import ResCNN
from dataset import MEL10
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import matplotlib.pyplot as plt

genre_map = [
    "blues",
    "classical",
    "country",
    "disco",
    "hiphop",
    "jazz",
    "metal",
    "pop",
    "reggae",
    "rock",
]

config = {
    "trainset_path": "../dataset/train.pkl",
    "valset_path": "../dataset/val.pkl",
    "testset_path": "../dataset/test.pkl",
    "weight_path": "../models/rescnn/best.ckpt",
    "logger_path": "../logger/rescnn",
    "confusion_matrix_path": "../performance/rescnn.png",


    "batch_size": 32,
    "max_epoch": 400,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "lr": 1e-4,
}


def get_param_num(model):
    num_param = sum(param.numel() for param in model.parameters())
    return num_param


def init_weights(m):
    if isinstance(m, nn.Module):
        torch.nn.init.xavier_uniform_(m.weight)


def train(model):
    # check dir
    if not os.path.exists(os.path.dirname(config['weight_path'])):
        os.mkdir(os.path.dirname(config['weight_path']))

    if not os.path.exists(config['logger_path']):
        os.mkdir(config['logger_path'])

    # Get model
    model = model.to(config["device"])
    # model.apply(init_weights)
    print("Number of params:", get_param_num(model))

    # Get dataset
    print("Loading dataset")
    with open(config['trainset_path'], 'rb') as dataset_file:
        train_set = pickle.load(dataset_file)

    with open(config["valset_path"], 'rb') as dataset_file:
        val_set = pickle.load(dataset_file)

    train_loader = DataLoader(
        train_set, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(
        val_set, batch_size=config['batch_size'], shuffle=True)

    # optimizer
    optimizer = torch.optim.AdamW([
        {'params': model.parameters(), 'weight_decay': 1e-9},
    ],
        lr=config['lr'],
        betas=(0.9, 0.999),
        eps=1e-9)

    # writer
    writer = SummaryWriter(log_dir=config['logger_path'])

    # start training
    print("start training")
    min_val_loss = 1000000

    for epoch in range(config['max_epoch']):
        print(f"----- Epoch {epoch} -----")
        model.train()
        loss_log = 0
        acc_log = 0

        for idx, batch in tqdm(enumerate(train_loader)):
            # suppose MEL26 dataset
            mfcc_input = batch[0].to(config['device'])

            # fine-genre in compare to coarse genre
            f_genre = batch[1].to(device=config['device'], dtype=torch.float32)
            f_genre_pred = model(mfcc_input).to(config['device'])

            # find loss and acc
            crit = torch.nn.CrossEntropyLoss()
            loss = crit(f_genre_pred, f_genre)
            loss_log += loss.item()
            acc_log += (f_genre.argmax(dim=-1) ==
                        f_genre_pred.argmax(dim=-1)).float().mean().item()

            # backward propogation
            loss.backward()
            nn.utils.clip_grad_norm_(
                model.parameters(), 1.0)

            optimizer.step()

        loss_log = loss_log / len(train_loader)
        acc_log = acc_log / len(train_loader)

        # validation
        model.eval()
        with torch.no_grad():
            val_loss_log = 0
            val_acc_log = 0
            for idx, batch in enumerate(val_loader):
                mfcc_input = batch[0].to(config['device'])
                # fine-genre in compare to coarse genre
                f_genre = batch[1].to(
                    device=config['device'], dtype=torch.float32)
                f_genre_pred = model(mfcc_input).to(config['device'])

                # find loss and acc
                crit = torch.nn.CrossEntropyLoss()
                loss = crit(f_genre_pred, f_genre)
                val_loss_log += loss.item()
                val_acc_log += (f_genre.argmax(dim=-1) ==
                                f_genre_pred.argmax(dim=-1)).float().mean().item()

        val_loss_log = val_loss_log / len(val_loader)
        val_acc_log = val_acc_log / len(val_loader)

        # save model
        if val_loss_log / len(val_loader) < min_val_loss:
            min_val_loss = val_loss_log / len(val_loader)

            print("saving model")
            torch.save(model.state_dict(), config['weight_path'])

            print("Train loss |", loss_log)
            print("Train Accuracy |", acc_log)
            print("Val loss |", val_loss_log)
            print("Val Accuracy |", val_acc_log)

        writer.add_scalars(
            "loss", {"train": loss_log, "valid": val_loss_log}, epoch)
        writer.add_scalars(
            "acc", {"train": acc_log, "valid": val_acc_log}, epoch)


def test(model):
    pred_list = []
    label_list = []

    audio_files = []
    for genre in genre_map:
        for index in range(10):
            path = f"../dataset/GTZAN/test/{genre}/{genre}.000{index}0.wav"
            audio_files.append(path)

    model_path = config["weight_path"]
    slice = 10

    checkpoint = torch.load(model_path, map_location=config['device'])
    model.load_state_dict(checkpoint, strict=True)
    model.eval()

    tot_score = 0
    for audio in audio_files:

        score_board = torch.zeros(10)
        y, sr = librosa.load(audio)

        for start in range(slice - 1):
            for i in range(start, 100, slice):
                # slice segments
                if sr * (i + slice) > y.shape[0]:
                    break

                segment = y[i * sr:(i+slice) * sr]

                S = librosa.feature.mfcc(y=segment, sr=sr)  # (bin, frames)
                S = (S - S.mean()) * 0.9 / S.std()

                S = S.transpose(0, 1)  # (frames, bin)

                model_input = torch.Tensor(S).unsqueeze(0)
                genre = model(model_input)
                genre = int(genre.argmax(dim=-1).item())
                score_board[genre] += 1

        genre = int(score_board.argmax(dim=-1).item())
        genre = genre_map[genre]
        pred_list.append(genre)
        print("PREDICTION:", genre)

        label = audio.split("/")[-2]
        label_list.append(label)
        print("LABLE:", label, end='\n')
        print("="*15)

        if genre == label:
            tot_score += 1

    print(f"acc: {tot_score * 100/ len(audio_files)}%")

    matrix = confusion_matrix(label_list, pred_list, labels=genre_map)
    display = ConfusionMatrixDisplay(matrix).plot()
    plt.savefig(config["confusion_matrix_path"])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--mode", required=True,
                        help="[train, test, predict]")

    args = parser.parse_args()
    mode = args.mode

    model = ResCNN()
    if mode == 'train':
        train(model=model)

    elif mode == 'test':
        test(model=model)
