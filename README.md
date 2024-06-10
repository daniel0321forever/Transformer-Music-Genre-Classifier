# Computer vision on music genre classification
Apply computer vision model to classify the music genre by their MFCC cepstrum diagram.

## Model Training
The model are using GTZAN dataset. To train the model, one should prepare **GTZAN** dataset and slice the dataset into `trian, valid, test` folders manually.

After setting up, one could train the model by typing `python3 pipeline.py --mode train` in ``src`` directory, and test the performance by typing `python3 pipelin3.py --model test`
in the same directory. The configuration could be set in the `config` variable in pipeline.py.
