cd src

python3 dataset.py -o ../dataset/train.pkl -i ../dataset/GTZAN/train
python3 dataset.py -o ../dataset/val.pkl -i ../dataset/GTZAN/val
python3 dataset.py -o ../dataset/test.pkl -i ../dataset/GTZAN/test