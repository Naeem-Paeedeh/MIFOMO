#!/bin/bash

python3 train.py --settings_file ablation/configs/PCA_Bands_30/Indian_Pines.json
python3 train.py --settings_file ablation/configs/PCA_Bands_30/PaviaU.json
python3 train.py --settings_file ablation/configs/PCA_Bands_30/Salinas.json
python3 train.py --settings_file ablation/configs/PCA_Bands_30/Houston.json

python3 train.py --settings_file ablation/configs/PCA_Bands_80/Indian_Pines.json
python3 train.py --settings_file ablation/configs/PCA_Bands_80/PaviaU.json
python3 train.py --settings_file ablation/configs/PCA_Bands_80/Salinas.json
python3 train.py --settings_file ablation/configs/PCA_Bands_80/Houston.json

python3 train.py --settings_file ablation/configs/PCA_Bands_100/Indian_Pines.json
python3 train.py --settings_file ablation/configs/PCA_Bands_100/PaviaU.json
python3 train.py --settings_file ablation/configs/PCA_Bands_100/Salinas.json
python3 train.py --settings_file ablation/configs/PCA_Bands_100/Houston.json
