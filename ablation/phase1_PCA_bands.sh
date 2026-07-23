#!/bin/bash

python3 train.py --settings_file ablation/configs/PCA_Bands_30/phase_1.json
python3 train.py --settings_file ablation/configs/PCA_Bands_80/phase_1.json
python3 train.py --settings_file ablation/configs/PCA_Bands_100/phase_1.json
