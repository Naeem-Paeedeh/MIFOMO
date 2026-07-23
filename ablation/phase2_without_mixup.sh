#!/bin/bash

python3 train.py --settings_file ablation/configs/without_mixup/Indian_Pines.json
python3 train.py --settings_file ablation/configs/without_mixup/PaviaU.json
python3 train.py --settings_file ablation/configs/without_mixup/Salinas.json
python3 train.py --settings_file ablation/configs/without_mixup/Houston.json