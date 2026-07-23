#!/bin/bash

python3 train.py --settings_file ablation/configs/source_dataset/Indian_Pines.json
python3 train.py --settings_file ablation/configs/source_dataset/PaviaU.json
python3 train.py --settings_file ablation/configs/source_dataset/Salinas.json
python3 train.py --settings_file ablation/configs/source_dataset/Houston.json
