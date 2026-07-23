#!/bin/bash

python3 train.py --settings_file ablation/configs/without_CP/Indian_Pines.json
python3 train.py --settings_file ablation/configs/without_CP/PaviaU.json
python3 train.py --settings_file ablation/configs/without_CP/Salinas.json
python3 train.py --settings_file ablation/configs/without_CP/Houston.json