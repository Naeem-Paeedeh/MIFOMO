#!/bin/bash

python3 train.py --settings_file ablation/configs/prompt/Indian_Pines.json
python3 train.py --settings_file ablation/configs/prompt/PaviaU.json
python3 train.py --settings_file ablation/configs/prompt/Salinas.json
python3 train.py --settings_file ablation/configs/prompt/Houston.json
