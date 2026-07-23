#!/bin/bash

python3 train.py --settings_file ablation/configs/Full_fine-tuning/Indian_Pines.json
python3 train.py --settings_file ablation/configs/Full_fine-tuning/PaviaU.json
python3 train.py --settings_file ablation/configs/Full_fine-tuning/Salinas.json
python3 train.py --settings_file ablation/configs/Full_fine-tuning/Houston.json
