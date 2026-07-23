#!/bin/bash

python3 train.py --settings_file ablation/configs/perturbation_range=0.3/Indian_Pines.json
python3 train.py --settings_file ablation/configs/perturbation_range=0.3/PaviaU.json
python3 train.py --settings_file ablation/configs/perturbation_range=0.3/Salinas.json
python3 train.py --settings_file ablation/configs/perturbation_range=0.3/Houston.json
