#!/bin/bash

python3 train.py --settings_file ablation/configs/main_components_are_disabled/Indian_Pines.json
python3 train.py --settings_file ablation/configs/main_components_are_disabled/PaviaU.json
python3 train.py --settings_file ablation/configs/main_components_are_disabled/Salinas.json
python3 train.py --settings_file ablation/configs/main_components_are_disabled/Houston.json
