#!/bin/bash

python3 train.py --settings_file ablation/configs/without_CP/Indian_Pines.json
python3 train.py --settings_file ablation/configs/without_CP/PaviaU.json
python3 train.py --settings_file ablation/configs/without_CP/Salinas.json
python3 train.py --settings_file ablation/configs/without_CP/Houston.json

python3 train.py --settings_file ablation/configs/without_mixup/Indian_Pines.json
python3 train.py --settings_file ablation/configs/without_mixup/PaviaU.json
python3 train.py --settings_file ablation/configs/without_mixup/Salinas.json
python3 train.py --settings_file ablation/configs/without_mixup/Houston.json

python3 train.py --settings_file ablation/configs/main_components_are_disabled/Indian_Pines.json
python3 train.py --settings_file ablation/configs/main_components_are_disabled/PaviaU.json
python3 train.py --settings_file ablation/configs/main_components_are_disabled/Salinas.json
python3 train.py --settings_file ablation/configs/main_components_are_disabled/Houston.json