#!/bin/bash

python3 train.py --settings_file ablation/configs/ptemperature_mixup_scheduler=0.1/Indian_Pines.json
python3 train.py --settings_file ablation/configs/ptemperature_mixup_scheduler=0.1/PaviaU.json
python3 train.py --settings_file ablation/configs/ptemperature_mixup_scheduler=0.1/Salinas.json
python3 train.py --settings_file ablation/configs/ptemperature_mixup_scheduler=0.1/Houston.json
