#!/bin/bash

python3 train.py --settings_file ablation/configs/ptemperature_mixup_scheduler=0.02/Indian_Pines.json
python3 train.py --settings_file ablation/configs/ptemperature_mixup_scheduler=0.02/PaviaU.json
python3 train.py --settings_file ablation/configs/ptemperature_mixup_scheduler=0.02/Salinas.json
python3 train.py --settings_file ablation/configs/ptemperature_mixup_scheduler=0.02/Houston.json
