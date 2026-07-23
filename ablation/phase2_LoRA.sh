#!/bin/bash

python3 train.py --settings_file ablation/configs/LoRA/Indian_Pines.json
python3 train.py --settings_file ablation/configs/LoRA/PaviaU.json
python3 train.py --settings_file ablation/configs/LoRA/Salinas.json
python3 train.py --settings_file ablation/configs/LoRA/Houston.json
