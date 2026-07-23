#!/bin/bash

python3 train.py --settings_file ablation/configs/Label_propagation/alpha=0.5/Indian_Pines.json
python3 train.py --settings_file ablation/configs/Label_propagation/alpha=0.5/PaviaU.json
python3 train.py --settings_file ablation/configs/Label_propagation/alpha=0.95/Indian_Pines.json
python3 train.py --settings_file ablation/configs/Label_propagation/sigma=10/Indian_Pines.json
python3 train.py --settings_file ablation/configs/Label_propagation/sigma=100/Indian_Pines.json

python3 train.py --settings_file ablation/configs/Label_propagation/alpha=0.95/PaviaU.json
python3 train.py --settings_file ablation/configs/Label_propagation/sigma=10/PaviaU.json
python3 train.py --settings_file ablation/configs/Label_propagation/sigma=100/PaviaU.json

python3 train.py --settings_file ablation/configs/Label_propagation/alpha=0.95/Salinas.json
python3 train.py --settings_file ablation/configs/Label_propagation/alpha=0.5/Salinas.json
python3 train.py --settings_file ablation/configs/Label_propagation/sigma=10/Salinas.json
python3 train.py --settings_file ablation/configs/Label_propagation/sigma=100/Salinas.json

python3 train.py --settings_file ablation/configs/Label_propagation/alpha=0.95/Houston.json
python3 train.py --settings_file ablation/configs/Label_propagation/alpha=0.5/Houston.json
python3 train.py --settings_file ablation/configs/Label_propagation/sigma=10/Houston.json
python3 train.py --settings_file ablation/configs/Label_propagation/sigma=100/Houston.json

