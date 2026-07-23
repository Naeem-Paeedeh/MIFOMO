#!/bin/bash

python3 train.py --settings_file ablation/configs/One_CP_for_all_heads/Indian_Pines.json
python3 train.py --settings_file ablation/configs/One_CP_for_all_heads/PaviaU.json
python3 train.py --settings_file ablation/configs/One_CP_for_all_heads/Salinas.json
python3 train.py --settings_file ablation/configs/One_CP_for_all_heads/Houston.json
