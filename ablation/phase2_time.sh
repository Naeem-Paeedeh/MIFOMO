#!/bin/bash

python3 train.py --settings_file ablation/configs/Time/Indian_Pines_CP.json
python3 train.py --settings_file ablation/configs/Time/PaviaU_CP.json
python3 train.py --settings_file ablation/configs/Time/Salinas_CP.json
python3 train.py --settings_file ablation/configs/Time/Houston_CP.json

python3 train.py --settings_file ablation/configs/Time/Indian_Pines_LoRA.json
python3 train.py --settings_file ablation/configs/Time/PaviaU_LoRA.json
python3 train.py --settings_file ablation/configs/Time/Salinas_LoRA.json
python3 train.py --settings_file ablation/configs/Time/Houston_LoRA.json

python3 train.py --settings_file ablation/configs/Time/Indian_Pines_Prompt.json
python3 train.py --settings_file ablation/configs/Time/PaviaU_Prompt.json
python3 train.py --settings_file ablation/configs/Time/Salinas_Prompt.json
python3 train.py --settings_file ablation/configs/Time/Houston_Prompt.json

python3 train.py --settings_file ablation/configs/Time/Indian_Pines_full_fine-tuning.json
python3 train.py --settings_file ablation/configs/Time/PaviaU_full_fine-tuning.json
python3 train.py --settings_file ablation/configs/Time/Salinas_full_fine-tuning.json
python3 train.py --settings_file ablation/configs/Time/Houston_full_fine-tuning.json
