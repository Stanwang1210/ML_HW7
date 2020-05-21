#!/bin/bash

#PBS -l select=1:ncpus=8:ngpus=1
#PBS -q v100
#PBS -o ./b07701209/HW7/pbs_output.o
#PBS -e ./b07701209/HW7/pbs_error
#PBS -m ea
#PBS -M b07701209@ntu.edu.tw

cd ./b07701209
#source ./ml_venv/bin/activate

cd ./HW7
# python3 src/hw7_knowledge_distillation.py
# python3 src/hw7_weight_quantization.py
# python3 src/hw7_validate.py
# python3 src/hw7_from_scratch.py
# python3 src/hw7_test.py ./food-11 ./predict.csv
# bash hw7_test.sh ./food-11 ./predict_2.csv
bash python3 HW7_Kaggle_Liu.py 

# cd ./hw9
# python3 src/hw9_unsupervised.py