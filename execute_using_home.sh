#!/bin/bash
#SBATCH -A aryaman.g
#SBATCH --gres=gpu:2
#SBATCH --mem=51000
#SBATCH --time=4-00:00:00
#SBATCH --mail-type=END

source ~/packages/keras_tf_venv3/bin/activate
module load cuda/10.0
module load cudnn/7-cuda-10.0

python pred_analysis.py --movie=21_jump_street --threshold=0.90
python main.py --epochs=100 --lr=0.001 --batch_size=64 --net="extendNet"  --save-checkpoint=0
