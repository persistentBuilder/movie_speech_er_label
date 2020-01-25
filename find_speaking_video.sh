#!/bin/bash
#SBATCH -A aryaman.g
#SBATCH --mem=11000
#SBATCH --time=8-00:00:00
#SBATCH --mail-type=END

source /home/aryaman.g/packages/keras_tf_venv3/bin/activate
python pred_analysis.py --movie=21_jump_street

