#!/bin/bash
#SBATCH -A aryaman.g
#SBATCH --mem=11000
#SBATCH --gres=gpu:1
#SBATCH --time=8-00:00:00
#SBATCH --mail-type=END

source /home/aryaman.g/packages/keras_tf_venv3/bin/activate
module load cuda/10.0
module load cudnn/7-cuda-10.0
python3 run_classifier.py --load mlstm_semeval.clf --fp16 --data movie_subtitles_csv/21_jump_street_subtitles.csv
mv clf_results.npy movie_subtitles_prediction/21_jump_street_results.npy
mv clf_results.npy.prob.npy movie_subtitles_prediction/21_jump_street_results_prob.npy

