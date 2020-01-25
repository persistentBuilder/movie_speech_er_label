#!/bin/bash
#SBATCH -A aryaman.g
#SBATCH --mem=11000
#SBATCH --nodelist=gnode03
#SBATCH --gres=gpu:1
#SBATCH --time=8-00:00:00
#SBATCH --mail-type=END

source /home/aryaman.g/packages/keras_tf_venv3/bin/activate
module load cuda/10.0
module load cudnn/7-cuda-10.0

movie_name=IT
movie_base_path=/ssd_scratch/cvit/aryaman.g/movies/$movie_name
clips_dir=${movie_base_path}/${movie_name}-clips
path_for_clips=$clips_dir

python visual_pred_analysis.py  --model-path="/home/aryaman.g/projects/all_code/simple_net_fer/runs/affectnet_model/model_best.pth.tar"\
        --threshold=-0.33 --base-path=${path_for_clips} --movie=${movie_name}  

