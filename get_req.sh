
#!/bin/bash
#SBATCH -A aryaman.g
#SBATCH --gres=gpu:2
#SBATCH --mem=51000
#SBATCH --cpus-per-task=10
#SBATCH --time=4-00:00:00
#SBATCH --mail-type=END

source ~/packages/keras_tf_venv3/bin/activate
module load cuda/10.0
module load cudnn/7-cuda-10.0
module load ffmpeg/4.2.1


pip freeze > requirements.txt
