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

movie_name=crazy_stupid_love
movie_name=IT
movie_base_path=/ssd_scratch/cvit/aryaman.g/movies/$movie_name
home_movie_dir=/home/aryaman.g/projects/FER/video_shot_annotation/Movies

mkdir -p /ssd_scratch/cvit/aryaman.g
mkdir -p /ssd_scratch/cvit/aryaman.g/movies
mkdir -p /ssd_scratch/cvit/aryaman.g/movies/$movie_name
ls /ssd_scratch/cvit/aryaman.g/movies/

echo "create csv for groud truth label of subtitle sentences"
python get_subtitles_csv.py --subtitles-path=${home_movie_dir}/${movie_name}/${movie_name}.srt

mp4File=${movie_base_path}/${movie_name}.mp4
srtFile=${movie_base_path}/${movie_name}.srt
cp ${home_movie_dir}/${movie_name}/${movie_name}.mp4 $mp4File
cp ${home_movie_dir}/${movie_name}/${movie_name}.srt $srtFile

echo "modify starttime for faster seek time in ffmpeg"
python modify_start_time_subtitles.py -10.00 $srtFile

echo "predict label from  subtitle sentences"
python3 sentiment-discovery/run_classifier.py --load sentiment-discovery/mlstm_semeval.clf --fp16 --data movie_subtitles_csv/${movie_name}_subtitles.csv
mv clf_results.npy movie_subtitles_prediction/${movie_name}_results.npy
mv clf_results.npy.prob.npy movie_subtitles_prediction/${movie_name}_results_prob.npy

clips_dir=${movie_base_path}/${movie_name}-clips
#if [ ! -d "$clips_dir" ]; then
echo "starting video cutting based on timestamp in subtitles"
printf %s\\n {0..9} | xargs -n 1 -P 10 bash fast-srt-split.sh $mp4File $srtFile mp4
#fi

path_for_clips=$clips_dir
echo "choose strongly labeled text predictions"
python text_pred_analysis.py --movie=${movie_name} --threshold=0.90 --base-path=${path_for_clips}

#echo "predict from clips and choose strongly labeled clips majority predictions"
#python visual_pred_analysis.py --model-path="/home/aryaman.g/projects/all_code/simple_net_fer/runs/affectnet_model/model_best.pth.tar"\
#        --threshold=-0.33 --base-path=${path_for_clips} --movie=${movie_name}

echo "train and test sound using labels assigned to clips with above workflow"
python main.py --epochs=100 --lr=0.0001 --batch_size=32 --net="simpleNet" --movie=${movie_name} --save-checkpoint=0
