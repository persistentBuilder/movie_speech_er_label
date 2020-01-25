#!/bin/bash
#SBATCH -A aryaman.g
#SBATCH --mem=31000
#SBATCH --cpus-per-task=10
#SBATCH --time=8-00:00:00
#SBATCH --mail-type=END

module load ffmpeg/4.2.1

#for i in {0..9} 
#do
#    bash fast-srt-split.sh 21_Jump_Street.mp4 21_Jump_Street.en.srt mp4 $i &
#done

#wait
movie_base_path=/home/aryaman.g/projects/FER/video_shot_annotation/Movies/21_jump_street
printf %s\\n {0..9} | xargs -n 1 -P 10 bash fast-srt-split.sh $movie_base_path/21_Jump_Street.mp4 21_Jump_Street.en.srt mp4
#printf %s\\n {0..99} | parallel bash fast-srt-split.sh 21_Jump_Street.mp4 21_Jump_Street.en.srt mp4


