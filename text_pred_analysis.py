import numpy as np
import argparse
from check_lip_movement import just_check_speaking
import pandas as pd


def get_video_path_for_seq(movie, seq):
    if movie == 'avengers':
        path = base_path + map_folders[movie] + '/shots/' + map_folders[movie] + "_shot_" + str(seq) + '.mp4'
    elif movie == 'erin':
        path = base_path + map_folders[movie] + '/ErinBrockovich/' + 'ErinBrockavich' + "_shot_" + str(seq) + '.mp4'
    elif movie == 'inception':
        path = base_path + map_folders[movie] + '/8_min_segments/shots/' +  "Inception_shot_" + str(seq) + '.mp4'
    elif movie == 'mi_ii':
        path = base_path + map_folders[movie] + '/shots/' + map_folders[movie]  + "_shot_" + str(seq) + '.mp4'
    else: 
        path = base_path + '/' + """{:04d}""".format(seq) + '-' + movie + '.mp4'  
    return path    

def main():
    movies_names = ['avengers', 'erin', 'inception', 'mi_ii']
    if args.movie != "":
        movies_names = [args.movie]
    th = args.threshold
    labels_map = {0: "anger", 1: "anticipation", 2: "disgust", 3: "fear", 4: "joy", 5: "sadness", 6: "surprise", 7: "trust"}
    
    for movie in movies_names:
        prob_file = 'movie_subtitles_prediction/' + movie + '_results_prob.npy'
        prob = np.load(prob_file)
        pred = np.argmax(prob, axis=1)
        strongness = np.max(prob, axis=1)
        # pred_file = 'movie_subtitles_prediction/' + movie + '_prediction.npy'
        # strongness_file = 'movie_subtitles_prediction/' + movie + '_strongness.npy'
        # np.save(pred_file, pred)
        # np.save(strongness_file, strongness)
        th_gt_index = np.argwhere(strongness > th)
        th_gt_index = th_gt_index.squeeze()
        th_gt_prediction = pred[th_gt_index]
        labels, counts_elements = np.unique(th_gt_prediction, return_counts=True)
        print(movie)
        print("total strong labels with threshold", th, " : ", th_gt_prediction.shape[0])
        print("total elements", pred.shape[0])
        
        for i in range(0,len(labels)):
            print(labels[i], labels_map[labels[i]], counts_elements[i])
        dialogue_file = 'movie_subtitles_csv/' + movie + '_subtitles.csv'
        #sequences = pd.read_csv(dialogue_file)['Unnamed: 0']
        sequences = pd.read_csv(dialogue_file).iloc[:,0]
        f = open("expression_for_video/expression_video_" + movie + ".txt", "w")
        for i in range(len(th_gt_index)):
            seq = sequences[th_gt_index[i]]
            label = pred[th_gt_index[i]]
            expression = labels_map[label]
            video_path = get_video_path_for_seq(movie, seq)
            print(video_path)
            #try:
                #if just_check_speaking(video_path, 2):
            f.write(video_path +  ',' + expression + '\n')
            #except:
            #    continue
        f.close()
        print("=======================================================")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=0.75)
    parser.add_argument("--movie", type=str, default="")
    parser.add_argument("--base-path", type=str, default="")
    args = parser.parse_args()
    base_path = args.base_path
    map_folders = {'avengers': 'Avengers', 'erin': 'ErinB', 'inception': 'Inception_2010', 'mi_ii': 'MI_II'}
    main()

