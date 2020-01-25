import os
import numpy as np
import argparse
import pandas as pd
import dlib
import torch
import cv2
from torch.autograd import Variable
import torchvision.transforms as transforms
import torch.nn as nn
from extendNet import extendNet

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

def resize_face_image(img):
    return cv2.resize(img, (image_resize_width, image_resize_height), interpolation=cv2.INTER_CUBIC)

def load_model(model_path, model=None):
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['state_dict'])
    return model

def predict_on_faces_using_emotion_model(faces, model): 
    faces = torch.stack(faces, dim=0)        
    faces = Variable(faces)
    prediction = model(faces)    
    return prediction.detach().clone().cpu().numpy()
    
def get_clips_paths(movie_name):
        
    files = os.listdir(args.base_path)
    clips_paths = []
    check_str = movie_name+".mp4"
    for fl in files:
        if fl.split("-",1)[-1]==check_str:
            clips_paths.append(args.base_path+'/'+fl)
    return clips_paths

def get_face_from_rect(rect, img):
    return img[max(0,rect.top()):min(rect.bottom(), img.shape[0]), max(rect.left(),0):min(rect.right(), img.shape[1])]

def get_faces_for_clip(clip_path, detector):
    cap = cv2.VideoCapture(clip_path)
    frame_count = 0
    face_imgs = []
    while cap.isOpened():
        ret, frame = cap.read()
        if ret is not True:
            break
        if frame is None:
            continue
        frame_count = frame_count + 1
        if frame_count % frame_interval != 0:
            continue
        face_rects = detector(frame, 1)
        for face_rect in face_rects:
            face_img = get_face_from_rect(face_rect, frame)
            face_imgs.append(transform(resize_face_image(face_img)))
    cap.release()
    cv2.destroyAllWindows()
    return face_imgs

def main():

    movies_names = ['avengers', 'erin', 'inception', 'mi_ii']
    if args.movie != "":
        movies_names = [args.movie]
    th = args.threshold
    text_labels_map = {0: "anger", 1: "anticipation", 2: "disgust", 3: "fear", 4: "joy", 5: "sadness", 6: "surprise", 7: "trust"}
    visual_labels_map = {0: "neutral", 1: "joy", 2: "sadness", 3: "surprise", 4: "fear", 5: "disgust", 6: "anger", 7: "contempt"}
    
    model = extendNet(num_classes=8)
    model = nn.DataParallel(model)

    model = load_model(
        model_path=args.model_path,
        model=model)
    model.eval()
    detector = dlib.get_frontal_face_detector()
    for movie in movies_names:
        clips_paths = get_clips_paths(movie)
        f = open("expression_for_video/expression_video_visual_" + movie + ".txt", "w")
        for clip_path in clips_paths:
            faces = get_faces_for_clip(clip_path, detector)
            if len(faces) == 0:
                continue
            #faces = process_faces_for_emotion_model(faces)
            prediction_on_faces = predict_on_faces_using_emotion_model(faces, model)
            pred = np.argmax(prediction_on_faces, axis=1)
            strongness = np.max(prediction_on_faces, axis=1)
            th_gt_index = np.argwhere(strongness > th)
            th_gt_index = th_gt_index.squeeze()
            th_gt_prediction = pred[th_gt_index]
            labels, counts_elements = np.unique(th_gt_prediction, return_counts=True)
            max_label = -1
            max_label_count = 0
            for i, label in enumerate(labels):
                if counts_elements[i] > max_label_count:
                    max_label = label
                    max_label_count = counts_elements[i]
            if max_label != -1:
                f.write(clip_path+','+visual_labels_map[max_label]+'\n')
        f.close()
        print("=======================================================")

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True, help="path to model for predicting emotion")
    parser.add_argument("--threshold", type=float, default=-0.33)
    parser.add_argument("--movie", type=str, default="")
    parser.add_argument("--base-path", type=str, default="")
    args = parser.parse_args()
    base_path = args.base_path
    frame_interval = 5
    image_resize_width = 224
    image_resize_height = 224
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor()
    ])
    map_folders = {'avengers': 'Avengers', 'erin': 'ErinB', 'inception': 'Inception_2010', 'mi_ii': 'MI_II'}
    main()

