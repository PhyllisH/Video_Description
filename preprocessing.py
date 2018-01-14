#-*- coding: utf-8 -*-
'''
According to the paper, the authors extracted upto 80 frames from each video,
they did not mention if they grabbed first 80 frames, or sampled 80 frames with same intervals,
but anyway I did the latter.
'''
import cv2
import os
import ipdb
import numpy as np
import pandas as pd
import skimage
from cnn_util import *


def preprocess_frame(image, target_height=224, target_width=224):

    if len(image.shape) == 2:
        image = np.tile(image[:,:,None], 3)
    elif len(image.shape) == 4:
        image = image[:,:,:,0]

    image = skimage.img_as_float(image).astype(np.float32)
    height, width, rgb = image.shape
    if width == height:
        resized_image = cv2.resize(image, (target_height,target_width))

    elif height < width:
        resized_image = cv2.resize(image, (int(width * float(target_height)/height), target_width))
        cropping_length = int((resized_image.shape[1] - target_height) / 2)
        resized_image = resized_image[:,cropping_length:resized_image.shape[1] - cropping_length]

    else:
        resized_image = cv2.resize(image, (target_height, int(height * float(target_width) / width)))
        cropping_length = int((resized_image.shape[0] - target_width) / 2)
        resized_image = resized_image[cropping_length:resized_image.shape[0] - cropping_length,:]

    return cv2.resize(resized_image, (target_height, target_width))

def main():
    num_frames = 80
    vgg_model = '/DB/rhome/yhu/tensorflow/video_to_sequence/VGG_ILSVRC_19_layers.caffemodel'
    vgg_deploy = '/DB/rhome/yhu/tensorflow/video_to_sequence/VGG_ILSVRC_19_layers_deploy.prototxt'
    video_path = '/DATA2/data/yhu/YouTubeClips'
    video_save_path = '/DATA2/data/yhu/attn_YouTubeFeats'
    attn_video_path = '/DATA2/data/yhu/attn_video_YouTubeFeats'
    #video_path = '/DATA2/data/yhu/hollywood/videoclips'
    #video_save_path = '/DATA2/data/yhu/hollywood/videoFeats'
    videos = os.listdir(video_path)
    video_generated = os.listdir(video_save_path)
    videos = filter(lambda x: x.endswith('avi'), videos)

    cnn = CNN(model=vgg_model, deploy=vgg_deploy, width=224, height=224)
    #cnn = CNN(width=224, height=224)

    for video in videos:
        print video
        #if video == "yREFkmrrYiw_51_57.avi":
            #if video+'.npy' in video_generated:
            #    pass
            #else:
        if os.path.exists( os.path.join(video_save_path, video+'.npy') ):
            print "Already processed ... "
            continue

        video_fullpath = os.path.join(video_path, video)
        try:
            cap  = cv2.VideoCapture( video_fullpath )
            print "video_fullpath: ",video_fullpath
        except:
            pass

        frame_count = 0
        frame_list = []

        while True:
            ret, frame = cap.read()
            #print ret,frame
            if ret is False:
                break

            frame_list.append(frame)
            frame_count += 1

        count=0
        lack = 80 - frame_count
        #frame_list = np.array(frame_list)
        print "frame_count: ", frame_count

        if frame_count > 80:
            frame_list = np.array(frame_list)
            frame_indices = np.linspace(0, frame_count, num=num_frames, endpoint=False).astype(int)
            frame_list = frame_list[frame_indices]
        elif frame_count <80:
            frame = frame_list[-1]
            while count<lack:
                frame_list.append(frame)
                count += 1
                #print count
            frame_list = np.array(frame_list)
        else:
            frame_list = np.array(frame_list)

        print frame_list.shape

        #ipdb.set_trace()
        
        cropped_frame_list = np.array(map(lambda x: preprocess_frame(x), frame_list))
        all_attn_frames = cnn.get_attention(cropped_frame_list)
        #cropped_frame_list = cnn.get_attention(cropped_frame_list)

        attns = all_attn_frames.max(axis=1)
        for x in range(0,all_attn_frames.shape[0]):
            attns[x] = (attns[x] - attns[x].min()) / attns[x].max()
        for x in range(0,cropped_frame_list.shape[-1]):
            cropped_frame_list[:,:,:,x] = np.multiply(cropped_frame_list[:,:,:,x], attns)

        #ipdb.set_trace()
        #attn_save_path = os.path.join(attn_video_path, video + '.npy')
        #np.save(attn_save_path, cropped_frame_list)
        #np.save(attn_save_path, attns)
        
        feats = cnn.get_features(cropped_frame_list)
        #print feats

        save_full_path = os.path.join(video_save_path, video + '.npy')
        np.save(save_full_path, feats)

if __name__=="__main__":
    main()
