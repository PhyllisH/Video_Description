#check the performence of different trainning epochs
#check fcn_model
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import numpy as np
import os
import ipdb
import csv
import sys
import cv2

from tensorflow.contrib import rnn
from keras.preprocessing import sequence
sys.path.append('/DB/rhome/yhu/tensorflow/video_to_sequence/fcn_inner')
#from fcn_attnwrapper import *
from model_changed import *
#from model_withFrameCaption import *
#from model_rm_decodeLSTM1 import *
#from activity_model import *
sys.path.append('/DB/rhome/yhu/tensorflow/caption-eval/coco-caption/')
sys.path.append('/DB/rhome/yhu/tensorflow/caption-eval/')

from run_evaluations import *

base_path = '/DB/rhome/yhu/tensorflow/video_to_sequence/fcn_inner/'
video_feat_path = '/DATA2/data/yhu/YouTubeFeats'
video_data_path='/DB/rhome/yhu/tensorflow/video_to_sequence/MSR Video Description Corpus.csv'
mapping_ref = '/DB/rhome/yhu/tensorflow/video_to_sequence/data/youtube_video_to_id_mapping.txt'
references_json = '/DB/rhome/yhu/tensorflow/video_to_sequence1.0/data/test_1000_0.25_reference.json'
#references_json = '/DB/rhome/yhu/tensorflow/video_to_sequence/fcn_inner/test_670_reference.json'

#basic
#fcn_feat_path = '/DATA2/data/yhu/YouTubeFeats'
model_path = '/DB/rhome/yhu/tensorflow/VideoDescription/models/'
test_file_path = '/DB/rhome/yhu/tensorflow/VideoDescription/results/'

#attnwrapper
#fcn_feat_path = '/DATA3_DB7/data/yhu/fcn_conv3_4_YouTubeFeats'
#model_path = '/DB/rhome/yhu/tensorflow/video_to_sequence/fcn_inner/fcn_attnwrapper/'
#test_file_path = '/DB/rhome/yhu/tensorflow/video_to_sequence/fcn_inner/fcn_attnwrapper_results'

#conv3_1
#fcn_feat_path = '/DATA3_DB7/data/yhu/fcn_conv3_1_YouTubeFeats'
#model_path = '/DB/rhome/yhu/tensorflow/video_to_sequence/fcn_inner/conv3_1_models/'
#test_file_path = '/DB/rhome/yhu/tensorflow/video_to_sequence/fcn_inner/conv3_1_results'

#conv5_4
#fcn_feat_path = '/DATA3_DB7/data/yhu/fcn_conv5_4_YouTubeFeats'
#model_path = '/DB/rhome/yhu/tensorflow/video_to_sequence/fcn_inner/conv5_4_models/'
#test_file_path = '/DB/rhome/yhu/tensorflow/video_to_sequence/fcn_inner/conv5_4_results'

#pool4
#fcn_feat_path = '/DATA3_DB7/data/yhu/fcn_pool4_YouTubeFeats'
#model_path = '/DB/rhome/yhu/tensorflow/video_to_sequence/fcn_inner/pool4_models/'
#test_file_path = '/DB/rhome/yhu/tensorflow/video_to_sequence/fcn_inner/pool4_results'

#references_json = '/DB/rhome/yhu/tensorflow/video_to_sequence/data/test_youtube_0.65_reference.json'
#references_json = '/DB/rhome/yhu/tensorflow/video_to_sequence/data/test_1000_A_0.1_reference.json'
#model_path = '/DB/rhome/yhu/tensorflow/video_to_sequence/activity_models/'
#video_feat_path = '/DATA3_DB7/data/yhu/ActivityNetFeats'
#video_data_path = '/DB/rhome/yhu/tensorflow/video_to_sequence/captions/all_video_infos.csv'
#video_attn_feat_path = '/DATA2/data/yhu/attn_YouTubeFeats'
#test_file_path = '/DB/rhome/yhu/tensorflow/video_to_sequence/model_test_result'
#mapping_ref = '/DB/rhome/yhu/tensorflow/video_to_sequence/data/activitynet_video_to_id_mapping.txt'
#references_json = '/DB/rhome/yhu/tensorflow/video_to_sequence/data/test_ActivityNet_0.1_reference.json'
input_file = '/DB/rhome/yhu/tensorflow/video_to_sequence/data/test_ActivityNet_300_0.1_new.txt'
test_file = '/DB/rhome/yhu/tensorflow/video_to_sequence/data/test_ActivityNet_300_0.1.txt'

def get_fcn_video_data(video_data_path, video_feat_path, fcn_feat_path, train_ratio=0.9):
    video_data = pd.read_csv(video_data_path, sep=',')
    video_data = video_data[video_data['Language'] == 'English']
    video_data['video_path'] = video_data.apply(lambda row: row['VideoID']+'_'+str(int(row['Start']))+'_'+str(int(row['End']))+'.avi.npy', axis=1)
    video_data['video_path'] = video_data['video_path'].map(lambda x: os.path.join(video_feat_path, x))

    video_data['fcn_feat_path'] = video_data.apply(lambda row: row['VideoID']+'_'+str(int(row['Start']))+'_'+str(int(row['End']))+'.avi.npy', axis=1)
    video_data['fcn_feat_path'] = video_data['video_path'].map(lambda x: os.path.join(fcn_feat_path, x))
    video_data = video_data[video_data['fcn_feat_path'].map(lambda x: os.path.exists( x ))]
    
    video_data = video_data[video_data['video_path'].map(lambda x: os.path.exists( x ))]
    video_data = video_data[video_data['Description'].map(lambda x: isinstance(x, str))]

    unique_filenames = video_data['video_path'].unique()
    train_len = int(len(unique_filenames)*train_ratio)

    train_vids = unique_filenames[:train_len]
    test_vids = unique_filenames[train_len:]

    train_data = video_data[video_data['video_path'].map(lambda x: x in train_vids)]
    test_data = video_data[video_data['video_path'].map(lambda x: x in test_vids)]

    return train_data, test_data

def get_ActivityNet_video_data(video_data_path, video_feat_path, train_ratio=0.9):
    video_data = pd.read_csv(video_data_path, sep=',')
    #video_data = video_data[video_data['Language'] == 'English']
    #video_data['video_path'] = video_data.apply(lambda row: row['VideoID']+'_'+str(int(row['Start']))+'_'+str(int(row['End']))+'.avi.npy', axis=1)
    #video_data['video_path'] = video_data['video_path'].map(lambda x: os.path.join(video_feat_path, x))
    video_data = video_data[video_data['video_path'].map(lambda x: os.path.exists( x ))]
    video_data = video_data[video_data['Description'].map(lambda x: isinstance(x, str))]

    unique_filenames = video_data['video_path'].unique()
    train_len = int(len(unique_filenames)*train_ratio)

    train_vids = unique_filenames[:train_len]
    test_vids = unique_filenames[train_len:]

    train_data = video_data[video_data['video_path'].map(lambda x: x in train_vids)]
    test_data = video_data[video_data['video_path'].map(lambda x: x in test_vids)]

    return train_data, test_data

def get_video_data(video_data_path, video_feat_path, train_ratio=0.9):
    video_data = pd.read_csv(video_data_path, sep=',')
    video_data = video_data[video_data['Language'] == 'English']
    video_data['video_path'] = video_data.apply(lambda row: row['VideoID']+'_'+str(int(row['Start']))+'_'+str(int(row['End']))+'.avi.npy', axis=1)
    video_data['video_path'] = video_data['video_path'].map(lambda x: os.path.join(video_feat_path, x))
    video_data = video_data[video_data['video_path'].map(lambda x: os.path.exists( x ))]
    video_data = video_data[video_data['Description'].map(lambda x: isinstance(x, str))]

    unique_filenames = video_data['video_path'].unique()
    train_len = int(len(unique_filenames)*train_ratio)

    train_vids = unique_filenames[:train_len]
    test_vids = unique_filenames[train_len:]

    train_data = video_data[video_data['video_path'].map(lambda x: x in train_vids)]
    test_data = video_data[video_data['video_path'].map(lambda x: x in test_vids)]

    return train_data, test_data


def video_to_id_mapping( mapping_ref = mapping_ref, test_file=test_file):
    f = open(mapping_ref,'r')
    lines = f.readlines()
    check_dict={}

    for line in lines:
        realname = line.split(' ')[0]
        newid = line.split(' ')[1]
        check_dict[realname] = newid

    f.close()

    fold = open(test_file,'r')
    fnew = open(os.path.join(test_file_path, os.path.basename(test_file).split('.txt')[0]+'_new.txt'),'w')

    old_lines = fold.readlines()
    lenvid = len('vid')

    for old_line in old_lines:
        oldname = old_line.split('\t')[0]
        newname = check_dict[oldname]
        #print int(newname[lenvid:])
        #if int(newname[lenvid:]) <1300:
            #pass
        #else:
        new_line = newname.split('\n')[0] + '\t' + old_line.split('\t')[1]
        fnew.write(new_line)

    fold.close()
    fnew.close()


def evaluate(input_file=input_file, references_json=references_json):
    HASH_IMG_NAME = True
    pylab.rcParams['figure.figsize'] = (10.0, 8.0)
    json.encoder.FLOAT_REPR = lambda o: format(o, '.3f')

    prediction_file = input_file
    reference_file = references_json
    json_predictions_file = '{0}.json'.format(prediction_file)

    #ipdb.set_trace()
    crf = CocoResFormat()
    crf.read_file(prediction_file, HASH_IMG_NAME)
    crf.dump_json(json_predictions_file)

    # create coco object and cocoRes object.
    coco = COCO(reference_file)
    cocoRes = coco.loadRes(json_predictions_file)

    # create cocoEval object.
    cocoEval = COCOEvalCap(coco, cocoRes)

    # evaluate results
    cocoEval.evaluate()
    #ipdb.set_trace()
    # print output evaluation scores
    metric_dict={}
    for metric, score in cocoEval.eval.items():
        print '%s: %.3f'%(metric, score)
        metric_dict[metric] = score

    return metric_dict

def test_model():
    #ipdb.set_trace()
    models_name = os.listdir(model_path)
    models_name = filter(lambda x: x.endswith('meta'), models_name)
    models_name = map(lambda x: x.split('.meta')[0], models_name)
    csvfile = open(os.path.join(test_file_path,'test.csv'),'wb')
    fieldnames = ['ModelName', 'Bleu_1', 'Bleu_2','Bleu_3','Bleu_4','METEOR','CIDEr','ROUGE_L']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    #plot_M = []

    #build model
    #train_data, test_data = get_ActivityNet_video_data(video_data_path, video_feat_path, train_ratio=0.9)
    #train ratio should be the same as trained model
    train_data, test_data = get_video_data(video_data_path, video_feat_path, train_ratio=0.75)
    #train_data, test_data = get_fcn_video_data(video_data_path, video_feat_path, fcn_feat_path, train_ratio=0.75)
    test_videos = test_data['video_path'].unique()
    captions = train_data['Description'].values
    captions = map(lambda x: x.replace('.', ''), captions)
    captions = map(lambda x: x.replace(',', ''), captions)
    wordtoix, ixtoword, bias_init_vector = preProBuildWordVocab(captions, word_count_threshold=10)
    np.save(base_path+'ixtoword', ixtoword)
    
    ixtoword = pd.Series(np.load(base_path+'ixtoword.npy').tolist())
    model = Video_Caption_Generator(
            dim_image=dim_image,
            n_words=len(ixtoword),
            dim_hidden=dim_hidden,
            batch_size=batch_size,
            n_lstm_steps=n_frame_step,
            drop_out_rate=drop_out_rate,
            bias_init_vector=None)
    
    video_tf, video_mask_tf, caption_tf, probs_tf, last_embed_tf = model.build_generator()
    #video_tf, video_mask_tf, image_caption_tf, caption_tf, probs_tf, last_embed_tf = model.build_generator()
    saver = tf.train.Saver()

    for model in models_name:
        #ipdb.set_trace()
        test_file = 'test_' + model + '.txt'
        #test(model_path=model_path + model, test_file=os.path.join(test_file_path,test_file))
        with tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))) as sess:
            saver.restore(sess, model_path+model)
            f=open(os.path.join(test_file_path,test_file),'w')

            check_dict={}

            for video in test_videos:
                #ipdb.set_trace()
                #vid = video.split('/')[-1].split('.')[:-1]
                #realvid ='.'.join(vid)
                vid = video.split('/')[-1].split('.')[0]
                realvid =vid
                print video.split('/')[-1]
                if realvid in check_dict:
                    pass
                else:
                    #video_path = video_feat_path + '/' + video
                    video_path = video

                    #image_caption = video_frame_caption(os.path.basename(video))
                    #image_caption = image_caption.lower().split(' ')
                    #image_caption_id = []
                    #for x in image_caption:
                    #   if x in wordtoix:
                    #       image_caption_id.append(wordtoix[x])
                    #image_caption_matrix = np.zeros([1,n_frame_step])
                    #image_caption_matrix[0,:len(image_caption_id)] = image_caption_id

                    video_feat = np.load(video_path)[None,...]
                    video_mask = np.ones((video_feat.shape[0], video_feat.shape[1]))

                    #for fcn models
                    #attn_video_path = fcn_feat_path + '/' + video.split('/')[-1]
                    #attn_video_feat = np.load(attn_video_path)[None,...]

                    #generated_word_index = sess.run(caption_tf, feed_dict={video_tf:video_feat, attn_video_tf:attn_video_feat, video_mask_tf:video_mask})
                    #probs_val = sess.run(probs_tf, feed_dict={video_tf:video_feat, attn_video_tf:attn_video_feat, video_mask_tf:video_mask})
                    #embed_val = sess.run(last_embed_tf, feed_dict={video_tf:video_feat, attn_video_tf:attn_video_feat, video_mask_tf:video_mask})

                    #current_send_feat = np.zeros([video_feat.shape[0],video_feat.shape[1],video_feat.shape[2]*2])

                    #for x in range(0,len(attn_video_feat[0])):
                    #   current_send_feat[0,x,:video_feat.shape[2]] = video_feat[0,x,:]
                    #   current_send_feat[0,x,video_feat.shape[2]:] = attn_video_feat[0,x,:]
                    #generated_word_index = sess.run(caption_tf, feed_dict={video_tf:current_send_feat, video_mask_tf:video_mask})
                    #probs_val = sess.run(probs_tf, feed_dict={video_tf:current_send_feat})
                    #embed_val = sess.run(last_embed_tf, feed_dict={video_tf:current_send_feat})

                    generated_word_index = sess.run(caption_tf, feed_dict={video_tf:video_feat, video_mask_tf:video_mask})
                    probs_val = sess.run(probs_tf, feed_dict={video_tf:video_feat})
                    embed_val = sess.run(last_embed_tf, feed_dict={video_tf:video_feat})
                    #generated_word_index = sess.run(caption_tf, feed_dict={
                    #   video_tf:video_feat, 
                    #   video_mask_tf:video_mask,
                    #   image_caption_tf:image_caption_matrix
                    #   })
                    #probs_val = sess.run(probs_tf, feed_dict={
                    #   video_tf:video_feat,
                    #   image_caption_tf:image_caption_matrix
                    #   })
                    #embed_val = sess.run(last_embed_tf, feed_dict={
                    #   video_tf:video_feat,
                    #   image_caption_tf:image_caption_matrix
                    #   })

                    generated_words = ixtoword[generated_word_index]

                    punctuation = np.argmax(np.array(generated_words) == '.')+1
                    generated_words = generated_words[:punctuation]

                    generated_sentence = ' '.join(generated_words)
                    print generated_sentence
                    check_dict[realvid] = generated_sentence
                    f.write(realvid+'\t'+generated_sentence+'\n')
                #ipdb.set_trace()
            f.close()

        #ipdb.set_trace()
        video_to_id_mapping(test_file = os.path.join(test_file_path,test_file))
        metric_dict = evaluate(input_file = os.path.join(test_file_path, 'test_'+model+'_new.txt'))
        writer.writerow({
            'ModelName': model, 
            'Bleu_1': metric_dict['Bleu_1'],
            'Bleu_2': metric_dict['Bleu_2'],
            'Bleu_3': metric_dict['Bleu_3'],
            'Bleu_4': metric_dict['Bleu_4'],
            'METEOR': metric_dict['METEOR'],
            'CIDEr': metric_dict['CIDEr'],
            'ROUGE_L': metric_dict['ROUGE_L']
            })
        #plot_M.append(metric_dict['METEOR'])

    csvfile.close()

    #t = range(1,len(plot_M)+1)
    #plt.plot(t,plot_M)
    #plt.show()


test_model()
