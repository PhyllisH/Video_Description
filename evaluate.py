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

from configure import *
from models import *
from data_processing import *

from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--mode', dest='mode', type=str, default='base', help='The method used, base or fcn flipped')
parser.add_argument('--FCNLayer', dest='FCNLayer', type=str, default='conv3_4', help='The layer where FCN added')

args = parser.parse_args()

def test_model():
    #ipdb.set_trace()
    test_file_path = os.path.join(test_base_path,'result_'+args.mode)
    if os.path.exists(test_file_path):
        pass
    else:
        os.mkdir(test_file_path)
    model_path = os.path.join(model_base_path,'model_'+args.mode)
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
    if args.mode == 'base' or args.mode == 'flipped':
		train_data, test_data = get_video_data(video_data_path, video_feat_path, train_ratio)
    elif args.mode == 'fcn':
        fcn_feat_path='/DATA3_DB7/data/yhu/fcn_'+args.FCNLayer+'_YouTubeFeats'
        train_data, test_data = get_fcn_video_data(video_data_path, video_feat_path, fcn_feat_path, train_ratio)
    else:
        print 'wrong mode'

    test_videos = test_data['video_path'].unique()
    captions = train_data['Description'].values
    captions = map(lambda x: x.replace('.', ''), captions)
    captions = map(lambda x: x.replace(',', ''), captions)
    wordtoix, ixtoword, bias_init_vector = preProBuildWordVocab(captions, word_count_threshold=10)
    np.save(base_path+'ixtoword', ixtoword)    
    ixtoword = pd.Series(np.load('data/ixtoword_'+str(train_ratio)+'.npy').tolist())

    if args.mode == 'base':
        model = Video_Caption_Generator(
            dim_image=dim_image,
            n_words=len(ixtoword),
            dim_hidden=dim_hidden,
            batch_size=batch_size,
            n_lstm_steps=n_frame_step,
            drop_out_rate=drop_out_rate,
            bias_init_vector=None)
        video_tf, video_mask_tf, caption_tf, probs_tf, last_embed_tf = model.build_generator()
    elif args.mode == 'flipped':
        model = Flipped_Video_Caption_Generator(
            dim_image=dim_image,
            n_words=len(ixtoword),
            dim_hidden=dim_hidden,
            batch_size=batch_size,
            n_lstm_steps=n_frame_step,
            drop_out_rate=drop_out_rate,
            bias_init_vector=None)
        video_tf, video_mask_tf, caption_tf, probs_tf, last_embed_tf = model.build_generator()
    elif args.mode == 'fcn':
        model = FCN_Video_Caption_Generator(
            dim_image=dim_image,
            n_words=len(ixtoword),
            dim_hidden=dim_hidden,
            batch_size=batch_size,
            n_lstm_steps=n_frame_step,
            drop_out_rate=drop_out_rate,
            bias_init_vector=None)

    	video_tf, attn_video_tf, video_mask_tf, caption_tf, probs_tf, last_embed_tf = model.build_generator()
    else:
    	pass
    #video_tf, video_mask_tf, image_caption_tf, caption_tf, probs_tf, last_embed_tf = model.build_generator()
    saver = tf.train.Saver()

    for model in models_name:
        #ipdb.set_trace()
        test_file = 'test_' + model + '.txt'
        #test(model_path=model_path + model, test_file=os.path.join(test_file_path,test_file))
        with tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))) as sess:
            saver.restore(sess, os.path.join(model_path,model))
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

                    if args.mode == 'base' or args.mode == 'flipped':
                    	generated_word_index = sess.run(caption_tf, feed_dict={video_tf:video_feat, video_mask_tf:video_mask})
                    	probs_val = sess.run(probs_tf, feed_dict={video_tf:video_feat})
                    	embed_val = sess.run(last_embed_tf, feed_dict={video_tf:video_feat})
                    elif args.mode == 'fcn':
                    	generated_word_index = sess.run(caption_tf, feed_dict={video_tf:video_feat, attn_video_tf:attn_video_feat, video_mask_tf:video_mask})
                    	probs_val = sess.run(probs_tf, feed_dict={video_tf:video_feat, attn_video_tf:attn_video_feat, video_mask_tf:video_mask})
                    	embed_val = sess.run(last_embed_tf, feed_dict={video_tf:video_feat, attn_video_tf:attn_video_feat, video_mask_tf:video_mask})
                    else:
                    	pass

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

def main():
    test_model()

if __name__ == '__main__':
    main()