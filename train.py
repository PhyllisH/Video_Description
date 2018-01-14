import tensorflow as tf
import os
import numpy as np
from configure import *
from models import *
from data_processing import *
import matplotlib.pyplot as plt

from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--mode', dest='mode', type=str, default='base', help='The method used, base, flipped, or fcn')
parser.add_argument('--FCNLayer', dest='FCNLayer', type=str, default='conv3_4', help='The layer where FCN added')

args = parser.parse_args()

def train(model_path):
    #get the train data
    train_data, _ = get_video_data(video_data_path, video_feat_path, train_ratio)

    #get the training video decriptions
    captions = train_data['Description'].values
    captions = map(lambda x: x.replace('.', ''), captions)
    captions = map(lambda x: x.replace(',', ''), captions)
    wordtoix, ixtoword, bias_init_vector = preProBuildWordVocab(captions, word_count_threshold)

    np.save('data/ixtoword_'+str(train_ratio), ixtoword)

    #build the model
    if args.mode=='base':
        model = Video_Caption_Generator(
                dim_image=dim_image,
                n_words=len(wordtoix),
                dim_hidden=dim_hidden,
                batch_size=batch_size,
                n_lstm_steps=n_frame_step,
                drop_out_rate = drop_out_rate,
                bias_init_vector=bias_init_vector)
    elif args.mode=='flipped':
        model = Flipped_Video_Caption_Generator(
                dim_image=dim_image,
                n_words=len(wordtoix),
                dim_hidden=dim_hidden,
                batch_size=batch_size,
                n_lstm_steps=n_frame_step,
                drop_out_rate = drop_out_rate,
                bias_init_vector=bias_init_vector)
    else:
        print 'wrong mode!'


    tf_loss, tf_video, tf_video_mask, tf_caption, tf_caption_mask, tf_probs = model.build_model()

    #saver = tf.train.Saver(max_to_keep=10)

    #optimizer= tf.train.AdamOptimizer(learning_rate=learning_rate)
    #train_op=optimizer.minimize(tf_loss)
    train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss=tf_loss)
    #train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(tf_loss)
    init_op = tf.initialize_all_variables()

    saver = tf.train.Saver(max_to_keep=10)

    sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))
    sess.run(init_op)
    #saver = tf.train.import_meta_graph('/DB/rhome/yhu/tensorflow/video_to_sequence/models_atten_phase1/model-900.meta')
    #saver.restore(sess, "/DB/rhome/yhu/tensorflow/video_to_sequence/models_atten_phase1/model-900")

    #for plot
    plot_loss=[]

    #ipdb.set_trace()

    for epoch in range(n_epochs):
        index = list(train_data.index)
        np.random.shuffle(index)
        train_data = train_data.ix[index]

        current_train_data = train_data.groupby('video_path').apply(lambda x: x.irow(np.random.choice(len(x))))
        current_train_data = current_train_data.reset_index(drop=True)

        print "current_train_data: ",len(current_train_data)

        for start,end in zip(
                range(0, len(current_train_data), batch_size),
                range(batch_size, len(current_train_data), batch_size)):
            print "start,end: ",start , end
            current_batch = current_train_data[start:end]
            current_videos = current_batch['video_path'].values

            current_feats = np.zeros((batch_size, n_frame_step, dim_image))
            current_feats_vals = map(lambda vid: np.load(vid), current_videos)

            current_video_masks = np.zeros((batch_size, n_frame_step))

            for ind,feat in enumerate(current_feats_vals):
                current_feats[ind][:len(current_feats_vals[ind])] = feat
                current_video_masks[ind][:len(current_feats_vals[ind])] = 1

            current_captions = current_batch['Description'].values
            # Remove '.' and ',' from caption
            for idx, cc in enumerate( current_captions ):
                current_captions[idx] = cc.replace('.', '').replace(',','')

            # Remove the [:-1] in this line!
            current_caption_ind  = map( lambda cap : [ wordtoix[word] for word in cap.lower().split(' ') if word in wordtoix], current_captions )

            #current_caption_ind = map(lambda cap: [wordtoix[word] for word in cap.lower().split(' ')[:-1] if word in wordtoix], current_captions)

            current_caption_matrix = sequence.pad_sequences(current_caption_ind, padding='post', maxlen=n_frame_step-1)
            current_caption_matrix = np.hstack( [current_caption_matrix, np.zeros( [len(current_caption_matrix),1]) ] ).astype(int)
            current_caption_masks = np.zeros((current_caption_matrix.shape[0], current_caption_matrix.shape[1]))
            nonzeros = np.array( map(lambda x: (x != 0).sum()+1, current_caption_matrix ))

            for ind, row in enumerate(current_caption_masks):
                row[:nonzeros[ind]] = 1

            probs_val = sess.run(tf_probs, feed_dict={
                tf_video:current_feats,
                tf_video_mask:current_video_masks,
                tf_caption: current_caption_matrix,
                tf_caption_mask: current_caption_masks
                })

            _, loss_val = sess.run(
                    [train_op, tf_loss],
                    feed_dict={
                        tf_video: current_feats,
                        tf_video_mask : current_video_masks,
                        tf_caption: current_caption_matrix,
                        tf_caption_mask: current_caption_masks
                        })

            print loss_val
            plot_loss.append(loss_val)
        if np.mod(epoch, 100) == 0:
            print "Epoch ", epoch, " is done. Saving the model ..."
            saver.save(sess, os.path.join(model_path, 'model'), global_step=epoch)
    t = range(1,len(plot_loss)+1)
    plt.plot(t,plot_loss)
    plt.show()

def test(model_path='models/model-900', video_feat_path=video_feat_path):
    #ipdb.set_trace()
    train_data, test_data = get_video_data(video_data_path, video_feat_path, train_ratio)
    test_videos = test_data['video_path'].unique()
    #test_videos = os.listdir(video_feat_path)
    ixtoword = pd.Series(np.load('data/ixtoword_'+str(train_ratio)+'.npy').tolist())

    #build the model
    if args.mode=='base':
        model = Video_Caption_Generator(
                dim_image=dim_image,
                n_words=len(wordtoix),
                dim_hidden=dim_hidden,
                batch_size=batch_size,
                n_lstm_steps=n_frame_step,
                drop_out_rate = drop_out_rate,
                bias_init_vector=bias_init_vector)
    elif args.mode=='flipped':
        model = Flipped_Video_Caption_Generator(
                dim_image=dim_image,
                n_words=len(wordtoix),
                dim_hidden=dim_hidden,
                batch_size=batch_size,
                n_lstm_steps=n_frame_step,
                drop_out_rate = drop_out_rate,
                bias_init_vector=bias_init_vector)
    else:
        print 'wrong mode!'

    video_tf, video_mask_tf, caption_tf, probs_tf, last_embed_tf = model.build_generator()
    sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))

    saver = tf.train.Saver()
    saver.restore(sess, model_path)

    f=open(os.path.basename(model_path)+'.txt','w')
    #f=open('hollywood_1000_atten.txt','w')

    check_dict={}

    for video in test_videos:
    #for video_feat_path in test_videos:
        #ipdb.set_trace()
        vid = video.split('/')[-1].split('.')[0]
        #word = vid.split('_')
        #lenth = len(word[-1])+len(word[-2])+2
        #realvid = vid[:-lenth]
        #realvid = video
        realvid = vid
        print video
        #print video_feat_path
        if realvid in check_dict:
            pass
        else:
            #video_path = video_feat_path + '/' + video
            video_path = video
            video_feat = np.load(video_path)[None,...]
            #video_feat = np.load(video_feat_path)[None,...]
            video_mask = np.ones((video_feat.shape[0], video_feat.shape[1]))

            generated_word_index = sess.run(caption_tf, feed_dict={video_tf:video_feat, video_mask_tf:video_mask})
            probs_val = sess.run(probs_tf, feed_dict={video_tf:video_feat, video_mask_tf:video_mask})
            embed_val = sess.run(last_embed_tf, feed_dict={video_tf:video_feat, video_mask_tf:video_mask})
            generated_words = ixtoword[generated_word_index]

            punctuation = np.argmax(np.array(generated_words) == '.')+1
            generated_words = generated_words[:punctuation]

            generated_sentence = ' '.join(generated_words)
            print generated_sentence
            check_dict[realvid] = generated_sentence
            f.write(realvid+'\t'+generated_sentence+'\n')
        #ipdb.set_trace()
    f.close()

def FCN_train(model_path, fcn_feat_path):
    train_data, _ = get_fcn_video_data(video_data_path, video_feat_path, fcn_feat_path, train_ratio)
    captions = train_data['Description'].values
    captions = map(lambda x: x.replace('.', ''), captions)
    captions = map(lambda x: x.replace(',', ''), captions)
    wordtoix, ixtoword, bias_init_vector = preProBuildWordVocab(captions, word_count_threshold)

    np.save('data/ixtoword_'+str(train_ratio), ixtoword)

    #g = tf.Graph()
    #with g.as_default():
    model = FCN_Video_Caption_Generator(
            dim_image=dim_image,
            n_words=len(wordtoix),
            dim_hidden=dim_hidden,
            batch_size=batch_size,
            n_lstm_steps=n_frame_step,
            drop_out_rate = drop_out_rate,
            bias_init_vector=bias_init_vector)

    tf_loss, tf_video, tf_attn_video, tf_video_mask, tf_caption, tf_caption_mask, tf_probs = model.build_model()

    #saver = tf.train.Saver(max_to_keep=10)

    #optimizer= tf.train.AdamOptimizer(learning_rate=learning_rate)
    #train_op=optimizer.minimize(tf_loss)
    train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss=tf_loss)
    #train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(tf_loss)
    init_op = tf.initialize_all_variables()

    saver = tf.train.Saver(max_to_keep=20)

    sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))
    sess.run(init_op)
    #saver.restore(sess, "/DB/rhome/yhu/tensorflow/video_to_sequence/fcn_28*28/fcn_concat1_models/model_concat-100")

    #for plot
    plot_loss=[]

    #ipdb.set_trace()

    for epoch in range(n_epochs):
        index = list(train_data.index)
        np.random.shuffle(index)
        train_data = train_data.ix[index]

        current_train_data = train_data.groupby('video_path').apply(lambda x: x.irow(np.random.choice(len(x))))
        current_train_data = current_train_data.reset_index(drop=True)

        print "current_train_data: ",len(current_train_data)

        for start,end in zip(
                range(0, len(current_train_data), batch_size),
                range(batch_size, len(current_train_data), batch_size)):
            print "start,end: ",start , end
            current_batch = current_train_data[start:end]
            current_videos = current_batch['video_path'].values

            current_attn_videos = current_batch['fcn_feat_path'].values
            current_attn_feats = np.zeros((batch_size, n_frame_step, dim_image))
            current_attn_feats_vals = map(lambda vid: np.load(vid), current_attn_videos)
            #for attn_ind,attn_feat in enumerate(current_attn_feats_vals):
            #    current_attn_feats[attn_ind][:len(current_attn_feats_vals[attn_ind])] = attn_feat
            #print "current_videos: ",current_videos

            current_feats = np.zeros((batch_size, n_frame_step, dim_image))
            current_feats_vals = map(lambda vid: np.load(vid), current_videos)

            current_video_masks = np.zeros((batch_size, n_frame_step))

            for ind,feat in enumerate(current_feats_vals):
                current_feats[ind][:len(current_feats_vals[ind])] = feat
                current_video_masks[ind][:len(current_feats_vals[ind])] = 1

            for attn_ind,attn_feat in enumerate(current_attn_feats_vals):
                current_attn_feats[attn_ind][:len(current_attn_feats_vals[attn_ind])] = attn_feat


            current_captions = current_batch['Description'].values
            # Remove '.' and ',' from caption
            for idx, cc in enumerate( current_captions ):
                current_captions[idx] = cc.replace('.', '').replace(',','')

            # Remove the [:-1] in this line!
            current_caption_ind  = map( lambda cap : [ wordtoix[word] for word in cap.lower().split(' ') if word in wordtoix], current_captions )

            #current_caption_ind = map(lambda cap: [wordtoix[word] for word in cap.lower().split(' ')[:-1] if word in wordtoix], current_captions)

            current_caption_matrix = sequence.pad_sequences(current_caption_ind, padding='post', maxlen=n_frame_step-1)
            current_caption_matrix = np.hstack( [current_caption_matrix, np.zeros( [len(current_caption_matrix),1]) ] ).astype(int)
            current_caption_masks = np.zeros((current_caption_matrix.shape[0], current_caption_matrix.shape[1]))
            nonzeros = np.array( map(lambda x: (x != 0).sum()+1, current_caption_matrix ))

            for ind, row in enumerate(current_caption_masks):
                row[:nonzeros[ind]] = 1

            probs_val = sess.run(tf_probs, feed_dict={
                tf_video:current_feats,
                tf_attn_video:current_attn_feats,
                tf_video_mask:current_video_masks,
                tf_caption: current_caption_matrix,
                tf_caption_mask: current_caption_masks
                })

            _, loss_val = sess.run(
                    [train_op, tf_loss],
                    feed_dict={
                        tf_video: current_feats,
                        tf_attn_video:current_attn_feats,
                        tf_video_mask : current_video_masks,
                        tf_caption: current_caption_matrix,
                        tf_caption_mask: current_caption_masks
                        })

            print loss_val
            plot_loss.append(loss_val)
        if np.mod(epoch, 50) == 0:
            print "Epoch ", epoch, " is done. Saving the model ..."
            saver.save(sess, os.path.join(model_path, 'model'), global_step=epoch)
    t = range(1,len(plot_loss)+1)
    plt.plot(t,plot_loss)
    plt.show()

def FCN_test(model_path='models/model-900', video_feat_path=video_feat_path):
    #ipdb.set_trace()
    train_data, test_data = get_fcn_video_data(video_data_path, video_feat_path, fcn_feat_path, train_ratio)
    test_videos = test_data['video_path'].unique()
    #test_videos = os.listdir(video_feat_path)
    ixtoword = pd.Series(np.load('data/ixtoword_'+str(train_ratio)+'.npy').tolist())

    model = FCN_Video_Caption_Generator(
            dim_image=dim_image,
            n_words=len(ixtoword),
            dim_hidden=dim_hidden,
            batch_size=batch_size,
            n_lstm_steps=n_frame_step,
            drop_out_rate=drop_out_rate,
            bias_init_vector=None)

    video_tf, attn_video_tf, video_mask_tf, caption_tf, probs_tf, last_embed_tf = model.build_generator()
    sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))

    saver = tf.train.Saver()
    saver.restore(sess, model_path)

    f=open(os.path.basename(model_path)+'.txt','w')
    #f=open('hollywood_1000_atten.txt','w')

    check_dict={}

    for video in test_videos:
    #for video_feat_path in test_videos:
        #ipdb.set_trace()
        vid = video.split('/')[-1].split('.')[0]
        #word = vid.split('_')
        #lenth = len(word[-1])+len(word[-2])+2
        #realvid = vid[:-lenth]
        #realvid = video
        realvid = vid
        print video
        #print video_feat_path
        if realvid in check_dict:
            pass
        else:
            #video_path = video_feat_path + '/' + video
            video_path = video
            video_feat = np.load(video_path)[None,...]
            attn_video_path = fcn_feat_path + '/' + video.split('/')[-1]
            attn_video_feat = np.load(attn_video_path)[None,...]

            #video_feat = np.load(video_feat_path)[None,...]
            video_mask = np.ones((video_feat.shape[0], video_feat.shape[1]))

            generated_word_index = sess.run(caption_tf, feed_dict={video_tf:video_feat, attn_video_tf:attn_video_feat, video_mask_tf:video_mask})
            probs_val = sess.run(probs_tf, feed_dict={video_tf:video_feat, attn_video_tf:attn_video_feat, video_mask_tf:video_mask})
            embed_val = sess.run(last_embed_tf, feed_dict={video_tf:video_feat, attn_video_tf:attn_video_feat, video_mask_tf:video_mask})
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


def main():
    model_path = os.path.join(model_base_path,'model_'+args.mode)
    if os.path.exists(model_path):
        pass
    else:
        os.mkdir(model_path)

    if args.mode == 'base' or args.mode == 'flipped':
        train(model_path)
        #test(model_path=os.path.join(model_path,'model-0'))
    elif args.mode == 'fcn':
        fcn_feat_path='/DATA3_DB7/data/yhu/fcn_'+args.FCNLayer+'_YouTubeFeats'
        FCN_train(model_path=model_path, fcn_feat_path=fcn_feat_path)
        #FCN_test(model_path=os.path.join(model_path,'model-0'))
    else:
        print 'wrong mode!'


if __name__ == '__main__':
    main()