############### Global Parameters ###############
#the video path
video_path = '/DATA2/data/yhu/YouTubeClips'
#the labels of videos
video_data_path='data/MSR Video Description Corpus.csv'
#the trained model
model_base_path = '/DATA3_DB7/data/yhu/VD_results/'
#the features of extracted 80 frames from the video
video_feat_path = '/DATA2/data/yhu/YouTubeFeats'
#the fcn features
fcn_feat_path = '/DATA3_DB7/data/yhu/fcn_conv3_4_YouTubeFeats'
#fcn_feat_path = '/DATA3_DB7/data/yhu/fcn_conv3_1_YouTubeFeats'
#fcn_feat_path = '/DATA3_DB7/data/yhu/fcn_pool4_YouTubeFeats'
#fcn_feat_path = '/DATA3_DB7/data/yhu/fcn_conv5_4_YouTubeFeats'

###############  data_preprocessing  ###############
train_ratio = 0.75
word_count_threshold = 10
mapping_ref = 'data/youtube_video_to_id_mapping.txt'
test_file = 'data/test_0.25.txt'
test_id_file = 'data/test_0.25_new.txt'

############## Train Parameters ###################
dim_image = 4096
dim_hidden= 128
n_frame_step = 80
n_epochs = 1000
batch_size = 100
learning_rate = 0.001
#drop_out_rate = 0.2
drop_out_rate = 0

############## Evaluation Parameters ################
base_path = '/DB/rhome/yhu/tensorflow/VideoDescription/'
allref_file = 'data/youtube_reference.txt'
references_json = 'data/test_0.25_reference.json'
test_base_path = '/DATA3_DB7/data/yhu/VD_results/'
