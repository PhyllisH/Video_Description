cnn_util.py  the CNN model, get the features of frames
preprocessing.py process the dataset, change the videos into features
model_changed.py the original model, 2lstm, implement of the structure
model_fcn.py the fcn+original model

#train or test, change the command in model_changed.py
#tensorflow+numpy+pandas
python model_changed.py


#2017/12/14
models.py 
three implements of VD task
video_caption_generator: the s2vt structure
flipped_video_caption_generator: in the decode phase, the words get in from lstm2 and get out from lstm1
fcn_video_caption_generator: the lstm1 gets two inputs one of which is original feature, the other is the feature*fcn(attention on human&20objects)

train.py
CUDA_VISIBLE_DEVICES=0 python train.py --mode=base (flipped or fcn)
the implement of loading data, train model and save model

evaluate.py
CUDA_VISIBLE_DEVICES=0 python evaluate.py
evaluate the models under the model_path(model_base_path+mode)
get the captions of the test videos and the grades of the standards
saved in test_file_path(test_base_path+mode)

configure.py
the configuration of varables, locations

