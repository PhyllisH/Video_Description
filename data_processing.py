# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import os
import sys
import ipdb
from configure import *

sys.path.append('/DB/rhome/yhu/tensorflow/caption-eval/coco-caption/')
sys.path.append('/DB/rhome/yhu/tensorflow/caption-eval/')

from run_evaluations import *
from create_json_references import *

def preProBuildWordVocab(sentence_iterator, word_count_threshold=5): # borrowed this function from NeuralTalk
    print 'preprocessing word counts and creating vocab based on word count threshold %d' % (word_count_threshold, )
    word_counts = {}
    nsents = 0
    for sent in sentence_iterator:
        nsents += 1
        for w in sent.lower().split(' '):
           word_counts[w] = word_counts.get(w, 0) + 1

    vocab = [w for w in word_counts if word_counts[w] >= word_count_threshold]
    print 'filtered words from %d to %d' % (len(word_counts), len(vocab))

    ixtoword = {}
    ixtoword[0] = '.'  # period at the end of the sentence. make first dimension be end token
    wordtoix = {}
    wordtoix['#START#'] = 0 # make first vector be the start token
    ix = 1
    for w in vocab:
        wordtoix[w] = ix
        ixtoword[ix] = w
        ix += 1

    word_counts['.'] = nsents
    bias_init_vector = np.array([1.0*word_counts[ixtoword[i]] for i in ixtoword])
    bias_init_vector /= np.sum(bias_init_vector) # normalize to frequencies
    bias_init_vector = np.log(bias_init_vector)
    bias_init_vector -= np.max(bias_init_vector) # shift to nice numeric range
    return wordtoix, ixtoword, bias_init_vector


def get_fcn_video_data(video_data_path, video_feat_path, fcn_feat_path, train_ratio=0.75):
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


def get_ActivityNet_video_data(video_data_path, video_feat_path, train_ratio=0.75):
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


def get_video_data(video_data_path, video_feat_path, train_ratio=0.75):
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
    tmp = os.path.split(test_file)
    fnew = open(os.path.join(tmp[0], tmp[1].split('.txt')[0]+'_new.txt'),'w')

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

def generate_ref(allref_file=allref_file, test_id_file=test_id_file):
    f_allref = open(allref_file,'r')
    f_test= open(test_id_file, 'r')
    testref = os.path.join(base_path,'data',test_id_file.split('_new.txt')[0]+'reference.txt')
    f_testref = open(testref,'w')

    lines = f_test.readlines()
    check_dict=[]

    for line in lines:
        testname = line.split('\t')[0]
        #newid = line.split(' ')[1]
        check_dict.append(testname)
    #ipdb.set_trace()
    old_lines = f_allref.readlines()

    for old_line in old_lines:
        oldname = old_line.split('\t')[0]
        if oldname in check_dict:
            f_testref.write(old_line)

    f_allref.close()
    f_test.close()
    f_testref.close()

    input_file = testref
    output_file = '{0}.json'.format(input_file)
    
    crf = CocoAnnotations()
    crf.read_file(input_file)
    crf.dump_json(output_file)
    print 'Created json references in %s' % output_file

def evaluate(input_file=test_id_file, references_json=references_json):
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

