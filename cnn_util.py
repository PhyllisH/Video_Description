import caffe
import ipdb
import cv2
import numpy as np
import skimage
import matplotlib.pyplot as plt

deploy = '/DB/rhome/yhu/caffe2public/models/bvlc_reference_caffenet/deploy.prototxt'
model = '/DB/rhome/yhu/caffe2public/models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'
mean = '/DB/rhome/yhu/caffe2public/python/caffe/imagenet/ilsvrc_2012_mean.npy'
#attn_deploy = '/DB/rhome/yhu/tensorflow/fcn/voc-fcn8s/deploy.prototxt'
#attn_model = '/DB/rhome/yhu/tensorflow/fcn/voc-fcn8s/fcn8s-heavy-pascal.caffemodel'
attn_deploy ='/DB/rhome/yhu/tensorflow/video_to_sequence/fcn_28*28/deploy.prototxt'
attn_model = '/DB/rhome/yhu/tensorflow/video_to_sequence/fcn_28*28/train_iter_50000_28x28_lr_1e-5.caffemodel'

class CNN(object):

    def __init__(self, deploy=deploy, model=model, attn_deploy=attn_deploy, attn_model=attn_model, mean=mean, batch_size=10, width=227, height=227):

        self.deploy = deploy
        self.model = model
        self.mean = mean

        self.attn_model = attn_model
        self.attn_deploy = attn_deploy

        self.batch_size = batch_size
        self.attn_net, self.net, self.transformer = self.get_net()
        self.net.blobs['data'].reshape(self.batch_size, 3, height, width)

        self.width = width
        self.height = height

    def get_net(self):
        caffe.set_mode_gpu()
        net = caffe.Net(self.deploy, self.model, caffe.TEST)
        attn_net = caffe.Net(self.attn_deploy, self.attn_model, caffe.TEST)

        transformer = caffe.io.Transformer({'data':net.blobs['data'].data.shape})
        transformer.set_transpose('data', (2,0,1))
        transformer.set_mean('data', np.load(self.mean).mean(1).mean(1))
        transformer.set_raw_scale('data', 255)
        transformer.set_channel_swap('data', (2,1,0))

        return attn_net, net, transformer

    def get_features(self, image_list, layers='fc7', layer_sizes=[4096]):
        iter_until = len(image_list) + self.batch_size
        all_feats = np.zeros([len(image_list)] + layer_sizes)

        for start, end in zip(range(0, iter_until, self.batch_size), \
                              range(self.batch_size, iter_until, self.batch_size)):

            image_batch = image_list[start:end]

            caffe_in = np.zeros(np.array(image_batch.shape)[[0,3,1,2]], dtype=np.float32)

            for idx, in_ in enumerate(image_batch):
                caffe_in[idx] = self.transformer.preprocess('data', in_)

            out = self.net.forward_all(blobs=[layers], **{'data':caffe_in})
            feats = out[layers]

            all_feats[start:end] = feats

        return all_feats

    #def get_attention(self, image_list, layers='score', layer_sizes=[224,224]):
    #    iter_until = len(image_list) + self.batch_size
        #layer_sizes = [image_list[0].shape[0],image_list[0].shape[1]]
        #all_attns = np.zeros([len(image_list)] + layer_sizes)
    #    all_attn_frames = np.zeros([image_list.shape[0],21,image_list.shape[1],image_list.shape[2]])
        #all_attns = np.zeros(image_list.shape[:-1])

    #    for start, end in zip(range(0, iter_until, self.batch_size), \
    #                          range(self.batch_size, iter_until, self.batch_size)):

     #       image_batch = image_list[start:end]

     #       caffe_in = np.zeros(np.array(image_batch.shape)[[0,3,1,2]], dtype=np.float32)

     #       for idx, in_ in enumerate(image_batch):
      #          caffe_in[idx] = self.transformer.preprocess('data', in_)
#
     #       attn = self.attn_net.forward_all(blobs=[layers], **{'data':caffe_in})
      #      attn_frame = attn[layers]
            #attns = attn[layers].max(axis=1)
            #for x in range(0,self.batch_size):
            #    attns[x] = (attns[x] - attns[x].min()) / attns[x].max()

            #all_attns[start:end] = attns
      #      all_attn_frames[start:end] = attn_frame
        #for x in xrange(1,image_list.shape[-1]+1):
        #    image_list[:,:,:,x-1] = np.multiply(image_list[:,:,:,x-1], all_attns)

        #return image_list
      #  return all_attn_frames

    def get_attention(self, image_list, layers='score_new', cnn_layer='conv3_4', cnn_out_layer='fc7',layer_sizes=[4096]):
        iter_until = len(image_list) + self.batch_size
        #layer_sizes = [image_list[0].shape[0],image_list[0].shape[1]]
        #all_attns = np.zeros([len(image_list)] + layer_sizes)
        all_feats = np.zeros([len(image_list)] + layer_sizes)
        #vgg_relu3_4 = np.zeros([image_list.shape[0],56,56])
        #all_attns = np.zeros(image_list.shape[:-1])

        for start, end in zip(range(0, iter_until, self.batch_size), \
                              range(self.batch_size, iter_until, self.batch_size)):

            image_batch = image_list[start:end]

            caffe_in = np.zeros(np.array(image_batch.shape)[[0,3,1,2]], dtype=np.float32)

            for idx, in_ in enumerate(image_batch):
                caffe_in[idx] = self.transformer.preprocess('data', in_)

            attn = self.attn_net.forward_all(blobs=[layers], **{'data':caffe_in})
            
            tmp = self.net.forward(start='input', end=cnn_layer, **{'data':caffe_in})
            vgg_conv3_4 = tmp[cnn_layer]
            vgg_conv3_4 = np.maximum(vgg_conv3_4,0)

            for i in range(0,vgg_conv3_4.shape[1]):
                vgg_conv3_4[:,i,:,:] = np.multiply(vgg_conv3_4[:,i,:,:], attn[layers][:,0,:,:])
            vgg_conv3_4.reshape([10,256,56,56])
            self.net.blobs['conv3_4'].data[...] = vgg_conv3_4
            out = self.net.forward(blobs=[cnn_out_layer],start='pool3')
            #out = self.net.forward(blobs=[cnn_out_layer],start='relu3_4', end='prob',**{'relu3_4':vgg_conv3_4})

            attn_frame = out[cnn_out_layer]
            #attns = attn[layers].max(axis=1)
            #for x in range(0,self.batch_size):
            #    attns[x] = (attns[x] - attns[x].min()) / attns[x].max()

            #all_attns[start:end] = attns
            all_feats[start:end] = attn_frame
        #for x in xrange(1,image_list.shape[-1]+1):
        #    image_list[:,:,:,x-1] = np.multiply(image_list[:,:,:,x-1], all_attns)

        #return image_list
        return all_feats




















