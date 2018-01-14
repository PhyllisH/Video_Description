#-*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn
from keras.preprocessing import sequence

class Video_Caption_Generator():
    def __init__(self, dim_image, n_words, dim_hidden, batch_size, n_lstm_steps, drop_out_rate, bias_init_vector=None):
        self.dim_image = dim_image
        self.n_words = n_words
        self.dim_hidden = dim_hidden
        self.batch_size = batch_size
        self.n_lstm_steps = n_lstm_steps
        self.drop_out_rate = drop_out_rate

        with tf.device("/gpu:0"):
            self.Wemb = tf.Variable(tf.random_uniform([n_words, dim_hidden], -0.1, 0.1), name='Wemb')

        self.lstm1 = rnn.BasicLSTMCell(dim_hidden,state_is_tuple=False,input_shape=self.dim_image)
        self.lstm2 = rnn.BasicLSTMCell(dim_hidden,state_is_tuple=False)
        self.lstm1_dropout = rnn.DropoutWrapper(self.lstm1,output_keep_prob=1 - self.drop_out_rate)
        self.lstm2_dropout = rnn.DropoutWrapper(self.lstm2,output_keep_prob=1 - self.drop_out_rate)

        #self.encode_image_W = tf.Variable( tf.random_uniform([dim_image, dim_hidden], -0.1, 0.1), name='encode_image_W')
        #self.encode_image_b = tf.Variable( tf.zeros([dim_hidden]), name='encode_image_b')

        self.embed_word_W = tf.Variable(tf.random_uniform([dim_hidden, n_words], -0.1,0.1), name='embed_word_W')
        if bias_init_vector is not None:
            self.embed_word_b = tf.Variable(bias_init_vector.astype(np.float32), name='embed_word_b')
        else:
            self.embed_word_b = tf.Variable(tf.zeros([n_words]), name='embed_word_b')

    def build_model(self):
        video = tf.placeholder(tf.float32, [self.batch_size, self.n_lstm_steps, self.dim_image]) # b * n * h
        video_mask = tf.placeholder(tf.float32, [self.batch_size, self.n_lstm_steps]) # b * n

        caption = tf.placeholder(tf.int32, [self.batch_size, self.n_lstm_steps]) # b * n
        caption_mask = tf.placeholder(tf.float32, [self.batch_size, self.n_lstm_steps]) # b * n

        video_flat = tf.reshape(video, [-1, self.dim_image])
        #image_emb = tf.nn.xw_plus_b( video_flat, self.encode_image_W, self.encode_image_b) # (batch_size*n_lstm_steps, dim_hidden)
        #image_emb = tf.reshape(image_emb, [self.batch_size, self.n_lstm_steps, self.dim_hidden]) # b * n * h
        image_emb = tf.reshape(video_flat, [self.batch_size, self.n_lstm_steps, self.dim_image])

        state1 = tf.zeros([self.batch_size, self.lstm1.state_size]) # b * s
        state2 = tf.zeros([self.batch_size, self.lstm2.state_size]) # b * s
        padding = tf.zeros([self.batch_size, self.dim_hidden]) # b * h

        probs = []

        loss = 0.0

        with tf.variable_scope(tf.get_variable_scope()) as scope:

            for i in range(self.n_lstm_steps): ## Phase 1 => only read frames
                if i == 0:
                    with tf.variable_scope("LSTM1"):
                        output1, state1 = self.lstm1_dropout( inputs=image_emb[:,i,:], state=state1 )

                    with tf.variable_scope("LSTM2"):
                        output2, state2 = self.lstm2_dropout( inputs=tf.concat([padding, output1],1), state=state2 )
                    #tf.get_variable_scope().reuse_variables()

                with tf.variable_scope("LSTM1",reuse = True):
                    output1, state1 = self.lstm1_dropout( inputs=image_emb[:,i,:], state=state1 )

                with tf.variable_scope("LSTM2",reuse = True):
                    output2, state2 = self.lstm2_dropout( inputs=tf.concat([padding, output1],1), state=state2 )

            # Each video might have different length. Need to mask those.
            # But how? Padding with 0 would be enough?
            # Therefore... TODO: for those short videos, keep the last LSTM hidden and output til the end.

            for i in range(self.n_lstm_steps): ## Phase 2 => only generate captions
                if i == 0:
                    current_embed = tf.zeros([self.batch_size, self.dim_hidden])    # b * h
                else:
                    with tf.device("/gpu:0"):
                        current_embed = tf.nn.embedding_lookup(self.Wemb, caption[:,i-1])

                #tf.get_variable_scope().reuse_variables()
                with tf.variable_scope("LSTM1",reuse = True):
                    output1, state1 = self.lstm1_dropout( padding, state1 )

                with tf.variable_scope("LSTM2",reuse = True):
                    output2, state2 = self.lstm2_dropout( tf.concat([current_embed, output1],1), state2 )

                labels = tf.expand_dims(caption[:,i], 1) # b*1
                indices = tf.expand_dims(tf.range(0, self.batch_size, 1), 1) # b x 1
                concated = tf.concat([indices, labels],1) # b x 2
                onehot_labels = tf.sparse_to_dense(concated, tf.stack([self.batch_size, self.n_words]), 1.0, 0.0) # b x w

                logit_words = tf.nn.xw_plus_b(output2, self.embed_word_W, self.embed_word_b) # b x w
                cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logit_words, labels=onehot_labels) # b x 1
                cross_entropy = cross_entropy * caption_mask[:,i] # b * 1

                probs.append(logit_words)

                current_loss = tf.reduce_sum(cross_entropy) # 1
                loss += current_loss

        loss = loss / tf.reduce_sum(caption_mask)
        return loss, video, video_mask, caption, caption_mask, probs


    def build_generator(self):
        video = tf.placeholder(tf.float32, [1, self.n_lstm_steps, self.dim_image])
        video_mask = tf.placeholder(tf.float32, [1, self.n_lstm_steps])

        video_flat = tf.reshape(video, [-1, self.dim_image])
        #image_emb = tf.nn.xw_plus_b( video_flat, self.encode_image_W, self.encode_image_b)
        #image_emb = tf.reshape(image_emb, [1, self.n_lstm_steps, self.dim_hidden])
        image_emb = tf.reshape(video_flat, [1, self.n_lstm_steps, self.dim_image])

        state1 = tf.zeros([1, self.lstm1.state_size])
        state2 = tf.zeros([1, self.lstm2.state_size])
        padding = tf.zeros([1, self.dim_hidden])

        generated_words = []

        probs = []
        embeds = []

        #with tf.variable_scope(tf.get_variable_scope()) as scope:
        for i in range(self.n_lstm_steps):
            if i > 0: 
                tf.get_variable_scope().reuse_variables()
                

            with tf.variable_scope("LSTM1"):
                output1, state1 = self.lstm1_dropout( image_emb[:,i,:], state1 )

            with tf.variable_scope("LSTM2"):
                output2, state2 = self.lstm2_dropout( tf.concat([padding,output1],1), state2 )

        for i in range(self.n_lstm_steps):

            tf.get_variable_scope().reuse_variables()

            if i == 0:
                current_embed = tf.zeros([1, self.dim_hidden])

            with tf.variable_scope("LSTM1"):
                output1, state1 = self.lstm1_dropout( padding, state1 )

            with tf.variable_scope("LSTM2"):
                output2, state2 = self.lstm2_dropout( tf.concat([current_embed,output1],1), state2 )

            logit_words = tf.nn.xw_plus_b( output2, self.embed_word_W, self.embed_word_b)
            max_prob_index = tf.argmax(logit_words, 1)[0]
            generated_words.append(max_prob_index)
            probs.append(logit_words)

            with tf.device("/gpu:0"):
                current_embed = tf.nn.embedding_lookup(self.Wemb, max_prob_index)
                current_embed = tf.expand_dims(current_embed, 0)

            embeds.append(current_embed)

        return video, video_mask, generated_words, probs, embeds


class FCN_Video_Caption_Generator():
    def __init__(self, dim_image, n_words, dim_hidden, batch_size, n_lstm_steps, drop_out_rate, bias_init_vector=None):
        self.dim_image = dim_image
        self.n_words = n_words
        self.dim_hidden = dim_hidden
        self.batch_size = batch_size
        self.n_lstm_steps = n_lstm_steps
        self.drop_out_rate = drop_out_rate

        with tf.device("/gpu:0"):
            self.Wemb = tf.Variable(tf.random_uniform([n_words, dim_hidden], -0.1, 0.1), name='Wemb')

        self.lstm1 = rnn.BasicLSTMCell(dim_hidden,state_is_tuple=False)
        self.lstm2 = rnn.BasicLSTMCell(dim_hidden,state_is_tuple=False)
        self.lstm1_dropout = rnn.DropoutWrapper(self.lstm1,output_keep_prob=1 - self.drop_out_rate)
        self.lstm2_dropout = rnn.DropoutWrapper(self.lstm2,output_keep_prob=1 - self.drop_out_rate)

        self.encode_image_W = tf.Variable( tf.random_uniform([dim_image, dim_hidden], -0.1, 0.1), name='encode_image_W')
        self.encode_image_b = tf.Variable( tf.zeros([dim_hidden]), name='encode_image_b')

        #self.encode_image_W_1 = tf.Variable( tf.random_uniform([dim_image, dim_hidden], -0.1, 0.1), name='encode_image_W_1')
        #self.encode_image_b_1 = tf.Variable( tf.zeros([dim_hidden]), name='encode_image_b_1')


        self.embed_word_W = tf.Variable(tf.random_uniform([dim_hidden, n_words], -0.1,0.1), name='embed_word_W')
        if bias_init_vector is not None:
            self.embed_word_b = tf.Variable(bias_init_vector.astype(np.float32), name='embed_word_b')
        else:
            self.embed_word_b = tf.Variable(tf.zeros([n_words]), name='embed_word_b')

    def build_model(self):
        video = tf.placeholder(tf.float32, [self.batch_size, self.n_lstm_steps, self.dim_image]) # b * n * h
        video_mask = tf.placeholder(tf.float32, [self.batch_size, self.n_lstm_steps]) # b * n

        caption = tf.placeholder(tf.int32, [self.batch_size, self.n_lstm_steps]) # b * n
        caption_mask = tf.placeholder(tf.float32, [self.batch_size, self.n_lstm_steps]) # b * n

        video_flat = tf.reshape(video, [-1, self.dim_image])
        image_emb = tf.nn.xw_plus_b( video_flat, self.encode_image_W, self.encode_image_b) # (batch_size*n_lstm_steps, dim_hidden)
        image_emb = tf.reshape(image_emb, [self.batch_size, self.n_lstm_steps, self.dim_hidden]) # b * n * h

        #used for attention
        attn_video = tf.placeholder(tf.float32, [self.batch_size, self.n_lstm_steps, self.dim_image]) # b * n * h
        attn_video_flat = tf.reshape(attn_video, [-1, self.dim_image])
        attn_image_emb = tf.nn.xw_plus_b( attn_video_flat, self.encode_image_W, self.encode_image_b) # (batch_size*n_lstm_steps, dim_hidden)
        attn_image_emb = tf.reshape(attn_image_emb, [self.batch_size, self.n_lstm_steps, self.dim_hidden]) # b * n * h
        

        state1 = tf.zeros([self.batch_size, self.lstm1.state_size]) # b * s
        state2 = tf.zeros([self.batch_size, self.lstm2.state_size]) # b * s
        padding = tf.zeros([self.batch_size, self.dim_hidden]) # b * h

        probs = []

        loss = 0.0

        with tf.variable_scope(tf.get_variable_scope()) as scope:

            for i in range(self.n_lstm_steps): ## Phase 1 => only read frames
                if i == 0:
                    with tf.variable_scope("LSTM1"):
                        output1, state1 = self.lstm1_dropout( inputs=tf.concat([image_emb[:,i,:],attn_image_emb[:,i,:]],1), state=state1 )

                    with tf.variable_scope("LSTM2"):
                        output2, state2 = self.lstm2_dropout( inputs=tf.concat([padding, output1],1), state=state2 )
                    #tf.get_variable_scope().reuse_variables()

                with tf.variable_scope("LSTM1",reuse = True):
                    output1, state1 = self.lstm1_dropout( inputs=tf.concat([image_emb[:,i,:],attn_image_emb[:,i,:]],1), state=state1 )

                with tf.variable_scope("LSTM2",reuse = True):
                    output2, state2 = self.lstm2_dropout( inputs=tf.concat([padding, output1],1), state=state2 )

            # Each video might have different length. Need to mask those.
            # But how? Padding with 0 would be enough?
            # Therefore... TODO: for those short videos, keep the last LSTM hidden and output til the end.

            for i in range(self.n_lstm_steps): ## Phase 2 => only generate captions
                if i == 0:
                    current_embed = tf.zeros([self.batch_size, self.dim_hidden])    # b * h
                else:
                    with tf.device("/gpu:0"):
                        current_embed = tf.nn.embedding_lookup(self.Wemb, caption[:,i-1])

                #tf.get_variable_scope().reuse_variables()
                with tf.variable_scope("LSTM1",reuse = True):
                    output1, state1 = self.lstm1_dropout( tf.concat([padding, padding],1), state1 )

                with tf.variable_scope("LSTM2",reuse = True):
                    output2, state2 = self.lstm2_dropout( tf.concat([current_embed, output1],1), state2 )

                labels = tf.expand_dims(caption[:,i], 1) # b*1
                indices = tf.expand_dims(tf.range(0, self.batch_size, 1), 1) # b x 1
                concated = tf.concat([indices, labels],1) # b x 2
                onehot_labels = tf.sparse_to_dense(concated, tf.stack([self.batch_size, self.n_words]), 1.0, 0.0) # b x w

                logit_words = tf.nn.xw_plus_b(output2, self.embed_word_W, self.embed_word_b) # b x w
                cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logit_words, labels=onehot_labels) # b x 1
                cross_entropy = cross_entropy * caption_mask[:,i] # b * 1

                probs.append(logit_words)

                current_loss = tf.reduce_sum(cross_entropy) # 1
                loss += current_loss

        loss = loss / tf.reduce_sum(caption_mask)
        return loss, video, attn_video, video_mask, caption, caption_mask, probs


    def build_generator(self):
        video = tf.placeholder(tf.float32, [1, self.n_lstm_steps, self.dim_image])
        video_mask = tf.placeholder(tf.float32, [1, self.n_lstm_steps])

        video_flat = tf.reshape(video, [-1, self.dim_image])
        image_emb = tf.nn.xw_plus_b( video_flat, self.encode_image_W, self.encode_image_b)
        image_emb = tf.reshape(image_emb, [1, self.n_lstm_steps, self.dim_hidden])

        state1 = tf.zeros([1, self.lstm1.state_size])
        state2 = tf.zeros([1, self.lstm2.state_size])
        padding = tf.zeros([1, self.dim_hidden])

        #used for attention
        attn_video = tf.placeholder(tf.float32, [1, self.n_lstm_steps, self.dim_image]) # b * n * h
        attn_video_flat = tf.reshape(attn_video, [-1, self.dim_image])
        attn_image_emb = tf.nn.xw_plus_b( attn_video_flat, self.encode_image_W, self.encode_image_b) # (batch_size*n_lstm_steps, dim_hidden)
        attn_image_emb = tf.reshape(attn_image_emb, [1, self.n_lstm_steps, self.dim_hidden]) # b * n * h

        generated_words = []

        probs = []
        embeds = []

        #with tf.variable_scope(tf.get_variable_scope()) as scope:
        for i in range(self.n_lstm_steps):
            if i > 0: 
                tf.get_variable_scope().reuse_variables()
                

            with tf.variable_scope("LSTM1"):
                output1, state1 = self.lstm1_dropout( tf.concat([image_emb[:,i,:],attn_image_emb[:,i,:]],1), state1 )

            with tf.variable_scope("LSTM2"):
                output2, state2 = self.lstm2_dropout( tf.concat([padding,output1],1), state2 )

        for i in range(self.n_lstm_steps):

            tf.get_variable_scope().reuse_variables()

            if i == 0:
                current_embed = tf.zeros([1, self.dim_hidden])

            with tf.variable_scope("LSTM1"):
                output1, state1 = self.lstm1_dropout( tf.concat([padding, padding],1), state1 )

            with tf.variable_scope("LSTM2"):
                output2, state2 = self.lstm2_dropout( tf.concat([current_embed,output1],1), state2 )


            logit_words = tf.nn.xw_plus_b( output2, self.embed_word_W, self.embed_word_b)
            max_prob_index = tf.argmax(logit_words, 1)[0]
            generated_words.append(max_prob_index)
            probs.append(logit_words)

            with tf.device("/gpu:0"):
                current_embed = tf.nn.embedding_lookup(self.Wemb, max_prob_index)
                current_embed = tf.expand_dims(current_embed, 0)

            embeds.append(current_embed)

        return video, attn_video, video_mask, generated_words, probs, embeds

class Flipped_Video_Caption_Generator():
    def __init__(self, dim_image, n_words, dim_hidden, batch_size, n_lstm_steps, drop_out_rate, bias_init_vector=None):
        self.dim_image = dim_image
        self.n_words = n_words
        self.dim_hidden = dim_hidden
        self.batch_size = batch_size
        self.n_lstm_steps = n_lstm_steps
        self.drop_out_rate = drop_out_rate

        with tf.device("/gpu:0"):
            self.Wemb = tf.Variable(tf.random_uniform([n_words, dim_hidden], -0.1, 0.1), name='Wemb')

        self.lstm1 = rnn.BasicLSTMCell(dim_hidden,state_is_tuple=False)
        self.lstm2 = rnn.BasicLSTMCell(dim_hidden,state_is_tuple=False)
        self.lstm1_dropout = rnn.DropoutWrapper(self.lstm1,output_keep_prob=1 - self.drop_out_rate)
        self.lstm2_dropout = rnn.DropoutWrapper(self.lstm2,output_keep_prob=1 - self.drop_out_rate)

        self.encode_image_W = tf.Variable( tf.random_uniform([dim_image, dim_hidden], -0.1, 0.1), name='encode_image_W')
        self.encode_image_b = tf.Variable( tf.zeros([dim_hidden]), name='encode_image_b')

        self.embed_word_W = tf.Variable(tf.random_uniform([dim_hidden, n_words], -0.1,0.1), name='embed_word_W')
        if bias_init_vector is not None:
            self.embed_word_b = tf.Variable(bias_init_vector.astype(np.float32), name='embed_word_b')
        else:
            self.embed_word_b = tf.Variable(tf.zeros([n_words]), name='embed_word_b')

    def build_model(self):
        video = tf.placeholder(tf.float32, [self.batch_size, self.n_lstm_steps, self.dim_image]) # b * n * h
        video_mask = tf.placeholder(tf.float32, [self.batch_size, self.n_lstm_steps]) # b * n

        caption = tf.placeholder(tf.int32, [self.batch_size, self.n_lstm_steps]) # b * n
        caption_mask = tf.placeholder(tf.float32, [self.batch_size, self.n_lstm_steps]) # b * n

        video_flat = tf.reshape(video, [-1, self.dim_image])
        image_emb = tf.nn.xw_plus_b( video_flat, self.encode_image_W, self.encode_image_b) # (batch_size*n_lstm_steps, dim_hidden)
        image_emb = tf.reshape(image_emb, [self.batch_size, self.n_lstm_steps, self.dim_hidden]) # b * n * h

        state1 = tf.zeros([self.batch_size, self.lstm1.state_size]) # b * s
        state2 = tf.zeros([self.batch_size, self.lstm2.state_size]) # b * s
        padding = tf.zeros([self.batch_size, self.dim_hidden]) # b * h

        probs = []

        loss = 0.0

        with tf.variable_scope(tf.get_variable_scope()) as scope:

            for i in range(self.n_lstm_steps): ## Phase 1 => only read frames
                if i == 0:
                    with tf.variable_scope("LSTM1"):
                        output1, state1 = self.lstm1_dropout( inputs=image_emb[:,i,:], state=state1 )

                    with tf.variable_scope("LSTM2"):
                        output2, state2 = self.lstm2_dropout( inputs=output1, state=state2 )
                    #tf.get_variable_scope().reuse_variables()

                with tf.variable_scope("LSTM1",reuse = True):
                    output1, state1 = self.lstm1_dropout( inputs=image_emb[:,i,:], state=state1 )

                with tf.variable_scope("LSTM2",reuse = True):
                    output2, state2 = self.lstm2_dropout( inputs=output1, state=state2 )

            # Each video might have different length. Need to mask those.
            # But how? Padding with 0 would be enough?
            # Therefore... TODO: for those short videos, keep the last LSTM hidden and output til the end.

            for i in range(self.n_lstm_steps): ## Phase 2 => only generate captions
                if i == 0:
                    current_embed = tf.zeros([self.batch_size, self.dim_hidden])    # b * h
                else:
                    with tf.device("/gpu:0"):
                        current_embed = tf.nn.embedding_lookup(self.Wemb, caption[:,i-1])

                #word gets in from lstm2 and get out from lstm1
                with tf.variable_scope("LSTM2",reuse = True):
                    output2, state2 = self.lstm2_dropout( current_embed, state2 )

                with tf.variable_scope("LSTM1",reuse = True):
                    output1, state1 = self.lstm1_dropout( output2, state1 )

                labels = tf.expand_dims(caption[:,i], 1) # b*1
                indices = tf.expand_dims(tf.range(0, self.batch_size, 1), 1) # b x 1
                concated = tf.concat([indices, labels],1) # b x 2
                onehot_labels = tf.sparse_to_dense(concated, tf.stack([self.batch_size, self.n_words]), 1.0, 0.0) # b x w

                logit_words = tf.nn.xw_plus_b(output1, self.embed_word_W, self.embed_word_b) # b x w
                cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logit_words, labels=onehot_labels) # b x 1
                cross_entropy = cross_entropy * caption_mask[:,i] # b * 1

                probs.append(logit_words)

                current_loss = tf.reduce_sum(cross_entropy) # 1
                loss += current_loss

        loss = loss / tf.reduce_sum(caption_mask)
        return loss, video, video_mask, caption, caption_mask, probs


    def build_generator(self):
        video = tf.placeholder(tf.float32, [1, self.n_lstm_steps, self.dim_image])
        video_mask = tf.placeholder(tf.float32, [1, self.n_lstm_steps])

        video_flat = tf.reshape(video, [-1, self.dim_image])
        image_emb = tf.nn.xw_plus_b( video_flat, self.encode_image_W, self.encode_image_b)
        image_emb = tf.reshape(image_emb, [1, self.n_lstm_steps, self.dim_hidden])

        state1 = tf.zeros([1, self.lstm1.state_size])
        state2 = tf.zeros([1, self.lstm2.state_size])
        padding = tf.zeros([1, self.dim_hidden])

        generated_words = []

        probs = []
        embeds = []

        #with tf.variable_scope(tf.get_variable_scope()) as scope:
        for i in range(self.n_lstm_steps):
            if i > 0: 
                tf.get_variable_scope().reuse_variables()
                

            with tf.variable_scope("LSTM1"):
                output1, state1 = self.lstm1_dropout( image_emb[:,i,:], state1 )

            with tf.variable_scope("LSTM2"):
                output2, state2 = self.lstm2_dropout( output1, state2 )

        for i in range(self.n_lstm_steps):

            tf.get_variable_scope().reuse_variables()

            if i == 0:
                current_embed = tf.zeros([1, self.dim_hidden])

            with tf.variable_scope("LSTM2"):
                output2, state2 = self.lstm2_dropout( current_embed, state2 )

            with tf.variable_scope("LSTM1"):
                output1, state1 = self.lstm1_dropout( output2, state1 )

            logit_words = tf.nn.xw_plus_b( output1, self.embed_word_W, self.embed_word_b)
            max_prob_index = tf.argmax(logit_words, 1)[0]
            generated_words.append(max_prob_index)
            probs.append(logit_words)

            with tf.device("/gpu:0"):
                current_embed = tf.nn.embedding_lookup(self.Wemb, max_prob_index)
                current_embed = tf.expand_dims(current_embed, 0)

            embeds.append(current_embed)

        return video, video_mask, generated_words, probs, embeds

