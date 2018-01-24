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

class BackUp_Hierarchical_Generator():
    def __init__(self, dim_image, n_words, dim_hidden, batch_size, n_lstm_steps, drop_out_rate,  step , bias_init_vector=None):
        self.dim_image = dim_image
        self.n_words = n_words
        self.dim_hidden = dim_hidden
        self.batch_size = batch_size
        self.n_lstm_steps = n_lstm_steps
        self.drop_out_rate = drop_out_rate
        self.step = step

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
        state_padding = tf.zeros([self.batch_size, self.lstm1.state_size]) # b * s

        probs = []

        loss = 0.0

        with tf.variable_scope(tf.get_variable_scope()) as scope:

            for i in range(self.n_lstm_steps): ## Phase 1 => only read frames
                if i == 0:
                    with tf.variable_scope("LSTM1"):
                        output1, state1 = self.lstm1_dropout( inputs=image_emb[:,i,:], state=state1 )
                    with tf.variable_scope("LSTM2"):
                        output2, state2 = self.lstm2_dropout( inputs=tf.concat([padding, output1],1), state=state2 )
                
                with tf.variable_scope("LSTM1",reuse = True):
                    output1, state1 = self.lstm1_dropout( inputs=image_emb[:,i,:], state=state1 )

                if (i+1)% self.step == 0:
                    with tf.variable_scope("LSTM2",reuse = True):
                        output2, state2 = self.lstm2_dropout( inputs=tf.concat([padding, output1],1), state=state2 )
                    with tf.variable_scope("LSTM1",reuse = True):
                        output1, state1 = self.lstm1_dropout( inputs=padding, state=state_padding )


            for i in range(self.n_lstm_steps): ## Phase 2 => only generate captions
                if i == 0:
                    current_embed = tf.zeros([self.batch_size, self.dim_hidden])    # b * h
                else:
                    with tf.device("/gpu:0"):
                        current_embed = tf.nn.embedding_lookup(self.Wemb, caption[:,i-1])

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
        image_emb = tf.nn.xw_plus_b( video_flat, self.encode_image_W, self.encode_image_b)
        image_emb = tf.reshape(image_emb, [1, self.n_lstm_steps, self.dim_hidden])

        state1 = tf.zeros([1, self.lstm1.state_size])
        state2 = tf.zeros([1, self.lstm2.state_size])
        padding = tf.zeros([1, self.dim_hidden])
        state_padding = tf.zeros([1, self.lstm1.state_size]) # b * s

        generated_words = []

        probs = []
        embeds = []

        #with tf.variable_scope(tf.get_variable_scope()) as scope:
        for i in range(self.n_lstm_steps):
            if i == 0:
                with tf.variable_scope("LSTM1"):
                    output1, state1 = self.lstm1_dropout( inputs=image_emb[:,i,:], state=state1 )
                with tf.variable_scope("LSTM2"):
                    output2, state2 = self.lstm2_dropout( inputs=tf.concat([padding, output1],1), state=state2 )
            
            with tf.variable_scope("LSTM1",reuse = True):
                output1, state1 = self.lstm1_dropout( inputs=image_emb[:,i,:], state=state1 )

            if (i+1) % self.step == 0:
                with tf.variable_scope("LSTM2",reuse = True):
                    output2, state2 = self.lstm2_dropout( inputs=tf.concat([padding, output1],1), state=state2 )
                with tf.variable_scope("LSTM1",reuse = True):
                    output1, state1 = self.lstm1_dropout( inputs=padding, state=state_padding )

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

class Hierarchical_Generator():
    def __init__(self, dim_image, n_words, dim_hidden, batch_size, n_lstm_steps, drop_out_rate,  step , bias_init_vector=None):
        self.dim_image = dim_image
        self.n_words = n_words
        self.dim_hidden = dim_hidden
        self.batch_size = batch_size
        self.n_lstm_steps = n_lstm_steps
        self.drop_out_rate = drop_out_rate
        self.step = step

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
        state_padding = tf.zeros([self.batch_size, self.lstm1.state_size]) # b * s

        probs = []

        loss = 0.0

        with tf.variable_scope(tf.get_variable_scope()) as scope:

            for i in range(self.n_lstm_steps): ## Phase 1 => only read frames
                #---layer 1------#
                if i == 0:
                    with tf.variable_scope("LSTM1"):
                        output1, state1 = self.lstm1_dropout( inputs=image_emb[:,i,:], state=state_padding )
                elif i % self.step == 0:
                    with tf.variable_scope("LSTM1",reuse = True):
                        output1, state1 = self.lstm1_dropout( inputs=image_emb[:,i,:], state=state_padding )
                else: 
                    with tf.variable_scope("LSTM1",reuse = True):
                        output1, state1 = self.lstm1_dropout( inputs=image_emb[:,i,:], state=state1 )

                #---layer 2------#
                if (i+1) == self.step:
                    with tf.variable_scope("LSTM2"):
                        output2, state2 = self.lstm2_dropout( inputs=tf.concat([padding, output1],1), state=state_padding )
                elif (i+1) % self.step == 0:
                    with tf.variable_scope("LSTM2",reuse = True):
                        output2, state2 = self.lstm2_dropout( inputs=tf.concat([padding, output1],1), state=state2 )
                else:
                    pass


            for i in range(self.n_lstm_steps): ## Phase 2 => only generate captions
                if i == 0:
                    current_embed = tf.zeros([self.batch_size, self.dim_hidden])    # b * h
                else:
                    with tf.device("/gpu:0"):
                        current_embed = tf.nn.embedding_lookup(self.Wemb, caption[:,i-1])

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
        image_emb = tf.nn.xw_plus_b( video_flat, self.encode_image_W, self.encode_image_b)
        image_emb = tf.reshape(image_emb, [1, self.n_lstm_steps, self.dim_hidden])

        state1 = tf.zeros([1, self.lstm1.state_size])
        state2 = tf.zeros([1, self.lstm2.state_size])
        padding = tf.zeros([1, self.dim_hidden])
        state_padding = tf.zeros([1, self.lstm1.state_size]) # b * s

        generated_words = []

        probs = []
        embeds = []

        #with tf.variable_scope(tf.get_variable_scope()) as scope:
        for i in range(self.n_lstm_steps):
            #---layer 1------#
            if i == 0:
                with tf.variable_scope("LSTM1"):
                    output1, state1 = self.lstm1_dropout( inputs=image_emb[:,i,:], state=state_padding )
            elif i % self.step == 0:
                with tf.variable_scope("LSTM1",reuse = True):
                    output1, state1 = self.lstm1_dropout( inputs=image_emb[:,i,:], state=state_padding )
            else: 
                with tf.variable_scope("LSTM1",reuse = True):
                    output1, state1 = self.lstm1_dropout( inputs=image_emb[:,i,:], state=state1 )

            #---layer 2------#
            if (i+1) == self.step:
                with tf.variable_scope("LSTM2"):
                    output2, state2 = self.lstm2_dropout( inputs=tf.concat([padding, output1],1), state=state_padding )
            elif (i+1) % self.step == 0:
                with tf.variable_scope("LSTM2",reuse = True):
                    output2, state2 = self.lstm2_dropout( inputs=tf.concat([padding, output1],1), state=state2 )
            else:
                pass

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

class SentenceVec_Hie_Generator():
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

        self.lstm_cap1 = rnn.BasicLSTMCell(dim_hidden,state_is_tuple=False)
        self.lstm_cap1_dropout = rnn.DropoutWrapper(self.lstm_cap1,output_keep_prob=1 - self.drop_out_rate)

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
        state_padding = tf.zeros([self.batch_size, self.lstm1.state_size]) # b * s

        state_cap1 = tf.zeros([self.batch_size, self.lstm_cap1.state_size]) # b * s

        probs = []

        loss = 0.0
        cap_loss = 0.0
        img_loss = 0.0
        img_cap_loss = 0.0

        with tf.variable_scope(tf.get_variable_scope()) as scope:
            #image encoder
            for i in range(self.n_lstm_steps): ## Phase 1 => only read frames
                if i == 0:
                    with tf.variable_scope("LSTM1"):
                        output1, state1 = self.lstm1_dropout( inputs=image_emb[:,i,:], state=state1 )

                    with tf.variable_scope("LSTM2"):
                        output2, state2 = self.lstm2_dropout( inputs=tf.concat([padding, output1],1), state=state2 )
                    #tf.get_variable_scope().reuse_variables()

                with tf.variable_scope("LSTM1",reuse = True):
                    output1, state1 = self.lstm1_dropout( inputs=image_emb[:,i,:], state=state1 )

                if (i+1) % 5 == 0:
                    with tf.variable_scope("LSTM2",reuse = True):
                        output2, state2 = self.lstm2_dropout( inputs=tf.concat([padding, output1],1), state=state2 )
                    with tf.variable_scope("LSTM1",reuse = True):
                        output1, state1 = self.lstm1_dropout( inputs=padding, state=state_padding)

            #--------out loss---------
            img_vec = output2
            #---------state loss-------------
            #img_vec = state2

            # caption encoder
            for i in range(self.n_lstm_steps): 
                if i == 0:
                    current_embed = tf.zeros([self.batch_size, self.dim_hidden])    # b * h
                    with tf.variable_scope("LSTM_cap1"):
                        output_cap1, state_cap1 = self.lstm_cap1_dropout( inputs=tf.concat([current_embed,padding],1), state=state_cap1 )

                else:
                    current_embed = tf.nn.embedding_lookup(self.Wemb, caption[:,i-1])  # b * word2vec_len

                    with tf.variable_scope("LSTM_cap1",reuse = True):
                        output_cap1, state_cap1 = self.lstm_cap1_dropout( inputs=tf.concat([current_embed,padding],1), state=state_cap1 )


            #------------output loss-----------
            sentence_vec = output_cap1
            #------------state loss-------------
            #sentence_vec = state_cap1

            #caption decoder & image decoder
            for i in range(self.n_lstm_steps): 
                if i == 0:
                    current_embed = tf.zeros([self.batch_size, self.dim_hidden])    # b * h

                else:
                    with tf.device("/gpu:0"):
                        current_embed = tf.nn.embedding_lookup(self.Wemb, caption[:,i-1])  # b * word2vec_len


                #-----------cap decoder-------------#
                with tf.variable_scope("LSTM_cap1",reuse = True):
                    output_cap1, state_cap1 = self.lstm_cap1_dropout( tf.concat([current_embed,padding],1), state_cap1 )

                #----------image decoder------------#
                with tf.variable_scope("LSTM1",reuse = True):
                    output1, state1 = self.lstm1_dropout( padding, state1 )

                with tf.variable_scope("LSTM2",reuse = True):
                    output2, state2 = self.lstm2_dropout( tf.concat([current_embed, output1],1), state2 )


                labels = tf.expand_dims(caption[:,i], 1) # b*1
                indices = tf.expand_dims(tf.range(0, self.batch_size, 1), 1) # b x 1
                concated = tf.concat([indices, labels],1) # b x 2
                onehot_labels = tf.sparse_to_dense(concated, tf.stack([self.batch_size, self.n_words]), 1.0, 0.0) # b x w

                #---------cap words------------#
                cap_logit_words = tf.nn.xw_plus_b(output_cap1, self.embed_word_W, self.embed_word_b) # b x w
                cap_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=cap_logit_words, labels=onehot_labels) # b x 1
                cap_cross_entropy = cap_cross_entropy * caption_mask[:,i] # b * 1
                cap_current_loss = tf.reduce_sum(cap_cross_entropy) # 1
                cap_loss += cap_current_loss

                #---------image words------------#
                logit_words = tf.nn.xw_plus_b(output2, self.embed_word_W, self.embed_word_b)
                cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logit_words, labels=onehot_labels) # b x 1
                cross_entropy = cross_entropy * caption_mask[:,i] # b * 1
                current_loss = tf.reduce_sum(cross_entropy) # 1
                img_loss += current_loss

                probs.append(logit_words)

                max_prob_index = tf.argmax(logit_words, 1) # b * 1
                if i == 0:
                    generated_word_index = tf.reshape(max_prob_index,[self.batch_size,1])
                else:
                    generated_word_index=tf.concat([generated_word_index,tf.reshape(max_prob_index,[self.batch_size,1])], axis=1) # b * n

        

        cap_loss = cap_loss / tf.reduce_sum(caption_mask)
        img_loss = img_loss / tf.reduce_sum(caption_mask)
        with tf.name_scope('image_loss'):
            x1_val = tf.nn.l2_normalize(img_vec, dim=1)
            #x1_val = tf.nn.l2_normalize(output_img2, dim=1)
            x2_val = tf.nn.l2_normalize(sentence_vec, dim=1)
            cos_sim = tf.matmul(x1_val, tf.transpose(x2_val, [1, 0]))
            img_cap_loss = self.batch_size - tf.trace(cos_sim)

        #img_cap_loss = tf.losses.cosine_distance(output_img2,output_cap2,dim=1)
        loss = img_loss + cap_loss + img_cap_loss/5.0
        #loss = img_cap_loss * 0 + cap_loss
        #return loss, video, video_mask, caption, caption_mask, probs, generated_word_index
        return loss, img_cap_loss, cap_loss, img_loss, video, video_mask, caption, caption_mask, probs


    def build_generator(self):
        video = tf.placeholder(tf.float32, [1, self.n_lstm_steps, self.dim_image])
        video_mask = tf.placeholder(tf.float32, [1, self.n_lstm_steps])

        video_flat = tf.reshape(video, [-1, self.dim_image])
        image_emb = tf.nn.xw_plus_b( video_flat, self.encode_image_W, self.encode_image_b)
        image_emb = tf.reshape(image_emb, [1, self.n_lstm_steps, self.dim_hidden])

        state1 = tf.zeros([1, self.lstm1.state_size])
        state2 = tf.zeros([1, self.lstm2.state_size])
        padding = tf.zeros([1, self.dim_hidden])
        state_padding = tf.zeros([1, self.lstm1.state_size]) # b * s

        state_cap1 = tf.zeros([1, self.lstm_cap1.state_size])

        generated_words = []

        probs = []
        embeds = []

        #image encoder
        for i in range(self.n_lstm_steps):
            if i == 0:
                with tf.variable_scope("LSTM1"):
                    output1, state1 = self.lstm1_dropout( inputs=image_emb[:,i,:], state=state1 )

                with tf.variable_scope("LSTM2"):
                    output2, state2 = self.lstm2_dropout( inputs=tf.concat([padding, output1],1), state=state2 )
                #tf.get_variable_scope().reuse_variables()

            with tf.variable_scope("LSTM1",reuse = True):
                output1, state1 = self.lstm1_dropout( inputs=image_emb[:,i,:], state=state1 )

            if (i+1) % 5 == 0:
                with tf.variable_scope("LSTM2",reuse = True):
                    output2, state2 = self.lstm2_dropout( inputs=tf.concat([padding, output1],1), state=state2 )
                with tf.variable_scope("LSTM1",reuse = True):
                    output1, state1 = self.lstm1_dropout( inputs=padding, state=state_padding)

        img_vec = output2

        #image decoder
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
                #current_embed = tf.expand_dims(current_vec, 0)

            embeds.append(current_embed)

        return video, video_mask, generated_words, probs, embeds

class f_Hierarchical_Generator():
    def __init__(self, dim_image, n_words, dim_hidden, batch_size, n_lstm_steps, drop_out_rate, bias_init_vector=None):
        self.dim_image = dim_image
        self.n_words = n_words
        self.dim_hidden = dim_hidden
        self.batch_size = batch_size
        self.n_lstm_steps = n_lstm_steps
        self.drop_out_rate = drop_out_rate

        with tf.device("/gpu:0"):
            self.Wemb = tf.Variable(tf.random_uniform([n_words, dim_hidden], -0.1, 0.1), name='Wemb')

        self.layers = 2
        self.lstm = [
        rnn.LSTMCell(
            num_units=dim_hidden,
            initializer=tf.orthogonal_initializer(), 
            ) for i in xrange(self.layers)
        ]

        self.lstm_dropout = [
        rnn.DropoutWrapper(self.lstm[i],
            output_keep_prob=1-self.drop_out_rate
            ) for i in xrange(self.layers)
        ]

        self.state_init = [
        self.lstm[i].zero_state(self.batch_size, tf.float32)
        for i in xrange(self.layers)
        ]

        self.generator_state_init = [
        self.lstm[i].zero_state(1, tf.float32)
        for i in xrange(self.layers)
        ]

        self.encode_image_W = tf.Variable( tf.random_uniform([dim_image, dim_hidden], -0.1, 0.1), name='encode_image_W')
        self.encode_image_b = tf.Variable( tf.zeros([dim_hidden]), name='encode_image_b')

        self.embed_word_W = tf.Variable(tf.random_uniform([dim_hidden, n_words], -0.1,0.1), name='embed_word_W')
        if bias_init_vector is not None:
            self.embed_word_b = tf.Variable(bias_init_vector.astype(np.float32), name='embed_word_b')
        else:
            self.embed_word_b = tf.Variable(tf.zeros([n_words]), name='embed_word_b')

    def _basic_hierarchy_layer(self, name, layer, step_input, steps, size, state, state_init):
        if steps == size**(layer-1)-1:
            with tf.variable_scope('LSTM{}'.format(layer)):
                output, state = self.lstm_dropout[layer-1](
                    inputs=step_input, state=state_init[layer-1]
                    )
                print '--------'
                print tf.get_variable_scope().name
        if ((steps != size**(layer-1)-1) 
            and ((steps+1) % size**(layer-1) == 0) 
            and ((steps+size**layer-size**(layer-1)+1) % size**layer == 0)):
            with tf.variable_scope('LSTM{}'.format(layer), reuse = True):
                output, state = self.lstm_dropout[layer-1](
                    inputs=step_input, state=state_init[layer-1]
                    )
        if ((steps != size**(layer-1)-1) 
            and ((steps+1) % size**(layer-1) == 0) 
            and ((steps+size**layer-size**(layer-1)+1) % size**layer != 0)):
            with tf.variable_scope('LSTM{}'.format(layer), reuse = True):
                output, state = self.lstm_dropout[layer-1](
                    inputs=step_input, state=state
                    )
        try:
            return output, state
        except:
            return 0

    def build_model(self):
        video = tf.placeholder(tf.float32, [self.batch_size, self.n_lstm_steps, self.dim_image]) # b * n * h
        video_mask = tf.placeholder(tf.float32, [self.batch_size, self.n_lstm_steps]) # b * n

        caption = tf.placeholder(tf.int32, [self.batch_size, self.n_lstm_steps]) # b * n
        caption_mask = tf.placeholder(tf.float32, [self.batch_size, self.n_lstm_steps]) # b * n

        video_flat = tf.reshape(video, [-1, self.dim_image])
        image_emb = tf.nn.xw_plus_b( video_flat, self.encode_image_W, self.encode_image_b) # (batch_size*n_lstm_steps, dim_hidden)
        image_emb = tf.reshape(image_emb, [self.batch_size, self.n_lstm_steps, self.dim_hidden]) # b * n * h

        
        padding = tf.zeros([self.batch_size, self.dim_hidden]) # b * h

        probs = []

        loss = 0.0

        state1, state2 = self.state_init

        for i in xrange(self.n_lstm_steps):

            try:
                output1, state1 = self._basic_hierarchy_layer(
                    'lstm1', 1, image_emb[:,i,:], i, 10, state1, self.state_init)
                print 'i have value in layer 1'
            except Exception, e:
                print e
                pass
            try:
                output2, state2 = self._basic_hierarchy_layer(
                    'lstm2', 2, tf.concat([padding, output1],1), i, 8, state2, self.state_init)
                print 'i have value in layer 2'
            except Exception, e:
                print e
                pass

        for i in range(self.n_lstm_steps): ## Phase 2 => only generate captions
            if i == 0:
                current_embed = tf.zeros([self.batch_size, self.dim_hidden])    # b * h
                
            else:
                with tf.device("/gpu:0"):
                    current_embed = tf.nn.embedding_lookup(self.Wemb, caption[:,i-1])

            with tf.variable_scope('LSTM1',reuse = True):
                output1, state1 = self.lstm_dropout[0]( padding, state1 )

            with tf.variable_scope('LSTM2',reuse = True):
                output2, state2 = self.lstm_dropout[1]( tf.concat([current_embed, output1],1), state2 )

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
        image_emb = tf.nn.xw_plus_b( video_flat, self.encode_image_W, self.encode_image_b)
        image_emb = tf.reshape(image_emb, [1, self.n_lstm_steps, self.dim_hidden])

        padding = tf.zeros([1, self.dim_hidden])

        generated_words = []

        probs = []
        embeds = []

        state1, state2 = self.generator_state_init

        for i in xrange(self.n_lstm_steps):

            try:
                output1, state1 = self._basic_hierarchy_layer(
                    'lstm1', 1, image_emb[:,i,:], i, 5, state1, self.generator_state_init)
                print 'i have value in layer 1'
            except:
                pass
            try:
                output2, state2 = self._basic_hierarchy_layer(
                    'lstm2', 2, tf.concat([padding, output1],1), i, 5, state2, self.generator_state_init)
                print 'i have value in layer 2'
            except:
                pass

        
        for i in range(self.n_lstm_steps):

            #tf.get_variable_scope().reuse_variables()

            if i == 0:
                current_embed = tf.zeros([1, self.dim_hidden])
                
            with tf.variable_scope('LSTM1',reuse=True):
                output1, state1 = self.lstm_dropout[0]( padding, state1 )
            with tf.variable_scope('LSTM2',reuse=True):
                output2, state2 = self.lstm_dropout[1]( tf.concat([current_embed,output1],1), state2 )

            logit_words = tf.nn.xw_plus_b( output2, self.embed_word_W, self.embed_word_b)
            max_prob_index = tf.argmax(logit_words, 1)[0]
            generated_words.append(max_prob_index)
            probs.append(logit_words)

            with tf.device("/gpu:0"):
                current_embed = tf.nn.embedding_lookup(self.Wemb, max_prob_index)
                current_embed = tf.expand_dims(current_embed, 0)

            embeds.append(current_embed)

        return video, video_mask, generated_words, probs, embeds

class Three_Hierarchical_Generator():
    def __init__(self, dim_image, n_words, dim_hidden, batch_size, n_lstm_steps, drop_out_rate,  step , bias_init_vector=None):
        self.dim_image = dim_image
        self.n_words = n_words
        self.dim_hidden = dim_hidden
        self.batch_size = batch_size
        self.n_lstm_steps = n_lstm_steps
        self.drop_out_rate = drop_out_rate
        self.step = step

        with tf.device("/gpu:0"):
            self.Wemb = tf.Variable(tf.random_uniform([n_words, dim_hidden], -0.1, 0.1), name='Wemb')

        self.lstm1 = rnn.BasicLSTMCell(dim_hidden,state_is_tuple=False)
        self.lstm2 = rnn.BasicLSTMCell(dim_hidden,state_is_tuple=False)
        self.lstm3 = rnn.BasicLSTMCell(dim_hidden,state_is_tuple=False)
        self.lstm1_dropout = rnn.DropoutWrapper(self.lstm1,output_keep_prob=1 - self.drop_out_rate)
        self.lstm2_dropout = rnn.DropoutWrapper(self.lstm2,output_keep_prob=1 - self.drop_out_rate)
        self.lstm3_dropout = rnn.DropoutWrapper(self.lstm3,output_keep_prob=1 - self.drop_out_rate)

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
        state3 = tf.zeros([self.batch_size, self.lstm3.state_size]) # b * s
        padding = tf.zeros([self.batch_size, self.dim_hidden]) # b * h
        state_padding = tf.zeros([self.batch_size, self.lstm1.state_size]) # b * s

        probs = []

        loss = 0.0

        with tf.variable_scope(tf.get_variable_scope()) as scope:

            for i in range(self.n_lstm_steps): ## Phase 1 => only read frames
                #---layer 1------#
                if i == 0:
                    with tf.variable_scope("LSTM1"):
                        output1, state1 = self.lstm1_dropout( inputs=image_emb[:,i,:], state=state_padding )
                elif i % self.step[0] == 0:
                    with tf.variable_scope("LSTM1",reuse = True):
                        output1, state1 = self.lstm1_dropout( inputs=image_emb[:,i,:], state=state_padding )
                else: 
                    with tf.variable_scope("LSTM1",reuse = True):
                        output1, state1 = self.lstm1_dropout( inputs=image_emb[:,i,:], state=state1 )

                #---layer 2------#
                if (i+1) == self.step[0]:
                    with tf.variable_scope("LSTM2"):
                        output2, state2 = self.lstm2_dropout( inputs=output1, state=state_padding )
                elif (i+1-self.step[0]) % (self.step[0] * self.step[1]) == 0:
                    with tf.variable_scope("LSTM2", reuse = True):
                        output2, state2 = self.lstm2_dropout( inputs=output1, state=state_padding )
                elif (i+1) % self.step[0] == 0:
                    with tf.variable_scope("LSTM2",reuse = True):
                        output2, state2 = self.lstm2_dropout( inputs=output1, state=state2 )
                else:
                    pass

                #---layer 3------#
                if (i+1) == self.step[0] * self.step[1]:
                    with tf.variable_scope("LSTM3"):
                        output3, state3 = self.lstm3_dropout( inputs=tf.concat([padding, output2],1), state=state_padding )
                elif (i+1) % (self.step[0] * self.step[1]) == 0:
                    with tf.variable_scope("LSTM3",reuse = True):
                        output3, state3 = self.lstm3_dropout( inputs=tf.concat([padding, output2],1), state=state3 )
                else:
                    pass


            for i in range(self.n_lstm_steps): ## Phase 2 => only generate captions
                if i == 0:
                    current_embed = tf.zeros([self.batch_size, self.dim_hidden])    # b * h
                else:
                    with tf.device("/gpu:0"):
                        current_embed = tf.nn.embedding_lookup(self.Wemb, caption[:,i-1])

                with tf.variable_scope("LSTM1",reuse = True):
                    output1, state1 = self.lstm1_dropout( padding, state1 )

                with tf.variable_scope("LSTM2",reuse = True):
                    output2, state2 = self.lstm2_dropout( output1, state2 )

                with tf.variable_scope("LSTM3",reuse = True):
                    output3, state3 = self.lstm3_dropout( inputs=tf.concat([current_embed, output2],1), state=state3 )

                labels = tf.expand_dims(caption[:,i], 1) # b*1
                indices = tf.expand_dims(tf.range(0, self.batch_size, 1), 1) # b x 1
                concated = tf.concat([indices, labels],1) # b x 2
                onehot_labels = tf.sparse_to_dense(concated, tf.stack([self.batch_size, self.n_words]), 1.0, 0.0) # b x w

                logit_words = tf.nn.xw_plus_b(output3, self.embed_word_W, self.embed_word_b) # b x w
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
        state_padding = tf.zeros([1, self.lstm1.state_size]) # b * s

        generated_words = []

        probs = []
        embeds = []

        #with tf.variable_scope(tf.get_variable_scope()) as scope:
        for i in range(self.n_lstm_steps):
            #---layer 1------#
            if i == 0:
                with tf.variable_scope("LSTM1"):
                    output1, state1 = self.lstm1_dropout( inputs=image_emb[:,i,:], state=state_padding )
            elif i % self.step[0] == 0:
                with tf.variable_scope("LSTM1",reuse = True):
                    output1, state1 = self.lstm1_dropout( inputs=image_emb[:,i,:], state=state_padding )
            else: 
                with tf.variable_scope("LSTM1",reuse = True):
                    output1, state1 = self.lstm1_dropout( inputs=image_emb[:,i,:], state=state1 )

            #---layer 2------#
            if (i+1) == self.step[0]:
                with tf.variable_scope("LSTM2"):
                    output2, state2 = self.lstm2_dropout( inputs=output1, state=state_padding )
            elif (i+1-self.step[0]) % (self.step[0] * self.step[1]) == 0:
                with tf.variable_scope("LSTM2", reuse = True):
                    output2, state2 = self.lstm2_dropout( inputs=output1, state=state_padding )
            elif (i+1) % self.step[0] == 0:
                with tf.variable_scope("LSTM2",reuse = True):
                    output2, state2 = self.lstm2_dropout( inputs=output1, state=state2 )
            else:
                pass

            #---layer 3------#
            if (i+1) == self.step[0] * self.step[1]:
                with tf.variable_scope("LSTM3"):
                    output3, state3 = self.lstm3_dropout( inputs=tf.concat([padding, output2],1), state=state_padding )
            elif (i+1) % (self.step[0] * self.step[1]) == 0:
                with tf.variable_scope("LSTM3",reuse = True):
                    output3, state3 = self.lstm3_dropout( inputs=tf.concat([padding, output2],1), state=state3 )
            else:
                pass

        for i in range(self.n_lstm_steps):

            tf.get_variable_scope().reuse_variables()

            if i == 0:
                current_embed = tf.zeros([1, self.dim_hidden])

            with tf.variable_scope("LSTM1"):
                output1, state1 = self.lstm1_dropout( padding, state1 )

            with tf.variable_scope("LSTM2"):
                output2, state2 = self.lstm2_dropout( output1, state2 )

            with tf.variable_scope("LSTM3"):
                output3, state3 = self.lstm3_dropout( inputs=tf.concat([current_embed, output2],1), state=state3 )

            logit_words = tf.nn.xw_plus_b( output3, self.embed_word_W, self.embed_word_b)
            max_prob_index = tf.argmax(logit_words, 1)[0]
            generated_words.append(max_prob_index)
            probs.append(logit_words)

            with tf.device("/gpu:0"):
                current_embed = tf.nn.embedding_lookup(self.Wemb, max_prob_index)
                current_embed = tf.expand_dims(current_embed, 0)

            embeds.append(current_embed)

        return video, video_mask, generated_words, probs, embeds