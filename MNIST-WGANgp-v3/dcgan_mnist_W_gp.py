# -*- coding: utf-8 -*-
"""
Created on Sat Sep  2 11:41:48 2017

@author: aaron
"""

import tensorflow as tf
import tensorflow.contrib.layers as layer
import numpy as np
import time
import matplotlib.pyplot as plt

#Define config class
class Config(object):
    def __init__(self, batch_size, latent_size, lr_G, lr_D, critic_num, epoch_num, alpha, beta1, beta2, lamb, save_per_epoch):
        self.batch_size = batch_size
        self.latent_size = latent_size
        self.epoch_num = epoch_num
        self.alpha = alpha
        self.lamb = lamb
        self.beta1 = beta1
        self.beta2 = beta2
        self.lr_G = lr_G
        self.lr_D = lr_D
        self.critic_num = critic_num
        self.summary_dir = "summary"
        self.model_dir = "models"
        self.is_training = True
        self.save_per_epoch = save_per_epoch
    def outfig(self):
        print('[   Elements in config   ]')
        print('Batch_size:     %10s' % self.batch_size)
        print('Latent_size:    %10s' % self.latent_size)
        print('Epoch_num:      %10s' % self.epoch_num)
        print('Lr_G:           %10s' % self.lr_G)
        print('Lr_D:           %10s' % self.lr_D)
        print('Summary_dir:    %10s' % self.summary_dir)
        print('Model_dir:      %10s' % self.model_dir)
        print('Alpha:          %10s' % self.alpha)
        print('Beta1:          %10s' % self.beta1)
        print('Beta2:          %10s' % self.beta2)
        print('Lambda:         %10s' % self.lamb)
        print('Critic_num      %10s' % self.critic_num)
        print('is_training:    %10s' % self.is_training)
        print('Save_per_epoch: %10s' % self.save_per_epoch)
        
#Define DCGAN class
class DCGAN(object):
    def __init__(self, sess, config):
        self.X, self.y = self.load_mnist_data()
        self.config = config
        self.sess = sess
        
    def generator(self, inputs, reuse=False, is_training=False, name='Generator'):
        with tf.variable_scope(name) as scope:
            if reuse:
                scope.reuse_variables()          
            #Project Latent_size vector into 7*7*64 tensor
            with tf.name_scope('G_layer0'):
                g_0 = tf.cast(
                        layer.fully_connected(inputs=inputs,
                                              num_outputs=7*7*64, 
                                              activation_fn=None, 
                                              scope='Linear'), 
                        dtype=tf.float32)
                g_0 = layer.batch_norm(g_0, is_training=is_training, scope='Batch_n')
                g_0 = tf.nn.relu(g_0, name='Relu')
                g_0 = tf.reshape(g_0, shape=[-1, 7, 7, 64])
            #Projector1
            g_1 = self.conv2d_transpose_bn(inputs=g_0, 
                                           output_num=32, 
                                           filter_size=5, 
                                           stride=2, 
                                           is_training=is_training, 
                                           name='G_layer1_conv')
            #Projector2
            g_2 = self.conv2d_transpose_bn(inputs=g_1, 
                                           output_num=16, 
                                           filter_size=5, 
                                           stride=1, 
                                           is_training=is_training, 
                                           name='G_layer2_conv')
            #Projector3
            g_3 = self.conv2d_transpose_bn(inputs=g_2, 
                                           output_num=8, 
                                           filter_size=5, 
                                           stride=2, 
                                           is_training=is_training, 
                                           name='G_layer3_conv')
            #Projector4
            g_4 = layer.conv2d_transpose(inputs=g_3,
                                         num_outputs=1, 
                                         kernel_size=5, 
                                         stride=1, 
                                         padding='SAME', 
                                         activation_fn=tf.nn.tanh, 
                                         scope='G_layer4_conv')
            return g_4
            
    def discriminator(self, inputs, y=None, reuse=False, is_training=False, name='Discriminator'):
        with tf.variable_scope(name) as scope:
            if reuse:
                scope.reuse_variables()
            d_0 = self.conv2d_bn(inputs=inputs, 
                                 output_num=4, 
                                 filter_size=5, 
                                 stride=1, 
                                 is_training=is_training, 
                                 name='D_layer0_conv')
            d_1 = self.conv2d_bn(inputs=d_0, 
                                 output_num=8, 
                                 filter_size=5, 
                                 stride=2, 
                                 is_training=is_training, 
                                 name='D_layer1_conv')
            d_2 = self.conv2d_bn(inputs=d_1, 
                                 output_num=32, 
                                 filter_size=5, 
                                 stride=2, 
                                 is_training=is_training, 
                                 name='D_layer2_conv')
            d_3 = self.conv2d_bn(inputs=d_2, 
                                 output_num=64, 
                                 filter_size=5, 
                                 stride=1, 
                                 is_training=is_training, 
                                 name='D_layer3_conv')
            with tf.name_scope('D_layer4'):
                d_4 = layer.flatten(d_3)
                d_4 = layer.fully_connected(d_4, 
                                            num_outputs=1, 
                                            activation_fn=None, 
                                            scope='Linear')
            return d_4

    #Process functions            
    def build(self):
        print('[    Building model...   ]')
        #Do some basic settings(Code_v3)
        if not tf.gfile.Exists(self.config.summary_dir):
            tf.gfile.MakeDirs(self.config.summary_dir)
        if not tf.gfile.Exists(self.config.summary_dir + "/train"):
            tf.gfile.MakeDirs(self.config.summary_dir + "/train")
        if not tf.gfile.Exists(self.config.summary_dir + "/test"):
            tf.gfile.MakeDirs(self.config.summary_dir + "/test")
        if not tf.gfile.Exists(self.config.model_dir):
            tf.gfile.MakeDirs(self.config.model_dir)
        #Define placeholder
        self.is_training = tf.placeholder(dtype=bool, 
                                          name='is_training')
        self.noise = tf.placeholder(dtype=tf.float32, 
                                    shape=[None, self.config.latent_size], 
                                    name='Noise')
        self.real_image = tf.placeholder(dtype=tf.float32, 
                                         shape=[self.config.batch_size, 28, 28, 1], 
                                         name='Real_images')
        self.R_image_sum = tf.summary.image('Real_images', self.real_image)
        #Orgnize network
        self.D_real_logits = self.discriminator(inputs=self.real_image, 
                                                             y=None, 
                                                             is_training=self.is_training, 
                                                             reuse=False)
        self.fake_image = self.generator(inputs=self.noise, 
                                         is_training=self.is_training)
        self.F_image_sum = tf.summary.image('Fake_images', self.fake_image)
        self.D_fake_logits = self.discriminator(inputs=self.fake_image, 
                                                             y=None, 
                                                             is_training=self.is_training, 
                                                             reuse=True)
        #Define loss
        with tf.name_scope('Loss'):
            with tf.name_scope('G_loss'):
                self.G_loss = tf.reduce_mean(tf.scalar_mul(-1, self.D_fake_logits))
#                self.G_loss = tf.reduce_mean(
#                        tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_fake_logits, 
#                                                                labels=tf.ones_like(self.D_fake)))
                self.G_loss_sum = tf.summary.scalar("G_loss", self.G_loss)
            with tf.name_scope('D_loss'):
                self.D_loss_fake = tf.reduce_mean(self.D_fake_logits)
#                self.D_loss_fake = tf.reduce_mean(
#                        tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_fake_logits, 
#                                                                labels=tf.zeros_like(self.D_fake)))
                self.D_loss_real = tf.reduce_mean(tf.scalar_mul(-1, self.D_real_logits))
#                self.D_loss_real = tf.reduce_mean(
#                        tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_real_logits, 
#                                                                labels=tf.ones_like(self.D_real)))
                self.D_gradients_penalty = self.gradient_penalty(self.real_image, self.fake_image)
                self.D_gradients_sum = tf.summary.scalar("D_gradients", self.D_gradients_penalty)
                self.D_loss = self.D_loss_fake + self.D_loss_real + self.config.lamb * self.D_gradients_penalty
                self.D_loss_sum = tf.summary.scalar("D_loss", self.D_loss)
                self.D_loss_sum_real = tf.summary.histogram("D_loss_real", self.D_loss_real)
                self.D_loss_sum_fake = tf.summary.histogram("D_loss_fake", self.D_loss_fake)
                
        #Define utils
        self.G_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope="Generator") 
        self.D_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope="Discriminator")
        self.updata_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        
        self.G_trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="Generator")
        self.D_trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="Discriminator")
        self.trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        
        self.G_sum = tf.summary.merge([self.F_image_sum, self.G_loss_sum])
        self.G_test = tf.summary.merge([self.F_image_sum])
        self.D_sum = tf.summary.merge([self.F_image_sum, self.R_image_sum, self.D_loss_sum, self.D_gradients_sum, self.D_loss_sum_real, self.D_loss_sum_fake])
        
        self.train_summary_writer = tf.summary.FileWriter(self.config.summary_dir + "/train", self.sess.graph)
        self.test_summary_writer = tf.summary.FileWriter(self.config.summary_dir + "/test")
        print('----Building Finished!----')
      
    def train(self):
        self.build()
        self.saver = tf.train.Saver()
        print('[    Start training...   ]')
        self.D_global_step = tf.Variable(0)
        self.G_global_step = tf.Variable(0)
        with tf.control_dependencies(self.D_update_ops):
            D_opt = tf.train.AdamOptimizer(learning_rate=self.config.lr_D,
                                           beta1=self.config.beta1,
                                           beta2=self.config.beta2) \
                            .minimize(self.D_loss, var_list=self.D_trainable_vars, global_step=self.D_global_step)
        with tf.control_dependencies(self.G_update_ops):
            G_opt = tf.train.AdamOptimizer(learning_rate=self.config.lr_G,
                                           beta1=self.config.beta1,
                                           beta2=self.config.beta2) \
                            .minimize(self.G_loss, var_list=self.G_trainable_vars, global_step=self.G_global_step)
        self.sess.run(tf.global_variables_initializer())
        #start iteration
            #Add time counting(Code_v3)
        start_time = time.time()
        for epoch in range(self.config.epoch_num):
            batch_ix = 0
            for X_batch, _ in self.minibatches(self.X, self.y, self.config.batch_size, True):
                #Make Discriminator close to earth mover distance
                D_iternum = self.config.critic_num
                #Discriminator training
                for _ in range(D_iternum):           
                    noise = np.random.uniform(-1, 1, [self.config.batch_size, self.config.latent_size])
                    feed_dict = {
                            self.real_image: X_batch.astype(np.float32), 
                            self.noise: noise,
                            self.is_training: True
                            }
                    _, train_D_loss, D_global_step, sum_str_D = self.sess.run(
                            [D_opt, self.D_loss, self.D_global_step, self.D_sum], feed_dict=feed_dict)
                #Generator training
                noise = np.random.uniform(-1, 1, [self.config.batch_size, self.config.latent_size])
                feed_dict = {
                        self.real_image: X_batch.astype(np.float32), 
                        self.noise: noise,
                        self.is_training: True
                        }
                _, train_G_loss, G_global_step, sum_str_G = self.sess.run(
                    [G_opt, self.G_loss, self.G_global_step, self.G_sum], feed_dict=feed_dict)
                    #Reorgnize the train & test procedure(Code_v3)
                if batch_ix % 10 == 0:#Log train data
                    self.train_summary_writer.add_summary(sum_str_D, D_global_step)
                    self.train_summary_writer.add_summary(sum_str_G, G_global_step)
                if batch_ix % 100 == 0:#Log test data
                    noise = np.random.uniform(-1, 1, [self.config.batch_size, self.config.latent_size])
                    feed_dict = { 
                            self.noise: noise,
                            self.is_training: False
                            }
                    sum_str_test = self.sess.run(self.G_test, feed_dict=feed_dict)
                    self.test_summary_writer.add_summary(sum_str_test, batch_ix)
                batch_ix += 1
                print("Epoch[%02d] %4d/%4d | G_loss: %.4f, D_loss: %.4f, Time: %.4f" 
                      % (epoch, batch_ix, 70000 // self.config.batch_size, train_G_loss, train_D_loss, time.time() - start_time))
            if (epoch  +  1) % self.config.save_per_epoch == 0:
                print('[     Saving model...    ]')
                self.saver.save(self.sess, self.config.model_dir + "/saved_model.ckpt", global_step=D_global_step)
                print('[  Epoch %02d model saved  ]' % epoch)
        print('----Training Finished!----')
    
    def reload(self):
        self.build()
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()
        print('[    Reloading model...  ]')
        if tf.gfile.ListDirectory(self.config.model_dir) == []:
            print('----Loading Failed!!!!----')
            assert 1 == 0
        else:
            self.saver.recover_last_checkpoints(self.config.model_dir)
        print('-----Loading Finished-----')
    
    def generate(self, num):
        print('[   Start generating...  ]')
        noise = np.random.uniform(-1, 1, [num, self.config.latent_size])
        fake_image = self.sess.run(self.fake_image,
                             feed_dict={
                                 self.noise: noise,
                                 self.is_training: False
                             })
        print('----Generating Finished---')
        return fake_image.reshape(num, 28, 28)
        
    #Define some functions
    def conv2d_transpose_bn(self, inputs, output_num, filter_size, stride, is_training, activate=True, name=None):
        with tf.variable_scope(name):
            conv = layer.conv2d_transpose(inputs=inputs,
                                          num_outputs=output_num, 
                                          kernel_size=filter_size, 
                                          stride=stride, 
                                          padding='SAME', 
                                          activation_fn=None, 
                                          scope='Conv2_trans')
            bn = layer.batch_norm(inputs=conv,
                                  is_training=is_training, 
                                  activation_fn=None, 
                                  scope='Batch_n')
            if activate:
                return tf.nn.relu(bn, name='Relu')
            else:
                return bn
            
    def conv2d_bn(self, inputs, output_num, filter_size, stride, is_training=False, name=None):
        with tf.variable_scope(name):
            conv = layer.conv2d(inputs=inputs,
                                num_outputs=output_num, 
                                kernel_size=filter_size, 
                                stride=stride, 
                                padding='SAME', 
                                activation_fn=None, 
                                scope='Conv2')
            bn = layer.batch_norm(inputs=conv,
                                  is_training=is_training, 
                                  activation_fn=None, 
                                  scope='Batch_n')
            #leakyRelu
            #return tf.nn.Relu(bn)
            return self.leakyRelu(bn)
    
    def gradient_penalty(self, real, fake):
        x = self.interpolate(real, fake)
        pred = self.discriminator(inputs=x,
                                  y=None, 
                                  is_training=self.is_training, 
                                  reuse=True)
        gradients = tf.gradients(pred, x)[0]
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
        gp = tf.reduce_mean((slopes - 1.)**2)
        return gp
    
    def interpolate(self, real, fake):
        shape = tf.concat((tf.shape(real)[0:1], tf.tile([1], [real.shape.ndims - 1])), axis=0)
        alpha = tf.random_uniform(shape=shape,
                                  minval=0.,
                                  maxval=1.)
        inter = real + alpha * (fake - real)
        inter.set_shape(real.get_shape().as_list())
        return inter
    
    def leakyRelu(self, inputs, name='leakyRelu'):
        return tf.maximum(inputs, inputs*self.config.alpha, name=name)
    
    def load_mnist_data(self, path="mnist.npz"):
        data = np.load(path)
        X = (np.concatenate((data["x_train"], data["x_test"]), axis=0)) / 255.0
        y = (np.concatenate((data["x_train"], data["x_test"]), axis=0)) / 255.0
        X = np.expand_dims(X, axis=3)
        return X, y
    
    def minibatches(self, inputs=None, targets=None, batch_size=None, shuffle=False):
        assert len(inputs) == len(targets)
        if shuffle:
            indices = np.arange(len(inputs))
            np.random.shuffle(indices)
        for start_idx in range(0, len(inputs) - batch_size + 1, batch_size):
            if shuffle:
                excerpt = indices[start_idx:start_idx + batch_size]
            else:
                excerpt = slice(start_idx, start_idx + batch_size)
            yield inputs[excerpt], targets[excerpt]

        
#The main function
if __name__ == "__main__":
    #Define the basic config
    config = Config(
	    batch_size=64, 
	    latent_size=100, 
	    lr_G=1e-4, 
      lr_D=1e-4,  
	    epoch_num=30, 
	    critic_num=20, 
	    alpha=0.2, 
      beta1=0.5,
      beta2=0.9,
      lamb=10.0,
	    save_per_epoch=2)
    config.outfig()
    #Define is train or test(Code_v3)
    Train_ops = 1
    #Define the number of image to generate
    NUM_GENERATED = 9
    #Define Interactivite session
    sess = tf.Session()
    #Define network
    net = DCGAN(sess, config)
    if Train_ops:
        #Start training
        net.train()
    else:
        #Reload model
        net.reload()
        #Generate images
        generate_images = net.generate(NUM_GENERATED)
        #Plot images
        Lens = int(np.sqrt(NUM_GENERATED))
        f, axarr = plt.subplots(Lens, Lens, figsize=(5, 5))
        for i in range(Lens):
            for j in range(Lens):
                axarr[i, j].imshow(generate_images[i*Lens + j], cmap='Greys', interpolation='nearest')
        f.savefig("output.png")
    
    