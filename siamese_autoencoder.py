from imports import *


class SiameseFC(object):

    def __init__(self, sess,args):
        self.args=args
        self.sess=sess
        self.stddev=0.02
        self.build_model()
        print_total_num_of_variables()
        print_total_count_of_varaibles()

    def build_model(self):
        self.encoded_feature_1 = tf.placeholder(tf.float32, [None, self.args.input_size], name='Feature_Vector1')
        self.encoded_feature_2 = tf.placeholder(tf.float32, [None, self.args.input_size], name='Feature_Vector2')
        self.label = tf.placeholder(tf.float32, [None], name='label')

        with tf.name_scope('Siamese_Network1'):
            with tf.name_scope('Code_vector1'):
                self.code_vector1 = self.siamese(self.encoded_feature_1, reuse=False)

        with tf.name_scope('Siamese_Network2'):
            with tf.name_scope('Code_vector2'):
                self.code_vector2 = self.siamese(self.encoded_feature_2, reuse=True)

        with tf.name_scope('cost'):
            self.true_distance = tf.reduce_mean(self.label * tf.reduce_sum(tf.square(  self.code_vector2 - self.code_vector1   ),axis=1)) # mean
            self.false_distance = tf.reduce_mean((1-self.label) * tf.reduce_sum(tf.square(  self.code_vector2 - self.code_vector1   ),axis=1))
            self.true_pair_loss =  tf.square(self.true_distance)
            self.false_pair_loss =  tf.square(tf.maximum(1300 - self.false_distance, 0))

            # self.true_pair_loss = 0.0008*tf.reduce_mean(self.label * self.true_distance+tf.log(1 + tf.exp(-self.true_distance)))
            # self.false_pair_loss = 10000000000000*tf.reduce_mean((1-self.label)*tf.log(1 + tf.exp(-self.false_distance)))
            self.loss = self.true_pair_loss + self.false_pair_loss

        self.loss_sum = tf.summary.scalar("Loss", self.loss)
        self.false_distance_sum = tf.summary.scalar("False_pair_distance", self.false_distance)
        self.true_distance_sum = tf.summary.scalar("True_pair_distance", self.true_distance)
        #self.distance_differences= tf.summary.scalar("Distance ", self.distance)
        self.true_pair_loss_sum = tf.summary.scalar("True_Pair_Loss", self.true_pair_loss)
        self.false_pair_loss_sum = tf.summary.scalar("False_Pair_Loss", self.false_pair_loss)
        trainable_var = tf.trainable_variables()

        with tf.name_scope('optimizers'):
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.args.lr)
            grads = self.optimizer.compute_gradients(self.loss, var_list=trainable_var)
            for grad, var in grads:
                tf.summary.histogram(var.op.name + "/gradient_siamese", grad)
            self.siamese_optimizer = self.optimizer.apply_gradients(grads)

        self.saver = tf.train.Saver()
        self.summary_op = tf.summary.merge_all()

        self.writer = tf.summary.FileWriter(self.args.checkpoint_dir, self.sess.graph)

    def train(self):
        """Train pix2pix"""
        print "Statrt"
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)
        if self.load():
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        with open(self.args.train_dataset_path,'r') as f:
            data = csv.reader(f)
            data = np.array([ row for row in data])
            print 'data.shape', data.shape

        print len(data[0]) ,'data0'
        counter = 1
        start_time = time.time()

        features_h5_open =h5py.File(self.args.h5features, 'r')
        features = np.array(features_h5_open['features']).astype(np.float32)
        labels  =  list(np.array(features_h5_open['labels']))
        #labels = [ i.replace('/home/compvis/notebooks/User_clustering/train_data_dir/','/home/ubuntuuser/datasets/image_datasets/VOC/sipan_resized/train_data_dir/')for i in labels]
        #labels=  [ i+'.jpg' for i in labels]
        print len(labels), features.shape , 'h5file_succ'

        features_h5_open.close()

        for epoch in xrange(self.args.epoch):
            np.random.shuffle(data)
            batch_idxs = min(len(data), self.args.train_size) // self.args.batch_size

            for idx in xrange(0, batch_idxs):

                batch_files = np.array(data[idx * self.args.batch_size:(idx + 1) * self.args.batch_size])
                checks= [ pair[2] in labels and pair[3] in labels  for pair in batch_files ]
                batch_files = batch_files[np.nonzero(checks)[0]]
                batch_feat1 = np.array([ features[labels.index(pair[2])] for pair in batch_files])
                batch_feat2 = np.array([  features[labels.index(pair[3]) ] for pair in batch_files])
                label = np.array([ pair[-1]  for pair in batch_files]).astype(np.uint8)
                print np.count_nonzero(label) ,'n_nonzer'
                if np.count_nonzero(checks)>0:

                    _, summary_str_g = self.sess.run([self.siamese_optimizer, self.summary_op],
                                                     feed_dict={
                                                         self.encoded_feature_1: batch_feat1,
                                                         self.encoded_feature_2: batch_feat2,
                                                         self.label: label
                                                     })
                    self.writer.add_summary(summary_str_g, counter)



                    true_pair_loss = self.true_pair_loss.eval({self.encoded_feature_1 : batch_feat1,
                                                         self.encoded_feature_2 : batch_feat2,
                                                         self.label :  label})
                    false_pair_loss = self.false_pair_loss.eval({self.encoded_feature_1 : batch_feat1,
                                                         self.encoded_feature_2 : batch_feat2,
                                                         self.label :  label} )
                    loss = self.loss.eval({self.encoded_feature_1 : batch_feat1,
                                                          self.encoded_feature_2 : batch_feat2,
                                                          self.label :  label} )

                    true_pair_distance = self.true_distance.eval({self.encoded_feature_1 : batch_feat1,
                                                         self.encoded_feature_2 : batch_feat2,
                                                         self.label :  label} )
                    false_pair_distance = self.false_distance.eval({self.encoded_feature_1 : batch_feat1,
                                                         self.encoded_feature_2 : batch_feat2,
                                                         self.label :  label} )

                    counter+=1
                    a=[epoch, idx, batch_idxs, time.time() - start_time, true_pair_loss , false_pair_loss, loss,true_pair_distance,false_pair_distance ]
                    print true_pair_distance
                    if np.mod(counter, 0) == 0:
                        print("Epoch: [%2d] [%4d/%4d] time: %.8f, true_pair_loss : %.8f, false_pair_loss: %.8f,loss: %.8f,true_pair_distance : %.8f, false_pair_distance: %.8f" % (
                            epoch, idx, batch_idxs, time.time() - start_time, true_pair_loss , false_pair_loss, loss,true_pair_distance,false_pair_distance ))

                    if np.mod(counter, 500) == 0:
                        self.save(self.args.checkpoint_dir, counter)
                else:
                    continue

    def test(self):

        """Test Autoencoder"""

        sample_files = glob(self.args.test_dataset_path)
        print 'sa', len(sample_files)
        # load testing input
        print("Loading testing images ...")
        sample = [load_data(sample_file, is_test=True, prior=False) for sample_file in sample_files]
        sample_images = np.array(sample)
        print(sample_images.shape)

        if self.load():
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        for i, sample_image in enumerate(sample_images[:10]):
            sample_image = np.array([sample_image])
            idx = i + 1
            print("sampling image ", idx)
            samples = self.sess.run(self.decoded1, feed_dict={self.image_batch: sample_image})
            save_images(sample_image, samples, [1, 1], '{}/test_{:04d}.jpeg'.format(self.args.test_dir, idx))
            # imsave(inverse_transform(samples),[self.args.batch_size, 1],
            # '{}/test_{:04d}.png'.format(self.args.test_dir, idx))

    def get_feature_vectors(self, feat_vec_path, data_path):

        """Test Autoencoder"""
        h5_file = h5py.File(feat_vec_path, 'w')

        features_h5_open = h5py.File(self.args.h5features, 'r')
        features = np.array(features_h5_open['features']).astype(np.float32)
        files = list(np.array(features_h5_open['labels']))
        files = [i.replace('/home/compvis/notebooks/User_clustering/train_data_dir/',
                            '/home/ubuntuuser/datasets/image_datasets/VOC/sipan_resized/train_data_dir/') for i in
                  files]
        files = [i + '.jpg' for i in files]
        # load testing input
        print features.shape,


        if self.load():
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")
        vecs = []
        labels = []
        batch_size = 4000
        n_batches= int(len(files)/4000)
        print n_batches
        for i in range(n_batches):
            samples = self.sess.run(self.code_vector1, feed_dict={self.encoded_feature_1:  features[i*batch_size:(i+1)*batch_size]  })
            vecs.extend(samples)
            labels.extend( files[ i*batch_size:(i+1)*batch_size])
        h5_file['label'] = labels
        h5_file['vecs'] = vecs
        h5_file.close()


    def save(self, checkpoint_dir, step):
        model_name = "Autoencoder.model"
        model_dir = "%s_%s" % (model_name, self.args.batch_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def load(self):
        print(" [*] Reading checkpoint...")
        model_name = "Autoencoder.model"
        model_dir = "%s_%s" % (model_name, self.args.batch_size)
        print 'model_dir', model_dir
        checkpoint_dir = os.path.join(self.args.checkpoint_dir, model_dir)
        print 'check_dir', checkpoint_dir
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        print 'ckpt', ckpt
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            return False

    def siamese(self, feature_vector, reuse):
        print feature_vector.get_shape() ,'fe'
        network = tf.layers.dense(
            inputs=feature_vector,
            units=1024,
            reuse=reuse,
            activation=tf.nn.relu,
            # kernel_initializer=tf.truncated_normal_initializer(stddev=self.stddev),
            kernel_regularizer=tf.contrib.layers.l1_regularizer(scale=0.8),
            name='Dense_Layer1')
        print network.get_shape()
        network = tf.layers.dense(
            inputs=network,
            units=2048,
            activation= tf.nn.relu,
            reuse=reuse,
            # activation=tf.nn.tanh,
            # kernel_initializer=tf.truncated_normal_initializer(stddev=self.stddev),
            # kernel_regularizer=tf.contrib.layers.l1_regularizer(scale=0.8),
            name='Dense_Layer2')
        print network.get_shape()

        return network

