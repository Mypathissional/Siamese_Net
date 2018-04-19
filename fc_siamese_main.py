from imports import *
from tmep import SiameseFC

parser = argparse.ArgumentParser(description='')
parser.add_argument('--input_size', dest='input_size', type=int, default=2048, help='# size of the input picture')
parser.add_argument('--train_dataset_path', dest='train_dataset_path', default='/home/ubuntuuser/datasets/train_data_dir_new_70_false_700_false.csv', help='# size of the input picture')

# parser.add_argument('--train_dataset_path', dest='train_dataset_path',
#         default='/home/ubuntuuser/datasets/train_data_dir_new_small_100.csv, help='# size of the input picture' )
#parser.add_argument('--train_dataset_path', dest='train_dataset_path', default='/home/ubuntuuser/datasets/image_datasets/VOC/sipan_resized/train_data_dir/*/*.jpg', help='# size of the input picture')

parser.add_argument('--h5features', dest='h5features', default='/home/ubuntuuser/PycharmProjects/keras-adversarial/examples/result256/second_training/train_features.h5', help='# size of the input picture')

parser.add_argument('--epoch', dest='epoch', type=int, default=1500, help='# of epoch')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=5000, help='# images in batch')
parser.add_argument('--train_size', dest='train_size', type=int, default=1e8, help='# images used to train')
parser.add_argument('--load_size', dest='load_size', type=int, default=256, help='scale images to this size')
parser.add_argument('--lr', dest='lr', type=float, default=0.0002, help='initial learning rate for adam')
parser.add_argument('--phase', dest='phase', default='train', help='train, test,pretrain')
parser.add_argument('--train_mode', dest='train_mode', default='siamese', help='autoencoder,siamese')

parser.add_argument('--save_epoch_freq', dest='save_epoch_freq', type=int, default=20, help='save a model every save_epoch_freq epochs (does not overwrite previously saved models)')
parser.add_argument('--save_latest_freq', dest='save_latest_freq', type=int, default=20, help='save the latest model every latest_freq sgd iterations (overwrites the previous lates')
#parser.add_argument('--checkpoint_dir', dest='checkpoint_dir', default='./logs_r11', help='models are saved here')
parser.add_argument('--checkpoint_dir', dest='checkpoint_dir', default='./fc_simaese_based_autoencoderfeatures/', help='models are saved here')

parser.add_argument('--test_dir', dest='test_dir', default='./test_1', help='test sample are saved here')

args = parser.parse_args()

def main(_):

    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)
    if not os.path.exists(args.test_dir):
        os.makedirs(args.test_dir)

    with tf.Session() as sess:
        model = SiameseFC(sess=sess, args=args)

        if args.phase == 'train':
            model.train()
        else:
            model.test()

        #model.get_feature_vectors(feat_vec_path='./fc_simaese_based_autoencoderfeatures/new_features2.h5',data_path='/home/ubuntuuser/PycharmProjects/keras-adversarial/examples/result256/second_training/train_features.h5')

if __name__ == '__main__':
    tf.app.run()

