from sklearn.metrics import confusion_matrix
import configparser
import numpy as np
import os
import pickle
import sys
import tensorflow as tf
import time


config = configparser.ConfigParser()
config.read('config.ini')


def load_data():
    # file_dicts, label_dicts = dict(), dict()
    file_lists, label_lists = list(), list()

    with open(config.get('PATH', 'TEST_MAL_FHS'), 'rb') as test_file:
        contents = pickle.load(test_file)
        # file_dicts.update(contents)
        # for md5 in contents.keys():
        #     label_dicts[md5] = np.array([0, 1])
        file_lists += contents
        label_lists += [np.array([0, 1]) for n in contents]
    with open(config.get('PATH', 'TEST_BEN_FHS'), 'rb') as test_file:
        contents = pickle.load(test_file)
        # file_dicts.update(contents)
        # for md5 in contents.keys():
        #     label_dicts[md5] = np.array([1, 0])
        file_lists += contents
        label_lists += [np.array([1, 0]) for n in contents]

    # return list(file_dicts.values()), list(label_dicts.values())  # dict
    return file_lists, label_lists


def initialize():
    # check model storage
    if not tf.gfile.Exists(config.get('CLASSIFIER', 'MODEL_STORAGE')):
        print('ERROR : you have to train classifer first.')
        sys.exit(1)

    # load data
    return load_data()


def eval():
    file_lists, label_lists = initialize()
    number_of_data = len(file_lists)

    # network architecture
    x = tf.placeholder(tf.float32, shape=[None, int(config.get('CLASSIFIER', 'INPUT_SIZE'))])
    x_image = tf.reshape(x, [-1, int(config.get('CLASSIFIER', 'INPUT_SIZE')), 1, 1])

    conv_layer_1 = tf.layers.conv2d(inputs=x_image, filters=2, kernel_size=[3, 1], padding="same",
                                    activation=tf.nn.relu)
    pool_layer_1 = tf.layers.max_pooling2d(inputs=conv_layer_1, pool_size=[2, 1], padding="valid", strides=2)

    conv_layer_2 = tf.layers.conv2d(inputs=pool_layer_1, filters=4, kernel_size=[3, 1], padding="same",
                                    activation=tf.nn.relu)
    pool_layer_2 = tf.layers.max_pooling2d(inputs=conv_layer_2, pool_size=[2, 1], padding="valid", strides=2)

    conv_layer_3 = tf.layers.conv2d(inputs=pool_layer_2, filters=8, kernel_size=[3, 1], padding="same",
                                    activation=tf.nn.relu)
    pool_layer_3 = tf.layers.max_pooling2d(inputs=conv_layer_3, pool_size=[2, 1], padding="valid", strides=2)

    conv_layer_4 = tf.layers.conv2d(inputs=pool_layer_3, filters=16, kernel_size=[3, 1], padding="same",
                                    activation=tf.nn.relu)
    pool_layer_4 = tf.layers.max_pooling2d(inputs=conv_layer_4, pool_size=[2, 1], padding="valid", strides=2)

    convert_flat = tf.reshape(pool_layer_4, [-1, 768 * 1 * 16])

    dense_layer_1 = tf.layers.dense(inputs=convert_flat, units=4096, activation=tf.nn.relu)

    dense_layer_2 = tf.layers.dense(inputs=dense_layer_1, units=512, activation=tf.nn.relu)

    dense_layer_3 = tf.layers.dense(inputs=dense_layer_2, units=64, activation=tf.nn.relu)

    dense_layer_4 = tf.layers.dense(inputs=dense_layer_3, units=8, activation=tf.nn.relu)

    y_ = tf.layers.dense(inputs=dense_layer_4, units=int(config.get('CLASSIFIER', 'OUTPUT_SIZE')))

    y_test = tf.nn.softmax(y_)

    # testing session start
    model_saver = tf.train.Saver()
    init = tf.global_variables_initializer()

    tf_config = tf.ConfigProto(allow_soft_placement=False, log_device_placement=False)

    # 실행 과정에서 요구되는 만큼의 GPU memory만 할당
    with tf.Session(config=tf_config) as sess:
        sess.run(init)
        model_saver.restore(sess, os.path.normpath(
            os.path.abspath('./{}/model.ckpt'.format(config.get('CLASSIFIER', 'MODEL_STORAGE')))))

        acc_cnt = 0
        print('total: {}'.format(number_of_data))
        print('--testing start--')
        start_time = time.time()

        for i in range(number_of_data):
            try:
                pred_y = sess.run(y_test, feed_dict={x: [file_lists[i]]})
                pred_label = np.array(pred_y).reshape([-1]).argmax(-1)  # 1 if malware else 0
                actual_label = label_lists[i].argmax(-1)
                # print('pred label: {pred}, actual label: {act}'.format(pred=pred_label, act=actual_label))

                if (pred_label == actual_label):
                    acc_cnt += 1
                if (i % 1000 == 0):
                    print(('{cnt} : {acc}'.format(cnt=i, acc=acc_cnt / (i + 1))))
            except:
                # print('error in {i}th: {err}'.format(i=i, err=e))
                pass

        print('------finish------')
        print('test time : {}'.format(time.time() - start_time))
        print('accuracy : {}'.format(acc_cnt / number_of_data))
    pass


#  KISnet
if __name__ == '__main__':
    eval()
    pass