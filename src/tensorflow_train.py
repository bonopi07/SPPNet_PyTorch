import configparser
import numpy as np
import os
import pickle
import tensorflow as tf
import time


config = configparser.ConfigParser()
config.read('config.ini')


def get_mini_batch(mal_lists, ben_lists):
    while True:
        half_batch_size = int(int(config.get('CLASSIFIER', 'BATCH_SIZE')) / 2)
        batch_data_lists, batch_label_lists = list(), list()

        np.random.shuffle(mal_lists)
        np.random.shuffle(ben_lists)

        batch_data_lists += mal_lists[:half_batch_size]
        batch_data_lists += ben_lists[:half_batch_size]

        # create label vector
        for i in range(half_batch_size):
            batch_label_lists.append(np.array([0, 1]))  # malware
        for i in range(half_batch_size):
            batch_label_lists.append(np.array([1, 0]))  # benign

        yield (batch_data_lists, batch_label_lists)


def load_data(train_flag=True):
    if train_flag:
        mal_file_path = config.get('PATH', 'TRAIN_MAL_FHS')
        ben_file_path = config.get('PATH', 'TRAIN_BEN_FHS')
    else:
        mal_file_path = config.get('PATH', 'TEST_MAL_FHS')
        ben_file_path = config.get('PATH', 'TEST_BEN_FHS')

    file_lists, label_lists = list(), list()
    with open(mal_file_path, 'rb') as mal_file:
        contents = pickle.load(mal_file)
        file_lists += contents
        label_lists += [np.array([0, 1]) for n in contents]
    with open(ben_file_path, 'rb') as ben_file:
        contents = pickle.load(ben_file)
        file_lists += contents
        label_lists += [np.array([1, 0]) for n in contents]

    return file_lists, label_lists


def create_model_snapshot(step):
    # create model storage
    model_storage = config.get('CLASSIFIER', 'MODEL_STORAGE{}'.format(step))
    if not tf.gfile.Exists(model_storage):
        tf.gfile.MkDir(model_storage)

    return model_storage


def put_log(sequence, log):
    print(sequence)
    log.append(sequence)


def inference(x, prob, train_flag=True):
    x_image = tf.reshape(x, [-1, int(config.get('CLASSIFIER', 'INPUT_SIZE')), 1, 1])

    conv_layer_1 = tf.layers.conv2d(inputs=x_image, filters=2, kernel_size=[3, 1], padding="same", activation=tf.nn.relu)
    pool_layer_1 = tf.layers.max_pooling2d(inputs=conv_layer_1, pool_size=[2, 1], padding="valid", strides=2)
    if train_flag:
        pool_layer_1 = tf.nn.dropout(pool_layer_1, keep_prob=prob)
    conv_layer_2 = tf.layers.conv2d(inputs=pool_layer_1, filters=4, kernel_size=[3, 1], padding="same", activation=tf.nn.relu)
    pool_layer_2 = tf.layers.max_pooling2d(inputs=conv_layer_2, pool_size=[2, 1], padding="valid", strides=2)
    if train_flag:
        pool_layer_2 = tf.nn.dropout(pool_layer_2, keep_prob=prob)
    conv_layer_3 = tf.layers.conv2d(inputs=pool_layer_2, filters=8, kernel_size=[3, 1], padding="same", activation=tf.nn.relu)
    pool_layer_3 = tf.layers.max_pooling2d(inputs=conv_layer_3, pool_size=[2, 1], padding="valid", strides=2)
    if train_flag:
        pool_layer_3 = tf.nn.dropout(pool_layer_3, keep_prob=prob)
    conv_layer_4 = tf.layers.conv2d(inputs=pool_layer_3, filters=16, kernel_size=[3, 1], padding="same", activation=tf.nn.relu)
    pool_layer_4 = tf.layers.max_pooling2d(inputs=conv_layer_4, pool_size=[2, 1], padding="valid", strides=2)
    if train_flag:
        pool_layer_4 = tf.nn.dropout(pool_layer_4, keep_prob=prob)

    convert_flat = tf.reshape(pool_layer_4, [-1, 768 * 1 * 16])

    dense_layer_1 = tf.layers.dense(inputs=convert_flat, units=4096, activation=tf.nn.relu)
    if train_flag:
        dense_layer_1 = tf.nn.dropout(dense_layer_1, prob)
    dense_layer_2 = tf.layers.dense(inputs=dense_layer_1, units=512, activation=tf.nn.relu)
    if train_flag:
        dense_layer_2 = tf.nn.dropout(dense_layer_2, prob)
    dense_layer_3 = tf.layers.dense(inputs=dense_layer_2, units=64, activation=tf.nn.relu)
    if train_flag:
        dense_layer_3 = tf.nn.dropout(dense_layer_3, prob)
    dense_layer_4 = tf.layers.dense(inputs=dense_layer_3, units=8, activation=tf.nn.relu)
    if train_flag:
        dense_layer_4 = tf.nn.dropout(dense_layer_4, prob)

    y_ = tf.layers.dense(inputs=dense_layer_4, units=int(config.get('CLASSIFIER', 'OUTPUT_SIZE')))

    if train_flag:
        return y_
    else:
        return tf.nn.softmax(y_)


def train(step):
    # initialize
    create_model_snapshot(step)
    log = list()

    print('load data start')
    file_lists, label_lists = load_data()
    print('load data finish')

    # network architecture
    x = tf.placeholder(tf.float32, shape=[None, int(config.get('CLASSIFIER', 'INPUT_SIZE'))])
    y = tf.placeholder(tf.float32, shape=[None, int(config.get('CLASSIFIER', 'OUTPUT_SIZE'))])
    prob = tf.placeholder(tf.float32)

    # loss function: softmax, cross-entropy
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=inference(x, prob=prob), labels=y))

    # optimizer: Adaptive momentum optimizer
    optimizer = tf.train.AdamOptimizer(float(config.get('CLASSIFIER', 'LEARNING_RATE'))).minimize(cost)

    # predict
    prediction = tf.equal(tf.argmax(y_, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))

    # training session start
    model_saver = tf.train.Saver()
    init = tf.global_variables_initializer()
    train_iter = get_mini_batch(mal_lists, ben_lists)
    tf_config = tf.ConfigProto(allow_soft_placement=False, log_device_placement=False)

    with tf.Session(config=tf_config) as sess:
        sess.run(init)
        print('LEARNING START')
        start_time = time.time()
        for i in range(int(config.get('CLASSIFIER', 'ITER'))):
            (training_data, training_label) = next(train_iter)
            sess.run(optimizer, feed_dict={x: training_data, y: training_label, prob: float(config.get('CLASSIFIER', 'DROPOUT_PROB'))})
            if (i % 100 == 0):
                print(i, sess.run(accuracy, feed_dict={x: training_data, y: training_label, prob: float(config.get('CLASSIFIER', 'DROPOUT_PROB'))}))
                if (i % 1000 == 0):
                    model_saver.save(sess, os.path.normpath(os.path.abspath('./{}/model.ckpt'.format(config.get('CLASSIFIER', 'MODEL_STORAGE')))))
        print('------finish------')
        print('learning time : {}'.format(time.time() - start_time))
        model_saver.save(sess, os.path.normpath(os.path.abspath('./{}/model.ckpt'.format(config.get('CLASSIFIER', 'MODEL_STORAGE')))))
    pass


def run():
    for step in range(1, 11):
        train(step)
        evaluate(step)

if __name__ == '__main__':
    run()
    pass
