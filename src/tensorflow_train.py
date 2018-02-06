# v0.1 tensorflow CNN 모델 검증

import tensorflow as tf
import numpy as np
import os, sys, time, csv, pickle


def get_mini_batch(mal_lists, ben_lists):
    while True:
        half_batch_size = int(BATCH_SIZE / 2)
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


def load_data():
    # FH file
    with open(TRAIN_MALWARE_FHS_PATH, 'rb') as mal_file:
        mal_dicts = pickle.load(mal_file)
    with open(TRAIN_BENIGN_FHS_PATH, 'rb') as ben_file:
        ben_dicts = pickle.load(ben_file)

    return list(mal_dicts.values()), list(ben_dicts.values())


def initialize():
    # create model storage
    if not tf.gfile.Exists(MODEL_STORAGE):
        tf.gfile.MkDir(MODEL_STORAGE)

    # load data
    return load_data()


def train():
    mal_lists, ben_lists = initialize()

    # network architecture
    prob = tf.placeholder(tf.float32)

    x = tf.placeholder(tf.float32, shape=[None, INPUT_SIZE])
    y = tf.placeholder(tf.float32, shape=[None, OUTPUT_SIZE])
    x_image = tf.reshape(x, [-1, INPUT_SIZE, 1, 1])

    conv_layer_1 = tf.layers.conv2d(inputs=x_image, filters=2, kernel_size=[3, 1], padding="same",
                                    activation=tf.nn.relu)
    pool_layer_1 = tf.layers.max_pooling2d(inputs=conv_layer_1, pool_size=[2, 1], padding="valid", strides=2)
    pool_drop_1 = tf.nn.dropout(pool_layer_1, keep_prob=prob)
    # (13607, 1, 1) -> (6533, 1, 2)

    conv_layer_2 = tf.layers.conv2d(inputs=pool_drop_1, filters=4, kernel_size=[3, 1], padding="same",
                                    activation=tf.nn.relu)
    pool_layer_2 = tf.layers.max_pooling2d(inputs=conv_layer_2, pool_size=[2, 1], padding="valid", strides=2)
    pool_drop_2 = tf.nn.dropout(pool_layer_2, keep_prob=prob)
    # (6533, 1, 2) -> (3266, 1, 4)

    conv_layer_3 = tf.layers.conv2d(inputs=pool_drop_2, filters=8, kernel_size=[3, 1], padding="same",
                                    activation=tf.nn.relu)
    pool_layer_3 = tf.layers.max_pooling2d(inputs=conv_layer_3, pool_size=[2, 1], padding="valid", strides=2)
    pool_drop_3 = tf.nn.dropout(pool_layer_3, keep_prob=prob)
    # (3266, 1, 4) -> (1633, 1, 8)

    conv_layer_4 = tf.layers.conv2d(inputs=pool_drop_3, filters=16, kernel_size=[3, 1], padding="same",
                                    activation=tf.nn.relu)
    pool_layer_4 = tf.layers.max_pooling2d(inputs=conv_layer_4, pool_size=[2, 1], padding="valid", strides=2)
    pool_drop_4 = tf.nn.dropout(pool_layer_4, keep_prob=prob)
    # (1633, 1, 8) -> (816, 1, 16)

    convert_flat = tf.reshape(pool_drop_4, [-1, 816 * 1 * 16])
    # (816, 1, 16) -> (13056)

    dense_layer_1 = tf.layers.dense(inputs=convert_flat, units=4096, activation=tf.nn.relu)
    dense_drop_1 = tf.nn.dropout(dense_layer_1, prob)
    # (13056) -> (4096)

    dense_layer_2 = tf.layers.dense(inputs=dense_drop_1, units=512, activation=tf.nn.relu)
    dense_drop_2 = tf.nn.dropout(dense_layer_2, prob)
    # (4096) -> (512)

    dense_layer_3 = tf.layers.dense(inputs=dense_drop_2, units=64, activation=tf.nn.relu)
    dense_drop_3 = tf.nn.dropout(dense_layer_3, prob)
    # (512) -> (64)

    dense_layer_4 = tf.layers.dense(inputs=dense_drop_3, units=8, activation=tf.nn.relu)
    dense_drop_4 = tf.nn.dropout(dense_layer_4, prob)
    # (64) -> (8)

    y_ = tf.layers.dense(inputs=dense_drop_4, units=OUTPUT_SIZE)
    # (8) -> (2)

    # loss function: softmax, cross-entropy
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_, labels=y))

    # optimizer: Adaptive momentum optimizer
    optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cost)

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
        for i in range(ITER):
            (training_data, training_label) = next(train_iter)
            sess.run(optimizer, feed_dict={x: training_data, y: training_label, prob: DROPOUT_PROB})
            if (i % 10 == 0):
                print(i, sess.run(accuracy, feed_dict={x: training_data, y: training_label, prob: DROPOUT_PROB}))
                if (i % 1000 == 0):
                    model_saver.save(sess, os.path.normpath(os.path.abspath('./{}/model.ckpt'.format(MODEL_STORAGE))))
        print('------finish------')
        print('learning time : {}'.format(time.time() - start_time))
        model_saver.save(sess, os.path.normpath(os.path.abspath('./{}/model.ckpt'.format(MODEL_STORAGE))))
    pass


#  KISnet
if __name__ == '__main__':
    train()
    pass
