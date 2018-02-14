import configparser
import csv
import numpy as np
import os
import pickle
import random
import tensorflow as tf
import time


config = configparser.ConfigParser()
config.read('config.ini')


def get_mini_batch(file_lists, label_lists, batch_size):
    while True:
        batch_file_lists, batch_label_lists = list(), list()

        idx_lists = list(range(len(file_lists)))
        random.shuffle(idx_lists)

        batched_idx_lists = idx_lists[:batch_size]

        for idx in batched_idx_lists:
            batch_file_lists.append(file_lists[idx])
            batch_label_lists.append(label_lists[idx])

        yield (batch_file_lists, batch_label_lists)


def load_data(train_flag=True):
    if train_flag:
        mal_file_path = config.get('PATH', 'TRAIN_MAL_FHS')
        ben_file_path = config.get('PATH', 'TRAIN_BEN_FHS')
    else:
        mal_file_path = config.get('PATH', 'EVAL_MAL_FHS')
        ben_file_path = config.get('PATH', 'EVAL_BEN_FHS')

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


def get_model_snapshot_path(step):
    # create model storage
    model_storage = config.get('CLASSIFIER', 'MODEL_STORAGE')+str(step)
    if not tf.gfile.Exists(model_storage):
        tf.gfile.MkDir(model_storage)

    return os.path.normpath(os.path.abspath('./{}/model.ckpt'.format(model_storage)))


def inference(x, prob=0.0, train_flag=True):
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


def train(step, log):
    print('# {}'.format(step))

    # load data
    print('load data start')
    start_time = time.time()
    file_lists, label_lists = load_data()
    load_time = time.time() - start_time
    print('load data finish : {} seconds'.format(load_time))

    # network architecture
    with tf.device('/gpu:0'):
        tf.reset_default_graph()

        x = tf.placeholder(tf.float32, shape=[None, int(config.get('CLASSIFIER', 'INPUT_SIZE'))])
        y = tf.placeholder(tf.float32, shape=[None, int(config.get('CLASSIFIER', 'OUTPUT_SIZE'))])
        prob = tf.placeholder(tf.float32)

        y_ = inference(x, prob=prob)

        # loss function: softmax, cross-entropy
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_, labels=y))

        # optimizer: Adaptive momentum optimizer
        optimizer = tf.train.AdamOptimizer(float(config.get('CLASSIFIER', 'LEARNING_RATE'))).minimize(cost)

        # predict
        prediction = tf.equal(tf.argmax(y_, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))

    # create model snapshot
    model_path = get_model_snapshot_path(step)

    # training session start
    keep_prob = float(1 - float(config.get('CLASSIFIER', 'DROPOUT_PROB')))
    model_saver = tf.train.Saver()
    init = tf.global_variables_initializer()
    train_iter = get_mini_batch(file_lists, label_lists, int(config.get('CLASSIFIER', 'BATCH_SIZE')))
    tf_config = tf.ConfigProto(allow_soft_placement=False, log_device_placement=False)

    with tf.Session(config=tf_config) as sess:
        sess.run(init)
        print('LEARNING START')
        start_time = time.time()
        for i in range(int(config.get('CLASSIFIER', 'ITER'))):
            (training_data, training_label) = next(train_iter)
            if i % 100 == 0:
                print(i, sess.run(accuracy, feed_dict={x: training_data, y: training_label, prob: keep_prob}))
                if i % 1000 == 0:
                    model_saver.save(sess, model_path)
            else:
                sess.run(optimizer, feed_dict={x: training_data, y: training_label, prob: keep_prob})
        else:
            (training_data, training_label) = next(train_iter)
            end_loss = sess.run(cost, feed_dict={x: training_data, y: training_label, prob: keep_prob})
        print('------finish------')
        train_time = time.time() - start_time
        print('training time : {}'.format(train_time))
        model_saver.save(sess, model_path)

        log += [step, end_loss, load_time, train_time]
    pass


def evaluate(step, log):
    # load data
    print('load data start')
    start_time = time.time()
    file_lists, label_lists = load_data(train_flag=False)
    load_time = time.time() - start_time
    print('load data finish : {} seconds'.format(load_time))

    number_of_data = len(file_lists)

    # network architecture
    with tf.device('/gpu:0'):
        tf.reset_default_graph()

        x = tf.placeholder(tf.float32, shape=[None, int(config.get('CLASSIFIER', 'INPUT_SIZE'))])
        y_ = inference(x, train_flag=False)

    # restore model snapshot
    model_path = get_model_snapshot_path(step)

    # evaluating session start
    model_saver = tf.train.Saver()
    init = tf.global_variables_initializer()
    tf_config = tf.ConfigProto(allow_soft_placement=False, log_device_placement=False)

    print('total: {}'.format(number_of_data))
    with tf.Session(config=tf_config) as sess:
        sess.run(init)
        model_saver.restore(sess, model_path)

        print('--evaluating start--')
        acc_cnt = 0
        start_time = time.time()
        for i in range(number_of_data):
            try:
                pred = sess.run(y_, feed_dict={x: [file_lists[i]]})
                pred_label = np.array(pred).reshape([-1]).argmax(-1)  # 1 if malware else 0
                actual_label = label_lists[i].argmax(-1)

                if pred_label == actual_label:
                    acc_cnt += 1
                if i % 1000 == 0:
                    print(('{cnt} : {acc}'.format(cnt=i, acc=acc_cnt / (i + 1))))
            except:
                pass
        print('------finish------')
        evaluation_time = time.time() - start_time
        total_accuracy = acc_cnt / number_of_data
        print('test time : {}'.format(evaluation_time))
        print('accuracy : {}'.format(total_accuracy))

        log += [load_time, evaluation_time, total_accuracy]
    pass


def run():
    for step in range(1, 2):
        log = list()
        train(step, log)
        evaluate(step, log)
        with open(config.get('BASIC_INFO', 'TF_LOG_FILE_NAME'), 'a', encoding='utf-8', newline='') as f:
            wr = csv.writer(f)
            wr.writerow(log)
    pass


if __name__ == '__main__':
    run()
    pass
