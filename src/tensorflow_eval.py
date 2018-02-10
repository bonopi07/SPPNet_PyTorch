from sklearn.metrics import confusion_matrix
import configparser
import numpy as np
import os
import pickle
import sys
import tensorflow as tf
import time


def eval():
    file_lists, label_lists = initialize()
    number_of_data = len(file_lists)

    # network architecture


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