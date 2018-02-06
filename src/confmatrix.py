# URL: http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py

import numpy as np
import matplotlib.pyplot as plt
import pickle
import matplotlib.patches as mpatches
import os, sys
import itertools

INPUT_FILE = 'ANN_FH_BC.cm'
class_name_legend = ['benign', 'malware']

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    # plt.xticks(tick_marks, classes, rotation=90)
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    # information about each block's value
    fmt = '.7f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    ## insert legend information
    # patches = [mpatches.Patch(color='white', label='G{num} = {group}'.format(num=i+1, group=class_name_legend[i])) for i in range(len(class_name_legend))]
    # plt.legend(handles=patches, bbox_to_anchor=(-0.60, 1), loc=2, borderaxespad=0.)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    pass

if __name__ == '__main__':
    ## import some data to play with
    # iris = datasets.load_iris()
    # X = iris.data
    # y = iris.target
    # class_names = iris.target_names

    ## Split the data into a training set and a test set
    # X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    ## Run classifier, using a model that is too regularized (C too low) to see
    ## the impact on the results
    # classifier = svm.SVC(kernel='linear', C=0.01)
    # y_pred = classifier.fit(X_train, y_train).predict(X_test)

    ## Compute confusion matrix
    # cnf_matrix = confusion_matrix(y_test, y_pred)
    np.set_printoptions(precision=2)

    # class_name = list()
    # for i in range(class_name_legend.__len__()):
    #     class_name.append('G{num}'.format(num=i+1))
    class_name = ['benign', 'malware']

    # for path, dir, files in os.walk(os.path.normpath('C:\\Users\\seongmin\\Desktop\\11_15_Analytics_Experiment\\5. CNN(Assembly, Binary)\\time_20_data')):
    #     for file in files:
    #         print(os.path.join(path, file))
    #         num_date = int(file.split('_')[1]) + 1
    #
    #         with open(os.path.join(path, file), 'rb') as f:
    #             mat_info = np.array(pickle.load(f))
    #
    #             # Plot normalized confusion matrix
    #             plt.figure()
    #             plot_confusion_matrix(mat_info, classes=class_name, normalize=True,
    #                                   title='Normalized confusion matrix, test: 9/{num}'.format(num=num_date))
    #             plt.show()

    with open(INPUT_FILE, 'rb') as f:
        mat_info = np.array(pickle.load(f))

        # Plot normalized confusion matrix
        plt.figure()
        plot_confusion_matrix(mat_info, classes=class_name, normalize=True,
                              title='ANN, Feature Hashing, Binary Classification (normalized)')
        plt.show()