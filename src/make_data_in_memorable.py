import os, time
import pickle

DIR_LISTS = [os.path.normpath(os.path.abspath('../data/fh/train_mal')), os.path.normpath(os.path.abspath('../data/fh/train_ben')), os.path.normpath(os.path.abspath('../data/fh/eval_mal')), os.path.normpath(os.path.abspath('../data/fh/eval_ben'))]
RES_LISTS = [os.path.normpath(os.path.abspath('../data/fh/train_mal.fhs')), os.path.normpath(os.path.abspath('../data/fh/train_ben.fhs')), os.path.normpath(os.path.abspath('../data/fh/eval_mal.fhs')), os.path.normpath(os.path.abspath('../data/fh/eval_ben.fhs'))]

def run():
    print('In-memorable data will generate')
    start_time = time.time()

    for target_dir, target_res in zip(DIR_LISTS, RES_LISTS):
        file_data_dicts = dict()
        for path, _, files in os.walk(target_dir):
            for file in files:
                with open(os.path.join(path, file), 'rb') as f:
                    file_name = os.path.splitext(file)[0]
                    file_data = pickle.load(f)
                    file_data_dicts[file_name] = file_data
        print('{target} file count: {cnt}'.format(target=target_dir, cnt=len(file_data_dicts)))

        with open(target_res, 'wb') as f:
            pickle.dump(list(file_data_dicts.values()), f)  # return list

    print('In-memorable data generated : {}'.format(time.time() - start_time))
    pass


if __name__ == '__main__':
    # run()

    pass