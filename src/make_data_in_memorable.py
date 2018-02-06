import configparser
import os, time
import pickle

def run():
    config = configparser.ConfigParser()
    config.read('config.ini')

    print('In-memorable data will generate')
    start_time = time.time()

    for target_dir, target_res in zip(INMEMORY_DIR_LISTS, INMEMORY_RES_LISTS):
        file_data_dicts = dict()
        for path, _, files in os.walk(target_dir):
            for file in files:
                with open(os.path.join(path, file), 'rb') as f:
                    file_name = os.path.splitext(file)[0]
                    file_data = pickle.load(f)
                    file_data_dicts[file_name] = file_data
        print('{target} file count: {cnt}'.format(target=target_dir, cnt=len(file_data_dicts)))

        with open(target_res, 'wb') as f:
            pickle.dump(file_data_dicts, f)

    print('In-memorable data generated : {}'.format(time.time() - start_time))
    pass


if __name__ == '__main__':
    run()
    pass