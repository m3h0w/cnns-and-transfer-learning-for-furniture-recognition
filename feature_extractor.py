import cv2
import numpy as np
from joblib import Parallel, delayed
import concurrent.futures
import matplotlib.pyplot as plt
from tqdm import tqdm
from parallel import parallel_process
import time
import glob, os, sys
import pickle

from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.models import load_model

import numpy as np

from keras import backend as K
# model = load_model('model.h5')
model = ResNet50(weights='imagenet', include_top=False)
# get_features_layer_output = K.function([model.layers[0].input],
#                                   [model.layers[-3].output])
n_samples = 12000
progress_bar = tqdm(total=n_samples)

def get_features(file, multiply):
    print(file)
    img = image.load_img(file, target_size=(224, 224))
    # print(img)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    # layer_output = get_features_layer_output([x])[0]
    progress_bar.update(multiply)
    return [file, model.predict(x).reshape(-1)]

# n_samples = 30000


def get_hist(file, multiply):
    # img = cv2.imread('./data/train/'+str(i)+'.jpg',0)
    img = cv2.imread(file, 0)
    hist, _ = np.histogram(img, density=False)
    # progress_bar.update(multiply)
    return hist/hist.sum()

if __name__ == '__main__':
    print(sys.argv[1])
    # n_iters = 500
    # # n_samples = 1000 # change up before the progress bar on top of this file as well
    # i = 0
    # j = 0
    if not sys.argv[1]:
        mode = 'train' # train / test / valid
    else:
        mode = sys.argv[1]

    data_path = '/furniture-data/'
    pickle_name_core = data_path + 'features/cnn_features_'+mode

    if mode == 'test':
        pickle_name = pickle_name_core + '.pkl'
        try:
            dict_prev = pickle.load(open(pickle_name, 'rb'))
        except:
            dict_prev = {}
        dict_len = len(dict_prev.keys())
        
        if(dict_prev):
            print(pickle_name, " already exists.")
            sys.exit(0)
        else:
            print("Extracting features for: ", mode)

        files = glob.glob(data_path + mode + "/*")
        progress_bar.total = len(files)
        n_jobs = 4
        print("FILES:", len(files))
        X = Parallel(n_jobs=n_jobs)(delayed(get_features)(file, n_jobs) for file in files)
        print("len of X", len(X))
        
        if(len(X) == 0):
            raise
        # if(len(X) != n_samples)
        #     n_iters += 1
        # print("--- %s seconds ---" % (time.time() - start_time))
        dict_new = {item[0]: item[1] for item in X}

        with open(pickle_name, 'wb') as f:
            pickle.dump(dict_new, f)
        sys.exit(0)

    for batch_i in range(1,129):
        
        pickle_name = pickle_name_core + '_' + str(batch_i) + '.pkl'
        try:
            dict_prev = pickle.load(open(pickle_name, 'rb'))
        except:
            dict_prev = {}
        dict_len = len(dict_prev.keys())
        
        if(dict_prev):
            print(pickle_name, " already exists.")
            continue
        else:
            print("Extracting features for: ", mode, batch_i)
        
        X = list()
        start_time = time.time()
        files = glob.glob(data_path + mode + "/" + str(batch_i) + "/*")
        progress_bar.total = len(files)
        n_jobs = 4
        print("FILES:", len(files))
        X = Parallel(n_jobs=n_jobs)(delayed(get_features)(file, n_jobs) for file in files)
        print("len of X", len(X))
        
        if(len(X) == 0):
            raise
        # if(len(X) != n_samples)
        #     n_iters += 1
        print("--- %s seconds ---" % (time.time() - start_time))
        dict_new = {item[0]: item[1] for item in X}

        with open(pickle_name, 'wb') as f:
            pickle.dump(dict_new, f)
        
    # X = pickle.load(open('hist_features.pkl', 'rb'))
    # print(len(X))