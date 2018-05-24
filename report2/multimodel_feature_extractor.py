import numpy as np
import time
import glob, os, sys
import pickle
from subprocess import call

import numpy as np
from keras import backend as K
from keras.preprocessing import image
import keras

from keras.applications.resnet50 import ResNet50 as resnet
from keras.applications.resnet50 import preprocess_input as resnet_pp

from keras.applications.densenet import DenseNet201 as densenet
from keras.applications.densenet import preprocess_input as densenet_pp

from keras.applications.vgg16 import VGG16 as vgg16
from keras.applications.vgg16 import preprocess_input as vgg16_pp

from keras.applications.inception_v3 import InceptionV3 as inception
from keras.applications.inception_v3 import preprocess_input as inception_pp

from keras.applications.xception import Xception as xception
from keras.applications.xception import preprocess_input as xception_pp

# Creates a dictionary of available models
def create_models_dictionary():
    models = {}
    _create_model_entry(models, 'resnet', resnet, resnet_pp, 244)
    _create_model_entry(models, 'densenet', densenet, densenet_pp, 244)
    _create_model_entry(models, 'vgg16', vgg16, vgg16_pp, 244)
    _create_model_entry(models, 'inception', inception, inception_pp, 299)
    _create_model_entry(models, 'xception', xception, xception_pp, 299)
    return models

# Utility for creating one entry in the models dictionary
def _create_model_entry(models, name, cnn, pp, img_size):
    models[name] = dict(cnn_object=cnn, pp_function=pp, img_size=img_size)

# Build a model by adding preprocessing before the pretrained CNN
def get_feature_extraction_model(model_name):
    cnn_object, pp_function, img_size = _get_pretrained_model(model_name)
    model = keras.models.Sequential()
    cnn_model = cnn_object(weights='imagenet', include_top=False)
    model.add(keras.layers.Lambda(pp_function, name='preprocessing', input_shape=(img_size, img_size, 3)))
    model.add(cnn_model)
    return model

# Unpacking information from the models dictionary
def _get_pretrained_model(model_name):
    cnn_object = models[model_name]['cnn_object']
    pp_function = models[model_name]['pp_function']
    img_size = models[model_name]['img_size']
    return cnn_object, pp_function, img_size

# Forward-propagate through the model to get the features of provided files
def get_features(files, model):
    # Load images based on the size of the Lambda layer 
    # provided as the first layer before the pretrained CNN
    x = np.array([image.img_to_array(image.load_img(file, target_size=(model.layers[0].input_shape[1], model.layers[0].input_shape[1]))) for file in files])
    return model.predict(x)

# Check if the pickle with features already exists and try to extract
# features by the CNN if it doesn't. Save them in a pickle.
def _extract_features(pickle_name, model, files_path):
    try:
        dict_prev = pickle.load(open(pickle_name, 'rb'))
        print(pickle_name, " already exists.")
        return 0       
    except:
        pass 
        
    start_time = time.time()
    files = glob.glob(files_path)
    if(len(files) <= 0):
        print('Extracting features failed because no files exist in the path ', files_path)
        return -1

    print("FILES:", len(files))
    X = get_features(files, model)   
    if(len(X) == 0):
        print('Extracting features failed because 0 features were extracted from ', files_path)
        return -1
    
    print("len of X", len(X))
    print("--- %s seconds ---" % (time.time() - start_time))

    dict_new = {file: X[i] for i, file in enumerate(files)}

    with open(pickle_name, 'wb') as f:
        pickle.dump(dict_new, f)
    
    return 0

# Decide how to extract features
# for test: one big pickle
# for valid and train: pickle per class
# Available modes: train, valid, test
# (MODES HAVE TO FIT THE NAMES OF THE FOLDERS IN DATA DIRECTORY)
# - data_path:
#   - test: images
#   - train: class_folders: images
#   - valid: class_folders: images
def extract_features_main(mode, model, data_path, max_label=None):
    
    # data_path = '/furniture-data/' - absolute path for the ubuntu VM
    data_path = data_path
    pickle_name_core = data_path + '/features_' + model_name + '/cnn_features_' + mode
    call(['mkdir', '-p', data_path + '/' + pickle_name_core.split('/')[-2]])

    if mode == 'test':
        pickle_name = pickle_name_core + '.pkl'
        files_path = data_path + '/' + mode + '/*'
        print("Extracting features for: ", mode)
        _extract_features(pickle_name, model, files_path)
        return

    if not max_label:
        max_label = 129
    for batch_i in range(1,max_label+1):
        pickle_name = pickle_name_core + '_' + str(batch_i) + '.pkl'
        files_path = data_path + '/' + mode + "/" + str(batch_i) + "/*"
        print("Extracting features for: ", mode, batch_i)
        _extract_features(pickle_name, model, files_path)
    
    return

if __name__ == '__main__':
    models = create_models_dictionary()

    model_name = sys.argv[1]  
    data_path = sys.argv[2]
    if(len(sys.argv) < 4):
        max_label = None
    else:
        max_label = int(sys.argv[3])

    model = get_feature_extraction_model(model_name)

    extract_features_main('valid', model, data_path, max_label)
    extract_features_main('train', model, data_path, max_label)
    extract_features_main('test', model, data_path)
