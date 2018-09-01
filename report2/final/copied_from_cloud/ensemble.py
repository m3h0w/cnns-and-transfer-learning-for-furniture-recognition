import numpy as np
import time
import glob, os, sys
import pickle
from subprocess import call

import numpy as np
from keras import backend as K
from keras.preprocessing import image
import keras

from keras.preprocessing.image import ImageDataGenerator
from customgenerator import DataGenerator

from keras.applications.resnet50 import ResNet50 as resnet
from keras.applications.resnet50 import preprocess_input as resnet_pp

from keras.applications.densenet import DenseNet201 as densenet
from keras.applications.densenet import preprocess_input as densenet_pp

DENSENET_IMG_SIZE = 224
RESNET_IMG_SIZE = 244

DENSENET_FEAT_SIZE = (7,7,1920)
RESNET_FEAT_SIZE = (2048,)

N_CLASSES = 128

# Build a model by adding preprocessing before the pretrained CNN
def get_feature_extraction_model(model_name):
    if(model_name == 'densenet'):
        cnn_object, pp_function, img_size = densenet, densenet_pp, DENSENET_IMG_SIZE

    if(model_name == 'resnet'):
        cnn_object, pp_function, img_size = resnet, resnet_pp, RESNET_IMG_SIZE

    model = keras.models.Sequential()
    cnn_model = cnn_object(weights='imagenet', include_top=False)
    model.add(keras.layers.Lambda(pp_function, name='preprocessing', input_shape=(img_size, img_size, 3)))
    model.add(cnn_model)
    return model

def _get_id_from_path(path):
    return int(path.split('/')[-1].split('.')[0])

def get_test_extraction_datagen(test_dir, img_size, aug_i = 0):
    if(aug_i > 0):
        test_datagen = ImageDataGenerator(
            rotation_range=10,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True
        )
    else:
        test_datagen = ImageDataGenerator()

    return test_datagen.flow_from_directory(
        test_dir,
        target_size=(img_size, img_size),
        batch_size=1,
        class_mode='categorical'
    )

def extract_and_save_features(extraction_model_name='densenet', test_data_dir='/furniture-data/test', img_size = 224, aug_i = 0):
    
    generator = get_test_extraction_datagen(test_data_dir, img_size, aug_i)
    model = get_feature_extraction_model(extraction_model_name)

    start_time = time.time()
    X = model.predict_generator(generator, len(generator.filenames))
    if(len(X) == 0):
        print('Extracting features failed')
        return -1
    print("len of X", len(X))
    print("--- %s seconds ---" % (time.time() - start_time))

    print("Saving files.")
    for i, file in enumerate(generator.filenames):
        print(i)
        np.save('../data_'+extraction_model_name+'_test_'+str(aug_i)+'/' + str(_get_id_from_path(file)), X[i])

    ids = [str(_get_id_from_path(file)) for file in generator.filenames]
    np.save('../data_'+extraction_model_name+'/meta/test_ids_'+str(aug_i), ids)

    return 0

def get_test_datagen(model_name = 'densenet', aug_i=0):
    if(model_name == 'densenet'):
        dim = DENSENET_FEAT_SIZE
    if(model_name == 'resnet'):
        dim = RESNET_FEAT_SIZE
    else:
        dim = DENSENET_FEAT_SIZE

    test_params = {
          'dim': dim,
          'batch_size': 1,
          'n_classes': N_CLASSES,
          'n_channels': None,
          'shuffle': False,
          'data_path': '../data_'+model_name+'_test_'+str(aug_i)+'/',
          'no_labels': True
        }

    test_ids = np.load('../data_'+model_name+'/meta/test_ids_0.npy')
    return DataGenerator(test_ids, None, **test_params)

def predict_proba_for_batch(model_path='densenet_top_4000_1000_5.h5', aug_i=0):
    model = keras.models.load_model(model_path)
    generator = get_test_datagen(aug_i)
    return generator.filenames, model.predict_generator(generator, len(generator.filenames))

if __name__ == '__main__':

    if(len(sys.argv) < 2):
        aug_n = 1
    else:
        aug_n = sys.argv[1]
    # model_name = sys.argv[1]
    # data_path = sys.argv[2]
    # mode = sys.argv[3]
    
    for i in range(aug_n):
        extract_and_save_features(aug_i=i)
    
    # filenames, y_pred_proba = predict_proba_for_batch(aug_i=0)
    # np.save('filenames', filenames)
    # np.save('y_pred_proba', y_pred_proba)

