from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.models import load_model

import numpy as np

from keras import backend as K
model = load_model('model.h5')
get_features_layer_output = K.function([model.layers[0].input],
                                  [model.layers[-3].output])


def get_features(file, model, multiply):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    layer_output = get_features_layer_output([x])[0]
    return layer_output.reshape(2048)