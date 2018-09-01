# from keras.applications.densenet import DenseNet201, DenseNet121
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.layers import Dense
from keras import layers, optimizers, models
import keras
from keras import backend as K
# from keras.applications.densenet import preprocess_input
import time

from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input
import matplotlib.pyplot as plt

from keras.models import load_model
from keras.applications import imagenet_utils
from keras.applications.imagenet_utils import _preprocess_symbolic_input

import argparse
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--load', dest='load', type=bool, default=False,
                    help='set to true if you wish to load pretrained model')

args = parser.parse_args()
load = args.load
print("load is", load)

target_size = (224,224)
validation_dir = './data/valid'
train_dir = './data/train'

# datagen = image.ImageDataGenerator(
#     rotation_range=20,
#     width_shift_range=0.2,
#     height_shift_range=0.2,
#     horizontal_flip=True)

# test_datagen = image.ImageDataGenerator(rescale=1./255)

# train_generator = datagen.flow_from_directory(
#         'data/train', # Change to train when actually training
#         target_size=target_size,
#         batch_size=32,
#         class_mode='categorical')

# validation_generator = datagen.flow_from_directory(
#         'data/valid',
#         target_size=target_size,
#         batch_size=32,
#         class_mode='categorical')

train_datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True
      )
 
validation_datagen = ImageDataGenerator()
 
# Change the batchsize according to your system RAM
train_batchsize = 200
val_batchsize = 10
 
train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=target_size,
        batch_size=train_batchsize,
        class_mode='categorical')
 
validation_generator = validation_datagen.flow_from_directory(
        validation_dir,
        target_size=target_size,
        batch_size=val_batchsize,
        class_mode='categorical',
        shuffle=False)

# if not load:
# cnn = DenseNet121(weights='imagenet', include_top=False)
cnn = ResNet50(weights='imagenet', include_top=False)

for layer in cnn.layers:
    layer.trainable = False

for layer in cnn.layers:
    print(layer.name, layer.trainable)

# create the base pre-trained model
model = keras.models.Sequential()
# model.add(keras.layers.Lambda(preprocess_input, name='preprocessing', input_shape=(224, 224, 3)))
model.add(keras.layers.InputLayer(input_shape=(224,224,3)))
model.add(cnn)
model.add(layers.Flatten(name='flatten'))
model.add(layers.Dense(1024, activation='relu', name='hidden_1024'))
# model.add(layers.Dropout(0.5))
model.add(layers.Dense(128, activation='softmax', name='classification'))

# if load:
#     model.load_weights('learnopencv5_checkpoint.h5', by_name=True)

# for layer in model.layers[1].layers[-22:]:
#     layer.trainable = True
#     print(layer.name, layer.trainable)

# Show a summary of the model. Check the number of trainable parameters
for layer in model.layers:
    print(layer, layer.trainable)

for layer in model.layers[1].layers:
    print(layer, layer.trainable)

model.layers[1].summary()
model.summary()

# optim = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
# optim = keras.optimizers.RMSprop(lr=0.001)
optim = keras.optimizers.SGD(lr=1e-4, momentum=0.9, nesterov=True)

# Compile the model
model.compile(loss='categorical_crossentropy',
            optimizer=optim,
            metrics=['acc'])

# # else:
# model = load_model('learnopencv1_checkpoint.h5', custom_objects={'imagenet_utils': imagenet_utils, '_preprocess_symbolic_input': _preprocess_symbolic_input})


# Train the model
history = model.fit_generator(
        train_generator,
        #   steps_per_epoch=train_generator.samples/train_generator.batch_size ,
        steps_per_epoch=200,
        epochs=30,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples/validation_generator.batch_size/2,
        verbose=1
        # callbacks=[keras.callbacks.ModelCheckpoint('learnopencv6_ft_checkpoint.h5', monitor='val_loss', verbose=2, save_best_only=True, mode='auto', period=1)]
    )

# Save the model
model.save('learnopencv.h5')

# acc = history.history['acc']
# val_acc = history.history['val_acc']
# loss = history.history['loss']
# val_loss = history.history['val_loss']

# epochs = range(len(acc))

# plt.plot(epochs, acc, 'b', label='Training acc')
# plt.plot(epochs, val_acc, 'r', label='Validation acc')
# plt.title('Training and validation accuracy')
# plt.legend()

# plt.figure()

# plt.plot(epochs, loss, 'b', label='Training loss')
# plt.plot(epochs, val_loss, 'r', label='Validation loss')
# plt.title('Training and validation loss')
# plt.legend()

# # plt.show()
# plt.savefig('plots.png', bbox_inches='tight')

###

# # compile the model (should be done *after* setting layers to non-trainable)
# model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

# # train the model on the new data for a few epochs
# model.fit_generator(
#         train_generator,
#         steps_per_epoch=50,
#         epochs=100,
#         validation_data=validation_generator,
#         validation_steps=10,
#         callbacks=[keras.callbacks.ModelCheckpoint('model2.h5', monitor='val_loss', verbose=0, save_best_only=True, mode='auto', period=1)])

# model.save('final_model2.h5')
# # at this point, the top layers are well trained and we can start fine-tuning
# # convolutional layers from inception V3. We will freeze the bottom N layers
# # and train the remaining top layers.

# # let's visualize layer names and layer indices to see how many layers
# # we should freeze:
# for i, layer in enumerate(base_model.layers):
#    print(i, layer.name)

# # we chose to train the top 2 inception blocks, i.e. we will freeze
# # the first 249 layers and unfreeze the rest:
# for layer in model.layers[:249]:
#    layer.trainable = False
# for layer in model.layers[249:]:
#    layer.trainable = True

# # we need to recompile the model for these modifications to take effect
# # we use SGD with a low learning rate
# from keras.optimizers import SGD
# model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy')

# # we train our model again (this time fine-tuning the top 2 inception blocks
# # alongside the top Dense layers
# model.fit_generator(...)