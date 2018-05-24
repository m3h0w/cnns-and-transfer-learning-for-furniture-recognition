from keras.applications.densenet import DenseNet201, DenseNet121
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import layers, optimizers, models
import keras
from keras import backend as K
from keras.applications.densenet import preprocess_input, decode_predictions

import matplotlib.pyplot as plt

target_size = (224,224)
validation_dir = 'report2/data/valid'
train_dir = 'report2/data/train'

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
      rotation_range=20,
      width_shift_range=0.2,
      height_shift_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest')
 
validation_datagen = ImageDataGenerator()
 
# Change the batchsize according to your system RAM
train_batchsize = 100
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

dense_net = DenseNet121(weights='imagenet', include_top=False)

for layer in dense_net.layers:
    layer.trainable = False

for layer in dense_net.layers:
    print(layer, layer.trainable)

# create the base pre-trained model
model = keras.models.Sequential()
model.add(keras.layers.Lambda(preprocess_input, name='preprocessing', input_shape=(224, 224, 3)))

# add a global spatial average pooling layer
model.add(dense_net)
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
# model.add(layers.Dropout(0.5))
model.add(layers.Dense(128, activation='softmax'))

# Show a summary of the model. Check the number of trainable parameters
for layer in model.layers:
    print(layer, layer.trainable)
model.summary()

# optim = optimizers.RMSprop(lr=1e-4)
optim = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

# Compile the model
model.compile(loss='categorical_crossentropy',
              optimizer=optim,
              metrics=['acc'])
# Train the model
history = model.fit_generator(
      train_generator,
    #   steps_per_epoch=train_generator.samples/train_generator.batch_size ,
      steps_per_epoch=3,
      epochs=30,
      validation_data=validation_generator,
      validation_steps=validation_generator.samples/validation_generator.batch_size,
      verbose=1,
      callbacks=[keras.callbacks.ModelCheckpoint('learnopencv_checkpoint.h5', monitor='val_loss', verbose=0, save_best_only=True, mode='auto', period=1)])
 
# Save the model
model.save('learnopencv1.h5')

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
 
epochs = range(len(acc))
 
plt.plot(epochs, acc, 'b', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
 
plt.figure()
 
plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
 
plt.show()

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