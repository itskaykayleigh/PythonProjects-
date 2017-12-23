import os
import sys
import glob
import h5py
import argparse
import matplotlib.pyplot as plt

from keras import __version__
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.models import Model
from keras.preprocessing import image
from keras.models import model_from_json
from keras.layers import Dense, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.training_utils import multi_gpu_model
from keras.optimizers import SGD


# Implementation
IM_WIDTH, IM_HEIGHT = 299, 299 #fixed size for InceptionV3
NB_EPOCHS = 20
BAT_SIZE = 256
FC_SIZE = 1024
NB_IV3_LAYERS_TO_FREEZE = 172
NB_CLASSES = 10
NB_TRAIN_SAMPLES = 71195
NB_VAL_SAMPLES = 18914

train_dir = '../../data/deepfashion/deepfashion_train/'
test_dir = '../../data/deepfashion/deepfashion_test/'

train_datagen = ImageDataGenerator(
   preprocessing_function=preprocess_input
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(
   preprocessing_function=preprocess_input
)

train_generator = train_datagen.flow_from_directory(
 train_dir,
 target_size=(IM_WIDTH, IM_HEIGHT),
 batch_size=BAT_SIZE,
)

validation_generator = test_datagen.flow_from_directory(
 test_dir,
 target_size=(IM_WIDTH, IM_HEIGHT),
 batch_size=BAT_SIZE,
)

# Instantiate the base pre-trained model 
# - the InceptionV3 network 
base_model = InceptionV3(weights='imagenet', include_top=False)


## Add and initialize a new last layer
def add_new_last_layer(base_model, nb_classes, FC_SIZE1, FC_SIZE2, FC_SIZE3):
    """Add last layer to the convnet
    Args:
    base_model: keras model excluding top
    nb_classes: # of classes
    Returns:
    new keras model with last layer
    """

    x = base_model.output
    
    # converts the (rows, cols, channels) tensor output into a (batch_size, channels) tensor 
    x = GlobalAveragePooling2D()(x) 
           
    # adds on a fully-connected Dense layer with 1024 hidden units 
    x = Dense(FC_SIZE, activation='relu')(x) 
    
    # adds a softmax function on the output to squeeze the values between [0,1]
    predictions = Dense(nb_classes, activation='softmax')(x) 
        
    model = Model(input=base_model.input, output=predictions)
    
    return model

 # Activate Multiple GPUs with Data parallelism
 # parallel_model = multi_gpu_model(model, gpus=8)

### Transfer learning
def transfer_learning(model, base_model):
	for layer in base_model.layers:
	    layer.trainable = False

	model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

### Fine Tune
def setup_to_finetune(model):
    """Freeze the bottom NB_IV3_LAYERS and retrain the remaining top layers.
    note: NB_IV3_LAYERS corresponds to the top 2 inception blocks in 
    the inceptionv3 architecture
    Args:
    model: keras model
    """
    for layer in model.layers[:NB_IV3_LAYERS_TO_FREEZE]:
        layer.trainable = False
    for layer in model.layers[NB_IV3_LAYERS_TO_FREEZE:]:
        layer.trainable = True
        
    model.compile(optimizer=SGD(lr=0.0001, momentum=0.9),   
                 loss='binary_crossentropy')


## Training
history = parallel_model.fit_generator(
 train_generator,
 samples_per_epoch=NB_TRAIN_SAMPLES,
 nb_epoch=NB_EPOCHS,
 validation_data=validation_generator,
 nb_val_samples=NB_VAL_SAMPLES)

Save the model into a json file 
model_json = model.to_json()
with open("../Data/incep_filter2.json", 'w') as json_file:
   json_file.write(model_json)

Save model weights 
model.save_weights('../Data/incep_weights2.h5')

### Visuals - Plot results 
def plot_training(history):
    """ Plots the training accuracies and loss using the history object. """
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(acc))

    plt.plot(epochs, acc, 'r.')
    plt.plot(epochs, val_acc, 'r')
    plt.title('Training and validation accuracy')

    plt.figure()
    plt.plot(epochs, loss, 'r.')
    plt.plot(epochs, val_loss, 'r-')
    plt.title('Training and validation loss')
    plt.show()
    
    
## Load model & weights     
json_file = open('../Data/incep_filter2.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

loaded_model.load_weights('../Data/incep_weights2.h5', by_name=False)


def predict(model, img, target_size, labels):
    """Run model prediction on image
    Args:
        model: keras model
        img: PIL format image
        target_size: (w,h) tuple
    Returns:
        list of predicted labels and their probabilities
    """
    
    if img.size != target_size:
        img = img.resize(target_size)
    
    # converts a PIL format image to a numpy array
    x = image.img_to_array(img)
    
    # converts our (3, 299, 299) size image to (1, 3, 299, 299)
    # model.predict function requires a 4 dimensional array as input,
    # where the 4th dimension corresponds to the batch size
    x = np.expand_dims(x, axis=0)
    
    # data normalization 
    # zero-centers image data using the mean channel values from the training dataset
    # extremely important step
    # if skipped, will cause all the predicted probabilities to be incorrect
    x = preprocess_input(x)
    
    # runs inference on data batch and returns predictions
    preds = model.predict(x)
    
    index, value = max(enumerate(preds[0]), key=operator.itemgetter(1))

    return preds[0], labels[index]


def plot_preds(image, preds, labels):  
    """Displays image and the top-n predicted probabilities in a bar graph.  
    Args:    
    image: PIL image
    preds: list of predicted labels and their probabilities  
    """  
    
    #image
    plt.imshow(image)
    plt.axis('off')

    #bar graph
    plt.figure()  
    order = list(range(len(preds)))
    plt.barh(labels, preds, alpha=0.5)
    plt.yticks(order, labels)
    plt.xlabel('Probability')
    plt.xlim(0, 1.01)
    plt.tight_layout()
    plt.show()

