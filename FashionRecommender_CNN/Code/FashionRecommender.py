import os 
import sys
import glob
import h5py
import pickle 
import requests
import operator
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image
from io import BytesIO
from keras.models import Model
import matplotlib.pyplot as plt
from keras.optimizers import SGD
from keras.preprocessing import image
from keras.models import model_from_json
from keras.layers import Dense, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.inception_v3 import InceptionV3, preprocess_input

from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA


def img_loader(image):
    """Read raw image into PIL format image.
    Args: 
        image: wild image from local file or online
    Returns: 
        image: PIL format image
    """
    if 'https' in image or 'http' in image: #online
        response = requests.get(image)
        img = Image.open(BytesIO(response.content))
        
    else: #local file
        img = Image.open(image)
        
    return img 


def img_to_vector(model, img, layer_name):
    """Return the FC hidden layer output
    Args:
        model: keras model
        img: PIL format image
        target_size: (w,h) tuple
    Returns:
        outputs of 1024 hidden units for each image - shape (1, 1024)
    """
    target_size = (299, 299) #fixed size for InceptionV3 architecture
        
    if img.size != target_size:
        img = img.resize(target_size)
    
    # converts a PIL format image to a numpy array
    x = image.img_to_array(img)
    
    # converts the (3, 299, 299) size image to (1, 3, 299, 299)
    x = np.expand_dims(x, axis=0)
    
    # data normalization 
    x = preprocess_input(x)
    
    # create a new Model that will output the layers 
    intermediate_layer_model = Model(inputs=model.input, 
                                outputs=model.get_layer(layer_name).output)
    
    # predict on the intermediate model for output 
    intermediate_output = intermediate_layer_model.predict(x)
    
    return intermediate_output


def intermediate_output(layer_name, model, generator, BAT_SIZE):
    """Return the FC hidden layer output
    Args:
        model: keras model
        layer_name: layer to extract output from 
        generator: image generator to generate batch images 
        BAT_SIZE: batch size for image generator 
    Returns:
        ouput in shape (number of images, layer hidden nodes size)
    """
    intermediate_layer_model = Model(inputs=model.input,
                                     outputs=model.get_layer(layer_name).output)

    intermediate_output = intermediate_layer_model.predict_generator(generator, 
                                                                     steps=BAT_SIZE)
    return intermediate_output


def optimal_pca_comp(training_vecotrs, n_comp):
    pca = PCA(n_components=n_comp, svd_solver='auto', random_state=7777)  
    pca_data = pca.fit_transform(training_vecotrs)  

    # find the optimal 
    #sns.set()
    sns.set_style('whitegrid')
    sns.set_context('talk')
    fig0 = plt.figure(figsize=(15,8));
    sns.barplot(y=pca.explained_variance_ratio_, 
                x=np.arange(1,n_comp+1), 
                data=None,
                color='deeppink', 
                saturation=.5)
    plt.plot(np.cumsum(pca.explained_variance_ratio_),'m',alpha=.5)
    plt.xlabel('Number of Components', fontsize=16)
    plt.ylabel('Ratio of Variance', fontsize=16);
    plt.title('Explained Variance Training', fontsize=16)
    

### Build recommender system 
def get_recommendations(new_img, model, training_vectors, layer_name):
    """ Recommends top 3 pieces of clothing that are most similar to the wild image,
    measured by cosine distance with feature extraction employing PCA.
    Args: 
        new_img: wild raw image 
        training_vectors: intermediate output from the pre-trained CNN model
    Returns:
        recommended list
    """
    
    img = img_loader(new_img)
    new_vec = img_to_vector(model, img, layer_name)

    # compute similarities between images using cosine distance 
    nn = NearestNeighbors(n_neighbors=3, 
                          metric='cosine',
                          algorithm='brute')
    nn.fit(training_vectors)
    results = nn.kneighbors(new_vec)
    
    return results[1][0]


def get_recommendations_pca(new_img, model, training_vectors, n_comp, n_neighbors, layer_name):
    """ Recommends top 3 pieces of clothing that are most similar to the wild image,
    measured by cosine distance with feature extraction employing PCA.
    """
    img = img_loader(new_img)
    vec = img_to_vector(model, img, layer_name)
    new_vectors = np.concatenate((training_vectors, vec), axis=0)
   
    pca = PCA(n_components=n_comp, svd_solver='auto', random_state=7777)  
    pca_data = pca.fit_transform(new_vectors)
    
    nn = NearestNeighbors(n_neighbors=n_neighbors,
                          metric='cosine',
                          algorithm='brute')
    nn.fit(pca_data[:-1])
    results = nn.kneighbors(pca_data[-1].reshape(1, -1))
    
    return results[1][0]

def visualize_recommendations(new_img, recommend_list, dir_path):
    """ Visualize clothing in recommended list. """
    
    clothing = os.listdir(dir_path)
    
    if 'http' in new_img or 'https' in new_img:
        list_im = []
        for clothing_index in recommend_list:
            clothing_path = dir_path + clothing[clothing_index]
            list_im.append(clothing_path)
        list_im.append(new_img)
          
    else: 
        list_im = [new_img]
        for clothing_index in recommend_list:
            clothing_path = dir_path + clothing[clothing_index]
            list_im.append(clothing_path)
    fig = plt.figure(figsize = (21, 14))
    
    for i, sample in enumerate(list_im):
        img = img_loader(sample)  
        fig.add_subplot(2, 2, i+1)
        plt.imshow(img)
        plt.tight_layout()
        plt.axis('off');