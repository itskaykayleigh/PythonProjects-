This project is about an image-based fashion recommender that can be served as a smarter search engine.

To read more about this project, click the link below:   
https://www.kayleighli.space/convolutional-neural-network-imagebased-fashion-recommender/   
To see a demo of the fashion recommender, click the link below:   
https://www.youtube.com/watch?v=HsqLlDyTUu8

The objective of the recommender is to:   

	1). Train a model to recognize clothing from an image, and   
	2). Build a recommender that can recommend clothing based on images that share similar fashion styles   

**Data:**   
 * Source: http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion.html
 * Data: DeepFashion 
 * Storage: AWS 

**Key skills:**    
 * Image processing 
 	* Image recognition 
	* Image similarity comparison 
 * Deep learning 
 	* Convolutional Neural Network
	* Image2Vec ("Style2Vec") 
	* Transfer Learning & Fine Tuning 

**Key Tools/Packages:**   
* Python
* AWS (Used GPUs for model training) 
* Keras & TensorFlow (*PyTorch for 1st round of modeling) 
* Nearest Neighbors 
* PCA 

**Machine Learnring problem:**   
Image detection     
Image-based recommender system

**Results & Findings:**   
The model is able to classify fashion images into 10 different fashion classes, such as coat, jacket, pants, dress, skirt and etc, with an accuracy of 95%. Given an image of any style in any background, the fashion recommender was able to return 3 images of clothing in simiar styles as recommendations. 
