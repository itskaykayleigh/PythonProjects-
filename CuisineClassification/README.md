 
This repo is for an ingredient-based cuisine classification project.

The project objective is using information scraped/downloaded from the web along with supervised machine learning techniques to gain insights from classifying cuisines based on ingredients. A flask app is also included in the repo to showcase the findings.

** Data:** 
 * Acquisition: flatfile download  
 * Source: Kaggle 
 * Data: 39,774 recipes from 20 countries (in total 90,000+ ingredients) 
 * Storage: PostgreSQL 
 
** Key skills:** 
 * Basics of the web (requests, HTML, CSS) 
 * Supervised machine learning 
 * SQL
 * Flask
 * Javascript  
 * PostgreSQL 
 * AWS 
	
**Machine Learning Problem:** 
Supervised - Classification   
Unsupervised - NPL 

**Analysis:** 
 * Supervised machine learning
	* Logistic Regression 
	* Bernoulli Naive Bayes 
 	* Linear Support Vector Classifier 
	* Random Forest Classifier 
	* K-Nearest Neighbors
	* Support Vector Machine
	
**Results & Findings:**  
 * Improved model performance by 50% (accuracy) 
 * The model works the best for well-represented cuisines such as Italian, Chinese, Indian and Mexican, while less so for underrepresented ones like Spanish, Russian or British
 * If given sufficient recipes, the model should be able to improve performance for the later classes as well 
 * Developed a flask app that demonstrate how the model works and can be applied
 * Potential application include auto-categorizing restaurants based on restaurant's cuisine specialization and building food recommendation system based on ingredients similarities 
