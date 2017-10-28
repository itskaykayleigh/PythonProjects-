 
This repo is for a cuisine classification project.

The project objective is using information scraped/downloaded from the web along with supervised learning techniques to gain insights from classifying cuisines based on ingredients. A flask app is also included in the repo to showcase the findings.

Data:
 * Acquisition: flatfile download  
 * Source: Kaggle 
 * Data: 39,774 recipes from 20 countries (in total 90,000+ ingredients) 
 * Storage: PostgreSQL 
 
Key skills: 
 * Basics of the web (requests, HTML, CSS) 
 * Supervised learning
 * SQL
 * Flask
 * Javascript  

Analysis: 
 * SQL  
 * Flask
 * Supervised machine learning
	* Logistic Regression 
	* Bernoulli Naive Bayes 
 	* Linear Support Vector Classifier 
	* Random Forest Classifier 

Results & Findings: 
 * Improved model performance by 50% (accuracy) 
 * The model works the best for well-represented cuisines such as Italian, Chinese, Indian and Mexican, while less so for underrepresented ones like Spanish, Russian or British
 * If given sufficient recipes, the model should be able to improve performance for the later classes as well 
 * Developed a flask app that demonstrate how the model works and can be applied
 * Potential application include auto-categorizing restaurants based on restaurant's cuisine specialization and building food recommendation system based on ingredients similarities 
----------------------------------------------Table of Contents------------------------------------------------
The repo contains five sections - Code, Data, Visuals, Flask App and Presentation. 

Code:
 * Data Cleaning (2 rounds of cleaning -> 2 rounds of modeling)
 * Modeling (2 rounds of cleaning -> 2 rounds of modeling)

Data: 
 * Pickle files (cleaned / samples)

Images:
 * EDA / Model Performance charts

Presentation:
 * Presentation slides (pdf) 
 

