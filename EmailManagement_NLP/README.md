This repo is for a project on building a personalized email management system which automatically groups emails into different buckets based on email content.  

To read more about this project, click the link: https://www.kayleighli.space/natural-language-processing-personalized-email-management/ 

The objective of the product is to help people work more efficiently and effectively by making it possible to prioritize emails that matter the most to users. 

**Data:**
 * Source: https://www.cs.cmu.edu/~enron/
 * Data: Enron Email Dataset 
 * Storage: MongoDB

**Key skills:**
 * Unsupervised Learning 
	* Clustering 
		* K-Means 
	* Dimensionality Reduction & Topic Modeling
		* LDA 
		* PCA (LSA, SVD)
		* NMF 

 * Natural Language Processing 
	* NLTK 
	* SpaCy
	* TextBlob 
	* Count Vectorizer 
	* TF-IDF Vectorizer 

**Key Tools/Packages:**
* Pandas, Numpy, Matplotlib, Seaborn   
* NLTK, SpaCy, TextBlob   
* Scikit-Learn  
	* TSNE, KMeans, Normalizer, CountVectorizer, LatentDirichletAllocation, NMF, PCA, TruncatedSVD

**Results & Findings:**
If given some emails, the automated pipeline will auto-categorize the emails into 2-4 groups based on email content. Each set of emails will end up with different groups. For instance, a personal email account will differ from a work email.

