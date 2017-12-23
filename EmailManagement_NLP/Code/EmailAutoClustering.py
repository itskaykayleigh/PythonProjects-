import pandas 
import numpy as np 
import nltk
from nltk.stem import porter
from nltk.stem import WordNetLemmatizer

from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer 

from sklearn.decomposition import NMF
from sklearn.preprocessing import Normalizer 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS as stopwords


class EmailClustering():
    """ Take in a corpus of emails (accepts pandas series), 
    preprocesses (cleans & tokenizes) and vectorizes them, and 
    conducts topic modeling to cluster each email into its 
    corresponding topic.
    
    OUT: DataFrame containing each email and its topic. 
    """
    
    def __init__(self, docs):
        self.docs = docs
        self.tokenizer = self.nltk_tokenizer
        self.count_vectorizer(self.docs, self.tokenizer)
        self.nmf_algo(self.norm_data, self.feature_names)
        

    def nltk_tokenizer(self, doc):
        """ Takes in a corpus of documents, cleans, and tokenizes:
        1. Remove numbers, punctuations and special charaters
        2. Tokenize into words using wordpunct
        3. Lemmatize
        4. Stem and lowercase
        5. Remove stop words

        OUT: cleaned text (tokens).
        """
        lemmatizer = WordNetLemmatizer()
        tokenizer = RegexpTokenizer(r'\w+')
        stop = nltk.corpus.stopwords.words('english')
        stop += ['.',',','(', ')',"'",'"']
        stop += ['_____________________________________________',
                 '__________________________________________________']
        stop += ['_________________________________________________________________']
        stop = set(stop) 
        
        doc = doc.lower()
        doc = doc.translate(str.maketrans('','','1234567890')) 
        tokens = tokenizer.tokenize(doc)
        tokens = [lemmatizer.lemmatize(i) for i in tokens]
        tokens = [i for i in tokens if i not in stop]
        tokens = [i for i in tokens if len(i)>3]
        
        return tokens 
             
    def count_vectorizer(self, docs, tokenizer, min_n=1, max_n=2, max_features=5000, max_df=0.6):
        """ This function takes in cleaned tokens and 
        returns vectorized and normalized data using count vectorizer.
        """

        self.vectorizer = CountVectorizer(tokenizer=tokenizer,
                                     ngram_range=(min_n,max_n),
                                     max_features=max_features,
                                     max_df=max_df) 
        
        self.vect_data = self.vectorizer.fit_transform(self.docs)
        self.norm_data = Normalizer().fit_transform(self.vect_data)
        self.feature_names = self.vectorizer.get_feature_names()
        
        
    def nmf_algo(self, norm_data, feature_names, n_comp=4, random_state=7777, no_top_words=20, topic_names=None):
        """ Returns a 1). NMF model and 2). transformed data
        given the parameters specified by user.
        """
        
        self.nmf = NMF(n_components=n_comp)
        
        self.nmf_data = self.nmf.fit_transform(self.norm_data)
        
        for ix, topic in enumerate(self.nmf.components_):
            if not topic_names or not topic_names[ix]:
                print("\nTopic ", ix)
            else:
                print("\nTopic: '",topic_names[ix],"'")
            print(", ".join([self.feature_names[i]
                            for i in topic.argsort()[:-no_top_words - 1:-1]]))
            
    def create_topic_space(self):
        """ Returns a pandas dataframe with emails as row and topics & content as column """
        self.df_topic = pandas.DataFrame(self.nmf_data)
        self.df_topic['topics'] = self.df_topic.idxmax(axis=1)
        df_topics = pandas.get_dummies(self.df_topic['topics'])
        self.df_topic = pandas.concat([self.df_topic, df_topics, self.docs.to_frame()], axis=1)
        self.df_topic.columns = ['t0_vec', 't1_vec','t2_vec','t3_vec', 'topics',
                                 'Email_Bucket_1', 'Email_Bucket_2','Email_Bucket_3',
                                 'Email_Bucket_4','Email']
        self.df_topic = self.df_topic.drop(['t0_vec', 't1_vec','t2_vec','t3_vec','topics'], axis=1)
        
        return self.df_topic
    
#         email_topic = defaultdict(list)

#         for topic, email in zip(df_topic['topics'][66:88],df_topic['Email'][66:88]):
#             if topic==0:
#                 email_topic['Corporation_Related'].append(email)
#             elif topic == 1: 
#                 email_topic['Meeting_Call_Appointment'].append(email)
#             elif topic == 2: 
#                 email_topic['IT_Related'].append(email)
#             else: 
#                 email_topic['Industry_Business_Market'].append(email)





