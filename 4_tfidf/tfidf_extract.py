import numpy as np
import tensorflow as tf
import random
import pickle

######################################## TF-IDF ############################################
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict

class TFIDF:
    def __init__(self):
        #load files
        a = open('./data/type_dict_khkim.pickle', 'rb')
        real_data = pickle.load(a)

        a = open('./data/pk_type2idx.pkl', 'rb')
        self.type_dict = pickle.load(a)

        # 각 type 별로 모든 문장을 하나의 리스트로 저장.
        self.real_data_by_type = [] 
        for key in real_data.keys():
            #print(len(real_data[key])) #how many sentences in one type?
            self.real_data_by_type.append([word[0] for sent in real_data[key] for word in sent if word != 'UNK'])

        dataset = [' '.join(type_sent) for type_sent in self.real_data_by_type] #속성 별 전체 문장, 인덱스 18개
        self.tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 1), min_df=1, sublinear_tf=True)
        
        # tfidfVectorizer 입력으로 전체 문장 데이터 (18개 속성 모두)가 들어감 (dataset)
        self.tfidf = self.tf.fit_transform(dataset) # 학습한 tfidfVectorizer tf를 기준으로 dataset 데이터의 tfidf 계산
        self.tfidf_matrix = np.array(self.tfidf.toarray())
        #tfidf_max = np.max(tfidf_matrix, axis=1) # max tfidf 구하는 코드
        self.tfidf_max=[]
        self.max_idx = [] # max tfidf 저장 변수
        self.feature_names = self.tf.get_feature_names() # feature extraction

    def keyword(self, idx):
        search= np.sort(self.tfidf_matrix[idx])[::-1] #sort in descend order
        for i in range (10): #10개 keyword
            self.tfidf_max.append(search[i])

        for i in range (len(self.tfidf_max)):
            self.max_idx.append(np.where(self.tfidf_matrix[idx] == self.tfidf_max[i])) #append keywords(max 10) for each type
        
        mylist=[]
        for i in range(len(self.max_idx)):
            for j in range(len(self.max_idx[i][0])):
                if (self.max_idx[i][0][j] not in mylist):
                        mylist.append(self.max_idx[i][0][j])
        
        words=[]
        feature_idx= np.array(mylist) #np.ndarray of feature idx
        for i in range(len(feature_idx)):
            words.append(self.feature_names[feature_idx[i]]) #append keywords(max 10) for each type
        
        return words