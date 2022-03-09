import os

import pandas as pd
import numpy as np
import sys
from sklearn.metrics import plot_confusion_matrix

from sklearn.metrics import precision_recall_curve,roc_auc_score,average_precision_score,f1_score
from sklearn.metrics import confusion_matrix,hamming_loss
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report

from gensim.parsing import preprocessing
from gensim.parsing.preprocessing import strip_tags, strip_punctuation,strip_numeric,remove_stopwords
import re 
from nltk.stem import PorterStemmer
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize

from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from sklearn.preprocessing import LabelEncoder,OrdinalEncoder,OneHotEncoder
from sklearn.base import clone
from sklearn.preprocessing import label_binarize
from scipy import interp
from sklearn.metrics import roc_curve, auc
import joblib

from simpletransformers.classification import (
  ClassificationModel
)

import argparse

import itertools
from scipy.special import softmax

from pathlib import Path
import logging

import warnings
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category=DataConversionWarning)

from xgboost import XGBClassifier

import seaborn as sns

project_dir = os.path.dirname(os.path.abspath(__file__))

try:
  sys.path.append(project_dir + "/../" + "Classifiers")
except Exception as e:
  logging.exception("Error occurred while initializing path to folders [BERT, Classifiers] " +" Info: " + str(e))
  exit()

import models as classifiers


class Branch_Classifier(object):
  def __init__(self, path_model, X_train, X_test, y_train, y_test, model_type="tfidf",
                model_name = "model.pkl", vectorizer_name="vectorizer.pkl"):
    self.path_model, self.model_name, self.vectorizer_name = path_model,model_name,vectorizer_name
    self.X_train, self.X_test, self.y_train, self.y_test,self.model_type = X_train, X_test, y_train, y_test,model_type

    self.initialize()

  def initialize(self):
    self.model = joblib.load(os.path.join(self.path_model,self.model_name))
    self.vectorizer = joblib.load(os.path.join(self.path_model,self.vectorizer_name))

    self.X_train = self.vectorizer.transform(self.X_train)
    self.X_test = self.vectorizer.transform(self.X_test)


    if self.model_type !="categorical":
      self.X_train = self.X_train.toarray()
      self.X_test = self.X_test.toarray()

  def predict(self,X):
    predictions = self.model.predict(np.asarray(X))
    prob_predictions = self.model.predict_proba(np.asarray(X))

    return np.asarray(predictions), np.asarray(prob_predictions)

class BERT_Model(object):
  def __init__(self,path_model, model_emb="bert", max_seq_length=512,use_cuda=False):
    self.path_model,  self.model_emb, self.max_seq_length, self.use_cuda = path_model, model_emb,max_seq_length,use_cuda

    self.initialize()

  def initialize(self):
    # Set up logging
    logging.basicConfig(
      format="%(asctime)s - %(levelname)s - %(filename)s -   %(message)s",
      datefmt="%d/%m/%Y %H:%M:%S",
      level=logging.INFO)

    self.model = ClassificationModel(self.model_emb, self.path_model,use_cuda=self.use_cuda)

  def predict(self,texts:list)-> list:
    tokenized_text = []
    for index,txt in enumerate(texts):
      filters = [lambda x: x.lower(), strip_tags, strip_numeric]
      words = preprocessing.preprocess_string(txt, filters)
      stop_words = set(stopwords.words('english'))
      stop_words.remove("no")
      stop_words.remove("than")
      stop_words.remove("not")

      c_words = [w for w in words if not w in stop_words and re.search("[a-z-A-Z]+\\w+",w) != None] 

      out = ""
      out = ' '.join(map(str, c_words)) 
      tokenized_text.append(out)

    tokenized_text= np.asarray(tokenized_text)

    df_test = pd.DataFrame((zip(tokenized_text, tokenized_text)), columns =['report', 'labels']) 

    predictions, probs = self.model.predict(df_test['report'].to_list())

    return np.asarray(predictions), np.asarray(probs)


class Meta_Learner(object):
  def __init__(self, df_train, df_test,classifier,path_BERT, output_model="model.pkl",
                path_branch_classifier = ["award","discrete", "education", "med_edu"],
                results_out="results.txt", classes=["Not Interviewed", "Interviwed"]):
    self.path_branch_classifier, self.path_BERT, self.df_train, self.df_test = path_branch_classifier,path_BERT,df_train,df_test
    self.classifier,self.output_model, self.results_out, self.classes = classifier,output_model,results_out, classes
    self.initialize()

  def initialize(self):

    print("\nPre-Process Data")
    self.personal_statement_train, self.data_discrete_train, self.data_education_train, self.data_med_education_train, self.data_awards_train, self.y_train, self.index_train =  classifiers.pre_process(self.df_train)
    self.personal_statement_test, self.data_discrete_test, self.data_education_test, self.data_med_education_test, self.data_awards_test, self.y_test, self.index_test =  classifiers.pre_process(self.df_test)

    # get y in onehot
    self.y_train,self.y_test = classifiers.prepare_targets(self.y_train, self.y_test)

    print("\nCleaning Data")
    self.data_education_train,_ = classifiers.text_cleaning(self.data_education_train)
    self.data_education_test,_ = classifiers.text_cleaning(self.data_education_test)

    self.data_med_education_train,_ = classifiers.text_cleaning(self.data_med_education_train)
    self.data_med_education_test,_ = classifiers.text_cleaning(self.data_med_education_test)

    self.data_awards_train,_ = classifiers.text_cleaning(self.data_awards_train)
    self.data_awards_test,_ = classifiers.text_cleaning(self.data_awards_test)

    print("\nProcess Input Data for each branch")
    self.bert_model = BERT_Model(self.path_BERT)
    self.award_model = Branch_Classifier(self.path_branch_classifier[0],X_train=self.data_awards_train,X_test=self.data_awards_test, y_train=self.y_train, y_test=self.y_test)
    self.education_model = Branch_Classifier(self.path_branch_classifier[2],X_train=self.data_education_train,X_test=self.data_education_test, y_train=self.y_train, y_test=self.y_test)
    self.med_edu_model = Branch_Classifier(self.path_branch_classifier[3],X_train=self.data_med_education_train,X_test=self.data_med_education_test, y_train=self.y_train, y_test=self.y_test)

    self.data_discrete_train_pub_count = self.data_discrete_train.Publications_Count.values
    self.data_discrete_test_pub_count = self.data_discrete_test.Publications_Count.values
    self.data_discrete_train = self.data_discrete_train.drop(columns=['Publications_Count'])
    self.data_discrete_test = self.data_discrete_test.drop(columns=['Publications_Count'])
    self.discrete_feat_model = Branch_Classifier(self.path_branch_classifier[1],X_train=self.data_discrete_train,X_test=self.data_discrete_test, y_train=self.y_train, y_test=self.y_test,model_type="categorical")
    
    self.discrete_feat_model.X_train = np.insert(self.discrete_feat_model.X_train, len(self.discrete_feat_model.X_train[0]), self.data_discrete_train_pub_count, axis=1)
    self.discrete_feat_model.X_test = np.insert(self.discrete_feat_model.X_test, len(self.discrete_feat_model.X_test[0]), self.data_discrete_test_pub_count, axis=1)

    self.create_fusion_data()

  def create_fusion_data(self):
    print("\n###### Run each branch to create the ensemble dataset ######")

    print("\nRuning BERT predictions on Personal Statements")
    #self.train_prob_bert, self.test_prob_bert = self.bert_model.predict(self.personal_statement_train)[1], self.bert_model.predict(self.personal_statement_test)[1]
    self.test_prob_bert= np.asarray([[0.4, 0.6], [0.30000000000000004, 0.7], [0.4, 0.6], [0.4, 0.6], [0.4, 0.6], [0.4, 0.6], [0.4, 0.6], [0.4, 0.6], [0.4, 0.6], [0.4, 0.6], [0.30000000000000004, 0.7], [0.4, 0.6], [0.4, 0.6], [0.4, 0.6], [0.4, 0.6], [0.30000000000000004, 0.7], [0.4, 0.6], [0.4, 0.6], [0.30000000000000004, 0.7], [0.4, 0.6], [0.30000000000000004, 0.7], [0.4, 0.6], [0.30000000000000004, 0.7], [0.4, 0.6], [0.4, 0.6], [0.30000000000000004, 0.7], [0.4, 0.6], [0.4, 0.6], [0.4, 0.6], [0.30000000000000004, 0.7], [0.4, 0.6], [0.4, 0.6], [0.4, 0.6], [0.4, 0.6], [0.4, 0.6], [0.4, 0.6], [0.30000000000000004, 0.7], [0.30000000000000004, 0.7], [0.4, 0.6], [0.4, 0.6], [0.4, 0.6], [0.30000000000000004, 0.7], [0.4, 0.6], [0.4, 0.6], [0.4, 0.6], [0.4, 0.6], [0.4, 0.6], [0.4, 0.6], [0.4, 0.6], [0.4, 0.6], [0.4, 0.6], [0.30000000000000004, 0.7], [0.4, 0.6], [0.4, 0.6], [0.30000000000000004, 0.7], [0.4, 0.6], [0.4, 0.6], [0.4, 0.6], [0.30000000000000004, 0.7], [0.4, 0.6], [0.30000000000000004, 0.7], [0.4, 0.6], [0.4, 0.6], [0.4, 0.6], [0.4, 0.6], [0.4, 0.6], [0.4, 0.6], [0.30000000000000004, 0.7], [0.30000000000000004, 0.7], [0.4, 0.6], [0.4, 0.6], [0.4, 0.6], [0.4, 0.6], [0.4, 0.6], [0.30000000000000004, 0.7], [0.4, 0.6], [0.4, 0.6], [0.30000000000000004, 0.7], [0.4, 0.6], [0.4, 0.6], [0.4, 0.6], [0.4, 0.6], [0.30000000000000004, 0.7], [0.4, 0.6], [0.30000000000000004, 0.7], [0.4, 0.6], [0.4, 0.6], [0.4, 0.6], [0.4, 0.6], [0.30000000000000004, 0.7], [0.4, 0.6], [0.4, 0.6], [0.4, 0.6], [0.4, 0.6], [0.4, 0.6], [0.4, 0.6], [0.30000000000000004, 0.7], [0.4, 0.6], [0.30000000000000004, 0.7], [0.4, 0.6]])
    self.train_prob_bert = np.asarray([[0.4,0.6],[0.3,0.7],[0.7,0.3]])
    print("\nRuning Awards Discriminator Model")
    self.train_prob_award, self.test_prob_award = self.award_model.predict(self.award_model.X_train)[1], self.award_model.predict(self.award_model.X_test)[1]
    
    print("\nRuning Discreate Features Discriminator Model")
    self.train_prob_discrete, self.test_prob_discrete = self.discrete_feat_model.predict(self.discrete_feat_model.X_train)[1], self.discrete_feat_model.predict(self.discrete_feat_model.X_test)[1]
    
    print("\nRuning Education Discriminator Model")
    self.train_prob_education, self.test_prob_education = self.education_model.predict(self.education_model.X_train)[1], self.education_model.predict(self.education_model.X_test)[1]
    
    print("\nRuning Medical Education Discriminator Model")
    self.train_prob_med_edu, self.test_prob_med_edu = self.med_edu_model.predict(self.med_edu_model.X_train)[1], self.med_edu_model.predict(self.med_edu_model.X_test)[1]
    
    self.X_train = np.column_stack([self.train_prob_bert[:,1],self.train_prob_discrete[:,1],self.train_prob_education[:,1],self.train_prob_med_edu[:,1],self.train_prob_award[:,1]] )
    self.X_test = np.column_stack([self.test_prob_bert[:,1],self.test_prob_discrete[:,1],self.test_prob_education[:,1],self.test_prob_med_edu[:,1],self.test_prob_award[:,1]] )

  def train(self):
    
    print("\nTraining Meta-Learner")
    self.classifier.fit(self.X_train, np.asarray(self.y_train))
    joblib.dump(self.classifier, self.output_model)
    return classifier


  def test(self):

    print("\nRunning Evaluation Set")
    predictions = self.classifier.predict(np.asarray(self.X_test))
    prob_predictions = self.classifier.predict_proba(np.asarray(self.X_test))
    acc, f1_micro, f1_wght,rec_micro,rec_w,prec_micro,prec_w = 0, 0, 0,0,0,0,0 

    f1_micro = f1_score(np.asarray(self.y_test), predictions, average='micro')

    f1_wght = f1_score(np.asarray(self.y_test), predictions, average='weighted')

    acc = accuracy_score(np.asarray(self.y_test), predictions,normalize=True)

    cr = classification_report(np.asarray(self.y_test), predictions, target_names=self.classes)

    rec_w = recall_score(np.asarray(self.y_test), predictions, average='weighted')
    prec_w = precision_score(np.asarray(self.y_test), predictions, average='weighted')

    rec_micro = recall_score(np.asarray(self.y_test), predictions, average='micro')
    prec_micro = precision_score(np.asarray(self.y_test), predictions, average='micro')

    cm= confusion_matrix(np.asarray(self.y_test), predictions )

    print("\nSaving Results at:", self.results_out)
    with open(self.results_out, 'w') as f:
      s = (
          '\nAccuracy : {}\n' 
           'F1 Micro : {}\n' 
           'F1 Weighted : {}\n'
           'Recall Micro : {}\n' 
           'Recall Weighted : {}\n' 
           'Precision Micro : {}\n' 
           'Precision Weighted : {}\n' 
           '\t\t\t\tclassification_report\n {}\n' 
           'Confusion Matrix : {}\n' 
          '-----------------------------------------------------------------------------\n\n'  
          )
      output_string = s.format(acc, f1_micro, f1_wght,rec_micro,rec_w,prec_micro,prec_w, cr,cm)
      f.write(output_string)

    # Save all Predictions and probabilities

    dict_results = {
                  "Probability_not_interviewd": prob_predictions[:,0],
                  "Probability_interviewd": prob_predictions[:,1],
                  "Prediction":predictions,
                  }



    df = pd.DataFrame(dict_results)
    results_out = self.results_out.split("performance")[0] + "interview_probabilities.csv"

    df.to_csv(results_out,index=False)
    print("\nSaving CSV Performance at:", results_out)


    return acc, f1_micro, f1_wght,rec_micro,rec_w,prec_micro,prec_w 


  def run_pipeline(self, run='train'):
    
    if run == 'train':
      self.train()
      print(self.test())

    if run == 'test':
      acc, f1_micro, f1_wght,rec_micro,rec_w,prec_micro,prec_w = self.test()


def parse_args():
  parser = argparse.ArgumentParser()


  parser.add_argument('--execution', type=str, default='train',
                        help='Choose execution type - train, test')

  parser.add_argument('--model_type', type=str, default='logistic',
                        help='Choose model to execute - logistic, sgb, rf,xgbost,knn,gb')

  parser.add_argument('--path_bert_model', type=str, default='../BERT/checkpoints/train/seed_100/bio_clinicalbert/best_model/',
            help='Provide path to BERT model')

  parser.add_argument('--path_discrete_feat_model', type=str, default='../Classifiers/checkpoints/seed_100/discrete_feat/xgbost/',
            help='Provide path to discrete features model (Classifier and Vectorizer must be in the same folder) ')

  parser.add_argument('--path_education_model', type=str, default='../Classifiers/checkpoints/seed_100/education/xgbost/',
            help='Provide path to education model (Classifier and Vectorizer must be in the same folder)')

  parser.add_argument('--path_med_education_model', type=str, default='../Classifiers/checkpoints/seed_100/med_education/xgbost/',
            help='Provide path to medical education model (Classifier and Vectorizer must be in the same folder)')

  parser.add_argument('--path_award_model', type=str, default='../Classifiers/checkpoints/seed_100/award/xgbost/',
            help='Provide path to awards model (Classifier and Vectorizer must be in the same folder)')

  parser.add_argument('--path_metalearner', type=str, default='../checkpoints/seed_100/logistic/model_logistic.pkl',
            help='Provide path to metalearner model')

  parser.add_argument( "--seed", type=int, default=100, help="Random Seed")

  parser.add_argument( "--path_train", type=str,default="/Users/thiago/Github/Residency_AI/src/training/example_data/train.csv",
                        help="Path to Train File")
  parser.add_argument( "--path_test", type=str,default="/Users/thiago/Github/Residency_AI/src/training/example_data/test.csv",
                        help="Path to Test File")


  return parser.parse_args()

if __name__ == '__main__':

  warnings.filterwarnings(action='ignore', category=DataConversionWarning)
  warnings.filterwarnings("ignore")

  with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    args = parse_args()
    step = args.execution.lower()
    model_type = args.model_type.lower()
    seed = int(args.seed)
    seed_out = "seed_" + str(seed)

    try:
      traindf = pd.read_csv(args.path_train)
      testdf = pd.read_csv(args.path_test)
    except Exception as e:
      logging.exception("Error occurred while reading Training and Testing Files" +" Info: " + str(e))
      exit()

    # Random shuffle data
    traindf = traindf.sample(frac=1,random_state=seed)
    testdf = testdf.sample(frac=1,random_state=seed)
    output_model = os.path.join("checkpoints",seed_out,model_type)
    results_out = os.path.join( "results",seed_out,model_type)
    
    # Definition of models with hyperparamters after a cv search
    if  model_type == "logistic": 
      classifier = LogisticRegression(penalty="l2",C=1.5,max_iter=500,class_weight='balanced')
    elif model_type == "sgb" :
      classifier = SGDClassifier(loss='modified_huber', max_iter=10000, tol=1e-3,   n_iter_no_change=200, early_stopping=True, n_jobs=-1 ,class_weight='balanced')
    elif model_type == "rf":
      classifier = RandomForestClassifier(n_estimators = 5000,class_weight='balanced')
    elif model_type == "xgbost":
      classifier = XGBClassifier(n_estimators=800,random_state=seed,class_weight='balanced')
    elif model_type == "knn":
      classifier = KNeighborsClassifier(n_neighbors=10, weights='distance', n_jobs=-1)
    elif model_type == "gb":
      classifier = GradientBoostingClassifier(n_estimators=1000, learning_rate=0.001, random_state=seed)


    output_model_ = output_model + "/model_" + str(model_type) + ".pkl"
    results_out_  = results_out + "/performance.txt"

    if step =='train':
      # make sure path is created if it doesn't exist
      Path(output_model).mkdir(parents=True, exist_ok=True)
      Path(results_out).mkdir(parents=True, exist_ok=True)

    if step =='test':
      classifier = joblib.load(args.path_metalearner)
    

    pipeline = Meta_Learner(df_train=traindf,df_test=testdf, classifier=classifier,output_model=output_model_, 
                            results_out=results_out_, path_BERT=args.path_bert_model,classes =["Not Interviewed", "Interviwed"],
                            path_branch_classifier=[args.path_award_model,args.path_discrete_feat_model, args.path_education_model, args.path_med_education_model])
    
    pipeline.run_pipeline(run=step)




