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

from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn import svm, tree
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from sklearn.preprocessing import LabelEncoder,OrdinalEncoder,OneHotEncoder
from sklearn.base import clone
from sklearn.preprocessing import label_binarize
from scipy import interp
from sklearn.metrics import roc_curve, auc
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
import joblib

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


def text_cleaning(data, steam=False, lemma = True, clean=True,min_lenght=2):
  words_sentences = []
  sentences = []

  for txt in data:
    orig_txt = txt
    txt = re.sub("none|other", "other",txt)
    filters = [lambda x: x.lower(), strip_tags, strip_punctuation, strip_numeric]

    words = preprocessing.preprocess_string(txt, filters)
    stop_words = set(stopwords.words('english'))
    stop_words.remove("no")
    stop_words.remove("than")
    stop_words.remove("not")
    if clean:
      words = [w for w in words if not w in stop_words and re.search("[a-z-A-Z]+\\w+",w) != None and len(w) >min_lenght ] 
    else:
      words = [w for w in words if re.search("[a-z-A-Z]+\\w+",w) != None and len(w) >1 ] 

    c_words = words

    if steam:
      porter = PorterStemmer()
      c_words = [porter.stem(word) for word in c_words]

    if lemma:
      lem = nltk.stem.wordnet.WordNetLemmatizer()
      c_words = [lem.lemmatize(word) for word in c_words]
    
    words_sentences.append(c_words)
  

    out = ""
    out = ' '.join(map(str, c_words)) 
    sentences.append(out)

  sentences= np.asarray(sentences)

  return sentences, words_sentences

def pre_process(data_,seed=0):

  data = data_.copy()

  data.drop(['year'],axis = 1, inplace=True) 

  # make sure to sample data
  data = data.sample(frac=1, random_state=seed)


  data[['gender']] = data['gender'].fillna('no_identified')
  data[['self_identification']] = data['self_identification'].fillna('Other')
  data[['Authorized_to_Work']] = data['Authorized_to_Work'].fillna('Other')


  # Discrete Features
  data = data.replace(to_replace='Other', value='Other', regex=True)
  data['self_identification'] = np.where((data.self_identification.str.contains('Other')),
                                        'Other',data.self_identification)
  data['self_identification'] = np.where((data.self_identification.str.contains('Hispanic')),
                                        'Hispanic',data.self_identification)
  data['self_identification'] = np.where((data.self_identification.str.contains('White')),
                                        'White',data.self_identification)
  data['self_identification'] = np.where((data.self_identification.str.contains('Asian')),
                                        'Asian',data.self_identification)
  data['self_identification'] = np.where((data.self_identification.str.contains('African American')),
                                        'African American',data.self_identification)
  data['self_identification'] = np.where((data.self_identification.str.contains('Prefer not to say')),
                                        'Prefer not to say',data.self_identification)
  data['self_identification'] = np.where((data.self_identification.str.contains('American Indian')),
                                        'American Indian',data.self_identification)
  data['self_identification'] = np.where((data.self_identification.str.contains('Native')),
                                        'Native',data.self_identification)
  data['self_identification'] = np.where((data.self_identification.str.contains('Mailing Address')),
                                        'Other',data.self_identification)
  data['Misdemeanor'] = np.where((data.Misdemeanor.str.contains('No')),
                                        'No',data.Misdemeanor)
  data['Misdemeanor'] = np.where((data.Misdemeanor.str.contains('Yes')),
                                        'Yes',data.Misdemeanor)
  data['Felony'] = np.where((data.Felony.str.contains('No')),
                                        'No',data.Felony)
  data['Felony'] = np.where((data.Felony.str.contains('Yes')),
                                        'Yes',data.Felony)
  data['Authorized_to_Work'] = np.where((data.Authorized_to_Work.str.contains('F-2')),
                                        'F-2',data.Authorized_to_Work)
  data['Authorized_to_Work'] = np.where((data.Authorized_to_Work.str.contains('B-1')),
                                        'B-1',data.Authorized_to_Work)
  data['Authorized_to_Work'] = np.where((data.Authorized_to_Work.str.contains('H-1B')),
                                        'H-1B',data.Authorized_to_Work)
  data['Authorized_to_Work'] = np.where((data.Authorized_to_Work.str.contains('J-1')),
                                        'J-1',data.Authorized_to_Work)
  data['Authorized_to_Work'] = np.where((data.Authorized_to_Work.str.contains('F-1')),
                                        'F-1',data.Authorized_to_Work)
  data['Authorized_to_Work'] = np.where((data.Authorized_to_Work.str.contains('B-2')),
                                        'B-2',data.Authorized_to_Work)
  data['Authorized_to_Work'] = np.where((data.Authorized_to_Work.str.contains('H-4')),
                                        'H-4',data.Authorized_to_Work)
  data['Authorized_to_Work'] = np.where((data.Authorized_to_Work.str.contains('J-2')),
                                        'J-2',data.Authorized_to_Work)
  data['Authorized_to_Work'] = np.where((data.Authorized_to_Work.str.contains('H-1')),
                                        'H-1',data.Authorized_to_Work)
  data['Authorized_to_Work'] = np.where((data.Authorized_to_Work.str.contains('TN')),
                                        'TN',data.Authorized_to_Work)
  data['Authorized_to_Work'] = np.where((data.Authorized_to_Work.str.contains('EAD|Diplomatic')),
                                        'Other',data.Authorized_to_Work)
  data['Authorized_to_Work'] = np.where((data.Authorized_to_Work.str.contains('EAD|Diplomatic')),
                                        'Other',data.Authorized_to_Work)
  data['gender'] = np.where((data.gender.str.contains('Female')),
                                        'Female',data.gender)
  data['gender'] = np.where((data.gender.str.contains('Male')),
                                        'Male',data.gender)


  data_discrete = data.copy()


  data_discrete.drop(['name','PS','interview','Medical_Education','Education','Awards', 'Unnamed: 0', 'Unnamed',
                            'Certification_Licensure','Publications','Page2'],axis = 1, inplace=True,errors='ignore')

  # Education
  data[['Education']] = data['Education'].fillna('no_identified')
  data_education = data.Education.str.lower().values

  # Medical Education
  data[['Medical_Education']] = data['Medical_Education'].fillna('no_identified')
  data_med_education = data.Medical_Education.str.lower().values

  # Award
  data[['Awards']] = data['Awards'].fillna('no_identified')
  data_awards = data.Awards.str.lower().values

  # Personal Statement
  data[['PS']] = data['PS'].fillna('no personal statement available');
  ps = data.PS.str.lower().values

  labels =data['interview'].tolist()
  index_data = data.index.values

  return ps, data_discrete, data_education, data_med_education, data_awards, labels, index_data


def tf_idf(X_t, X_v, n_gram=(2,3),max_features = 2000, save="tfidf_vectorizer.pkl"): 
  vectorizer = TfidfVectorizer(analyzer='word',ngram_range=n_gram, min_df=2,)
  vectorizer.fit(X_t)
  X_train = vectorizer.transform(X_t).toarray()
  X_val = vectorizer.transform(X_v).toarray()
  joblib.dump(vectorizer, save)
  return X_train,X_val

# OrdinalEncoder()  or OneHotEncoder(sparse=False)
def categorical_inputs(encoding,X_train, X_test,save="categorical_vectorizer.pkl"):
  X_t_pub = X_train.Publications_Count.values
  X_v_pub = X_test.Publications_Count.values
  # those aren't categorical
  X_t_d = X_train.drop(columns=['Publications_Count'])
  X_v_d = X_test.drop(columns=['Publications_Count'])
  
  all_d = pd.concat([X_t_d, X_v_d])
  
  encoding.fit(all_d)
  
  X_train_enc = encoding.transform(X_t_d)
  X_test_enc = encoding.transform(X_v_d)
  
  X_test_enc = np.insert(X_test_enc, len(X_test_enc[0]), X_v_pub, axis=1)
  
  X_train_enc = np.insert(X_train_enc, len(X_train_enc[0]), X_t_pub, axis=1)
  
  joblib.dump(encoding, save)

  return X_train_enc, X_test_enc

def prepare_targets(y_train, y_test):
  encoding = LabelEncoder()
  encoding.fit(y_train)
  y_train_enc = encoding.transform(y_train)
  y_test_enc = encoding.transform(y_test)
  return y_train_enc, y_test_enc


def train(classifier,X_train,y_train, output_model=""):
  classifier.fit(np.asarray(X_train), np.asarray(y_train))
  joblib.dump(classifier, output_model)
  return classifier


def test(classifier,X_test,y_test,years_set,results_out="", test_index =[],classes=["Not Interviewed", "Interviwed"],train_set=False):
  predictions = classifier.predict(np.asarray(X_test))
  prob_predictions = classifier.predict_proba(np.asarray(X_test))
  acc, f1_micro, f1_wght,rec_micro,rec_w,prec_micro,prec_w = 0, 0, 0,0,0,0,0 

  if not train_set:
    f1_micro = f1_score(np.asarray(y_test), predictions, average='micro')

    f1_wght = f1_score(np.asarray(y_test), predictions, average='weighted')

    acc = accuracy_score(np.asarray(y_test), predictions,normalize=True)

    cr = classification_report(np.asarray(y_test), predictions, target_names=classes)

    rec_w = recall_score(np.asarray(y_test), predictions, average='weighted')
    prec_w = precision_score(np.asarray(y_test), predictions, average='weighted')

    rec_micro = recall_score(np.asarray(y_test), predictions, average='micro')
    prec_micro = precision_score(np.asarray(y_test), predictions, average='micro')

    cm= confusion_matrix(np.asarray(y_test), predictions )

    logging.info("Saving Results at:", results_out)
    with open(results_out, 'w') as f:
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


    # let's also save per year
    unique_years = np.unique(years_set)
    out = {}
    for x in unique_years:
      out[x] = [[],[]]

    # create pred and test for by year
    for index, y in enumerate(y_test):
      p = predictions[index]
      year = years_set[index]
      out[year][0].append(y) # true
      out[year][1].append(p) # pred

    # save performance
    for key,val in out.items():
      true  = val[0]
      preds = val[1]
      f1_micro_year = f1_score(true, preds, average='micro')
      f1_wght_year = f1_score(true, preds, average='weighted')
      acc_year = accuracy_score(true, preds,normalize=True)
      cr_year = classification_report(true, preds, target_names=classes)
      rec_w_year = recall_score(true, preds, average='weighted')
      prec_w_year = precision_score(true, preds, average='weighted')
      rec_micro_year = recall_score(true, preds, average='micro')
      prec_micro_year = precision_score(true, preds, average='micro')
      cm_year= confusion_matrix(true, preds )

      with open(results_out, 'a') as f:
        f.write("\n\nPerformance for the year of " + str(key))
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
        output_string = s.format(acc_year, f1_micro_year, f1_wght_year,rec_micro_year,rec_w_year,prec_micro_year,prec_w_year,cr_year,cm_year)
        f.write(output_string)

  # Save all Predictions and probabilities
  if train_set:    
    logging.info("\n\ntrain_set")
    dict_results = {
                  "Probability_not_interviewd": prob_predictions[:,0],
                  "Probability_interviewd": prob_predictions[:,1],
                  "Prediction":predictions,
                  "Label":y_test,
                  "Year_Application":years_set}
  else:
    logging.info("\n\ntest_set")
    dict_results = {"Data_Index":test_index,
                  "Probability_not_interviewd": prob_predictions[:,0],
                  "Probability_interviewd": prob_predictions[:,1],
                  "Prediction":predictions,
                  "Label":y_test,
                  "Year_Application":years_set}



  df = pd.DataFrame(dict_results)
  if train_set:
    results_out = results_out.split(".txt")[0] + "probabilities_train_set.csv"
  else:
    results_out = results_out.split(".txt")[0] + "probabilities.csv"

  df.to_csv(results_out,index=False)
  logging.info("Saving CSV Performance at:", results_out)

  return acc, f1_micro, f1_wght,rec_micro,rec_w,prec_micro,prec_w 


def run_model(classifier,X_train, X_test,y_train, y_test, y_train_years, y_test_years,
              run='train', test_index =[],
              classes=["Not Interviewed", "Interviwed"], output_model="", results_out=""):
  
  if run == 'train':
    classifier = train(classifier,X_train,y_train, output_model=output_model)

  # test with test set
  acc, f1_micro, f1_wght,rec_micro,rec_w,prec_micro,prec_w = test(classifier,X_test,y_test,y_test_years,results_out=results_out,test_index=test_index,classes=classes)

  # also run test with training, to save probabilities
  acc, f1_micro, f1_wght,rec_micro,rec_w,prec_micro,prec_w = test(classifier,X_train,y_train,y_train_years,results_out=results_out,test_index=test_index,classes=classes,train_set=True)


def parse_args():
  parser = argparse.ArgumentParser()


  parser.add_argument('--execution', type=str, default='train',
                        help='Choose execution type - train, test')

  parser.add_argument('--model_type', type=str, default='xgbost',
                        help='Choose model to execute - logistic, sgb, rf,xgbost,knn,gb')
  parser.add_argument('--data_type', type=str, default='all',
            help='Choose type execution - personal_statement, discrete_feat, education, med_education, award, all ')
  parser.add_argument( "--seed", type=int, default=100, help="Random Seed")
  parser.add_argument( "--path_train", type=str,default="train.csv",
                        help="Path to Train File")
  parser.add_argument( "--path_test", type=str,default="test.csv",
                        help="Path to Test File")


  return parser.parse_args()

if __name__ == '__main__':

  warnings.filterwarnings(action='ignore', category=DataConversionWarning)

  with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    args = parse_args()
    step = args.execution.lower()
    model_type = args.model_type.lower()
    data_type = args.data_type.lower()
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

    y_train_years = traindf['year'].tolist()
    y_test_years  = testdf['year'].tolist()

    # Preprocess Data
    ps_train, data_discrete_train, data_education_train, data_med_education_train, data_awards_train, y_train, index_train =  pre_process(traindf,seed=seed)
    ps_test, data_discrete_test, data_education_test, data_med_education_test, data_awards_test, y_test, index_test =  pre_process(testdf,seed=seed)

    # Clean String data
    logging.info("Cleaning Data")
    ps_train,_ = text_cleaning(ps_train)
    ps_test,_ = text_cleaning(ps_test)

    data_education_train,_ = text_cleaning(data_education_train)
    data_education_test,_ = text_cleaning(data_education_test)

    data_med_education_train,_ = text_cleaning(data_med_education_train)
    data_med_education_test,_ = text_cleaning(data_med_education_test)

    data_awards_train,_ = text_cleaning(data_awards_train)
    data_awards_test,_ = text_cleaning(data_awards_test)

    # get y in onehot
    y_train,y_test = prepare_targets(y_train, y_test)

    # compute class weights
    class_wghts = compute_class_weight('balanced',np.unique(y_train),y_train).tolist()

    # configure for each type of run
    if data_type == 'all':
      runs = ["personal_statement", "discrete_feat", "education", "med_education", "award"]
    else:
      runs = [data_type]

    for run in runs:
      logging.info("\nRunning Pipeline for Type: " + str(run))
      output_model = os.path.join("checkpoints",seed_out,run,model_type)
      results_out = os.path.join( "results",seed_out,run,model_type)
      
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


      results_out_  = results_out + "/performance.txt"
      output_model_ = output_model + "/model_" + str(model_type) + ".pkl"

      if step =='train':
        # make sure path is created if it doesn't exist
        Path(output_model).mkdir(parents=True, exist_ok=True)
        Path(results_out).mkdir(parents=True, exist_ok=True)

      if step =='test':
        classifier = joblib.load(output_model_)

      # make data
      if run =='personal_statement':
        
        X_train,X_test = tf_idf(ps_train, ps_test, save=output_model+"/vectorizer.pkl")
      elif run =='discrete_feat':           # OneHotEncoder(sparse=False)
        
        X_train,X_test = categorical_inputs(OrdinalEncoder(), data_discrete_train, data_discrete_test,save=output_model+"/vectorizer.pkl")
      elif run =='education':
        X_train,X_test = tf_idf(data_education_train, data_education_test,save=output_model+"/vectorizer.pkl")
      elif run =='med_education':
        X_train,X_test = tf_idf(data_med_education_train, data_med_education_test,save=output_model+"/vectorizer.pkl")
      else:
        X_train,X_test = tf_idf(data_awards_train, data_awards_test,save=output_model+"/vectorizer.pkl")

      # run model
      logging.info("Runing Program for: ",run)
      
      run_model(classifier,X_train, X_test,y_train, y_test,y_train_years, y_test_years,
                 run=step, test_index = index_test,output_model=output_model_, results_out=results_out_)
    
    


