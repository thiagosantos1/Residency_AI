import os
import torch

import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

from simpletransformers.classification import (
  ClassificationModel
)
import logging

import sys
import re
from sklearn.metrics import plot_confusion_matrix

from sklearn.metrics import precision_recall_curve,roc_auc_score,average_precision_score,f1_score
from sklearn.metrics import confusion_matrix,hamming_loss
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report
from sklearn.utils.class_weight import compute_class_weight

from gensim.parsing import preprocessing
from gensim.parsing.preprocessing import strip_tags, strip_punctuation,strip_numeric,remove_stopwords
from nltk.corpus import stopwords

import argparse

import itertools

from scipy.special import softmax

from pathlib import Path

import warnings
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category=DataConversionWarning)


class BERT_SimpTrans:
  def __init__(self, X_train,y_train, X_test, y_test,  y_train_years, y_test_years,
               classes=["Not Interviewed", "Interviwed"],do_stopwords=False,
               input_model ="checkpoints/", execution='train', index_test=[],
               output_model = "checkpoints/", results_out = "BERT_residency.txt",add_wandb = False,
               model_emb = 'bert', model_name = "emilyalsentzer/Bio_ClinicalBERT",
               n_epocs = 8, n_batches = 16, l_rate = 2e-5, decay = 1e-7, cuda = False,
               save_steps = 2000, evaluate_during_training_steps = 500, max_seq_length = 512,
               early_stopping_patience = 5, early_stopping_delta = 0.01 ):

    self.X_train, self.y_train, self.X_test, self.y_test = X_train,y_train, X_test, y_test
    self.y_train_years, self.y_test_years = y_train_years, y_test_years
    self.load_model = False
    self.execution,self.input_model,self.output_model,self.output_best_model = execution, input_model,output_model,output_model
    self.model_emb,self.model_name,self.results_out,self.n_epocs,self.n_batches = model_emb,model_name,results_out,n_epocs,n_batches
    self.l_rate,self.decay,self.cuda,self.index_test,self.add_wandb,self.do_stopwords = l_rate,decay,cuda,index_test,add_wandb,do_stopwords
    self.save_steps, self.evaluate_during_training_steps, self.max_seq_length = save_steps, evaluate_during_training_steps, max_seq_length
    self.early_stopping_patience,self.early_stopping_delta = early_stopping_patience,early_stopping_delta

    if "test" in execution:
      self.model_type = self.input_model
      self.load_model = True
    else:
      self.model_type = self.model_name
    
    self.best_model = self.output_model + "/best_model"

    self.classes = classes

    self.initialize()


  def initialize(self):

    self.num_classes = len(self.classes)
    if self.do_stopwords:
      self.X_train = self.text_cleaning(self.X_train)
      self.X_test = self.text_cleaning(self.X_test)

    self.df_train = pd.DataFrame(list(zip(self.X_train, self.y_train)), 
                columns =['report', 'labels']) 

    self.df_test = pd.DataFrame((zip(self.X_test, self.y_test)), 
                columns =['report', 'labels'])   


    if self.load_model:
      train_args = {
        'eval_batch_size': 32,
      }
      
      self.model = ClassificationModel(self.model_emb, os.path.join(self.output_model,"best_model"),use_cuda=self.cuda)

    else:
      class_wghts = []
      with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        class_wghts = compute_class_weight('balanced',np.unique(self.y_train),self.y_train).tolist()

      args={'learning_rate':self.l_rate, 'num_train_epochs': self.n_epocs,
            'overwrite_output_dir': True,'output_dir':self.output_model,"reprocess_input_data": True, 
            "use_multiprocessing": True,"evaluate_during_training": True,
            "evaluate_during_training_steps": self.evaluate_during_training_steps,"save_eval_checkpoints": False,
            "save_steps": self.save_steps,"save_model_every_epoch": False,'max_seq_length': self.max_seq_length,
            'output_hidden_states':False,'do_lower_case':True, 'fp16':self.cuda, "n_gpu": int(self.cuda==True),
            'train_batch_size':self.n_batches,  'weight_decay':self.decay,"best_model_dir":self.best_model,
            'wandb_project':'residency-transformers',
            'wandb_kwargs' :  {'name': f'{self.model_emb}-{self.model_name}'},
            "early_stopping_patience": self.early_stopping_patience,"early_stopping_delta": self.early_stopping_delta, "early_stopping_metric": "eval_loss",
            "early_stopping_metric_minimize": True,'early_stopping_consider_epochs':True, 'use_early_stopping':False,
            
            }

      if not self.add_wandb:
        args['wandb_project'] = None 
        args['wandb_kwargs'] = {} 

      self.model = ClassificationModel(self.model_emb, self.model_name, num_labels=self.num_classes, 
                                       use_cuda=self.cuda,args=args, weight=class_wghts,cuda_device=5)
      

  def text_cleaning(self,data, min_lenght=2):
    sentences = []

    for txt in data:
      filters = [lambda x: x.lower(), strip_tags, strip_numeric]

      words = preprocessing.preprocess_string(txt, filters)
      stop_words = set(stopwords.words('english'))
      stop_words.remove("no")
      stop_words.remove("than")
      stop_words.remove("not")

      c_words = [w for w in words if not w in stop_words and re.search("[a-z-A-Z]+\\w+",w) != None and len(w) >min_lenght ] 

      out = ""
      out = ' '.join(map(str, c_words)) 
      sentences.append(out)

    sentences= np.asarray(sentences)

    return sentences

  def train(self):
    logging.info("\tTraining model and saving at: " + self.output_model)

    self.model.train_model(self.df_train,  eval_df=self.df_test)
    logging.info("Testing on Test set")
    

  def hamming_score(self,y_true, y_pred, normalize=True, sample_weight=None):

    acc_list = []
    for i in range(len(y_true)):
      set_true = set( np.where(y_true[i])[0] )
      set_pred = set( np.where(y_pred[i])[0] )
      tmp_a = None
      if len(set_true) == 0 and len(set_pred) == 0:
        tmp_a = 1
      else:
        tmp_a = len(set_true.intersection(set_pred))/\
                float( len(set_true.union(set_pred)) )
      acc_list.append(tmp_a)
    return np.mean(acc_list)



  def test(self,test_data,years_set,train_set=False):

    logging.info("\n\tEvaluating model for test data of size: ", test_data.shape[0])

    logging.info("\n\tTesting model")
    
    predictions, probs = self.model.predict(test_data['report'].to_list())
    probs = [softmax(prb) for prb in probs]

    probs_0 = [x[0] for x in probs]
    probs_1 = [x[1] for x in probs]

    cm= confusion_matrix(test_data['labels'], predictions )


    labels = []
    for l in test_data['labels']:
      labels.append(l)

    f1_micro = f1_score(labels, predictions, average='micro')

    f1_wght = f1_score(labels, predictions, average='weighted')

    acc = accuracy_score(labels, predictions,normalize=True)

    h_l = hamming_loss(labels, predictions)
    h_l_score = self.hamming_score(labels, predictions)
  
    cr = classification_report(labels, predictions, target_names=self.classes)

    rec_w = recall_score(labels, predictions, average='weighted')
    prec_w = precision_score(labels, predictions, average='weighted')

    rec_micro = recall_score(labels, predictions, average='micro')
    prec_micro = precision_score(labels, predictions, average='micro')


    logging.info("Accuracy: {}\nF1 Micro: {}\nF1 Weighted: {}\n".format(acc, f1_micro, f1_wght))


    if not train_set: # save results only for test set
     
      output_file = self.results_out 
      logging.info("Saving Results at:", output_file)
      with open(output_file, 'w') as f:
        s = (
            '\nAccuracy : {}\n' 
             'F1 Micro : {}\n' 
             'F1 Weighted : {}\n'
             'Recall Micro : {}\n' 
             'Recall Weighted : {}\n' 
             'Precision Micro : {}\n' 
             'Precision Weighted : {}\n' 
             'Hamming Loss : {}\n' 
             'Hamming Score : {}\n' 
             '\t\t\t\tclassification_report\n {}\n' 
             'Confusion Matrix : {}\n' 
            '-----------------------------------------------------------------------------\n\n'  
            )
        output_string = s.format(acc, f1_micro, f1_wght,rec_micro,rec_w,prec_micro,prec_w, h_l,h_l_score,cr,cm)
        f.write(output_string)


      # let's also save per year
      unique_years = np.unique(years_set)
      out = {}
      for x in unique_years:
        out[x] = [[],[]]

      # create pred and test for by year
      for index, y in enumerate(labels):
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
        cr_year =  classification_report(true, preds, target_names=self.classes)
        rec_w = recall_score(true, preds, average='weighted')
        prec_w = precision_score(true, preds, average='weighted')
        rec_micro_year = recall_score(true, preds, average='micro')
        prec_micro_year = precision_score(true, preds, average='micro')
        cm_year= confusion_matrix(true, preds )

        with open(output_file, 'a') as f:
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
          output_string = s.format(acc_year, f1_micro_year, f1_wght_year,rec_micro_year,rec_w,prec_micro_year,prec_w,cr_year,cm_year)
          f.write(output_string)
    
    logging.info("\n\nData: ",len(self.index_test),"\nProbability_not_interviewd: ",len(probs_0))
    logging.info("Probability_interviewd: ", len(probs_1), "\nPrediction: ",len(predictions))
    logging.info("Label: ", len(labels), "\nYear_Application: ",len(years_set))
    if train_set:    
      logging.info("\n\ntrain_set")
      dict_results = {
                    "Probability_not_interviewd": probs_0,
                    "Probability_interviewd": probs_1,
                    "Prediction":predictions,
                    "Label":labels,
                    "Year_Application":years_set}
    else:
      logging.info("\n\ntest_set")
      dict_results = {"Data_Index":self.index_test,
                    "Probability_not_interviewd": probs_0,
                    "Probability_interviewd": probs_1,
                    "Prediction":predictions,
                    "Label":labels,
                    "Year_Application":years_set}

    
    df = pd.DataFrame(dict_results)

    output_file = self.results_out 
    if train_set:
      logging.info("Saving raining Training probabilities")
      output_file = output_file.split(".txt")[0] + "probabilities_train_set.csv"
    else:
      output_file = output_file.split(".txt")[0] + "probabilities.csv"

    df.to_csv(output_file,index=False)

    logging.info("Saving CSV Performance at:", output_file)

    return acc, f1_micro, f1_wght,rec_micro,rec_w,prec_micro,prec_w 

  def plot_confusion_matrix(self,cm,classes=[], ylabel='True label', xlabel = 'Predicted label',
                                   title='Confusion matrix',normalize=False, cmap=plt.cm.Blues, 
                                   png_output=None):
     
        
    # Calculate chart area size
    leftmargin = 0.5 # inches
    rightmargin = 0.5 # inches
    categorysize = 0.5 # inches
    figwidth = leftmargin + rightmargin + (len(classes) * categorysize)           

    f = plt.figure(figsize=(figwidth, figwidth))

    # Create an axes instance and ajust the subplot size
    ax = f.add_subplot(111)
    ax.set_aspect(1)
    f.subplots_adjust(left=leftmargin/figwidth, right=1-rightmargin/figwidth, top=0.94, bottom=0.1)

    res = ax.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)
    plt.colorbar(res,fraction=0.046, pad=0.04)
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha='right')
    ax.set_yticklabels(classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, format(cm[i, j], fmt),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black")

    
    plt.tight_layout()
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.xticks(rotation=90)
    #plt.show()
    
    if png_output is not None:
      f.savefig(png_output, bbox_inches='tight')


  def run(self):
    
    if self.execution.lower() == "train":
      self.train()
      self.test(self.df_test,self.y_test_years)
      self.test(self.df_train,self.y_train_years,train_set=True)
    elif self.execution.lower() == "test":
      self.test(self.df_test,self.y_test_years)




def parse_args():
  parser = argparse.ArgumentParser()


  parser.add_argument('--execution', type=str, default='train',
                        help='Choose execution type - train, test')
  parser.add_argument( "--do_stopwords", type=int,default=1,
                        help="Whether to apply stopwords to PS"
                      )

  parser.add_argument( "--path_train", type=str,default="train.csv",
                        help="Path to Train File"
                      )

  parser.add_argument( "--path_test", type=str,default="test.csv",
                        help="Path to Test File"
                      )

  parser.add_argument('--cuda', type=str, default='false',
                        help='set cuda available type - true, false')


  parser.add_argument('--model_type', type=str, default='bio_clinicalbert',
            help='Choose bert of execution - bert, roberta, distilbert,distilroberta,electra, biobert, bio_clinicalbert, biomednlp')

  parser.add_argument( "--seed", type=int, default=100, help="Random Seed"
    )

  parser.add_argument( "--n_epocs", type=int, default=8, help="n_epocs"
    )
  parser.add_argument( "--n_batches", type=int, default=16, help="n_batches"
    )
  parser.add_argument( "--save_steps", type=int, default=2000, help="Save Steps"
    )
  parser.add_argument( "--evaluate_during_training_steps", type=int, default=500, help="Evaluation Steps"
    )
  parser.add_argument( "--max_seq_length", type=int, default=512, help="Max sentence length"
    )
  parser.add_argument( "--early_stopping_patience", type=int, default=5, help="Early Stoping Patience"
    )
  parser.add_argument( "--l_rate", type=float, default=2e-5, help="Learning Rate"
    )
  parser.add_argument( "--decay", type=float, default=1e-7, help="Decay"
    )
  parser.add_argument( "--early_stopping_delta", type=float, default=0.01, help="Early Stoping Dealta"
    )


  return parser.parse_args()



if __name__ == '__main__':

  warnings.filterwarnings(action='ignore', category=DataConversionWarning)

  args = parse_args()
  step = args.execution.lower()
  do_stopwords = args.do_stopwords

  fold_out = step

  cuda = args.cuda.lower() == 'true' 

  seed = int(args.seed)

  logging.info( "Reading and Pre-Processing Data")

  try:
    traindf = pd.read_csv(args.path_train)
    testdf = pd.read_csv(args.path_test)
  except Exception as e:
    logging.exception("Error occurred while reading Training and Testing Files" +" Info: " + str(e))
    exit()


  # Random shuffle data
  traindf = traindf.sample(frac=1,random_state=seed)
  testdf = testdf.sample(frac=1,random_state=seed)

  traindf[['PS']] = traindf['PS'].fillna('no personal statement available');
  traindf = traindf.sample(frac=1, random_state=seed)

  testdf[['PS']] = testdf['PS'].fillna('no personal statement available');
  testdf = testdf.sample(frac=1, random_state=seed)

  X_train = traindf.PS.str.lower().values
  X_test = testdf.PS.str.lower().values
  index_test = testdf.index.values


  y_train =traindf['interview'].tolist()
  y_test =testdf['interview'].tolist()

  y_train_years = traindf['year'].tolist()
  y_test_years  = testdf['year'].tolist()

  seed_out = "seed_" + str(seed)

  model_type = args.model_type.lower() 

  if model_type =="bert":
    # BERT and tokenizer to be used
    bert_model =  'bert-base-uncased'
    model_emb = 'bert'
    output_model = os.path.join("checkpoints",
                                fold_out,
                                seed_out,
                                'bert')
    results_out = os.path.join( "results",
                                fold_out,
                                seed_out,
                                "bert")

  elif model_type =="roberta":
    model_emb = 'roberta'
    # BERT and tokenizer to be used
    bert_model =  'roberta-base'
    output_model = os.path.join("checkpoints",
                                fold_out,
                                seed_out,
                                'roberta')
    results_out = os.path.join( "results",
                                fold_out,
                                seed_out,
                                "roberta")

  elif model_type =="distilbert":
    model_emb = 'bert'
    # BERT and tokenizer to be used
    bert_model =  'distilbert-base-cased'
    output_model = os.path.join("checkpoints",
                                fold_out,
                                seed_out,
                                'distilbert')
    results_out = os.path.join( "results",
                                fold_out,
                                seed_out,
                                "distilbert")

  elif model_type =="distilroberta":
    model_emb = 'roberta'
    # BERT and tokenizer to be used
    bert_model =  'distilroberta-base'
    output_model = os.path.join("checkpoints",
                                fold_out,
                                seed_out,
                                'distilroberta')
    results_out = os.path.join( "results",
                                fold_out,
                                seed_out,
                                "distilroberta")

  elif model_type =="electra":
    model_emb = 'electra'
    # BERT and tokenizer to be used
    bert_model =  'google/electra-small-discriminator'
    output_model = os.path.join("checkpoints",
                                fold_out,
                                seed_out,
                                'electra')
    results_out = os.path.join( "results",
                                fold_out,
                                seed_out,
                                "electra")


  elif model_type =="biobert":   # BIO BERTs
    model_emb = 'bert'
    bert_model = "dmis-lab/biobert-base-cased-v1.1"
    output_model = os.path.join("checkpoints",
                                fold_out,
                                seed_out,
                                 'biobert'
                                 )
    results_out  = os.path.join( "results",
                                fold_out,
                                seed_out,
                                "biobert"
                                )

  elif model_type =="bio_clinicalbert": 
    model_emb = 'bert'
    bert_model = "emilyalsentzer/Bio_ClinicalBERT"
    output_model  = os.path.join("checkpoints",
                                fold_out,
                                seed_out,
                                'bio_clinicalbert'
                                )
    results_out  = os.path.join( "results",
                                fold_out,
                                seed_out,
                                "bio_clinicalbert"
                                )

  elif model_type =="biomednlp": 
    model_emb = 'bert'
    bert_model = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
    output_model  = os.path.join("checkpoints",
                                fold_out,
                                seed_out,
                                'biomednlp/'
                                )
    results_out  = os.path.join( "results",
                                fold_out,
                                seed_out,
                                "biomednlp"
                                )
  else:
    logging.info("Please choose a correct BERT model")


  if step =='train':
    # make sure path is created if it doesn't exist
    Path(output_model).mkdir(parents=True, exist_ok=True)
    Path(results_out).mkdir(parents=True, exist_ok=True)


  logging.info("\n\n\t\t Runing Classification for: ", model_type,"\n\n")
  results_out_run = results_out + "/performance.txt"

  model = BERT_SimpTrans(X_train,y_train, X_test, y_test, y_train_years, y_test_years,
          model_name=bert_model, model_emb=model_emb,do_stopwords=do_stopwords,
          input_model=output_model,output_model=output_model, cuda=cuda, execution=step, 
          results_out=results_out_run,index_test=index_test, n_epocs = args.n_epocs, n_batches = args.n_batches,
          save_steps = args.save_steps, evaluate_during_training_steps = args.evaluate_during_training_steps, max_seq_length=args.max_seq_length,
          early_stopping_patience = args.early_stopping_patience, early_stopping_delta=args.early_stopping_delta, l_rate=args.l_rate, decay=args.decay)

  model.run()






