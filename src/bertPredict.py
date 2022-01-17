import os

import pandas as pd
import numpy as np

from simpletransformers.classification import (
    ClassificationModel
)

import seaborn as sn
import matplotlib.pyplot as plt

from gensim.parsing.preprocessing import strip_tags, strip_punctuation, strip_numeric, remove_stopwords,preprocess_string
from nltk.corpus import stopwords


from scipy.special import softmax

from pathlib import Path
import logging
import joblib
import warnings
from sklearn.exceptions import DataConversionWarning

warnings.filterwarnings(action='ignore', category=DataConversionWarning)

from highlight_text import HighlightText, ax_text, fig_text
import html, random

from IPython.core.display import display, HTML
import nltk
from nltk.tokenize import word_tokenize
import re
from lime import lime_text
from lime.lime_text import LimeTextExplainer


nltk.download('stopwords')
nltk.download('punkt')


class BERT(object):

    """BERT Classifier - Input: Personal Statement"""
    def __init__(self, path_model,use_cuda=False,cuda_device=0):
        self.path_model = path_model
        self.use_cuda = use_cuda
        self.cuda_device = cuda_device
        self.model = ClassificationModel('bert', path_model, use_cuda=use_cuda,cuda_device=cuda_device)
        self.model.model.config.output_hidden_states = True
        self.model.model.config.silent = True
        
    def text_cleaning(self,txt, min_lenght=2):
        filters = [lambda x: x.lower(), strip_tags, strip_punctuation, strip_numeric]

        words = preprocess_string(txt, filters)
        stop_words = set(stopwords.words('english'))
        stop_words.remove("no")
        stop_words.remove("than")
        stop_words.remove("not")

        c_words = [w for w in words if
                   not w in stop_words and re.search("[a-z-A-Z]+\\w+", w) != None and len(w) > min_lenght]

        out = ""
        out = ' '.join(map(str, c_words))

        return out

    def predict(self,text):
        preds, raw_outputs, _, _ = self.model.predict([text])
        probs = [softmax(prb) for prb in raw_outputs]
        return np.array(probs[0])


class LIME_Analysis(object):

    color_classes = {0: '65, 137, 225',  # blue
                              1: "234, 131, 4",  # orange
                             }
        
    # function to normalize, if applicable
    def normalize_MinMax(self,arr, t_min=0, t_max=1):
        norm_arr = []
        diff = t_max - t_min
        diff_arr = max(arr) - min(arr)
        for i in arr:
            temp = (((i - min(arr)) * diff) / diff_arr) + t_min
            norm_arr.append(temp)
        return norm_arr


    def html_escape(self,text):
        return html.escape(text)


    def highlight_full_data(self,lime_weights, data, pred):
        words_p = [x[0] for x in lime_weights if x[1] > 0]
        weights_p = np.asarray([x[1] for x in lime_weights if x[1] > 0])
        if len(weights_p) > 1:
            weights_p = self.normalize_MinMax(weights_p, t_min=min(weights_p), t_max=1)
        else:
            weights_p = [1]
        words_n = [x[0] for x in lime_weights if x[1] < 0]
        weights_n = np.asarray([x[1] for x in lime_weights if x[1] < 0])

        if pred == 0:
            opposite = 1
        else:
            opposite = 0

        # positive values
        df_coeff = pd.DataFrame(
            {'word': words_p,
             'num_code': weights_p
             })
        word_to_coeff_mapping_p = {}
        for row in df_coeff.iterrows():
            row = row[1]
            word_to_coeff_mapping_p[row[0]] = row[1]

        # negative values
        df_coeff = pd.DataFrame(
            {'word': words_n,
             'num_code': weights_n
             })

        word_to_coeff_mapping_n = {}
        for row in df_coeff.iterrows():
            row = row[1]
            word_to_coeff_mapping_n[row[0]] = row[1]

        max_alpha = 1
        highlighted_text = []
        highlighted_text.append("<font size=+2> ")
        
        data = re.sub("-", " ", data)
        data = re.sub("/", "", data)
        for word in word_tokenize(data):
            if word.lower() in word_to_coeff_mapping_p or word.lower() in word_to_coeff_mapping_n:
                if word.lower() in word_to_coeff_mapping_p:
                    weight = word_to_coeff_mapping_p[word.lower()]
                else:
                    weight = word_to_coeff_mapping_n[word.lower()]

                if weight > 0:
                    color = self.color_classes[pred]
                else:
                    color = self.color_classes[opposite]
                    weight *= -1
                    weight *= 10

                highlighted_text.append('<span font-size:40px; ; style="background-color:rgba(' + color + ',' + str(
                    weight) + ');">' + self.html_escape(word) + '</span>')

            else:
                highlighted_text.append(word)

        highlighted_text.append("</font> ")

        highlighted_text = ' '.join(highlighted_text)

        return highlighted_text


    def plotProbabilities(self,pred_prob):
        fig, ax = plt.subplots(figsize=(5, 0.8))
        labels = ('Not Interviewed', 'Interviewed')
        y_pos = np.arange(len(labels))
        performance =pred_prob

        # performance = [0.37,0.63]
        ax.barh(y_pos, performance, color=['royalblue', 'darkorange'])
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels)
        ax.invert_yaxis()  # labels read top-to-bottom

        for i in ax.patches:
            plt.text(i.get_width() + 0.05, i.get_y() + 0.55,
                     str(round((i.get_width()), 2)),
                     fontsize=10)

        ax.set_xlabel('Prediction probabilities for only Personal statement')
        plt.xlim(0, 1)
        plt.savefig('static/lime_probability.png',bbox_inches ="tight")
        plt.clf()
        plt.cla()
        plt.close()


    def display_LIME(self,model,data_original, data_clean, class_names=["Not Interviewed", "Interviwed"], save_to="lime.html",prediction="Not Interviewed"):

        # LIME Predictor Function
        def predict(texts):
            results = []
            for text in texts:
                preds, raw_outputs, _, _ = model.predict([text])
                probs = [softmax(prb) for prb in raw_outputs]
                results.append(probs[0])

            return np.array(results)

        explainer = LimeTextExplainer(class_names=class_names)
        exp = explainer.explain_instance(data_clean, predict, num_features=60,
                                         num_samples=50, top_labels=2)
        l = exp.available_labels()
        run_info = exp.as_list(l[0])
        pred = l[0]

        if prediction == "Interviwed":
            run_info = exp.as_list(l[1])
            pred = l[1]

        pred_prob=exp.predict_proba

        self.plotProbabilities(pred_prob)

        html_w = self.highlight_full_data(run_info, data_original, pred)
        with open(save_to, "w") as file:
            file.write(html_w)



def BERT_predict(data,save_lime=True):
    # This is a path to a BERT model
    # Our best model can be found at:
    path_model = "../models/personal_statement/best_model"
    plt.switch_backend('Agg') 
    model = BERT(path_model=path_model)
    lime_analysis = LIME_Analysis()

    data_clean = model.text_cleaning(data)
    class_names = ["Not Interviewed", "Interviwed"]
    res=model.predict(data_clean)
    prediction = class_names[np.argmax(res)]

    if save_lime:
        # Lime Example
        lime_analysis.display_LIME(model.model,data,data_clean,save_to="static/lime.html",prediction=prediction)

    return res[1]


if __name__ == '__main__':
    pass


