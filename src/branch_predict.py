import numpy as np
import joblib
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from matplotlib import pyplot as plt
import shap
import pandas as pd
from gensim.parsing import preprocessing
from gensim.parsing.preprocessing import strip_tags, strip_punctuation, strip_numeric, remove_stopwords
import re
from nltk.stem import PorterStemmer
import nltk
from nltk.corpus import stopwords
nltk.download('wordnet')
shap.initjs()


def text_cleaning(data, steam=True, lemma=True, clean=True, min_lenght=2):
    words_sentences = []
    sentences = []

    for txt in data:
        orig_txt = txt
        txt = re.sub("none|other", "other", txt)
        filters = [lambda x: x.lower(), strip_tags, strip_punctuation, strip_numeric]

        words = preprocessing.preprocess_string(txt, filters)
        stop_words = set(stopwords.words('english'))
        stop_words.remove("no")
        stop_words.remove("than")
        stop_words.remove("not")
        if clean:
            words = [w for w in words if
                     not w in stop_words and re.search("[a-z-A-Z]+\\w+", w) != None and len(w) > min_lenght]
        else:
            words = [w for w in words if re.search("[a-z-A-Z]+\\w+", w) != None and len(w) > 1]

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

    sentences = np.asarray(sentences)

    return sentences


def parse(CV):
    awards = ''
    medical_edu = ''
    edu = ''
    pub_count = 0

    PS = ''
    gender = ''
    self_iden = ''
    misdemeanor = 'No'
    felony = 'No'
    authorization = 'F-1'
    medical_edu = ''
    edu = ''
    awards = ''
    cert = ''
    pub = ''
    pub_count = 0

    CV = CV.replace('\r', '\n')
    CV = CV.replace('\n\n', '\n')
    CV = CV.replace('\n\n', '\n')
    CV = CV.split('\n')
    footer = False
    for j in range(len(CV)):
        line = CV[j]
        if line[:len('Medical Education')] == 'Medical Education':
            medical_edu = ' '.join(CV[j + 1:j + 4])
        if line[:len('Education')] == 'Education':
            # edu = CV[j+1]
            edu = ''
            k = 0
            while len(CV) > j + k + 1 and (CV[j + k + 1] != 'Membership and Honorary/Professional Societies' and CV[
                j + k + 1] != 'Medical School Awards' and CV[j + k + 1] != 'Volunteer Experience' and CV[
                                               j + k + 1] != 'Certification/Licensure') and CV[
                j + k + 1] != 'Current/Prior Training' and CV[j + k + 1] != 'Work Experience':
                edu += CV[j + k + 1]
                edu += '\n'
                k += 1
        if line == 'Medical School Awards':
            awards = ''
            k = 0
            while (len(CV) > j + k + 1) and CV[j + k + 1] != 'Volunteer Experience' and CV[
                j + k + 1] != 'Average Hours/Week: ' and CV[j + k + 1] != 'Curriculum Vitae' and CV[
                j + k + 1] != 'Research Experience' and CV[j + k + 1] != 'Certification/Licensure' and CV[
                j + k + 1] != 'Current/Prior Training' and CV[j + k + 1] != 'Work Experience':
                awards += CV[j + k + 1]
                awards += '\n'
                k += 1
        if line == 'Certification/Licensure':
            cert = CV[j + 1]

        if line == 'Publications':
            pub = ''
            k = 0
            footer = False
            pub_count = 0
            while (len(CV) > j + k + 1) and CV[j + k + 1] != 'Hobbies & Interests':
                if CV[j + k + 1][
                   :len(
                       'Emory University Program, Radiology-Diagnostic')] == 'Emory University Program, Radiology-Diagnostic':
                    footer = True
                if CV[j + k + 1][:len('Curriculum Vitae')] == 'Curriculum Vitae':
                    footer = False
                if footer == False:
                    pub += CV[j + k + 1]
                    pub += '\n'
                    if ('published' in CV[j + k + 1].lower()) or ('submitted' in CV[j + k + 1].lower()) or (
                            'presented at' in CV[j + k + 1].lower()) or ('presentation at' in CV[j + k + 1].lower()):
                        pub_count += 1
                k += 1

        if line[:len('Gender')] == 'Gender':
            gender = line[6:]
        if line[:len('Self Identification:')] == 'Self Identification:':
            self_iden = CV[j+1]
        if line[:len('Misdemeanor Conviction in the United States?')] == 'Misdemeanor Conviction in the United States?':
            misdemeanor = line[len('Misdemeanor Conviction in the United States?'):]
        if line[:len('Felony Conviction in the United States?')] == 'Felony Conviction in the United States?':
            felony = line[len('Felony Conviction in the United States?'):]
        if line == 'Current Visa / Employment Authorization Status:':
            authorization = CV[j+1]


    return pub_count, awards, medical_edu, edu, gender,pub,self_iden,misdemeanor,felony,authorization


def pre_process(data):

    data[['Gender']] = data['Gender'].fillna('no_identified')
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
    data['Gender'] = np.where((data.Gender.str.contains('Female')),
                                        'Female',data.Gender)
    data['Gender'] = np.where((data.Gender.str.contains('Male')),
                                        'Male',data.Gender)

    return data

def predictCV(text):
    pub_count, awards, medical_edu, edu, gender,pub,self_iden,misdemeanor,felony,authorization = parse(text)

    
    pub_corpus_test = text_cleaning([pub])
    awd_corpus_test = text_cleaning([awards])
    edu_corpus_test = text_cleaning([edu])
    mededu_corpus_test = text_cleaning([medical_edu])

    award_vectorizer = joblib.load("../models/awards/vectorizer.pkl")
    awd_test = award_vectorizer.transform(awd_corpus_test)
    model_award = joblib.load("../models/awards/model.pkl")

    #model_award._le = LabelEncoder().fit([0, 1])
    pred_awards = model_award.predict_proba(awd_test)

    edu_vectorizer = joblib.load("../models/education/vectorizer.pkl")
    edu_test = edu_vectorizer.transform(edu_corpus_test)
    model_edu = joblib.load("../models/education/model.pkl")
    model_edu._le = LabelEncoder().fit([0, 1])
    pred_education = model_edu.predict_proba(edu_test)


    mededu_vectorizer = joblib.load("../models/med_education/vectorizer.pkl")
    mededu_test = mededu_vectorizer.transform(mededu_corpus_test)
    model_mededu = joblib.load("../models/med_education/model.pkl")
    model_mededu._le = LabelEncoder().fit([0, 1])
    pred_med_education = model_mededu.predict_proba(mededu_test)


    ## Discrete 
    df = pd.DataFrame(list(zip([gender],[self_iden],[misdemeanor],[felony],[authorization])),
               columns =['Gender', 'self_identification','Misdemeanor','Felony', 'Authorized_to_Work'])
    
    data_discrete = pre_process(df)

    pub_vectorizer = joblib.load("../models/discrete_feat/vectorizer.pkl")

    pub_test = pub_vectorizer.transform(data_discrete)
    pub_test = np.insert(pub_test, len(pub_test[0]), pub_count, axis=1)

    model_pub = joblib.load("../models/discrete_feat/model.pkl")
    model_pub._le = LabelEncoder().fit([0, 1])
    pred_discrete_feat = model_pub.predict_proba(pub_test)

    return [pred_discrete_feat[0][1], pred_education[0][1], pred_med_education[0][1], pred_awards[0][1]]


def metaLeaner_prediction(input_data,save_plot=True):
    final_model=joblib.load("../models/meta_learner/model.pkl")
    plt.switch_backend('Agg') 

    result = final_model.predict(input_data)

    explainer = shap.Explainer(final_model,feature_names=['Personal Statement', 'Discrete features', 'Education', 'Medical Education', 'Awards'])
    shap_values = explainer(input_data)
    shap.plots.bar(shap_values, show=False)

    if save_plot:
        if result[0] == 0:
            plt.savefig('static/try0.png',bbox_inches='tight',dpi=100)
        else:
            plt.savefig('static/try1.png',bbox_inches='tight',dpi=100)

        plt.clf()
        plt.cla()
        plt.close()
        
    return result[0]

if __name__ == '__main__':
    pass
