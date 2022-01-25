# An Automated NLP Tool to Rank Applications for Diagnostic Radiology Residency: Utility for Understanding Elements Associated with Selection for Interview

<!-- TOC --> 

- [Instruction Navigation](#an-automated-nlp-tool-to-rank-applications-for-diagnostic-radiology-residency)
    - [Publication](#publication)
    - [Web System Application](#web-system-application)
    - [Late Fusion Arquitecture and Experiment Results](#late-fusion-arquitecture-and-experiment-results)
    - [Installation](#installation)
    - [Download Models](#download-models)
    - [Demo App](#demo-app)
    - [Contributors](#contributors)

<!-- /TOC -->


## Publication

This is the code repository for the paper: An Automated NLP Tool to Rank Applications for Diagnostic Radiology Residency: Utility for Understanding Elements Associated with Selection for Interview
	
## Web System Application 

We developed a web system application for users to test our proposed pipilne for ranking Applications for Diagnostic Radiology Residency. An example of our Web System is illustraded bellow:

<table border=1>
<tr align='center' > 
<td><img src="https://github.com/thiagosantos1/Residency_AI/blob/main/imgs/web_b.png" width="500"                  title="NLP Ranking Tool"></td>         
<td><img src="https://github.com/thiagosantos1/Residency_AI/blob/main/imgs/web_a.png" width="500" title="NLP Ranking Tool"></td>
</tr>
</table>

## Late Fusion Arquitecture and Experiment Results
<table border=1>
<tr align='center' > 
<td><img src="https://github.com/thiagosantos1/Residency_AI/blob/main/imgs/pipeline.png" width="500"        title="Late Fusion Arquitecture"></td>         
<td><img src="https://github.com/thiagosantos1/Residency_AI/blob/main/imgs/meta_learner.png" width="400" title="Experiment Results"></td>
</tr>
</table>


## Installation

We recommend using a virtual environment

If you do not already have `conda` installed, you can install Miniconda from [this link](https://docs.conda.io/en/latest/miniconda.html#linux-installers) (~450Mb). Then, check that conda is up to date:

```bash
conda update -n base -c defaults conda
```

And create a fresh conda environment (~220Mb):

```bash
conda create python=3.8 --name=nlp_ranking
```

If not already activated, activate the new conda environment using:

```bash
conda activate nlp_ranking
```

Then install the following packages (~3Gb):

```bash
conda install pytorch cudatoolkit=10.2 -c pytorch
```

```bash
conda install pytorch torchvision torchaudio -c pytorch
```

#### Install Other Dependencies with pip:

```bash
pip3 install -r requirements.txt
```

### Mac Users - Install libomp 11.0

```bash
wget https://raw.githubusercontent.com/chenrui333/homebrew-core/0094d1513ce9e2e85e07443b8b5930ad298aad91/Formula/libomp.rb
```

```bash
brew unlink libomp
```

```bash
brew install --build-from-source ./libomp.rb
```

#### Check that version 11.1 is installed 

```bash
brew list --version libomp
```

### Download Models
For privacy reasons, the trained model weights can only be shared upon request. 

Please email me at: tpsanto@emory.edu with the following information:

Name:

Afiliation:

How are you intending to use the tool (Research or Comercial?):


## Training Pipeline
We also provide the code for all individual branch training as well as the Meta-Learner. You can find the code inside of folder src/training.

Example Script for training BERT Classifier

```shell
cd src/training/BERT

python3 main.py --path_train "train.csv" --path_test "test.csv" --execution train --model_type "bio_clinicalbert" --n_epocs 6
```

|Input Option|Available Options            |
|------------|-----------------------------|
|execution   | train, test                 |
|do_stopwords| True, False                 |
|cuda        | True False                  |
|model_type  | bert, roberta, distilbert,distilroberta,electra, biobert, bio_clinicalbert, biomednlp|
|seed        | int                         |
|n_epocs     | int                         |
|n_batches   | int                         |
|save_steps  | int                         |
|evaluate_during_training_steps| int                         |
|max_seq_length| int                         |
|early_stopping_patience| int                         |
|l_rate      | float                       |
|decay       | float                       |
|early_stopping_delta| float                       |
|path_train  | path_train                  |
|path_test   | path_test                   |



Example Script for training each branch 
```shell
cd src/training/Classifiers

python3 main.py --path_train "train.csv" --path_test "test.csv" --execution train --model_type xgbost --data_type education
```

|Input Option|Available Options            |
|------------|-----------------------------|
|execution   | train, test                 |
|model_type  | logistic, sgb, rf, xgbost, knn, gb|
|data_type        | personal_statement, discrete_feat, education, med_education, award, all |
|seed        | int                         |
|path_train  | path_train                  |
|path_test   | path_test                   |


Example Script for training meta-learner
```shell
cd src/training/Meta-Learner

python3 main.py --path_train "train.csv" --path_test "test.csv" --execution train --path_bert_model "path_to_bert" --path_award_model "path_to_award" ....
```

|Input Option|Available Options            |
|------------|-----------------------------|
|execution   | train, test                 |
|model_type  | logistic, sgb, rf, xgbost, knn, gb|
|seed        | int                         |
|path_bert_model  | path_bert_model                  |
|path_discrete_feat_model  | path_discrete_feat_model                  |
|path_education_model  | path_education_model                  |
|path_med_education_model  | path_med_education_model                  |
|path_award_model  | path_award_model                  |
|path_train  | path_train                  |
|path_test   | path_test                   |

### Data Format 
In order to train the pipeline on your own data, please follow the data input format provided in the examples inside the folder training/example_data. Format should be as follow:

|interview|PS                           |gender|self_identification                          |Misdemeanor|Felony|Authorized_to_Work|Medical_Education|Education|Awards   |Certification_Licensure|Publications|Publications_Count|year|
|---------|-----------------------------|------|---------------------------------------------|-----------|------|------------------|-----------------|---------|---------|-----------------------|------------|------------------|----|
|1        |Free Text                    |Female|  White                                      | No        | No   |No                |Free Text        |Free Text|Free Text|Free Text              |Free Text   |1                 |2016|
|1        |Free Text                    |Male  |  Asian                                      | No        | No   |Yes               |Free Text        |Free Text|Free Text|Free Text              |Free Text   |5                 |2019|
|0        |Free Text                    |Male  |  Asian                                      | No        | No   |Yes               |Free Text        |Free Text|Free Text|Free Text              |Free Text   |0                 |2018|


## Demo app

A minimal demo app is provided for you to play with the classification model!

<table border=1>
<tr align='center' > 
<td><img src="https://github.com/thiagosantos1/Residency_AI/blob/main/imgs/web_b.png" width="500"                  title="NLP Ranking Tool"></td>         
</table>

You can easily run your app in your default browser by running:

```shell
cd src
python3 app.py >/dev/null 2>&1
```

## Contributors

Ms. Thiago Santos

    Rishav Dhar

Dr. Amara Tariq

Dr. Imon Banerjee
