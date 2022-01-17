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
```bash
python3 app/src/download_models.py
```


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
