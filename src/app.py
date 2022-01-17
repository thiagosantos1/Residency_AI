from flask import Flask, render_template, request
from branch_predict import *
import numpy as np
from bertPredict import *

app = Flask(__name__)


@app.route('/')
def main():
    return render_template("index.html")


@app.route('/predict', methods=["POST"])
def predict():
    if request.method == "POST":
        text_ps = request.form['personal statement']
        text_cv = request.form['cv']
        
        predict_cv = predictCV(text_cv)
        predict_ps = BERT_predict(text_ps,save_lime=True)

        input_probs = [[predict_ps, predict_cv[0], predict_cv[1], predict_cv[2], predict_cv[3]]]

        output = metaLeaner_prediction(np.asarray(input_probs),save_plot=True)
        return render_template('result.html', classification=output)


if __name__ == "__main__":
    app.run(debug=False)
