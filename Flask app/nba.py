from flask import Flask, render_template, url_for, request
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, plot_confusion_matrix, log_loss
from sklearn.ensemble import VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from model import final_model
import pickle

image_folder = os.path.join('static', 'images')

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = image_folder

@app.route('/', methods=['GET', 'POST'])
def home():

    court = os.path.join(app.config['UPLOAD_FOLDER'], 'Court 1 smaller.png')
    return render_template("nba.html", user_image = court)

@app.route('/age', methods=['POST', 'GET'])
def get_more_data():
    
    court = os.path.join(app.config['UPLOAD_FOLDER'], 'Court 1 smaller.png')

    if request.method == 'POST':
        year_choice = request.form['year-form']

        return render_template("nba2.html", year_choice=year_choice, user_image = court)

@app.route('/teams', methods=['POST', 'GET'])
def get_even_more_data():
    
    court = os.path.join(app.config['UPLOAD_FOLDER'], 'Court 1 smaller.png')

    if request.method == 'POST':
        parameters = request.form['age-form']
        year_choice = parameters.split(", ")[1]
        max_age = parameters.split(", ")[0]

        return render_template("nba3.html", year_choice=year_choice, max_age=max_age, user_image = court)

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    
    court = os.path.join(app.config['UPLOAD_FOLDER'], 'Court 1 smaller.png')
    player_circle = os.path.join(app.config['UPLOAD_FOLDER'], 'Circle.png')

    updated_df = os.path.join(app.config['UPLOAD_FOLDER'], 'updated_df.pickle')
    model = os.path.join(app.config['UPLOAD_FOLDER'], 'model.pickle')
    ss = os.path.join(app.config['UPLOAD_FOLDER'], 'scaler.pickle')

    if request.method == 'POST':
        parameters = request.form['team-form']
        year_choice = parameters.split(", ")[2]
        max_age = parameters.split(", ")[1]
        if max_age == "all":
            max_age = 50
        teams = parameters.split(", ")[0]

        df = pickle.load(open(updated_df, "rb"))
        MODEL = pickle.load(open(model, "rb"))
        scaler = pickle.load(open(ss, "rb"))

        top_24 = final_model(MODEL, scaler, df, int(f'20{year_choice[-2:]}'), \
            max_age=int(max_age), teams=teams, first_timers_only=False)

        return render_template("nba_results.html", user_image=court, circle=player_circle, all_stars=top_24)

if __name__ == '__main__':
    app.run(debug=True)

