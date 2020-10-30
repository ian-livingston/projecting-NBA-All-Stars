def final_model(MODEL, scaler, df, year, max_age=50, teams="all", first_timers_only=False):

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
    import time
    import random

    if teams != "all":
        prediction_year_df = df.loc[(df["Year"] == year) & (df["Age"] <= max_age) & (df["Tm"] == teams)]
    else:
        prediction_year_df = df.loc[(df["Year"] == year) & (df["Age"] <= max_age)]

    features = ["All-Star?", "PTS/game", "AST/game", "Years from prime", "PER", "Trajectory", "Adjusted TV market value * GS", "TRB/game", "PTS+AST/game", "MP/game", "FT/game", "0", "1-3", "4-7", "8+"]
    
    if first_timers_only == True:
        prediction_year_df = df.loc[(df["Year"] == year) & (df["Age"] <= max_age) & (df["Past All-Star Games (incl this season)"] == 0)]

    X = prediction_year_df[features]
    y = prediction_year_df["All-Star next season?"]
    X_scaled = scaler.transform(X)

    predictions = MODEL.predict(X_scaled)
    prob_predictions = MODEL.predict_proba(X_scaled)
    prob_dict = dict(zip(list(X.index), list(prob_predictions)))

    eligible_players = [[player, list(chance)[1]] for player, chance in prob_dict.items()]
    if len(eligible_players) < 24:
        top_24 = [[player.split(", ")[0].upper(), f'{round((chance * 100), 1)}%'] for player, chance in sorted(eligible_players, key=lambda item: item[1], reverse=True)]
    else:
        top_24 = [[player.split(", ")[0].upper(), f'{round((chance * 100), 1)}%'] for player, chance in sorted(eligible_players, key=lambda item: item[1], reverse=True)][:24]
    
    return top_24