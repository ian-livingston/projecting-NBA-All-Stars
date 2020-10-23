

def nba_log_regression(df, list_of_feature_columns, target_feature, RandomOverSampler=False, RandomUnderSampler=False, sample=0.5, folds=5, loops=3, print_all=False):
    '''
    Parameters:
    -----------
    df (df): Data frame to pull data from (not split).

    list_of_feature_columns (list of strings): List of columns in df to be used as features.

    target_feature (string): Column in df to be used as target.

    RandomOverSampler (hyperparameter): If True, minority class will be oversampled via RandomOverSampler. Default value is False.

    RandomUnderSampler (hyperparameter): If True, majority class will be undersampled via RandomUnderSampler. Default value is False.

    sample (float): Float representing desired proportion of majority/minority to over/undersample data to. Default value is .5.
    
    folds (int): Number of n_splits used by KFold. Default value is 5.
    
    loops (int): Number of times to run cross_val_score using a different 
    random state (which begin at 0 and increment up to the value of loops).
        
    print_all: If True, function prints the list of all r2 scores as well. 
    Default value is False.
    
    Returns:
    --------
    score (float): F1 score (cross-validated)
    
    class_report (array): Classification report
    
    con_matrix (array): Confusion matrix (will be plotted if print_all=True)
    '''

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split, KFold, cross_val_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report, confusion_matrix, plot_confusion_matrix, log_loss
    from imblearn.over_sampling import RandomOverSampler
    from imblearn.under_sampling import RandomUnderSampler

    features = list_of_feature_columns
    X = df[features]
    y = df[target_feature]

    # Test-train split
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=10)

    # Oversampling on training set (if called by parameter)
    if RandomOverSampler == True:
        ros = RandomOverSampler(sampling_strategy=sample, random_state=10)
        X_train, y_train = ros.fit_sample(X_train, y_train)

    # Undersampling on training set (if called by parameter)
    elif RandomUnderSampler == True:
        rus = RandomUnderSampler(sampling_strategy=sample, random_state=10)
        X_train, y_train = rus.fit_sample(X_train, y_train)

    # Scaling training set (for now) and testing set (for later)
    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Instantiating and fitting
    logr = LogisticRegression()
    logr.fit(X_train_scaled, y_train)
    logr_predictions = logr.predict(X_test_scaled)
    logr_tn, logr_fp, logr_fn, logr_tp = confusion_matrix(y_test, logr_predictions).ravel()
    logr_prob_predictions = logr.predict_proba(X_test_scaled)
    logr_prob_dict = dict(zip(list(X_test.index), list(logr_prob_predictions)))

    # Cross-validating
    listed_scores = []
    for i in range(loops):
        kf = KFold(n_splits=folds, shuffle=True, random_state=i)
        listed_scores.extend(cross_val_score(logr, X_test_scaled, y_test, cv=kf, scoring="f1_weighted"))
    
    # Returns
    score = np.mean(listed_scores)
    class_report = classification_report(y_test, logr_predictions)
    con_matrix = confusion_matrix(y_test, logr_predictions)
    loss = log_loss(y_test, logr_predictions)
    
    if print_all == True:
        print(f'Score on test set ({folds}-fold validation): {np.mean(listed_scores)}\n')
        print(f'Log loss: {loss}\n')
        print(f'Classification report:\n {class_report}\n')
        fig, ax = plt.subplots(dpi=150)
        plot_confusion_matrix(logr, X_test_scaled, y_test, ax=ax, cmap="Oranges")
    
    return score, loss, class_report, con_matrix, scaler, logr


# For KNN
def nba_knn(df, list_of_feature_columns, target_feature, RandomOverSampler=False, RandomUnderSampler=False, sample=0.5, k=10, folds=5, loops=3, print_all=False):
    '''
    Parameters:
    -----------
    df (df): Data frame to pull data from (not split).

    list_of_feature_columns (list of strings): List of columns in df to be used as features.

    target_feature (string): Column in df to be used as target.

    RandomOverSampler (hyperparameter): If True, minority class will be oversampled via RandomOverSampler. Default value is False.

    RandomUnderSampler (hyperparameter): If True, majority class will be undersampled via RandomUnderSampler. Default value is False.

    sample (float): Float representing desired proportion of majority/minority to over/undersample data to. Default value is .5.

    k (int): Number of folds in KFold cross-validation.
    
    folds (int): Number of n_splits used by KFold. Default value is 5.
    
    loops (int): Number of times to run cross_val_score using a different 
    random state (which begin at 0 and increment up to the value of loops).
        
    print_all: If True, function prints the list of all r2 scores as well. 
    Default value is False.
    
    Returns:
    --------
    score (float): F1 score (cross-validated)
    
    class_report (array): Classification report
    
    con_matrix (array): Confusion matrix (will be plotted if print_all=True)
    '''

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split, KFold, cross_val_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
    from sklearn.metrics import classification_report, confusion_matrix, plot_confusion_matrix, log_loss
    from imblearn.over_sampling import RandomOverSampler
    from imblearn.under_sampling import RandomUnderSampler

    features = list_of_feature_columns
    X = df[features]
    y = df[target_feature]

    # Test-train split
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=10)

    # Oversampling on training set (if called by parameter)
    if RandomOverSampler == True:
        ros = RandomOverSampler(sampling_strategy=sample, random_state=10)
        X_train, y_train = ros.fit_sample(X_train, y_train)

    # Undersampling on training set (if called by parameter)
    elif RandomUnderSampler == True:
        rus = RandomUnderSampler(sampling_strategy=sample, random_state=10)
        X_train, y_train = rus.fit_sample(X_train, y_train)

    # Scaling training set (for now) and testing set (for later)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Instantiating and fitting
    knn = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)
    knn.fit(X_train_scaled, y_train)
    knn_predictions = knn.predict(X_test_scaled)
    knn_prob_predictions = knn.predict_proba(X_test_scaled)
 
    # Cross-validating
    listed_scores = []
    for i in range(loops):
        kf = KFold(n_splits=folds, shuffle=True, random_state=i)
        listed_scores.extend(cross_val_score(knn, X_test_scaled, y_test, cv=kf, scoring="f1_weighted"))
    
    # Returns
    score = np.mean(listed_scores)
    class_report = classification_report(y_test, knn_predictions)
    con_matrix = confusion_matrix(y_test, knn_predictions)
    loss = log_loss(y_test, knn_predictions)
    
    if print_all == True:
        print(f'Score on test set ({folds}-fold validation): {np.mean(listed_scores)}\n')
        print(f'Log loss: {loss}\n')
        print(f'Classification report:\n {class_report}\n')
        fig, ax = plt.subplots(dpi=150)
        plot_confusion_matrix(knn, X_test_scaled, y_test, ax=ax, cmap="Oranges")
    
    return score, loss, class_report, con_matrix


# For random forest
def nba_random_forest(df, list_of_feature_columns, target_feature, RandomOverSampler=False, RandomUnderSampler=False, sample=0.5, folds=5, loops=3, print_all=False):
    '''
    Parameters:
    -----------
    df (df): Data frame to pull data from (not split).

    list_of_feature_columns (list of strings): List of columns in df to be used as features.

    target_feature (string): Column in df to be used as target.

    RandomOverSampler (hyperparameter): If True, minority class will be oversampled via RandomOverSampler. Default value is False.

    RandomUnderSampler (hyperparameter): If True, majority class will be undersampled via RandomUnderSampler. Default value is False.

    sample (float): Float representing desired proportion of majority/minority to over/undersample data to. Default value is .5.
    
    folds (int): Number of n_splits used by KFold. Default value is 5.
    
    loops (int): Number of times to run cross_val_score using a different 
    random state (which begin at 0 and increment up to the value of loops).
        
    print_all: If True, function prints the list of all r2 scores as well. 
    Default value is False.
    
    Returns:
    --------
    score (float): F1 score (cross-validated)
    
    class_report (array): Classification report
    
    con_matrix (array): Confusion matrix (will be plotted if print_all=True)
    '''

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split, KFold, cross_val_score
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import classification_report, confusion_matrix, plot_confusion_matrix, log_loss
    from imblearn.over_sampling import RandomOverSampler
    from imblearn.under_sampling import RandomUnderSampler

    features = list_of_feature_columns
    X = df[features]
    y = df[target_feature]

    # Test-train split
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=10)

    # Oversampling on training set (if called by parameter)
    if RandomOverSampler == True:
        ros = RandomOverSampler(sampling_strategy=sample, random_state=10)
        X_train, y_train = ros.fit_sample(X_train, y_train)
    
    # Undersampling on training set (if called by parameter)
    elif RandomUnderSampler == True:
        rus = RandomUnderSampler(sampling_strategy=sample, random_state=10)
        X_train, y_train = rus.fit_sample(X_train, y_train)

    # Instantiating and fitting
    ran_for = RandomForestClassifier()
    ran_for.fit(X_train, y_train)
    ran_for_predictions = ran_for.predict(X_test)
    ran_for_prob_predictions = ran_for.predict_proba(X_test)
    
    # Cross-validating
    listed_scores = []
    for i in range(loops):
        kf = KFold(n_splits=folds, shuffle=True, random_state=i)
        listed_scores.extend(cross_val_score(ran_for, X_test, y_test, cv=kf, scoring="f1_weighted"))
    
    # Returns
    score = np.mean(listed_scores)
    class_report = classification_report(y_test, ran_for_predictions)
    con_matrix = confusion_matrix(y_test, ran_for_predictions)
    loss = log_loss(y_test, ran_for_predictions)
    
    if print_all == True:
        print(f'Score on test set ({folds}-fold validation): {np.mean(listed_scores)}\n')
        print(f'Log loss: {loss}\n')
        print(f'Classification report:\n {class_report}\n')
        fig, ax = plt.subplots(dpi=150)
        plot_confusion_matrix(ran_for, X_test, y_test, ax=ax, cmap="Oranges")
    
    return score, loss, class_report, con_matrix


# For SVC
def nba_svc(df, list_of_feature_columns, target_feature, SMOTE=False, RandomOverSampler=False, RandomUnderSampler=False, sample=0.5, folds=5, loops=3, print_all=False):
    '''
    Parameters:
    -----------
    df (df): Data frame to pull data from (not split).

    list_of_feature_columns (list of strings): List of columns in df to be used as features.

    target_feature (string): Column in df to be used as target.

    SMOTE (hyperparameter): If True, SMOTE will be applied to data in both train and test sets before 
    modeling. Default value is False.

    RandomOverSampler (hyperparameter): If True, minority class will be oversampled via RandomOverSampler. Default value is False.

    RandomUnderSampler (hyperparameter): If True, majority class will be undersampled via RandomUnderSampler. Default value is False.

    sample (float): Float representing desired proportion of majority/minority to over/undersample data to. Default value is .5.
    
    folds (int): Number of n_splits used by KFold. Default value is 5.
    
    loops (int): Number of times to run cross_val_score using a different 
    random state (which begin at 0 and increment up to the value of loops).
        
    print_all: If True, function prints the list of all r2 scores as well. 
    Default value is False.
    
    Returns:
    --------
    score (float): F1 score (cross-validated)
    
    class_report (array): Classification report
    
    con_matrix (array): Confusion matrix (will be plotted if print_all=True)
    '''

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split, KFold, cross_val_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import classification_report, confusion_matrix, plot_confusion_matrix, log_loss
    from sklearn.svm import SVC
    from imblearn.over_sampling import SMOTE
    from imblearn.over_sampling import RandomOverSampler
    from imblearn.under_sampling import RandomUnderSampler

    features = list_of_feature_columns
    X = df[features]
    y = df[target_feature]

    # Test-train split
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=10)

    # Applying on training set (if called by parameter)
    if SMOTE == True:
        X_train, y_train = SMOTE(random_state=10).fit_sample(X_train, y_train)

    # Oversampling on training set (if called by parameter)
    elif RandomOverSampler == True:
        ros = RandomOverSampler(sampling_strategy=sample, random_state=10)
        X_train, y_train = ros.fit_sample(X_train, y_train)

    # Undersampling on training set (if called by parameter)
    elif RandomUnderSampler == True:
        rus = RandomUnderSampler(sampling_strategy=sample, random_state=10)
        X_train, y_train = rus.fit_sample(X_train, y_train)

    # Scaling training set (for now) and testing set (for later)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Instantiating and fitting
    svc = SVC()
    svc.fit(X_train_scaled, y_train)
    svc_predictions = svc.predict(X_test_scaled)
    
    # Cross-validating
    listed_scores = []
    for i in range(loops):
        kf = KFold(n_splits=folds, shuffle=True, random_state=i)
        listed_scores.extend(cross_val_score(svc, X_test_scaled, y_test, cv=kf, scoring="f1_weighted"))
    
    # Returns
    score = np.mean(listed_scores)
    class_report = classification_report(y_test, svc_predictions)
    con_matrix = confusion_matrix(y_test, svc_predictions)
    loss = log_loss(y_test, svc_predictions)
    
    if print_all == True:
        print(f'Score on test set ({folds}-fold validation): {np.mean(listed_scores)}\n')
        print(f'Log loss: {loss}\n')
        print(f'Classification report:\n {class_report}\n')
        fig, ax = plt.subplots(dpi=150)
        plot_confusion_matrix(svc, X_test_scaled, y_test, ax=ax, cmap="Oranges")
    
    return score, loss, class_report, con_matrix




def nba_logr_model_predict(df, list_of_feature_columns, target_feature, max_age=50, RandomOverSampler=False, RandomUnderSampler=False, sample=0.5, folds=5, loops=3, print_all=False):
    '''
    Parameters:
    -----------
    df (df): Data frame to pull data from (not split).

    list_of_feature_columns (list of strings): List of columns in df to be used as features.

    target_feature (string): Column in df to be used as target.

    season_to_predict_on (int): The season (format: YYYY for year when season ends) on which the model will be run after fitting.

    RandomOverSampler (hyperparameter): If True, minority class will be oversampled via RandomOverSampler. Default value is False.

    RandomUnderSampler (hyperparameter): If True, majority class will be undersampled via RandomUnderSampler. Default value is False.

    sample (float): Float representing desired proportion of majority/minority to over/undersample data to. Default value is .5.
    
    folds (int): Number of n_splits used by KFold. Default value is 5.
    
    loops (int): Number of times to run cross_val_score using a different 
    random state (which begin at 0 and increment up to the value of loops).
        
    print_all: If True, function prints the list of all r2 scores as well. 
    Default value is False.
    
    Returns:
    --------
    score (float): F1 score (cross-validated)
    
    class_report (array): Classification report
    
    con_matrix (array): Confusion matrix (will be plotted if print_all=True)
    '''

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, plot_confusion_matrix, log_loss
    import time
    import random

    # Fitting a logr model and returning it
    score, loss, class_report, con_matrix, scaler, MODEL = nba_log_regression(df, list_of_feature_columns, target_feature, RandomOverSampler=RandomOverSampler, RandomUnderSampler=RandomUnderSampler, sample=sample, folds=folds, loops=loops, print_all=False)

    # Picking a full season of data to test on
    possible_seasons = [2000, 2001, 2002, 2010, 2011, 2012, 2013, 2014, 2015, 2008, 2009, 2003, 2004, 2005, 2006, 2007]
    test_season = random.choice(possible_seasons)
    print(f'Test season: {test_season}')
    time.sleep(3)
    test_season_df = df.loc[(df["Year"] == test_season) & (df["Age"] <= max_age)]
    X = test_season_df[list_of_feature_columns]
    y = test_season_df[target_feature]

    # Scaling based on fit from training/testing
    X_scaled = scaler.transform(X)

    # Generating predictions on test season
    predictions = MODEL.predict(X_scaled)
    prob_predictions = MODEL.predict_proba(X_scaled)
    prob_dict = dict(zip(list(X.index), list(prob_predictions)))

    eligible_players = [[player, list(chance)[1]] for player, chance in prob_dict.items()]
    top_24 = [[player, chance] for player, chance in sorted(eligible_players, key=lambda item: item[1], reverse=True)][:24]
    print(f'Top 24 most likely {test_season+1} NBA All-Stars:\n')
    for player in top_24:
        guy = player[0].split(", ")[0]
        percentage_chance = str(round((player[1] * 100), 2))
        if test_season_df.loc[test_season_df["Player"] == guy]["All-Star next season?"].values[0] == 1:
            real = f'Was a real {test_season} All-Star'
        else:
            real = f'Was not a real {test_season} All-Star'
        print(f'{guy}: {percentage_chance}% ...... ({real})\n')

    fig, ax = plt.subplots(dpi=150)
    plot_confusion_matrix(MODEL, X_scaled, y, ax=ax, cmap="Oranges")
    #print(f'Overall accuracy: {accuracy_score(y, predictions)}')
    #print(f'Precision: {precision_score(y, predictions, average="weighted")}')
    #print(f'Recall: {recall_score(y, predictions, average="weighted")}')
    #print(f'F!: {f1_score(y, predictions, average="weighted")}')
    