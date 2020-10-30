

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
def nba_knn(df, list_of_feature_columns, target_feature, RandomOverSampler=False, RandomUnderSampler=False, sample=0.5, k=9, folds=5, loops=3, print_all=False):
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
    from sklearn.neighbors import KNeighborsClassifier
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
    
    return score, loss, class_report, con_matrix, scaler, knn


# For random forest
def nba_random_forest(df, list_of_feature_columns, target_feature, max_depth=6, min_samples=9, RandomOverSampler=False, RandomUnderSampler=False, sample=0.5, folds=5, loops=3, print_all=False):
    '''
    Parameters:
    -----------
    df (df): Data frame to pull data from (not split).

    list_of_feature_columns (list of strings): List of columns in df to be used as features.

    target_feature (string): Column in df to be used as target.

    max_depth (int): Maximum depth of each split.

    min_samples (int): Min_samples_split parameter in RandomForest() model.

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
    ran_for = RandomForestClassifier(max_depth=6, min_samples_split=min_samples)
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
    
    return score, loss, class_report, con_matrix, ran_for


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
    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
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
    
    return score, loss, class_report, con_matrix, scaler, svc


# For making predictions on a single season using logistic regression
def nba_logr_model_predict(df, list_of_feature_columns, target_feature, max_age=50, print_all=False):
    '''
    Parameters:
    -----------
    df (df): Data frame to pull data from (not split).

    list_of_feature_columns (list of strings): List of columns in df to be used as features.

    target_feature (string): Column in df to be used as target.

    max_age (int): Max age to use as player filter
        
    print_all: If True, function prints the list of all r2 scores as well. 
    Default value is False.
    
    Returns:
    --------
    (none)

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
            real = f'WAS a real {test_season} All-Star'
        else:
            real = f'Was NOT a real {test_season} All-Star'
        print(f'{guy}: {percentage_chance}% ...... ({real})\n')

    fig, ax = plt.subplots(dpi=150)
    plot_confusion_matrix(MODEL, X_scaled, y, ax=ax, cmap="Oranges")
    #print(f'Overall accuracy: {accuracy_score(y, predictions)}')
    #print(f'Precision: {precision_score(y, predictions, average="weighted")}')
    #print(f'Recall: {recall_score(y, predictions, average="weighted")}')
    #print(f'F!: {f1_score(y, predictions, average="weighted")}')
    

# For ensembling
def nba_ensemble(df, list_of_feature_columns, target_feature, k=10, max_depth=6, min_samples=9, print_all=False, plot=False):
    '''
    Parameters:
    -----------
    
    df (df): Data frame to pull data from (not split).

    list_of_feature_columns (list of strings): List of columns in df to be used as features.

    target_feature (string): Column in df to be used as target.

    k (int): Number of folds in KFold cross-validation.

    min_samples (int): Min_samples_split parameter in RandomForest() model.

    print_all (bool): If True, prints scores for training and test sets for both "hard" and "soft"
    VotingClassifier models. Default value is False.
        
    plot (bool): If True, plots a confusion matrix based on predictions made using the testing set.
    Default value is False.
    
    Returns:
    --------
    scaler: fit instance of StandardScaler
    
    voting_classifier_soft: fit instance of VotingClassifier model
    
    '''

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import VotingClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, plot_confusion_matrix, log_loss

    X = df[list_of_feature_columns]
    y = df[target_feature]

    # Test-train split and scaling
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=10)
    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Instantiating contributing models and adding them to a list
    list_of_models = [("logr", LogisticRegression()), (f'KNN {k}', KNeighborsClassifier(n_neighbors=k, n_jobs=-1)), ("ran_for", RandomForestClassifier(min_samples_split=min_samples, max_depth=max_depth)), ("svc_model", SVC(probability=True))]

    # Instantiating VotingClassifier (hard) + training, testing, scoring
    voting_classifier_hard = VotingClassifier(estimators=list_of_models, voting='hard', n_jobs=-1)
    voting_classifier_hard.fit(X_train_scaled, y_train)
    hard_predictions = voting_classifier_hard.predict(X_train_scaled)
    if print_all == True:
        print(f'TRAIN ("hard" voting): Overall accuracy: {accuracy_score(y_train, hard_predictions)}')
        print(f'TRAIN ("hard" voting): Precision: {precision_score(y_train, hard_predictions)}')
        print(f'TRAIN ("hard" voting): Recall: {recall_score(y_train, hard_predictions)}')
        print(f'TRAIN ("hard" voting): F1: {f1_score(y_train, hard_predictions)}')
        print(f'TRAIN ("hard" voting): Log loss: {log_loss(y_train, hard_predictions)}\n')
    hard_test_predictions = voting_classifier_hard.predict(X_test_scaled)
    if print_all == True:
        print(f'TEST ("hard" voting): Overall accuracy: {accuracy_score(y_test, hard_test_predictions)}')
        print(f'TEST ("hard" voting): Precision: {precision_score(y_test, hard_test_predictions)}')
        print(f'TEST ("hard" voting): Recall: {recall_score(y_test, hard_test_predictions)}')
        print(f'TEST ("hard" voting): F1: {f1_score(y_test, hard_test_predictions)}')
        print(f'TEST ("hard" voting): Log loss: {log_loss(y_test, hard_test_predictions)}\n')

    # Instantiating VotingClassifier (soft)
    voting_classifier_soft = VotingClassifier(estimators=list_of_models, voting='soft', n_jobs=-1)
    voting_classifier_soft.fit(X_train_scaled, y_train)
    soft_predictions = voting_classifier_soft.predict(X_train_scaled)
    soft_prob_predictions = voting_classifier_soft.predict_proba(X_train_scaled)
    soft_prob_dict = dict(zip(list(X_train.index), list(soft_prob_predictions)))
    if print_all == True:
        print(f'TRAIN ("soft" voting): Overall accuracy: {accuracy_score(y_train, soft_predictions)}')
        print(f'TRAIN ("soft" voting): Precision: {precision_score(y_train, soft_predictions)}')
        print(f'TRAIN ("soft" voting): Recall: {recall_score(y_train, soft_predictions)}')
        print(f'TRAIN ("soft" voting): F1: {f1_score(y_train, soft_predictions)}')
        print(f'TRAIN ("soft" voting): Log loss: {log_loss(y_train, soft_predictions)}\n')
    soft_test_predictions = voting_classifier_soft.predict(X_test_scaled)
    soft_test_prob_predictions = voting_classifier_soft.predict_proba(X_test_scaled)
    soft = dict(zip(list(X_test.index), list(soft_test_prob_predictions)))
    if print_all == True:
        print(f'TEST ("soft" voting): Overall accuracy: {accuracy_score(y_test, soft_test_predictions)}')
        print(f'TEST ("soft" voting): Precision: {precision_score(y_test, soft_test_predictions)}')
        print(f'TEST ("soft" voting): Recall: {recall_score(y_test, soft_test_predictions)}')
        print(f'TEST ("soft" voting): F1: {f1_score(y_test, soft_test_predictions)}')
        print(f'TEST ("soft" voting): Log loss: {log_loss(y_test, soft_test_predictions)}\n')
     
    if plot == True:
        fig, ax = plt.subplots(dpi=150)
        plot_confusion_matrix(voting_classifier_hard, X_test_scaled, y_test, ax=ax, cmap="Oranges")
        #plot_decision_regions(X_train.values, y_train.values, voting_classifier)
        #plt.title('Max voting classifier')
        #plt.xlabel('Feature 1')
        #plt.ylabel('Feature 2')
        #plt.gcf().set_size_inches(12,8)

    return scaler, voting_classifier_soft


# For making predictions on a single season using ensemble
def nba_ensemble_predict(df, list_of_feature_columns, target_feature, max_age=50, k=9, print_all=False, plot=False):
    '''
    Parameters:
    -----------
    df (df): Data frame to pull data from (not split).

    list_of_feature_columns (list of strings): List of columns in df to be used as features.

    target_feature (string): Column in df to be used as target.

    max_age (int): Max age to use as player filter
    
    print_all (bool): If True, prints scores for training and test sets for both "hard" and "soft"
    VotingClassifier models. Default value is False.
        
    plot: If True, plots a confusion matrix based on predictions made using the testing set.
    Default value is False.
    
    Returns:
    --------
    (none)

    '''

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

    # Fitting an ensemble model and returning it
    scaler, MODEL = nba_ensemble(df, list_of_feature_columns, target_feature, k=9, print_all=print_all, plot=False)

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
            real = f'WAS a real {test_season} All-Star'
        else:
            real = f'Was NOT a real {test_season} All-Star'
        print(f'{guy}: {percentage_chance}% ...... ({real})\n')

    fig, ax = plt.subplots(dpi=150)
    plot_confusion_matrix(MODEL, X_scaled, y, ax=ax, cmap="Oranges")

# For Flask site
def final_model(MODEL, scaler, df, year, max_age=50, teams="all", position="all", first_timers_only=False):

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
    top_24 = [[player.split(", ")[0], chance] for player, chance in sorted(eligible_players, key=lambda item: item[1], reverse=True)][:24]
    
    return top_24