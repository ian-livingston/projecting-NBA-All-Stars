

def nba_log_regression(df, list_of_feature_columns, target_feature, folds=5, loops=3, print_all=False):
    '''
    Parameters:
    -----------
    df (df): Data frame to pull data from (not split).

    list_of_feature_columns (list of strings): List of columns in df to be used as features.

    target_feature (string): Column in df to be used as target.
    
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
    from sklearn.metrics import classification_report, confusion_matrix, plot_confusion_matrix

    features = list_of_feature_columns
    X = df[features]
    y = df[target_feature]

    # Test-train split
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=10)

    # Scaling training set (for now) and testing set (for later)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
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
    
    if print_all == True:
        print(f'Score on test set ({folds}-fold validation): {np.mean(listed_scores)}\n')
        print(f'Classification report:\n {class_report}\n')
        fig, ax = plt.subplots(dpi=150)
        plot_confusion_matrix(logr, X_test_scaled, y_test, ax=ax, cmap="Oranges")
    
    return score, class_report, con_matrix


# For KNN
def nba_knn(df, list_of_feature_columns, target_feature, k=10, folds=5, loops=3, print_all=False):
    '''
    Parameters:
    -----------
    df (df): Data frame to pull data from (not split).

    list_of_feature_columns (list of strings): List of columns in df to be used as features.

    target_feature (string): Column in df to be used as target.

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
    from sklearn.metrics import classification_report, confusion_matrix, plot_confusion_matrix

    features = list_of_feature_columns
    X = df[features]
    y = df[target_feature]

    # Test-train split
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=10)

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
    
    if print_all == True:
        print(f'Score on test set ({folds}-fold validation): {np.mean(listed_scores)}\n')
        print(f'Classification report:\n {class_report}\n')
        fig, ax = plt.subplots(dpi=150)
        plot_confusion_matrix(knn, X_test_scaled, y_test, ax=ax, cmap="Oranges")
    
    return score, class_report, con_matrix


# For random forest
def nba_random_forest(df, list_of_feature_columns, target_feature, folds=5, loops=3, print_all=False):
    '''
    Parameters:
    -----------
    df (df): Data frame to pull data from (not split).

    list_of_feature_columns (list of strings): List of columns in df to be used as features.

    target_feature (string): Column in df to be used as target.
    
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
    from sklearn.metrics import classification_report, roc_auc_score, roc_curve, confusion_matrix, plot_confusion_matrix

    features = list_of_feature_columns
    X = df[features]
    y = df[target_feature]

    # Test-train split
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=10)

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
    
    if print_all == True:
        print(f'Score on test set ({folds}-fold validation): {np.mean(listed_scores)}\n')
        print(f'Classification report:\n {class_report}\n')
        fig, ax = plt.subplots(dpi=150)
        plot_confusion_matrix(ran_for, X_test, y_test, ax=ax, cmap="Oranges")
    
    return score, class_report, con_matrix


# For SVC
def nba_svc(df, list_of_feature_columns, target_feature, SMOTE=False, RandomOverSampler=False, folds=5, loops=3, print_all=False):
    '''
    Parameters:
    -----------
    df (df): Data frame to pull data from (not split).

    list_of_feature_columns (list of strings): List of columns in df to be used as features.

    target_feature (string): Column in df to be used as target.

    SMOTE (hyperparameter): If True, SMOTE will be applied to data in both train and test sets before 
    modeling. Default value is False. 
    
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
    from sklearn.metrics import classification_report, confusion_matrix, plot_confusion_matrix
    from sklearn.svm import SVC
    from imblearn.over_sampling import SMOTE
    from imblearn.over_sampling import RandomOverSampler

    features = list_of_feature_columns
    X = df[features]
    y = df[target_feature]

    # Test-train split
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=10)

    # Scaling training set (for now) and testing set (for later)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    if SMOTE == True:
        X_train_scaled, y_train = SMOTE(random_state=10).fit_sample(X_train_scaled, y_train)

    elif RandomOverSampler == True:
        ros = RandomOverSampler(random_state=10)
        X_train_scaled, y_train = ros.fit_sample(X_train_scaled, y_train)

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
    
    if print_all == True:
        print(f'Score on test set ({folds}-fold validation): {np.mean(listed_scores)}\n')
        print(f'Classification report:\n {class_report}\n')
        fig, ax = plt.subplots(dpi=150)
        plot_confusion_matrix(svc, X_test_scaled, y_test, ax=ax, cmap="Oranges")
    
    return score, class_report, con_matrix