from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

def extra_trees_classifier(X_train, y_train, X_test, y_test):
    from sklearn.ensemble import ExtraTreesClassifier
    

    clf = ExtraTreesClassifier(n_estimators=80, random_state=42, bootstrap=True)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    
    return clf, report, cm



def logistic_regression_classifier(X_train, y_train, X_test, y_test):
    from sklearn.linear_model import LogisticRegression

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    clf = LogisticRegression(random_state=42, solver="saga", penalty = "l1", C = 0.01, class_weight = {0: 0.9, 1: 1})
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    
    return clf, report, cm



def randomforest_classifier(X_train, y_train, X_test, y_test):
    from sklearn.ensemble import RandomForestClassifier
    
    clf = RandomForestClassifier(random_state=42, criterion = "entropy", n_estimators=70)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    
    return clf, report, cm