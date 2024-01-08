import pandas as pd
from data_extraction import extract_data
from data_preprocessing import pre_processing_data


from sklearn.model_selection import train_test_split
from algo import logistic_regression_classifier, extra_trees_classifier, randomforest_classifier



def main():
    df = extract_data()
    
    X_train, y_train, X_test, y_test = pre_processing_data(df)


    
    clf_xtra, report_xtra, cm_xtra = extra_trees_classifier(X_train, y_train, X_test, y_test)
    print("Classification Report:\n", report_xtra)
    print("Confusion Matrix:\n", cm_xtra)
    
    clf_log, report_log, cm_log = logistic_regression_classifier(X_train, y_train, X_test, y_test)
    print("Classification Report:\n", report_log)
    print("Confusion Matrix:\n", cm_log)

    clf_rand, report_rand, cm_rand = randomforest_classifier(X_train, y_train, X_test, y_test)
    print("Classification Report:\n", report_rand)
    print("Confusion Matrix:\n", cm_rand)
    
    
    feature_importances = clf_rand.feature_importances_
    df_feature_importance = pd.DataFrame({'Feature': X_train.columns, 'Importance': feature_importances})
    df_feature_importance = df_feature_importance.sort_values(by='Importance', ascending=False)
    print(df_feature_importance.head(10))


    
    return


if __name__ == "__main__":
    main()