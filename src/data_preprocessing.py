import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler



def pre_processing_data(df):

    df= df.drop(columns=["PassengerId"])

    df['name_length'] = df['Name'].str.len()
    df= df.drop(columns=["Name"])

    df = pd.concat([df, pd.get_dummies(df['Sex'], prefix='Sex', drop_first=True)], axis=1).drop(columns=['Sex'])



    df, test_df = train_test_split(df, test_size=0.2, stratify=df["Survived"], random_state=42)
    imput_dict = {}

    imput_dict["Age"]=df["Age"].median()
    df["Age"] = df["Age"].fillna(df["Age"].median())

    def fill_in_missing_cabin(df):

        df['Cabin'] = df['Cabin'].fillna('M')
        df['Cabin'] = df['Cabin'].str[0]
        df_encoded = pd.get_dummies(df['Cabin'], prefix='cabin_first_char', drop_first=True)
        df = pd.concat([df, df_encoded], axis=1)
        df.drop(columns=['Cabin'], inplace=True)

        return df

    df = fill_in_missing_cabin(df)
    test_df = fill_in_missing_cabin(test_df)

    mode_value = df['Embarked'].mode().iloc[0]

    df['Embarked'] = df['Embarked'].fillna(mode_value)
    imput_dict["Embarked"] = mode_value


    df = pd.concat([df, pd.get_dummies(df['Embarked'], prefix='Embarked', drop_first=True)], axis=1).drop(columns=['Embarked'])

    
    tickets = list()

    for i in list(df.Ticket):
        if not i.isdigit():
            tickets.append(i.replace('.', '').replace('/', '').strip().split(' ')[0])
        else:
            tickets.append('x')
            
    df['Ticket'] = tickets
    df = pd.get_dummies(df, columns=['Ticket'], prefix='T', dtype=int)

    df = df.apply(pd.to_numeric, errors='coerce')  
    df = df.astype(float) 


    for col, imputed_value in imput_dict.items():
        test_df[col] = test_df[col].fillna(imputed_value)
    
    test_df = test_df.drop(columns=["Ticket"])    
    test_df = pd.concat([test_df, pd.get_dummies(test_df['Embarked'], prefix='Embarked', drop_first=True)], axis=1).drop(columns=['Embarked'])

    def fill_col(df, test_df):
        missing_col_test = set(df.columns) - set(test_df.columns)
        for col in missing_col_test:
            test_df[col] = 0
        
        missing_col_df = set(test_df.columns) - set(df.columns)

        for col in missing_col_df:
            df[column] = 0

        df = df.reindex(sorted(test_df.columns), axis=1)
        test_df = test_df.reindex(sorted(df.columns), axis=1)
        
        df = df.apply(pd.to_numeric, errors='coerce')  
        df = df.astype(float)  
        
        test_df = test_df.apply(pd.to_numeric, errors='coerce')  
        test_df = test_df.astype(float)  

        return df, test_df

    df, test_df = fill_col(df, test_df)

    X_train = df.drop('Survived', axis=1)
    y_train = df['Survived']
    X_test = test_df.drop('Survived', axis=1)
    y_test = test_df['Survived']


    return X_train, y_train, X_test, y_testa