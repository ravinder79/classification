import pandas as pd
import numpy as np
from pydataset import data
import acquire
import sklearn.impute
import sklearn.model_selection
import sklearn.preprocessing
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler

def drop_columns(df):
    return df.drop(columns = ['species_id', 'measurement_id'])

def rename_columns(df):
    return df.rename(columns = {'species_name' : 'species'})

def encode_iris(train, test):
    encoder = sklearn.preprocessing.OneHotEncoder(sparse = False)
    encoder.fit(train[['species']])
    m = encoder.transform(train[['species']])
    cols =  ['species_' + c for c in encoder.categories_[0]]
    train = pd.concat([train, pd.DataFrame(m, columns=cols, index=train.index)], axis =1).drop(columns = 'species')
    m = encoder.transform(test[['species']])
    test = pd.concat([test, pd.DataFrame(m, columns = cols, index = test.index)], axis =1).drop(columns = 'species')
    return train, test

def prep_iris(df):
    df = drop_columns(df)
    df = rename_columns(df)
    train, test = sklearn.model_selection.train_test_split(df, train_size=.8, random_state=123)
    train, test = encode_iris(train, test)
    return train, test


def label_encoder(train, test):
    encoder = LabelEncoder()
    encoder.fit(train.embarked)
    train.encoded = encoder.transform(train.embarked)
    test.encoded = encoder.transform(test.embarked)
    train_array = np.array(train.encoded).reshape(len(train.encoded),1)
    test_array = np.array(test.encoded).reshape(len(test.encoded),1)
    col_name = 'embarked'
    encoded_values = sorted(list(train[col_name].unique()))

    ohe = OneHotEncoder(sparse=False, categories='auto')
    train_ohe = ohe.fit_transform(train_array)
    test_ohe = ohe.transform(test_array)

        # Turn the array of new values into a data frame with columns names being the values
        # and index matching that of train/test
        # then merge the new dataframe with the existing train/test dataframe
    train_encoded = pd.DataFrame(data=train_ohe,
                            columns=encoded_values, index=train.index)
    train = train.join(train_encoded)

    test_encoded = pd.DataFrame(data=test_ohe,
                                   columns=encoded_values, index=test.index)
    test = test.join(test_encoded)
    return train, test

def scale_minmax(train, test, column_list):
    scaler = MinMaxScaler()
    column_list_scaled = [col + '_scaled' for col in column_list]
    train_scaled = pd.DataFrame(scaler.fit_transform(train[column_list]), 
                                columns = column_list_scaled, 
                                index = train.index)
    train = train.join(train_scaled)

    test_scaled = pd.DataFrame(scaler.transform(test[column_list]), 
                                columns = column_list_scaled, 
                                index = test.index)
    test = test.join(test_scaled)

    return train, test

def impute_titanic(train, test):
    imputer = sklearn.impute.SimpleImputer(strategy='mean')
    imputer.fit(train[['age']])
    train.age = imputer.transform(train[['age']])
    test.age = imputer.transform(test[['age']])
    return train, test

def prep_titanic(df):
    df.embark_town = df.embark_town.fillna('Southampton')
    df.embarked = df.embarked.fillna('S')
    df = df.drop(columns = 'deck')
    train, test = sklearn.model_selection.train_test_split(df, train_size=.8, random_state=123)
    train, test = label_encoder(train,test)
    train, test = impute_titanic(train, test)
    train, test = scale_minmax(train, test, ['age', 'fare'])
    return train, test