import pandas as pd
import numpy as np
import os
import tensorflow as tf
import math
from numpy.random import MT19937
from numpy.random import RandomState, SeedSequence

####### STUDENTS FILL THIS OUT ######
#Question 3
def reduce_dimension_ndc(df, ndc_df):
    '''
    df: pandas dataframe, input dataset
    ndc_df: pandas dataframe, drug code dataset used for mapping in generic names
    return:
        df: pandas dataframe, output dataframe with joined generic drug name
    '''
    
    ndc_dict = ndc_df.set_index("NDC_Code")["Non-proprietary Name"].to_dict()
    df["generic_drug_name"] = df.ndc_code.replace(ndc_dict)
    
    return df

#Question 4
def select_first_encounter(df):
    '''
    df: pandas dataframe, dataframe with all encounters
    return:
        - first_encounter_df: pandas dataframe, dataframe with only the first encounter for a given patient
    '''
    # convert data to encounter level and split at patient level
    group_col = ["patient_nbr", "encounter_id"]
    agg_cols = [col for col in df.columns if col not in group_col]
    df = df.sort_values(by="patient_nbr").groupby(group_col)[agg_cols].agg(lambda x: ','.join(str(s) for s in set(x)))
    df = pd.DataFrame(df)
    
    # save encounter IDs in separate column
    df.insert(0,"encounter_id",df.index.get_level_values(1))
    
    # rename encounter index starting from 0 to make slicing easier
    df.index = pd.MultiIndex.from_arrays([df.index.get_level_values(0), df.groupby(level=0).cumcount()], names=['patient_nbr', 'encounter_id'])

    # select only first encounter for each patient
    first_encounter_df = df.xs(0, level=1)
    first_encounter_df.reset_index(inplace=True)
    
    return first_encounter_df


#Question 6
def patient_dataset_splitter(df, patient_key='patient_nbr'):
    '''
    df: pandas dataframe, input dataset that will be split
    patient_key: string, column that is the patient id

    return:
     - train: pandas dataframe,
     - validation: pandas dataframe,
     - test: pandas dataframe,
    '''
    rs = RandomState(MT19937(SeedSequence(42)))
    index = df.index.to_numpy().copy()
    rs.shuffle(index)
    
    df = df.iloc[index].copy()
    
    n_train = math.ceil(len(df)*.6)
    n_val = n_train + math.ceil(len(df)*.2)
    n_test = n_val + math.ceil(len(df)*.2)
    
    train = df.iloc[index[:n_train]]
    validation = df.iloc[index[n_train:n_val]]
    test = df.iloc[index[n_val:n_test]]
    return train, validation, test

#Question 7

def create_tf_categorical_feature_cols(categorical_col_list,
                              vocab_dir='./diabetes_vocab/'):
    '''
    categorical_col_list: list, categorical field list that will be transformed with TF feature column
    vocab_dir: string, the path where the vocabulary text files are located
    return:
        output_tf_list: list of TF feature columns
    '''
    output_tf_list = []
    for c in categorical_col_list:
        vocab_file_path = os.path.join(vocab_dir,  c + "_vocab.txt")
        '''
        Which TF function allows you to read from a text file and create a categorical feature
        You can use a pattern like this below...
        tf_categorical_feature_column = tf.feature_column.......

        '''
        output_tf_list.append(tf_categorical_feature_column)
    return output_tf_list

#Question 8
def normalize_numeric_with_zscore(col, mean, std):
    '''
    This function can be used in conjunction with the tf feature column for normalization
    '''
    return (col - mean)/std



def create_tf_numeric_feature(col, MEAN, STD, default_value=0):
    '''
    col: string, input numerical column name
    MEAN: the mean for the column in the training data
    STD: the standard deviation for the column in the training data
    default_value: the value that will be used for imputing the field

    return:
        tf_numeric_feature: tf feature column representation of the input field
    '''
    return tf_numeric_feature

#Question 9
def get_mean_std_from_preds(diabetes_yhat):
    '''
    diabetes_yhat: TF Probability prediction object
    '''
    m = '?'
    s = '?'
    return m, s

# Question 10
def get_student_binary_prediction(df, col):
    '''
    df: pandas dataframe prediction output dataframe
    col: str,  probability mean prediction field
    return:
        student_binary_prediction: pandas dataframe converting input to flattened numpy array and binary labels
    '''
    return student_binary_prediction
