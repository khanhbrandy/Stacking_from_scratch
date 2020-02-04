"""
Created on 2019-10-14
Creator: khanh.brandy

"""
import pandas as pd
import numpy as np
import time
import brandy_interest
import brandy_profile
import brandy_preprocess
import brandy_model
import joblib
import warnings

def preprocess_data(profile_link, level_list, seed):
    profile = brandy_profile.Profile()
    interest = brandy_interest.Interest()
    preprocess = brandy_preprocess.Preprocessor()    
    profile_raw = profile.get_profile(profile_link)
    interest_raw, ids = interest.data_merge(level_list)
    data = preprocess.finalize_data(profile_raw, interest_raw)
    X, y, X_train, y_train, X_test, y_test = preprocess.split_data(data, seed=seed, re=False)
    return X, y, X_train, y_train, X_test, y_test

def build_model(X_train, y_train, X_test, y_test, n_fold, seed):
    model = brandy_model.Model()
    clf_list = [model.clf_0, model.clf_1, model.clf_2]
    meta_train, meta_test = model.generate_metadata(X_train, y_train, X_test, y_test, clf_list, model.generate_oof, n_fold, seed)
    # Fit Meta classifier
    meta_clf = model.model_predict(model.clf_3, meta_train, y_train, meta_test, y_test, seed)
    print('Start dumping Meta classifier...')
    joblib.dump(meta_clf, 'meta_clf.pkl') 
    print('Done dumping Meta classifier ! \n')
    return meta_clf, meta_train, meta_test
if __name__=='__main__':
    print('*'*100) 
    print('*'*100+'\n')
    seed = 50
    n_fold = 5
    level_list = [
        {'level':'LV1', 'link':'training_data/MCREDIT_TRAINING_CLEAN_LV1.csv'},
        {'level':'LV2', 'link':'training_data/MCREDIT_TRAINING_CLEAN_LV2.csv'},
        {'level':'LV3', 'link':'training_data/MCREDIT_TRAINING_CLEAN_LV3.csv'},
        {'level':'LV4', 'link':'training_data/MCREDIT_TRAINING_CLEAN_LV4.csv'},
        {'level':'LV5', 'link':'training_data/MCREDIT_TRAINING_CLEAN_LV5.csv'}]
    model = brandy_model.Model()
    profile_link = 'training_data/MCREDIT_TRAINING_CLEAN_2_DEMO.csv'
    X, y, X_train, y_train, X_test, y_test = preprocess_data(profile_link, level_list, seed)
    warnings.filterwarnings('ignore', category=FutureWarning)
    meta_clf, meta_train, meta_test = build_model(X_train, y_train, X_test, y_test, n_fold, seed)
    print('*'*100) 
    print('*'*100+'\n')

