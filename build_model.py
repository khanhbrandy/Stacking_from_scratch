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


def build_model(data, clf_list, seed):
    model = brandy_model.Model()
    X, y, X_train, y_train, X_test, y_test = model.split_data(data, seed=seed, re=False)
    oof_train = []
    oof_test = []
    for clf in clf_list:
        clf_oof_train, clf_oof_test = model.generate_oof(clf, X_train, y_train, X_test, n_fold, seed)
        oof_train.append(pd.DataFrame(clf_oof_train))
        oof_test.append(pd.DataFrame(clf_oof_test))
    meta_train = model.generate_metadata(oof_train)
    meta_test = model.generate_metadata(oof_test)
    # Fit Meta classifier
    meta_clf = model.model_predict(model.clf_3, meta_train, y_train, meta_test, y_test, seed)
    print('Start dumping Meta classifier...')
    joblib.dump(meta_clf, 'meta_clf.pkl') 
    print('Done dumping Meta classifier ! \n')
    return meta_clf
if __name__=='__main__':
    print('*'*100) 
    print('*'*100+'\n')
    model = brandy_model.Model()
    profile = brandy_profile.Profile()
    interest = brandy_interest.Interest()
    preprocess = brandy_preprocess.Preprocessor()
    seed = 50
    n_fold = 5
    clf_list = [model.clf_0, model.clf_2]
    profile_raw = profile.get_profile()
    interest_raw, ids, fbids_lv1, fbids_lv2, fbids_lv3, fbids_lv4, fbids_lv5 = interest.data_merge()
    # interest_raw.to_excel('interest.xlsx')
    data = preprocess.finalize_data(profile_raw, interest_raw)
    warnings.filterwarnings('ignore', category=FutureWarning)
    meta_clf = build_model(data, clf_list, seed)
    print('*'*100) 
    print('*'*100+'\n')

