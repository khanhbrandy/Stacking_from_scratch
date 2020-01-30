"""
Created on 2019-10-14
Creator: khanh.brandy

"""
import pandas as pd
import numpy as np
import time
from copy import copy as make_copy
from sklearn import preprocessing

class Interest:
    def __init__(self):
        pass

    def get_interest(self, url):
        data=pd.read_csv(url, header = 0, converters={0:str,1:str})
        data = data.head(100) ### For testing purpose !!!!!
        data = self.interest_standardize(data, std=False)
        fbids=np.array(data['FACEBOOK_ID'])
        return data, fbids

    def data_merge(self, merge=True):
        print('Start getting interest data for training...')
        start = time.time()
        data_lv1, self.fbids_lv1=self.get_interest('training_data/MCREDIT_TRAINING_CLEAN_LV1.csv')
        data_lv2, self.fbids_lv2=self.get_interest('training_data/MCREDIT_TRAINING_CLEAN_LV2.csv')
        data_lv3, self.fbids_lv3=self.get_interest('training_data/MCREDIT_TRAINING_CLEAN_LV3.csv')
        data_lv4, self.fbids_lv4=self.get_interest('training_data/MCREDIT_TRAINING_CLEAN_LV4.csv')
        data_lv5, self.fbids_lv5=self.get_interest('training_data/MCREDIT_TRAINING_CLEAN_LV5.csv')
        print('Done getting interest data for training. Time taken = {:.1f}(s) \n'.format(time.time()-start))
        data_final = make_copy(data_lv1)
        dfs = [data_lv2, data_lv3, data_lv4, data_lv5]
        if merge:
            for df in dfs:
                data_final = data_final.merge(df, on=['FACEBOOK_ID'], how='left')
                interest_strength = data_final.fillna(0)
        else:
            interest_strength=data_lv4.fillna(0)
        def level_convert(fbid):
            if fbid in set(self.fbids_lv5):
                    return 5
            else:
                if fbid in set(self.fbids_lv4):
                    return 4
                else:
                    if fbid in set(self.fbids_lv3):
                        return 3
                    else:
                        if fbid in set(self.fbids_lv2):
                            return 2
                        else:
                            return 1
        interest_strength['INTEREST_LEVEL'] = interest_strength['FACEBOOK_ID'].map(level_convert)
        # Get ids 
        suMCREDIT_ids = interest_strength.sum(axis=0)
        suMCREDIT_ids[suMCREDIT_ids != 0]
        ids = suMCREDIT_ids[suMCREDIT_ids != 0].index
        return interest_strength, ids

    def interest_standardize(self, interest_data, ids=None, std=False):
        interest_data=interest_data.iloc[:,1:]
        if ids==None:
            interest_data=interest_data
        else:
            interest_data=interest_data[ids]
        if std:
            interest_data.set_index('FACEBOOK_ID', inplace=True)
            scaler = preprocessing.StandardScaler()
            interest_data_stded=scaler.fit_transform(interest_data.values)
            df_interest_data_stded=pd.DataFrame(interest_data_stded, index=interest_data.index,columns=interest_data.columns)
            df_interest_data_stded.reset_index(inplace=True)
        else:
            df_interest_data_stded=interest_data
        return df_interest_data_stded
