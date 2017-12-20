import pandas as pd

from src import params

userProfile_train = pd.read_csv(params['proj_path']+'data/trainingset/userProfile_train.csv')
userProfile_train['gender'] = userProfile_train['gender'].fillna('-1') #3
userProfile_train['province'] = userProfile_train['province'].fillna('-1') #-1,34
userProfile_train['age'] = userProfile_train['age'].fillna('-1') #-1,60后-90后,00后

