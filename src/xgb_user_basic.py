import pandas as pd

from src import params

userProfile_train = pd.read_csv(params['proj_path']+'data/trainingset/userProfile_train.csv')
userProfile_train['gender'] = userProfile_train['gender'].fillna('-1') #3
userProfile_train['province'] = userProfile_train['province'].fillna('-1') #-1,34
userProfile_train['age'] = userProfile_train['age'].fillna('-1') #-1,60后-90后,00后

gender = pd.get_dummies( userProfile_train['gender'],prefix='gender' )
province = pd.get_dummies( userProfile_train['province'],prefix='province' )
age = pd.get_dummies( userProfile_train['age'],prefix='age' )
tr_x = pd.concat([gender,province,age],axis=1,join='inner') #axis=1 是行
print( tr_x.head(3) )