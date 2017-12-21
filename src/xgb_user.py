import math
import pandas as pd
import time

from src import params

# userProfile_train = pd.read_csv(params['proj_path']+'data/trainingset/userProfile_train.csv')
# userProfile_train['gender'] = userProfile_train['gender'].fillna('-1') #3
# userProfile_train['province'] = userProfile_train['province'].fillna('-1') #-1,34
# userProfile_train['age'] = userProfile_train['age'].fillna('-1') #-1,60后-90后,00后
#
# gender = pd.get_dummies( userProfile_train['gender'],prefix='gender' )
# province = pd.get_dummies( userProfile_train['province'],prefix='province' )
# age = pd.get_dummies( userProfile_train['age'],prefix='age' )
# tr_x = pd.concat([gender,province,age],axis=1,join='inner') #axis=1 是行
# print( tr_x.head(3) )

## user action feat(stats feats,season feats)
def trans_data( userProfile_train ):
    end_time = userProfile_train['actionTime'].max()
    actionType = pd.get_dummies( userProfile_train['actionType'],prefix='actionType' )
    tr_x = pd.concat([userProfile_train,actionType],axis=1,join='inner') #axis=1 是行
    del tr_x['actionType']
    del tr_x['actionTime']
    tr_x = tr_x.groupby('userid',as_index=False).sum()
    vals = []
    for i in range(1,10):
        vals.append( tr_x['actionType_%s' % i].sum() )
    vals = list(map(lambda x:round(-math.log((1.0*(x-min(vals)+100)/(max(vals)-min(vals)+100*len(vals)))),4),vals))
    acttype2weight = {(idx+1):weight for idx,weight in enumerate(vals) }
    print(vals)
    # 半衰期为一个月
    userProfile_train['time_weight'] = userProfile_train['actionTime'].apply( lambda x:0.5**int((end_time-x)/(30*24*3600)) )
    userProfile_train['action_weight'] = userProfile_train['actionType'].apply( lambda x:acttype2weight[x] )
    userProfile_train['user_score'] = userProfile_train['time_weight']*userProfile_train['action_weight']
    user_score = userProfile_train[['userid','user_score']].groupby('userid',as_index=False).sum()
    tr_x = pd.merge( tr_x,user_score,on='userid' )


    #季节，月份，
    userProfile_train['actionDateTime'] = userProfile_train['actionTime'].apply(lambda t:time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(t)))
    userProfile_train['month'] = userProfile_train['actionDateTime'].str[5:7]
    month = pd.get_dummies(userProfile_train['month'], prefix='month')
    def month2season( month ):
        if month in [3,4,5]:
            return 'spring'
        if month in [6,7,8]:
            return 'summer'
        if month in [9,10,11]:
            return 'autumn'
        if month in [12,1,2]:
            return 'winter'
    userProfile_train['season'] = userProfile_train['month'].apply(month2season)
    season = pd.get_dummies(userProfile_train['season'], prefix='season')
    user_ord_time_feat = pd.concat( [userProfile_train[['userid']],month,season],axis=1,join='inner')
    user_ord_time_feat = user_ord_time_feat.groupby('userid',as_index=False).sum()
    tr_x = pd.merge(tr_x,user_ord_time_feat, on='userid')
    #转化率
    for i in range(2,10):
        for j in range(1,i):
            tr_x['conversion_rate_%s_%s' %(j,i)] = tr_x['actionType_%s' %i]/tr_x['actionType_%s' %j]
    return tr_x

userProfile_train = pd.read_csv(params['proj_path']+'data/trainingset/action_train.csv')
tr_x = trans_data( userProfile_train )
