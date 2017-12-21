import math
import pandas as pd
import time
import xgboost as xgb
from sklearn.model_selection import train_test_split

from src import params

## user action feat(stats feats,season feats)
def trans_data( user_profile,user_action,orderFuture_train ):
    user_profile['gender'] = user_profile['gender'].fillna('-1') #3
    user_profile['province'] = user_profile['province'].fillna('-1') #-1,34
    user_profile['age'] = user_profile['age'].fillna('-1') #-1,60后-90后,00后

    tr_x = user_profile['userid']
    gender = pd.get_dummies( user_profile['gender'],prefix='gender' )
    province = pd.get_dummies( user_profile['province'],prefix='province' )
    age = pd.get_dummies( user_profile['age'],prefix='age' )
    tr_x_ = pd.concat([tr_x,gender,province,age],axis=1,join='inner') #axis=1 是行

    end_time = user_action['actionTime'].max()
    actionType = pd.get_dummies( user_action['actionType'],prefix='actionType' )
    tr_x = pd.concat([user_action,actionType],axis=1,join='inner') #axis=1 是行
    tr_x = pd.merge(tr_x_,tr_x,on='userid')
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
    user_action['time_weight'] = user_action['actionTime'].apply( lambda x:0.5**int((end_time-x)/(30*24*3600)) )
    user_action['action_weight'] = user_action['actionType'].apply( lambda x:acttype2weight[x] )
    user_action['user_score'] = user_action['time_weight']*user_action['action_weight']
    user_score = user_action[['userid','user_score']].groupby('userid',as_index=False).sum()
    tr_x = pd.merge( tr_x,user_score,on='userid' )


    #季节，月份，
    user_action['actionDateTime'] = user_action['actionTime'].apply(lambda t:time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(t)))
    user_action['month'] = user_action['actionDateTime'].str[5:7]
    month = pd.get_dummies(user_action['month'], prefix='month')
    def month2season( month ):
        if month in [3,4,5]:
            return 'spring'
        if month in [6,7,8]:
            return 'summer'
        if month in [9,10,11]:
            return 'autumn'
        if month in [12,1,2]:
            return 'winter'
    user_action['season'] = user_action['month'].apply(month2season)
    season = pd.get_dummies(user_action['season'], prefix='season')
    user_ord_time_feat = pd.concat( [user_action[['userid']],month,season],axis=1,join='inner')
    user_ord_time_feat = user_ord_time_feat.groupby('userid',as_index=False).sum()
    tr_x = pd.merge(tr_x,user_ord_time_feat, on='userid')
    #转化率
    for i in range(2,10):
        for j in range(1,i):
            tr_x['conversion_rate_%s_%s' %(j,i)] = tr_x['actionType_%s' %i]/tr_x['actionType_%s' %j]
            tr_x['conversion_rate_%s_%s' % (j, i)] = tr_x['conversion_rate_%s_%s' %(j,i)].apply( lambda x:1.0 if x>1 else x)
    tr_x = tr_x.fillna(0)

    tr_x = pd.merge(tr_x, orderFuture_train, on='userid')
    return tr_x


def eval():
    action_train = pd.read_csv(params['proj_path'] + 'data/trainingset/action_train.csv')
    userProfile_train = pd.read_csv(params['proj_path']+'data/trainingset/userProfile_train.csv')
    orderFuture_train = pd.read_csv(params['proj_path'] + 'data/trainingset/orderFuture_train.csv')
    tr_x = trans_data(userProfile_train,action_train,orderFuture_train)
    print( 'trainset len:%s' %len(tr_x) )
    pos,neg = (len(tr_x[tr_x['orderType']==1]),len( tr_x[tr_x['orderType']==0] ))
    print( 'pos/neg:%s/%s' %(len(tr_x[tr_x['orderType']==1]),len( tr_x[tr_x['orderType']==0] )) )

    y = tr_x['orderType']
    del tr_x['userid']
    del tr_x['orderType']
    print(tr_x.head(5))
    X_train, X_test, y_train, y_test = train_test_split(tr_x, y, test_size=0.3, random_state=20171221)
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    # dtrain = xgb.DMatrix(tr_x, label=y)
    param = {'max_depth': 5
        , 'min_child_weight': 25  # 以前是5,过拟合
        #, 'gamma': 0.1
        , 'subsample': 1.0
        , 'colsample_bytree': 0.7
        , 'eta': 0.01
        , 'lambda': 100  # L2惩罚系数,过拟合
        , 'scale_pos_weight': float(neg) / pos  # 处理正负样本不平衡,
        , 'objective': 'binary:logistic'
        , 'eval_metric': 'auc'  # 注意目标函数和评分函数的对应
        , 'early_stopping_rounds': 100  # eval 得分没有继续优化 就停止了
        , 'seed': 8888
        , 'nthread': 4
        , 'silent': 0
        }
    evallist = [(dtest, 'eval'), (dtrain, 'train')]
    bst = xgb.train(param, dtrain, num_boost_round=1000, evals=evallist)
    bst.save_model(params['proj_path']+'model/xgb_user.model')

    clf = xgb.Booster({'nthread': 4})  # init model
    clf.load_model(params['proj_path']+'model/xgb_user.model')
    feat = tr_x.columns
    data = pd.DataFrame( feat,columns=['feature'] )
    data['col'] = data.index
    feature_score = clf.get_fscore()
    keys = []
    values = []
    for key in feature_score:
        keys.append( key )
        values.append( feature_score[key] )
    df = pd.DataFrame( keys,columns=['features'] )
    df['score'] = values
    df['col'] = df['features'].apply( lambda x:int(x[1:]) )
    s = pd.merge( df,data,on='col' )
    s = s.sort_values('score',ascending=False)[['feature','score']]
    s.to_csv(params['proj_path']+'cache/feature_scores_user.csv',index=False,encoding='utf-8')

# eval()

def xgb_sub():
    userProfile_train = pd.read_csv(params['proj_path'] + 'data/trainingset/userProfile_train.csv')
    action_train = pd.read_csv(params['proj_path'] + 'data/trainingset/action_train.csv')
    orderFuture_train = pd.read_csv(params['proj_path'] + 'data/trainingset/orderFuture_train.csv')
    tr_x = trans_data(userProfile_train,action_train,orderFuture_train)
    print('trainset len:%s' % len(tr_x))

    del tr_x['userid']
    y = tr_x['orderType']
    dtrain = xgb.DMatrix(tr_x,y)
    param = {'max_depth': 5
        , 'min_child_weight': 25  # 以前是5,过拟合
             # , 'gamma': 0.1
        , 'subsample': 1.0
        , 'colsample_bytree': 0.7
        , 'eta': 0.01
        , 'lambda': 100  # L2惩罚系数,过拟合
        , 'scale_pos_weight': 33682.0 / 6625  # 处理正负样本不平衡,
        , 'objective': 'binary:logistic'
        , 'eval_metric': 'auc'  # 注意目标函数和评分函数的对应
        , 'early_stopping_rounds': 10  # eval 得分没有继续优化 就停止了
        , 'seed': 8888
        , 'nthread': 4
        , 'silent': 0
    }
    bst = xgb.train(param, dtrain, num_boost_round=10000)
    bst.save_model(params['proj_path'] + 'model/xgb_user_action.model')
    userProfile_test = pd.read_csv(params['proj_path'] + 'data/test/userProfile_test.csv')
    userAction_test = pd.read_csv(params['proj_path'] + 'data/test/action_test.csv')
    orderFuture_test = pd.read_csv(params['proj_path'] + 'data/test/orderFuture_test.csv')
    te_x = trans_data(userProfile_test,userAction_test,orderFuture_test)
    sub = pd.DataFrame()
    sub['userid'] = te_x['userid']
    te_x = xgb.DMatrix(te_x)
    clf = xgb.Booster({'nthread': 4})  # init model
    clf.load_model(params['proj_path'] + 'model/xgb_user_action.model')

    sub['orderType'] = clf.predict(te_x)
    sub['orderType'] =  sub['orderType'].apply( lambda x:1 if x>0.5 else 0)
    sub.to_csv(params['proj_path']+'res/result_xgb_user{0}_{1}.csv'.format(10000, 5), index=False)
xgb_sub()