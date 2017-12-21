import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split

from src import params

def trans_data( userProfile_train,orderFuture_train ):
    # userProfile_train = pd.read_csv(params['proj_path']+'data/trainingset/userProfile_train.csv')
    userProfile_train['gender'] = userProfile_train['gender'].fillna('-1') #3
    userProfile_train['province'] = userProfile_train['province'].fillna('-1') #-1,34
    userProfile_train['age'] = userProfile_train['age'].fillna('-1') #-1,60后-90后,00后

    tr_x = userProfile_train['userid']
    gender = pd.get_dummies( userProfile_train['gender'],prefix='gender' )
    province = pd.get_dummies( userProfile_train['province'],prefix='province' )
    age = pd.get_dummies( userProfile_train['age'],prefix='age' )
    tr_x = pd.concat([tr_x,gender,province,age],axis=1,join='inner') #axis=1 是行
    print( tr_x.head(3) )

    # orderFuture_train = pd.read_csv(params['proj_path']+'data/trainingset/orderFuture_train.csv')
    tr_x = pd.merge( tr_x,orderFuture_train,on='userid' )
    return tr_x

def xgb_sub():
    userProfile_train = pd.read_csv(params['proj_path'] + 'data/trainingset/userProfile_train.csv')
    orderFuture_train = pd.read_csv(params['proj_path'] + 'data/trainingset/orderFuture_train.csv')
    tr_x = trans_data(userProfile_train,orderFuture_train)
    print('trainset len:%s' % len(tr_x))

    del tr_x['userid']
    y = tr_x['orderType']
    dtrain = xgb.DMatrix(tr_x,y)
    param = {'max_depth': 5
        , 'min_child_weight': 5  # 以前是5,过拟合
             # , 'gamma': 0.1
        , 'subsample': 1.0
        , 'colsample_bytree': 0.7
        # , 'eta': 0.01
        , 'lambda': 100  # L2惩罚系数,过拟合
        , 'scale_pos_weight': 33682.0 / 6625  # 处理正负样本不平衡,
        , 'objective': 'binary:logistic'
        , 'eval_metric': 'auc'  # 注意目标函数和评分函数的对应
        , 'early_stopping_rounds': 10  # eval 得分没有继续优化 就停止了
        , 'seed': 2000
        , 'nthread': 4
        , 'silent': 0
        }
    bst = xgb.train(param, dtrain, num_boost_round=100)
    bst.save_model(params['proj_path'] + 'model/xgb_user_basic.model')
    userProfile_test = pd.read_csv(params['proj_path'] + 'data/test/userProfile_test.csv')
    orderFuture_test = pd.read_csv(params['proj_path'] + 'data/test/orderFuture_test.csv')
    te_x = trans_data(userProfile_test, orderFuture_test)
    sub = pd.DataFrame()
    sub['userid'] = te_x['userid']
    te_x = xgb.DMatrix(te_x)
    clf = xgb.Booster({'nthread': 4})  # init model
    clf.load_model(params['proj_path'] + 'model/xgb_user_basic.model')

    sub['orderType'] = clf.predict(te_x)
    sub['orderType'] =  sub['orderType'].apply( lambda x:1 if x>0.5 else 0)
    sub.to_csv(params['proj_path']+'res/result_xgb_combine_{0}_{1}.csv'.format(100, 5), index=False)
# xgb_sub()


def eval():
    userProfile_train = pd.read_csv(params['proj_path']+'data/trainingset/userProfile_train.csv')
    tr_x = trans_data(userProfile_train)
    print( 'trainset len:%s' %len(tr_x) )
    pos,neg = (len(tr_x[tr_x['orderType']==1]),len( tr_x[tr_x['orderType']==0] ))
    print( 'pos/neg:%s/%s' %(len(tr_x[tr_x['orderType']==1]),len( tr_x[tr_x['orderType']==0] )) )

    y = tr_x['orderType']
    del tr_x['userid']
    del tr_x['orderType']
    X_train, X_test, y_train, y_test = train_test_split(tr_x, y, test_size=0.3, random_state=20170805)
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    # dtrain = xgb.DMatrix(tr_x, label=y)
    param = {'max_depth': 5
        , 'min_child_weight': 5  # 以前是5,过拟合
        #, 'gamma': 0.1
        , 'subsample': 1.0
        , 'colsample_bytree': 0.7
        #, 'eta': 0.01
        , 'lambda': 100  # L2惩罚系数,过拟合
        , 'scale_pos_weight': neg / pos  # 处理正负样本不平衡,
        , 'objective': 'binary:logistic'
        , 'eval_metric': 'auc'  # 注意目标函数和评分函数的对应
        , 'early_stopping_rounds': 10  # eval 得分没有继续优化 就停止了
        , 'seed': 2000
        , 'nthread': 4
        , 'silent': 0
        }
    evallist = [(dtest, 'eval'), (dtrain, 'train')]
    bst = xgb.train(param, dtrain, num_boost_round=100, evals=evallist)
    bst.save_model(params['proj_path']+'model/xgb_user_basic.model')

    clf = xgb.Booster({'nthread': 4})  # init model
    clf.load_model(params['proj_path']+'model/xgb_user_basic.model')
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
    s.to_csv(params['proj_path']+'cache/feature_scores.csv',index=False,encoding='utf-8')

eval()
