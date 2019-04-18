import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier,Dataset
from xgboost import XGBClassifier,DMatrix
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold




if __name__== '__main__':

    train = pd.read_csv('repeat buyer train data.csv',sep=',')
    predict = pd.read_csv('repeat buyer test data.csv', sep=',')
    col_del = train.ix[:,(train==0).all()].columns
    train = train.drop(col_del,axis=1)
    predict = predict.drop(col_del,axis=1)

    train = train.fillna(-1)
    predict = predict.fillna(-1)
    train[['age_range', 'gender']] = train[['age_range', 'gender']].astype('int8')
    predict[['age_range', 'gender']] = predict[['age_range', 'gender']].astype('int8')
    label = train['label']
    trainNew = train.drop(['label'],axis=1)

    cat_features = [0,1,2,3]

    # 构造训练集和测试集
    def trainData(train_df,label_df):
        skv = StratifiedKFold(n_splits=5, shuffle=True, random_state=620)
        trainX = []
        trainY = []
        testX = []
        testY = []
        for train_index, test_index in skv.split(X=train_df, y=label_df):
            train_x, train_y, test_x, test_y = train_df.iloc[train_index, :], label_df.iloc[train_index], \
                                               train_df.iloc[test_index, :], label_df.iloc[test_index]

            trainX.append(train_x)
            trainY.append(train_y)
            testX.append(test_x)
            testY.append(test_y)
        return trainX,trainY,testX,testY

    trainX,trainY,testX,testY = trainData(trainNew,label)

    # 将训练数据集划分分别训练5个lgbm,xgboost和catboost 模型
    # lightgbm模型
    
    pred_lgbms = []
    for i in range(5):
        lgbm = LGBMClassifier(n_estimators=2000,objective='binary',num_leaves=31,max_depth=5,learning_rate=0.03,
                              reg_lambda=1,metric=['auc'], random_state=10,n_jobs=-1)
        lgbm.fit(trainX[i],trainY[i],eval_set=[(testX[i],testY[i])],early_stopping_rounds=50,eval_metric='auc',
                 categorical_feature=cat_features)
        print(lgbm.evals_result_)
        pred = lgbm.predict_proba(predict,num_iteration=lgbm.best_iteration_)[:,1]
        pred_lgbms.append(pred)

    # catboost模型
    pred_cats = []
    for i in range(5):
    
        cat = CatBoostClassifier(learning_rate=0.02, iterations=5000, eval_metric='AUC', od_wait=50,
                                 od_type='Iter', random_state=10, thread_count=8, l2_leaf_reg=1)
        cat.fit(trainX[i], trainY[i], eval_set=[(testX[i], testY[i])], early_stopping_rounds=50,
                use_best_model=True,cat_features=cat_features)
        print(cat.evals_result_)
        pred = cat.predict_proba(predict, ntree_end=cat.best_iteration_)[:, 1]
        pred_cats.append(pred)
    
    # xgboost模型
    pred_xgbs = []
    pred_result = predict[['merchant_id','user_id']]
    for i in range(5):

        xgb = XGBClassifier(n_estimators=2000, max_depth=5, learning_rate=0.025, 
                          eval_metric='auc', reg_lambda=1, random_state=10, n_jobs=8)
        xgb.fit(train_x,trainY[i],eval_set=[(test_x,testY[i])],early_stopping_rounds=50,eval_metric='auc')
        print(xgb.evals_result_)
        pred = xgb.predict_proba(predict_x, ntree_limit = xgb.best_iteration)[:,1]
        pred_xgbs.append(pred)
    
    def sigmoid_ver(x):
        return np.log(x/(1-x))
    def sigmoid(x):
        return 1/(1 + np.e**(-x))

    pred_t = np.zeros(len(predict))
    
    for i in range(5):
        pred_t += (sigmoid_ver(pred_lgbms[i]) + sigmoid_ver(pred_cats[i]) + sigmoid_ver(pred_xgbs[i]))

    result = sigmoid(pred_t/15)
    pred_result['prob'] = result
    pred_result[['user_id','merchant_id','prob']].to_csv('submission result.csv',sep=',', index=False)

    
