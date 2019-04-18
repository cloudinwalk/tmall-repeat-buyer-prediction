import datetime
import numpy as np
import pandas as pd
from scipy import sparse
from collections import Counter
import time


class dataFeature(object):

    def __init__(self,u_log,train,test):
        # self.u_info = u_info
        self.u_log = u_log
        self.train = train
        self.test = test

    # 对各种特征进行统计
    def summary(self,key,gbname,pname,prefix,operator,dummy,ifratio):
        # user_info = self.u_info
        user_log = self.u_log
        train = self.train
        test = self.test
        # 根据不同特征选择不同的计算模式
        if operator==None:
            if not dummy:
                if prefix == None:# 对各种交互次数进行统计
                    df = user_log.groupby(key).size().reset_index().rename(columns={0:pname})
                else: # 对item cat brand等交互种类数进行统计 设置prefix = 1
                    df = user_log.groupby(key).agg({gbname: lambda x: len(set(x))}).reset_index().rename(columns=
                                                                        {gbname: pname})
            else:
                df = user_log.groupby(key +[gbname]).size().reset_index().rename(columns= {0: pname})
                df = pd.get_dummies(df, columns=[gbname], prefix=prefix)
                df = df.apply(pd.to_numeric, downcast='unsigned')
                columns = [i for i in df.columns.tolist() if prefix in i]
                for col in columns:
                    df[col] *= df[pname]
                df = df.groupby(key).sum().reset_index().drop([pname], axis=1)

        else:
            if not dummy:
                if prefix != None:
                    df = user_log.groupby(key + [gbname]).size().reset_index().rename(columns={0: pname})
                    df = df.groupby(key).agg({pname: operator}).reset_index()
                    df.columns = key + [ prefix + 'count', prefix + 'mean', prefix + 'max', prefix + 'min']
                else:
                    df = user_log.groupby(key).agg({gbname: operator}).reset_index()
                    df.rename(columns={gbname: pname}, inplace=True)
            else:
                if not ifratio:
                    df = user_log.groupby(key[0] + [gbname]).agg({key[1]: operator}).reset_index().rename(columns=
                                                                                                         {key[1]: pname})
                    df = pd.get_dummies(df, columns=[gbname], prefix=prefix)
                    df = df.apply(pd.to_numeric, downcast='unsigned')
                    columns = [i for i in df.columns.tolist() if prefix in i]
                    for col in columns:
                        df[col] *= df[pname]
                    df = df.groupby(key[0]).sum().reset_index().drop([pname], axis=1)
                else:
                    df = user_log.groupby(key).agg({gbname:lambda x:len(set(x))}).reset_index().rename(columns={gbname: 'all_cnt'})
                    df = df.merge(user_log.groupby(key).agg({gbname: operator}).reset_index(), on=key, how='left',copy=False)
                    df[pname] = df[gbname] / (df['all_cnt'] + 10)
                    df[pname] = df[pname].astype('float32')
                    columns = df[key[1]].unique().tolist()
                    df = pd.get_dummies(df, columns=[key[1]], prefix=prefix)
                    for col in columns:
                        df[prefix +'_'+ str(col)] *= df[pname]
                    df = df.groupby(key[0]).sum().reset_index().drop(['all_cnt', gbname, pname], 1)

        train = train.merge(df, on=key, how='left', copy=False)
        test = test.merge(df, on=key, how='left', copy=False)
        self.train = train
        self.test = test





if __name__ == '__main__':

    # 数据清洗
    user_info = pd.read_csv('E:\\repeat buyer\\user_info_format1.csv', usecols=['user_id', 'age_range',
                                                                                'gender']).drop_duplicates()
    user_info = user_info.dropna(axis=0,how='all')
    user_info[['age_range', 'gender']] = user_info[['age_range', 'gender']].astype('int8')
    user_log = pd.read_csv('E:\\repeat buyer\\user_log_format1.csv', usecols=['user_id', 'item_id', 'cat_id',
                          'seller_id', 'brand_id', 'time_stamp', 'action_type']).rename(columns={'seller_id': 'merchant_id'})
    user_log = user_log.dropna(axis=0,how='all')
    test = pd.read_csv('E:\\repeat buyer\\test_format1.csv', usecols=['user_id', 'merchant_id'])
    train = pd.read_csv('E:\\repeat buyer\\train_format1.csv', usecols=['user_id', 'merchant_id', 'label'])
    train = train.drop_duplicates(['user_id','merchant_id'])
    test = test.drop_duplicates(['user_id','merchant_id'])
    user_log = user_log.merge(user_info, on=['user_id'],how='left',copy=False)
    train = train.merge(user_info, on=['user_id'],how='left',copy=False)
    test = test.merge(user_info, on=['user_id'],how='left',copy=False)
    del user_info
    gc.collect()
    user_log['month'] = user_log['time_stamp'] // 100
    user_log['day'] = user_log['time_stamp'] % 100

    feature = dataFeature(user_log,train,test)
    # 用户每天互动的次数
    feature.summary('user_id','time_stamp','cnt','user_every_day_cnt',None,1,0)

    # 用户互动的总次数
    feature.summary('user_id',None,'user_id_cnt',None,None,0,0)

    # 用户每个月互动的次数
    feature.summary('user_id','month','user_month_cnt','um',None,1,0)
    feature.summary(['merchant_id'], 'month', 'merchant_month_cnt', 'mm',None, 1, 0)

    # merchant 被互动的次数
    feature.summary('merchant_id',None,'merchant_id_cnt',None,None,0,0)

    # 用户 merchant交互次数
    feature.summary(['user_id','merchant_id'],None,'user_id_merchant_id_cnt',None,None,0,0)

    # merchant 按gender统计交互次数
    feature.summary(['merchant_id','gender'], None, 'merchant_gender_cnt', None, None, 0,0)
    feature.summary(['merchant_id', 'age_range'], None, 'merchant_age_cnt', None, None, 0, 0)

    # 统计用户对每个 项目 品牌 merchant 品类 的交互的种类数
    feature.summary('user_id','item_id','user_query_item_id_cnt',1,None,0)
    feature.summary('user_id', 'cat_id', 'user_query_cat_id_cnt', 1, None, 0)
    feature.summary('user_id', 'merchant_id', 'user_query_merchant_id_cnt', 1, None, 0)
    feature.summary('user_id', 'brand_id', 'user_query_brand_id_cnt', 1, None, 0)

    # 用户时间差特征
    def timediff(t):
        delta = datetime.datetime.strptime(str(max(t)), '%m%d') - datetime.datetime.strptime(str(min(t)), '%m%d')
        return delta.days

    feature.summary('user_id','time_stamp','user_time_diff',None,timediff,0,0)
    feature.summary(['merchant_id'], 'time_stamp', 'merchant_time_diff', None, timediff, 0, 0)

    # 用户商铺 时间差特征
    feature.summary(['user_id', 'merchant_id'], 'time_stamp', 'user_merchant_time_diff', None, timediff, 0,0)

    # 用户商铺有几个月有互动的统计
    feature.summary(['user_id', 'merchant_id'],'month','user_merchant_month_cnt',1,None,0,0)

    # 用户商铺有多少天有互动的统计
    feature.summary(['user_id', 'merchant_id'],'day','user_merchant_day_cnt',1,None,0,0)

    # 用户与商铺具体有互动的月份每个月互动次数统计
    feature.summary(['user_id', 'merchant_id'],'month','cnt','month_act',None,1,0)

    # 商铺适合的年龄段和性别
    feature.summary(['merchant_id', 'gender', 'age_range'],None,'merchant_gender_age_cnt',None,None,0,0)

    # 商铺下有多少种 item,cat,brand 和有多少用户互动数以及每种互动的均值最值
    feature.summary('merchant_id', 'item_id', 'cnt', 'merchant_item_', [np.size, np.mean, np.max, np.min], 0,0)
    feature.summary('merchant_id', 'cat_id', 'cnt', 'merchant_cat_', [np.size, np.mean, np.max, np.min], 0,0)
    feature.summary('merchant_id', 'brand_id', 'cnt', 'merchant_brand_', [np.size, np.mean, np.max, np.min], 0,0)
    feature.summary('merchant_id', 'user_id', 'cnt', 'merchant_user_', [np.size, np.mean, np.max, np.min], 0,0)

    # 统计用户在商铺下互动了多少个item 以及每种item互动次数的均值 最值
    feature.summary(['user_id', 'merchant_id'],'item_id','cnt','user_merchant_item_',
                    [np.size, np.mean, np.max, np.min],0,0)

    # 统计用户在商铺下互动了多少个cat 以及每种cat互动次数的均值 最值
    feature.summary(['user_id', 'merchant_id'], 'cat_id', 'cnt', 'user_merchant_cat_',
                    [np.size, np.mean, np.max, np.min], 0,0)

    # 统计用户在商铺下互动了多少个brand 以及每种brand互动次数的均值 最值
    feature.summary(['user_id', 'merchant_id'], 'brand_id', 'cnt', 'user_merchant_brand_',
                    [np.size, np.mean, np.max, np.min], 0,0)

    # 用户在商铺下的行为习惯 点击、收藏、加购物以及购买的统计
    feature.summary(['user_id', 'merchant_id'],'action_type','cnt','um_action',None,1,0)

    # 用户在商铺下的行为比例
    train = feature.train
    test = feature.test
    train['um_action_0_ratio'] = train['um_action_0'] / (train['um_action_0'] + train['um_action_1'] +
                                                         train['um_action_2'] + train['um_action_3'] + 10)
    
    train['um_action_1_ratio'] = train['um_action_1'] / (train['um_action_0'] + train['um_action_1'] +
                                                         train['um_action_2'] + train['um_action_3'] + 10)
    
    train['um_action_2_ratio'] = train['um_action_2'] / (train['um_action_0'] + train['um_action_1'] +
                                                         train['um_action_2'] + train['um_action_3'] + 10)
    
    train['um_action_3_ratio'] = train['um_action_3'] / (train['um_action_0'] + train['um_action_1'] +
                                                         train['um_action_2'] + train['um_action_3'] + 10)
    
    test['um_action_0_ratio'] = test['um_action_0'] / (test['um_action_0'] + test['um_action_1'] +
                                                       test['um_action_2'] + test['um_action_3'] + 10)
    
    test['um_action_1_ratio'] = test['um_action_1'] / (test['um_action_0'] + test['um_action_1'] +
                                                       test['um_action_2'] + test['um_action_3'] + 10)
    
    test['um_action_2_ratio'] = test['um_action_2'] / (test['um_action_0'] + test['um_action_1'] +
                                                       test['um_action_2'] + test['um_action_3'] + 10)
    
    test['um_action_3_ratio'] = test['um_action_3'] / (test['um_action_0'] + test['um_action_1'] +
                                                       test['um_action_2'] + test['um_action_3'] + 10)

    feature.summary('user_id', 'item_id', 'cnt', 'user_item_', [np.size, np.mean, np.max, np.min], 0,0)
    feature.summary('user_id', 'cat_id', 'cnt', 'user_cat_', [np.size, np.mean, np.max, np.min], 0,0)
    feature.summary('user_id', 'brand_id', 'cnt', 'user_brand_',[np.size, np.mean, np.max, np.min], 0,0)
    feature.summary('user_id', 'merchant_id', 'cnt', 'user_merchant_', [np.size, np.mean, np.max, np.min], 0,0)

    # 用户在商铺的出现比例
    train = feature.train
    test = feature.test
    train['um_u_ratio'] = train['user_id_merchant_id_cnt'] / train['user_id_cnt']
    test['um_u_ratio'] = test['user_id_merchant_id_cnt'] / test['user_id_cnt']
    train['um_m_ratio'] = train['user_id_merchant_id_cnt'] / train['merchant_id_cnt']
    test['um_m_ratio'] = test['user_id_merchant_id_cnt'] / test['merchant_id_cnt']

    # 商铺下各种行为的统计
    feature.summary('merchant_id','action_type','cnt','merchant_action_cnt',None,1,0)
    feature.summary(['user_id'], 'action_type', 'cnt', 'user_action_cnt', None, 1, 0)


    # 商铺下年龄+性别 组合对item brand cat的次数统计
    feature.summary(['merchant_id', 'gender', 'age_range'],'item_id','cnt','merchant_gender_age_item_id_',
                    [np.size, np.mean, np.max, np.min], 0,0)
    feature.summary(['merchant_id', 'gender', 'age_range'], 'brand_id', 'cnt', 'merchant_gender_age_brand_id_',
                    [np.size, np.mean, np.max, np.min], 0,0)
    feature.summary(['merchant_id', 'gender', 'age_range'], 'cat_id', 'cnt', 'merchant_gender_age_cat_id_',
                    [np.size, np.mean, np.max, np.min], 0,0)

    # 店铺每个月有多少用户进行交互
    def count(x):
        return len(set(x))
   
    feature.summary(['merchant_id','user_id'],'month','cnt','merchant_month_user_cnt',count,1,0)
    train = feature.train
    test = feature.test

    # # 购买特征统计 # #
    feature = dataFeature(user_log[user_log['action_type']==2],train,test)

    feature.summary(['merchant_id'], 'time_stamp', 'merchant_buy_time_diff', None, timediff, 0, 0)

    feature.summary(['merchant_id'], 'month', 'merchant_month_cnt', 'merchant_month', None, 1, 0)

    # 商铺的回购率
    def rebuy(x):
        return len([i[0] for i in Counter(x).items() if i[1] > 1])
    feature.summary('merchant_id','user_id','merchant_allbuy_user_cnt',1,None,0,0)
    feature.summary('merchant_id','user_id','merchant_repeat_buy_user_cnt',None,rebuy,0,0)


    # 商铺+ cat的回购率
    feature.summary(['merchant_id', 'cat_id'], 'user_id', 'merchant_cat_repeat_ratio', 1,rebuy, 1,1)

    # 用户回购率
    feature.summary('user_id','merchant_id','user_repeat_buy_cnt',None,rebuy,0,0)
    train = feature.train
    test = feature.test
    train['user_repeat_buy_ratio'] = train['user_repeat_buy_cnt']/train['user_action_cnt_2']
    test['user_repeat_buy_ratio'] = test['user_repeat_buy_cnt']/test['user_action_cnt_2']

    # 判断是否重复购买用户 统计用户回购的总次数
    def rebuyuser(x):
        return {i[0]: i[1] for i in Counter(x).items() if i[1] > 1}
        
    feature.summary('merchant_id','user_id','repeat_user_list',None,rebuyuser,0,0)

    def extra_user_repeat_cnt(x):
        user_id = x['user_id']
        repeat_user_list = x['repeat_user_list']
        try:
            return repeat_user_list[user_id]
        except:
            return 0

    train = feature.train
    test = feature.test
    train['usr_repeat_cnt'] = train[['user_id', 'repeat_user_list']].apply(extra_user_repeat_cnt, axis=1)
    test['usr_repeat_cnt'] = test[['user_id', 'repeat_user_list']].apply(extra_user_repeat_cnt, axis=1)

