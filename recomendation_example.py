"""

Method Description:

This program is trying to predict yelp reviews scores from the yelp dataset.
This program uses two boosted tree models from the library catboost to make a recommendation system for a large scale dataset.
In order to parse the dataset in the first place it uses Apache Spark.
The first model is less effective but more useful in cases where there are cold starts and sparce data.
The second model uses more information and is more useful in the cases where there is existing data from which to make a recommendation.
Together the models take a weighted average to decide the final score.
Given the large dataset it takes a few minutes to run.

Error Distribution:
>=0 and <1: 102555
>=1 and <2: 32549
>=2 and <3: 6088
>=3 and <4: 848
>4: 4

RMSE:
0.9526919387608324

Execution Time:
482.4467318058014 ms

"""

import sys
import json
import time
import pandas as pd
import numpy as np
import math
import xgboost as xgb
from catboost import CatBoostRegressor
from collections import Counter
from pyspark import SparkContext, SparkConf
from sklearn import metrics

# spark function to split csv data files
def split_rdd(rdd):
    ith = rdd.first()
    return rdd.filter(lambda x:x!=ith).map(lambda x: x.split(","))

# spark function to load the data from the json files and use mapreduce to sieve needed data
def make_maps(rdd, sc, user, business, tip, checkin, photo):
    usr = sc.textFile(user).map(json.loads).map(lambda x:(x['user_id'],(x['review_count'],x['average_stars'],
                                                                         len(x['friends'].split(", ")),x['fans'],
                                                                         x['useful'],x['funny'],x['cool'],
                                                                         x['compliment_hot'], x['compliment_plain'],
                                                                         x['compliment_cool'], x['compliment_note'],
                                                                         x['compliment_photos']
                                                                         ))).collectAsMap()

    biz = sc.textFile(business).map(json.loads).map(lambda x:(x['business_id'],(x['review_count'],x['stars'],
                                                                         x['longitude'], x['latitude'],
                                                                         len(x["categories"].split(", ")) if x["categories"] is not None else 0,
                                                                         len(x.get("attributes")) if x["attributes"] is not None else 0
                                                                         ))).collectAsMap()
    tips = sc.textFile(tip).map(json.loads).persist()
    tips_pair= tips.map(lambda x: ((x['user_id'], x['business_id']), 1)).reduceByKey(lambda x,y: x+y).collectAsMap()
    tips_biz = tips.map(lambda x: (x['business_id'], 1)).reduceByKey(lambda x,y: x+y).collectAsMap()
    tips_user = tips.map(lambda x: (x['user_id'], 1)).reduceByKey(lambda x,y: x+y).collectAsMap()

    photos = sc.textFile(photo).map(json.loads).map(lambda x: (x['business_id'], 1)).reduceByKey(lambda x,y: x+y).collectAsMap()

    checkins = sc.textFile(checkin).map(json.loads).map(lambda x: (x['business_id'], len(x.get("time")) if x["time"] is not None else 0)).reduceByKey(lambda x,y: x+y).collectAsMap()

    user_all_dict = rdd.map(lambda x: (x[0], (x[1], x[2]))).groupByKey().mapValues(lambda x: dict(x)).collectAsMap()
    biz_all_dict = rdd.map(lambda x: (x[1], (x[0], x[2]))).groupByKey().mapValues(lambda x: dict(x)).collectAsMap()

    user_ave_dict = rdd.map(lambda x: (x[0], x[2])).groupByKey().mapValues(lambda x: (sum(x)/len(x))).collectAsMap()
    biz_ave_dict = rdd.map(lambda x: (x[1], x[2])).groupByKey().mapValues(lambda x: (sum(x)/len(x))).collectAsMap()

    return usr, user_ave_dict, biz, biz_ave_dict, tips_pair, tips_user, tips_biz, photos, checkins, user_all_dict, biz_all_dict

# combines information from the maps made by the spark functions
def process_info(x, usr_all, user_ave, user_ect, buisness_all, business_aves, business_ect):

    user_total_ave = sum(user_ave.values())/len(user_ave.values())
    usr_rev_count = 0

    bz_total_ave = sum(business_aves.values())/len(business_aves.values())
    bz_rev_count=0

    if user_ect.get(x[0]):
        usr_rev_count = user_ect[x[0]][0]
        usr_ave = user_ect[x[0]][1]
    else:
        if user_ave.get(x[0]):
            usr_ave = user_ave[x[0]]
        else:
            usr_ave = user_total_ave
        if usr_all.get(x[0]):
            usr_rev_count = len(usr_all[x[0]])

    if business_ect.get(x[1]):
        bz_rev_count = business_ect[x[1]][0]
        bz_ave = business_ect[x[1]][1]
    else:
        if business_aves.get(x[1]):
            bz_ave = business_aves[x[1]]
        else:
            bz_ave = bz_total_ave
        if buisness_all.get(x[1]):
            bz_rev_count = len(buisness_all[x[1]])

    return usr_ave, bz_ave, bz_rev_count, usr_rev_count

# processes info for model 2
def process_info_2(x, user_ect, business_ect, tips_pair, tips_user, tips_biz, photos, checkins):

    out_list = []
    usr = x[0]
    biz = x[1]

    usr_inf = user_ect.get(usr, [np.nan]*12)
    out_list+=usr_inf
    biz_inf = business_ect.get(biz, [np.nan]*6)
    out_list += biz_inf
    tip_inf = [tips_pair.get((usr, biz), np.nan), tips_user.get(usr, np.nan), tips_biz.get(biz, np.nan)]
    out_list += tip_inf
    photos_inf = photos.get(biz, np.nan)
    out_list += [photos_inf]
    checkins_inf = checkins.get(biz, np.nan)
    out_list += [checkins_inf]

    return out_list

# makes the sets compliant with what the catboost library wants
def make_XY_set_2(in_file, listy):
    in_arr= pd.read_csv(in_file)
    x_arr = np.array(listy)
    y = in_arr['stars'].to_numpy()
    return x_arr, y

# makes the sets compliant with what the catboost library wants
def make_XY_set(in_file, listy):
    base_df = pd.read_csv(in_file)
    usr_ave = []
    bz_ave = []
    usr_rev_count = []
    biz_rev_count = []
    for thing in listy:
        usr_ave.append(thing[0])
        bz_ave.append(thing[1])
        biz_rev_count.append(thing[2])
        usr_rev_count.append(thing[3])

    base_df['user_id_enc'] = pd.Categorical(base_df['user_id']).codes
    base_df['business_id_enc'] = pd.Categorical(base_df['business_id']).codes
    base_df["user_average"] = usr_ave
    base_df["bz_average"] = bz_ave
    base_df["user_review_count"] = usr_rev_count
    base_df["bz_review_count"] = biz_rev_count

    x_cols=['user_id_enc', 'business_id_enc', "user_average", "bz_average", "user_review_count", "bz_review_count"]
    y=['stars']
    return base_df[x_cols], base_df[y]

# this makes 
def weighted_ave_switcher(test_set, usr, bz, pred1, pred2):
    in_arr = pd.read_csv(test_set)

    preds=[]

    for i in range(len(in_arr)):
        if not usr.get(in_arr.iloc[i][0]) or not bz.get(in_arr.iloc[i][1]):
            out_pred = (pred1[i] * .4) + (pred2[i] * .6)
        else:
            out_pred = pred2[i]

        preds.append(out_pred)

    return preds

def project(folder, in_file, out_file):
    # start spark
    sc = SparkContext().getOrCreate()
    sc.setLogLevel("WARN")
    print("spark started")
    
    # load in datasets
    train_set = folder + "yelp_train.csv"
    test_set = in_file
    user = folder + "user.json"
    business = folder + "business.json"
    tip = folder + "tip.json"
    checkin = folder + "checkin.json"
    photo = folder+"photo.json"

    # build a spark rdd
    rdd = split_rdd(sc.textFile(train_set)).map(lambda x: (x[0], x[1],float(x[2]))).persist()
    
    # use map reduce to get necesary data and return maps
    usr, user_ave_dict, biz, biz_ave_dict, tips_pair, tips_user, tips_biz, photos, checkins, user_all_dict, biz_all_dict = make_maps(rdd, sc, user, business, tip, checkin, photo)

    # make rdd for test data and make maps
    rdd2 =split_rdd(sc.textFile(test_set))
    output_collection = rdd2.map(lambda x: (x[0], x[1])).collect()
    rdd2= rdd2.map(lambda x: (x[0], x[1])).persist()

    print("maps made")

    # process the maps into the train and test sets for the first model
    train_list = rdd.map(
        lambda x: process_info(x, user_all_dict, user_ave_dict, usr, biz_all_dict, biz_ave_dict, biz)).collect()
    test_list = rdd2.map(
        lambda x: process_info(x, biz_all_dict, user_ave_dict, usr, biz_all_dict, biz_ave_dict, biz)).collect()

    X_train, y_train = make_XY_set(train_set, train_list)
    X_test, y_test = make_XY_set(test_set, test_list)

    # build catboost model
    model_1 = CatBoostRegressor(objective="RMSE",learning_rate=.07,l2_leaf_reg=1.25,max_depth=6,random_seed=553,logging_level="Silent")
    model_1.fit(X_train, y_train)
    prediction_1 = model_1.predict(X_test)

    print(math.sqrt(metrics.mean_squared_error(y_test, prediction_1)))

    print("model 1 done")

    # build second more detailed model in the style of the first
    train_list_2 = rdd.map(
        lambda x: process_info_2(x, usr, biz, tips_pair, tips_user, tips_biz, photos, checkins)).collect()
    test_list_2 = rdd2.map(
        lambda x: process_info_2(x, usr, biz, tips_pair, tips_user, tips_biz, photos, checkins)).collect()

    X_train, y_train = make_XY_set_2(train_set, train_list_2)
    X_test, y_test = make_XY_set_2(test_set, test_list_2)

    model_2 = CatBoostRegressor(objective="RMSE", learning_rate=0.15, max_depth=6, iterations=1000, l2_leaf_reg=1.25,
                              random_seed=553, logging_level="Silent")

    model_2.fit(X_train, y_train)
    prediction_2 = model_2.predict(X_test)

    print(math.sqrt(metrics.mean_squared_error(y_test, prediction_2)))
    print("model 2 done")

    # take weighted average
    out_preds = weighted_ave_switcher(test_set, usr, biz, prediction_1, prediction_2)

    print("weighting done")

    print()
    # get error distribution
    er_counter = Counter(er1=0, er2=0, er3=0, er4=0, er5=0)

    for i in range(len(out_preds)):
        er = abs(out_preds[i] - y_test[i])
        if er < 1:
            er_counter.update(["er1"])
        elif er < 2:
            er_counter.update(["er2"])
        elif er < 3:
            er_counter.update(["er3"])
        elif er < 4:
            er_counter.update(["er4"])
        else:
            er_counter.update(["er5"])

    print("Error Distribution:")
    print(">=0 and <1: " + str(er_counter.get("er1")))
    print(">=1 and <2: " + str(er_counter.get("er2")))
    print(">=2 and <3: " + str(er_counter.get("er3")))
    print(">=3 and <4: " + str(er_counter.get("er4")))
    print(">4: " + str(er_counter.get("er5")))

    print()
    # final rmse
    print("RMSE:")
    print(math.sqrt(metrics.mean_squared_error(y_test, out_preds)))
    print()

    sc.stop()

    with open(out_file, "w") as f:
        f.writelines("user_id, business_id, prediction\n")
        for i in range(len(output_collection)):
            f.writelines(str(output_collection[i][0])+","+str(output_collection[i][1])+","+str(out_preds[i])+"\n")

if __name__ == '__main__':
    start_time = time.time()
    folder = "data/"
    in_file = "data/yelp_val.csv"
    out_file = "project_out.csv"

    project(folder, in_file, out_file)
    print("Execution Time:")
    print(time.time() - start_time)