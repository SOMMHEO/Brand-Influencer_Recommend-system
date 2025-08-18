import pandas as pd
import numpy as np
import re
from tqdm import tqdm
from datetime import datetime
import time

from dotenv import load_dotenv
import json

from inference_modules.Recommender import *
from inference_modules.Load_NecessaryThings_from_s3 import *
from inference_modules.DB_connection_for_inference import * 
from libraries.slack_hook import SlackHook

import sys
import warnings
warnings.filterwarnings(action='ignore')


def main(argv_seller_uid=None):
    start_time = time.time()
    start_dt = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    aws_access_key = os.getenv("aws_accessKey")
    aws_secret_key = os.getenv("aws_secretKey")
    bucket_name = 'flexmatch-data'
    region = 'ap-northeast-2'

    # load data, encoder, model
    Load_from_s3 = LoadNecessaryThings(aws_access_key, aws_secret_key, bucket_name, region)

    Load_from_s3.load_data()
    user_info_ori = Load_from_s3.user_info_ori
    product_info_ori = Load_from_s3.product_info_ori
    user_view_ori = Load_from_s3.user_view_ori

    # additional data preprocessing
    user_info_ori['user_uid'] = user_info_ori['user_uid'].astype(str)
    product_info_ori['item_uid'] = product_info_ori['item_uid'].astype(str)
    user_view_ori['item_uid'] = user_view_ori['item_uid'].astype(str)
    user_view_ori['user_uid'] = user_view_ori['user_uid'].astype(str)

    user_info_ori_del = user_info_ori[
    (~user_info_ori['user_id'].astype(str).str.contains("test", na=False)) &
    (~user_info_ori['user_id'].astype(str).str.contains("same", na=False)) &
    (~user_info_ori['user_id'].astype(str).str.contains("DD", na=False)) &
    (~user_info_ori['user_id'].astype(str).str.contains("sell", na=False)) &
    (~user_info_ori['user_id'].astype(str).str.contains("Ro", na=False)) &
    (~user_info_ori['user_id'].astype(str).str.contains("dong", na=False)) &
    (~user_info_ori['user_id'].astype(str).str.contains("ehddlf", na=False)) &
    (~user_info_ori['user_id'].astype(str).str.contains("flex", na=False)) &
    (~user_info_ori['user_id'].astype(str).str.contains("qqqq", na=False)) &
    (~user_info_ori['user_id'].astype(str).str.contains("sylvileo", na=False)) &
    (~user_info_ori['user_id'].astype(str).str.contains("influDong1", na=False)) &
    (~user_info_ori['user_id'].astype(str).str.contains("devel", na=False)) &
    (~user_info_ori['user_id'].astype(str).str.contains("admin", na=False)) &
    (~user_info_ori['user_id'].astype(str).str.contains("ssh8918", na=False)) &
    (user_info_ori['user_uid'].astype(str).str.len() >= 4) &
    (user_info_ori['group_key'] == 2) &
    (~user_info_ori['add1'].astype(str).str.contains("dddd", na=False))
    ]
    # 이미 group_key=2 기반인 것을 가져오고 있긴 하지만 한 번 더 체크해줌.

    # user_info preprocessing
    user_info_del_lst = user_info_ori_del['user_uid'].unique()
    user_info_ori_del['add1'] = user_info_ori_del['add1'].str.replace('https://www.instagram.com/', '')
    user_info_ori_del['add1'] = user_info_ori_del['add1'].str.replace('https://instagram.com/', '')
    user_info_ori_del['acnt_nm'] = user_info_ori_del['add1'].str.replace('/', '')

    # user_view preprocessing
    user_view_ori_del = user_view_ori[user_view_ori['user_uid'].isin(user_info_del_lst)]

    # load additional info
    product_info, user_sales_info, instagram_user_category_info, not_conn_instagram_user_info, conn_instagram_user_info = get_additional_DB_info()    
    
    # create instagram data
    insta_profile_info = pd.concat([not_conn_instagram_user_info, conn_instagram_user_info], axis=0)
    insta_profile_info.drop(['acnt_nm'], axis=1, inplace=True)
    total_insta_user_info = pd.merge(instagram_user_category_info, insta_profile_info, on='acnt_id', how='left')
    total_insta_user_info.drop_duplicates(['member_uid', 'user_id', 'acnt_id', 'acnt_nm'], inplace=True)

    total_insta_user_info_2 = total_insta_user_info.drop(['user_id'], axis=1)
    user_info_merge_with_main_category = pd.merge(user_info_ori_del, total_insta_user_info_2, on='acnt_nm', how='left')
    
    # merge with DB data and instagram data
    all_df = pd.merge(user_view_ori_del, user_info_merge_with_main_category, on='user_uid', how='left')
    user_sales_info['user_uid'] = user_sales_info['user_uid'].astype(str)
    final_df = pd.merge(all_df, user_sales_info, on='user_uid', how='left')

    # influencer matching
    # recommended_influencers, seller_main_category_top3 = recommend_influencers_by_all_brands(product_info_ori, final_df)
    recommended_influencers = recommend_influencers_by_all_brands(product_info_ori, final_df)

    # calculate metric
    brand_match_results, overall_avg = calculate_brand_to_user_category_match(
        user_data=final_df,
        recommended_dict=recommended_influencers,
        product_info=product_info_ori
    )
    
    # 추천 성능 평가 결과 출력
    for seller_uid, match_rate in brand_match_results.items():
        if argv_seller_uid:
                
            if(seller_uid == argv_seller_uid):
                print(f"[{seller_uid}] 추천 인플루언서 카테고리 매칭률: {match_rate:.2f}%")
                break
        else:
            print(f"[{seller_uid}] 추천 인플루언서 카테고리 매칭률: {match_rate:.2f}%")
                
    if not argv_seller_uid:
        print(f"\n 전체 브랜드 추천 인플루언서 매칭률: {overall_avg:.2f}%")

    serializable_result = {
                    brand: df.to_dict(orient="records")  # 각 DataFrame → 리스트[딕셔너리]
                    for brand, df in recommended_influencers.items()
                }
    
    # print(serializable_result)
    result_json = json.dumps(serializable_result, ensure_ascii=False)

    ### DB Connection 추가
    with open('config/accounts.json', 'r', encoding='utf-8') as f:
        acc = json.load(f)

    ssh_ip = acc['SSH_IP']
    ssh_id = acc['SSH_ID']
    ssh_pw = acc['SSH_PW']
    mysql_id = acc['DB_ID']
    mysql_pw = acc['DB_PW']
    db_name = acc['DB_NAME']

    connector = SSHMySQLConnector(ssh_ip, ssh_id, ssh_pw, mysql_id, mysql_pw, db_name)
    connector.connect()

    ## test 시에 DB Insert는 잠시 off
    for seller_uid, user_list in serializable_result.items():
        if argv_seller_uid and argv_seller_uid != seller_uid:
            continue
                
        seller_uid = int(seller_uid)
        print(json.dumps([seller_uid, user_list], ensure_ascii=False, indent=4))  
        
        # 유저 리스트를 JSON 문자열로 변환 
        recom_user_list_json = json.dumps(user_list, ensure_ascii=False, indent=4)
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        value = [{
            'seller_uid': seller_uid,
            'recom_user_list': recom_user_list_json,
            'regdate': now,
        }]

        connector.insert_query('op_mem_seller_recommendation', value)
    connector.close()

    # slack alarm
    slack = SlackHook()

    end_time = time.time()
    end_dt = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    elapsed_time = end_time - start_time

    minutes = int(elapsed_time // 60)
    seconds = int(elapsed_time % 60)
    elapsed_str = f"{minutes}분 {seconds}초"

    print(elapsed_str)

    message = (
    f":rocket: 인플루언서 추천 모델 평가 완료!\n\n"
    f"*Start Time (시작 시간):* {start_dt}\n"
    f"*End Time (완료 시간):* {end_dt}\n"
    f"*Elapsed Time (걸린 시간):* {elapsed_str}\n\n"
    )

    slack = SlackHook()
    slack.send_message(message)

# if __name__=="__main__":

#     if len(sys.argv) > 1:
#         argv_seller_uid = int(sys.argv[1])
#     else :
#         argv_seller_uid = None

if __name__=="__main__":
    argv_seller_uid = int(sys.argv[1]) if len(sys.argv) > 1 else None
    main(argv_seller_uid)


