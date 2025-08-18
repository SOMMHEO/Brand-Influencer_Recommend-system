import pandas as pd
import numpy as np
import re
from tqdm import tqdm

from datetime import datetime
import time

from dotenv import load_dotenv
import json

from inference_modules.Load_NecessaryThings_from_s3 import *
from inference_modules.DB_connection_for_inference import * 
from libraries.slack_hook import SlackHook

from collections import defaultdict
from itertools import chain

import sys
import warnings
warnings.filterwarnings(action='ignore')

def recommend_influencers_by_all_brands(product_info, all_user_view_and_interest):
    result_dict = {}

    # 전체 브랜드 목록 정제 및 추출
    product_info = product_info.dropna(subset=['brand'])
    product_info['item_name'] = product_info['item_name'].apply(
        lambda x: re.sub('[^A-Za-z0-9가-힣]', '', str(x)) if pd.notna(x) else '')
    product_info['brand'] = product_info['brand'].apply(
        lambda x: re.sub('[^A-Za-z0-9가-힣]', '', str(x)) if pd.notna(x) else '')
    product_info['item_name'] = product_info['item_name'].apply(
        lambda x : re.sub(r'\d+회차', '', str(x)) if pd.notna(x) else '')

    product_info = product_info.drop_duplicates(subset=['brand', 'item_name'])

    # seller가 올린 상품들 중에서 unique categroy count가 가장 큰 3개를 추출 -> 가장 자주 올린 상품 카테고리 top3개 추출
    category_counts = product_info.groupby(['seller_uid', 'product_category']).size().reset_index(name='count')
    seller_main_category = category_counts.sort_values(['seller_uid', 'count'], ascending=[True, False])
    seller_main_category_top3 = seller_main_category.groupby('seller_uid').head(3).reset_index(drop=True)

    seller_recommendation_pool = defaultdict(list)

    for _, row in seller_main_category_top3.iterrows():
        seller = row['seller_uid']
        product_category = row['product_category']

        # 각 seller를 기준으로 등록된 상품의 top3 카테고리 추출
        brand_products = product_info[(product_info['seller_uid'] == seller) & (product_info['product_category'] == product_category)]
        if brand_products.empty:
            continue

        brand = brand_products['brand'].iloc[0]
        
        brand_category = str(brand_products['product_category'].iloc[0])
        brand_item_uids = brand_products['item_uid'].astype(str).tolist()
        # print(brand_item_uids)

        # 유저 데이터 준비
        user_data = all_user_view_and_interest.copy()
        # user_data = pd.merge(user_data, brand_category, on='item_uid')
        user_data['item_uid'] = user_data['item_uid'].astype(str)

        for col in ['interestcategory_1', 'interestcategory_2', 'interestcategory_3']:
            user_data[col] = user_data[col].fillna('').astype(str).str.strip()

        # 1. 브랜드 상품 조회 점수
        viewed = user_data[user_data['item_uid'].isin(brand_item_uids)].copy()
        viewed.drop_duplicates(subset=['user_uid', 'item_uid'], inplace=True)
        viewed_grouped = viewed.groupby(['user_uid', 'item_uid'])['view_cnt'].sum().reset_index()
        viewed_grouped['view_score'] = viewed_grouped['view_cnt'] * 0.5

        user_data = user_data.merge(viewed_grouped[['user_uid', 'view_score']], on='user_uid', how='left')
        user_data['view_score'] = user_data['view_score'].fillna(0)

        # 2. 관심 카테고리 매칭 점수
        category_match_score = pd.Series(0, index=user_data.index)
        category_match_score += (user_data['interestcategory_1'] == brand_category) * 5 
        category_match_score += (user_data['interestcategory_2'] == brand_category) * 3
        category_match_score += (user_data['interestcategory_3'] == brand_category) * 1
        user_data['category_match_score'] = category_match_score

        # 3. 매칭 성공률 점수 -> 편차가 크기 때문에 너무 많은 가중치가 들어갈까봐 일단 0.1로 설정
        user_data['success_match_score'] = user_data['success_match_count'] * 0.1

        # 최종 점수
        user_data['final_score'] = (
            user_data['view_score'] +
            user_data['category_match_score'] +
            user_data['success_match_score']
        )

        # 추천 상위 top_n
        recommended = user_data[user_data['final_score'] > 0].copy()    
        recommended = recommended.sort_values(by='final_score', ascending=False)
        recommended = recommended[['user_uid', 'user_id', 'final_score', 'interestcategory_1', 'interestcategory_2', 'interestcategory_3']].drop_duplicates('user_uid').head(10) # 수정

        # result_dict[(seller, brand)] = recommended
        seller_recommendation_pool[seller].append(recommended)
        # print(seller_recommendation_pool)

    # 각 seller별로 카테고리 추천 결과를 interleave 방식으로 병합
    for seller, rec_lists in seller_recommendation_pool.items():
        interleaved = list(chain.from_iterable(zip(*[df.itertuples(index=False) for df in rec_lists if not df.empty])))

        seen = set()
        final_recommend = []
        for r in interleaved:
            if r.user_uid not in seen:
                seen.add(r.user_uid)
                # final_recommend.append({'user_uid': r.user_uid, 'user_id': r.user_id, 'final_score': r.final_score})
                final_recommend.append({
                                        'user_uid': r.user_uid,
                                        'user_id': r.user_id,
                                        'final_score': r.final_score,
                                        'interestcategory_1': r.interestcategory_1,
                                        'interestcategory_2': r.interestcategory_2,
                                        'interestcategory_3': r.interestcategory_3
                                    })
            if len(final_recommend) == 10:
                break

        result_dict[seller] = pd.DataFrame(final_recommend)

    return result_dict, seller_main_category_top3

# def calculate_brand_to_user_category_match(user_data, recommended_dict, product_info):
#     brand_match_results = {}

#     for seller_uid, recommended in recommended_dict.items():
#         # 브랜드의 전체 상품 카테고리 (중복 제거)
#         product_info['seller_uid'] = product_info['seller_uid'].astype(int)
#         brand_products = product_info[product_info['seller_uid'] == seller_uid]
#         brand_categories = set(brand_products['product_category'].dropna().astype(str).str.strip().unique())

#         # 추천된 유저 리스트
#         user_uids = recommended['user_uid'].tolist()

#         # 매칭 체크
#         match_count = 0
#         total_count = len(user_uids)

#         for uid in user_uids:
#             user_row = user_data[user_data['user_uid'] == uid].drop_duplicates('user_uid')
#             if user_row.empty:
#                 continue

#             interests = set(user_row.iloc[0][['interestcategory_1', 'interestcategory_2', 'interestcategory_3']].dropna())
#             interests = {str(cat).strip() for cat in interests}

#             # 교집합이 하나라도 있으면 매칭
#             if brand_categories & interests:
#                 match_count += 1

#         match_rate = match_count / total_count if total_count > 0 else 0
#         brand_match_results[seller_uid] = round(match_rate * 100, 2)

#     # 전체 평균 계산
#     all_rates = list(brand_match_results.values())
#     overall_avg = round(np.mean(all_rates), 2) if all_rates else 0.0

#     return brand_match_results, overall_avg

def calculate_brand_to_user_category_match(user_data, recommended_dict, product_info):
    brand_match_results = {}

    for seller_uid, recommended in recommended_dict.items():
        product_info['seller_uid'] = product_info['seller_uid'].astype(int)
        brand_products = product_info[product_info['seller_uid'] == seller_uid]
        brand_categories = set(brand_products['product_category'].dropna().astype(str).str.strip().unique())

        user_uids = recommended['user_uid'].unique()  # 중복 제거
        match_count = 0
        total_count = len(user_uids)

        for uid in user_uids:
            user_rows = user_data[user_data['user_uid'] == uid]
            if user_rows.empty:
                continue

            # 한 유저가 여러 row로 나뉘어 있을 경우 병합
            interest_cols = ['interestcategory_1', 'interestcategory_2', 'interestcategory_3']
            interests = set()
            for col in interest_cols:
                interests.update(user_rows[col].dropna().astype(str).str.strip().tolist())

            if brand_categories & interests:
                match_count += 1
            else:
                print(f"[NO MATCH] uid: {uid}, 관심사: {interests}, 브랜드 카테고리: {brand_categories}")

        match_rate = match_count / total_count if total_count > 0 else 0
        brand_match_results[seller_uid] = round(match_rate * 100, 2)

    all_rates = list(brand_match_results.values())
    overall_avg = round(np.mean(all_rates), 2) if all_rates else 0.0

    return brand_match_results, overall_avg



if __name__=="__main__":

    if len(sys.argv) > 1:
        argv_seller_uid = int(sys.argv[1])
    else :
        argv_seller_uid = None

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
    (~user_info_ori['user_id'].str.contains("test", na=False)) & (~user_info_ori['user_id'].str.contains("same")) & (~user_info_ori['user_id'].str.contains("DD")) &
    (~user_info_ori['user_id'].str.contains("sell")) & (~user_info_ori['user_id'].str.contains("Ro")) & (~user_info_ori['user_id'].str.contains("dong")) &
    (~user_info_ori['user_id'].str.contains("ehddlf")) & (~user_info_ori['user_id'].str.contains("flex")) & 
    (~user_info_ori['user_id'].str.contains("qqqq")) & (~user_info_ori['user_id'].str.contains("sylvileo")) & 
    (~user_info_ori['user_id'].str.contains("influDong1")) & (~user_info_ori['user_id'].str.contains("devel")) & 
    (~user_info_ori['user_id'].str.contains("admin")) & (user_info_ori['user_uid'].astype(str).str.len() >= 4) & (user_info_ori['group_key']==2)
    ] # 이미 group_key=2 기반인 것을 가져오고 있긴 하지만 한 번 더 체크해줌.

    user_info_del_lst = user_info_ori_del['user_uid'].unique()
    user_view_ori_del = user_view_ori[user_view_ori['user_uid'].isin(user_info_del_lst)]

    all_user_view_and_interest = pd.merge(user_view_ori_del, user_info_ori_del, on='user_uid')
    # all_user_view_and_interest.to_excel("result/brand_matching_all_user_view_and_interest.xlsx")

    # influencer matching
    recommended_influencers, seller_main_category_top3 = recommend_influencers_by_all_brands(product_info_ori, all_user_view_and_interest)

    # calculate metric
    brand_match_results, overall_avg = calculate_brand_to_user_category_match(
        user_data=all_user_view_and_interest,
        recommended_dict=recommended_influencers,
        product_info=seller_main_category_top3
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



