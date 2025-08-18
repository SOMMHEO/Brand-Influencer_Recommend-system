import pandas as pd
import re
import numpy as np
import json
import torch
from collections import defaultdict
from datetime import datetime
from inference_modules.DB_connection_for_inference import InsertQuery

## influencer-item recommendation
def generate_recommendations_and_matching_rate(user_encoder, item_encoder, neumf_model, user_view_ori, product_info_ori, user_info_ori, product_info, device, sp_user_uid):
    neumf_model.eval()  # 평가 모드로 전환
    num_users = len(user_encoder.classes_)
    num_items = len(item_encoder.classes_)

    category_matching_rates = []

    # 사용자 ID 통일
    user_view_ori['user_uid'] = user_view_ori['user_uid'].astype(str)
    user_encoder.classes_ = np.array([str(uid) for uid in user_encoder.classes_])

    # 사용자별 조회 아이템 (str 처리)
    user_seen_items = {
        user: set(str(item) for item in items)
        for user, items in user_view_ori.groupby("user_uid")["item_uid"].agg(list).items()
    }

    # 현재 아카이브 기준 인기 상품
    product_info['item_uid'] = product_info['item_uid'].astype(str)
    popular_items_by_category = (
        product_info.groupby("product_category")[["item_uid", "viewcount", "product_category"]]
        .apply(lambda x: x.nlargest(10, "viewcount"))
        .reset_index(drop=True)
    )

    current_items = set(product_info["item_uid"].unique())  # 현재 아카이브에 존재하는 아이템

    final_recommendations = {}

    with torch.no_grad():
        for user_id in range(num_users):
            real_user_id = user_encoder.inverse_transform([user_id])[0]
            
            if sp_user_uid != None and real_user_id != sp_user_uid:
                continue

            seen_items = user_seen_items.get(real_user_id, set())

            if real_user_id in user_view_ori["user_uid"].values:
                # NCF 추천
                user_tensor = torch.tensor([user_id] * num_items).to(device)
                item_tensor = torch.tensor(range(num_items)).to(device)

                scores = neumf_model(user_tensor, item_tensor).cpu().numpy()
                item_ids = item_encoder.inverse_transform(np.arange(num_items))
                item_scores = list(zip([str(item) for item in item_ids], scores))

                filtered = [(item, float(score)) for item, score in item_scores if item not in seen_items and item in current_items]
                top_k = sorted(filtered, key=lambda x: x[1], reverse=True)[:10]

                final_recommendations[str(real_user_id)] = top_k  # (item_uid, score)

                # 추천 결과 디비에 삽입
                InsertQuery(real_user_id, top_k)

            else:
                # Cold Start 추천
                user_row = user_info_ori[user_info_ori["user_uid"] == real_user_id]
                interest_categories = []

                if not user_row.empty:
                    interest_categories = user_row.iloc[0][
                        ["interestcategory_1", "interestcategory_2", "interestcategory_3"]
                    ].dropna().values

                recommended_items = []
                for category in interest_categories:
                    if category in popular_items_by_category["product_category"].unique():
                        top_items = popular_items_by_category[
                            popular_items_by_category["product_category"] == category
                        ]["item_uid"].tolist()
                        recommended_items.extend(top_items)

                filtered_recommendations = list(dict.fromkeys(recommended_items))[:10]

                if len(filtered_recommendations) < 10:
                    backup_items = product_info.sort_values(by="viewcount", ascending=False)["item_uid"].tolist()
                    backup_fill = [item for item in backup_items if item not in filtered_recommendations]
                    filtered_recommendations.extend(backup_fill[:10 - len(filtered_recommendations)])

                final_recommendations[str(real_user_id)] = [(item, None) for item in filtered_recommendations]
                
                # 추천 결과 삽입
                # InsertQuery(user_id, final_recommendations)
                InsertQuery(real_user_id, [(item, None) for item in filtered_recommendations])


    # 카테고리 매칭률 계산
    user_seen_items_dict = {
        user: set(items) for user, items in user_view_ori.groupby("user_uid")["item_uid"].agg(list).items()
    }

    for user_id, recommended in final_recommendations.items():
        user_row = user_info_ori[user_info_ori["user_uid"] == int(user_id)]

        if user_row.empty:
            continue  # 유저 정보가 없으면 스킵

        # 추천 리스트가 (item, score) 튜플인지 단일 아이템 리스트인지 판단
        if isinstance(recommended[0], tuple):
            recommended_items = [str(item) for item, _ in recommended]
        else:
            recommended_items = [str(item) for item in recommended]

        # 추천 아이템들의 카테고리 set 추출
        recommended_df = product_info[product_info["item_uid"].astype(str).isin(recommended_items)]
        recommended_categories = set(recommended_df["product_category"].values)

        seen_items = set(str(i) for i in user_seen_items_dict.get(str(user_id), set()))

        if seen_items:
            # NCF 추천 유저
            seen_categories = set(product_info_ori[product_info_ori["item_uid"].astype(str).isin(seen_items)]["product_category"].values)
            matching_rate = 1.0 if recommended_categories & seen_categories else 0.0
        else:
            # Cold Start 유저
            interest_categories = set(user_row.iloc[0][["interestcategory_1", "interestcategory_2", "interestcategory_3"]].dropna().values)
            matching_rate = 1.0 if recommended_categories & interest_categories else 0.0

        category_matching_rates.append(matching_rate)

    # 결과 출력
    print(json.dumps(final_recommendations, indent=2, ensure_ascii=False))
    if sp_user_uid == None:
        # print(f"전체 평균 카테고리 매칭률: {np.mean(category_matching_rates):.2%}")
        print(f"전체 유저의 추천 상품 매칭률: {np.mean(category_matching_rates):.2%}")
    else : 
        print(f"[{sp_user_uid}] 추천 상품 매칭률 : {np.mean(category_matching_rates):.2%}")
    return final_recommendations, np.mean(category_matching_rates)


## brand-influencer recommendation
def recommend_influencers_by_all_brands(product_info, final_df):
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

    category_counts = product_info.groupby(['seller_uid', 'product_category']).size().reset_index(name='count')
    seller_main_category = category_counts.sort_values(['seller_uid', 'count'], ascending=[True, False])
    # seller_main_category_top3 = seller_main_category.groupby('seller_uid').head(3).reset_index(drop=True)

    seller_recommendation_pool = defaultdict(list)

    for _, row in seller_main_category.iterrows():
        seller = row['seller_uid']
        product_category = row['product_category']

        # 각 seller를 기준으로 등록된 상품의 카테고리 추출
        brand_products = product_info[(product_info['seller_uid'] == seller) & (product_info['product_category'] == product_category)]
        if brand_products.empty:
            continue

        brand = brand_products['brand'].iloc[0]
        
        brand_category = str(brand_products['product_category'].iloc[0])
        brand_item_uids = brand_products['item_uid'].astype(str).tolist()
        # print(brand_item_uids)

        # 유저 데이터 준비
        user_data = final_df.copy()
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
        # 디딤돌 연구과제의 경우 5,3,1 점으로 조절했음
        category_match_score = pd.Series(0, index=user_data.index)
        category_match_score += (user_data['interestcategory_1'] == brand_category) * 3 
        category_match_score += (user_data['interestcategory_2'] == brand_category) * 2
        category_match_score += (user_data['interestcategory_3'] == brand_category) * 1
        user_data['category_match_score'] = category_match_score

        # 3. 매칭 성공률 점수 -> 편차가 크기 때문에 너무 많은 가중치가 들어갈까봐 일단 0.1로 설정
        user_data['success_match_score'] = user_data['success_match_count'] * 0.1

        # 4. 최근 가입한 유저일수록 가중치 부여 -> then 가중치를 어떻게 부여할 것인가?
        # 가입일에서 오늘까지의 경과일을 계산
        user_data['reg_datetime'] = pd.to_datetime(user_data['reg_datetime'], errors='coerce')
        user_data['days_since_reg'] = (datetime.now() - user_data['reg_datetime']).dt.days

        # 한 달이내 가입자 최대 5점, 60일~180일 3점, 180일~240일은 1점, 그 이상은 0점
        def calc_recent_join_score(days):
            if pd.isna(days):
                return 0
            if days <= 30:
                return 3
            elif 60 <= days <= 180:
                return 2
            elif 180 < days <= 240:
                return 1
            else:
                return 0

        user_data['recent_join_score'] = user_data['days_since_reg'].apply(calc_recent_join_score)
 
        # 5. 인스타그램 데이터가 있는 경우 팔로워 정보 반영
        def influencer_scale_type(count):
            if pd.isna(count):
                return None
            count = int(count)
            if count < 1000:
                return 'nano'
            elif 1000 <= count <= 10000:
                return 'micro'
            elif 10000 < count <= 100000:
                return 'mid'
            elif 100000 < count <= 500000:
                return 'macro'
            else:
                return 'mega'

        scale_weight = {
            'nano': 1,
            'micro': 2,
            'mid': 3,
            'macro': 4,
            'mega': 5
        }

        user_data['influencer_scale'] = user_data['follower_cnt'].apply(influencer_scale_type)
        user_data['follower_score'] = user_data['influencer_scale'].map(scale_weight).fillna(0)


        # 6. 인스타그램 데이터가 있는 경우 전문 카테고리 부분 반영
        # 카테고리 매핑 테이블
        category_map = {
            '다이어트/건강보조식품': '헬시',
            '유명장소/핫플': '푸드',
            '스포츠': '스포츠',
            'IT': 'IT',
            '뷰티': '뷰티',
            '베이비/키즈': '베이비/키즈',
            '패션': '패션'
            # 나머지는 서비스 매칭
        }

        def normalize_category(cat):
            return category_map.get(cat, '서비스')

        def top3_match_score(top3_str, brand_category):
            if pd.isna(top3_str) or pd.isna(brand_category):
                return 0
            user_cats = [normalize_category(c.strip()) for c in top3_str.split('@')]
            if normalize_category(brand_category) in user_cats:
                return 3  # 일치 시 가산점
            return 0

        user_data['top3_category_score'] = user_data['top_3_category'].apply(
            lambda x: top3_match_score(x, brand_category)
        )

        # 7. 스토어 방문자가 많은 사람한테 조금 더 가중치
        # 7. 스토어 방문 횟수 기반 가중치
        def calc_visit_score(visits):
            if pd.isna(visits) or visits <= 0:
                return 0
            elif visits <= 10:
                return 1
            elif visits <= 100:
                return 2
            elif visits <= 500:
                return 3
            else:
                return 4

        user_data['store_visit_score'] = user_data['total_visit'].apply(calc_visit_score)

        # 최종 점수
        user_data['final_score'] = (
            user_data['view_score'] +
            user_data['category_match_score'] +
            user_data['recent_join_score'] +
            user_data['success_match_score'] +
            user_data['follower_score'] + 
            user_data['top3_category_score'] +
            user_data['store_visit_score'])

        # 추천 상위 top_n
        recommended = user_data.copy()    # 변경
        recommended = recommended.sort_values(by='final_score', ascending=False)
        recommended = recommended[['user_uid', 'user_id', 'final_score']].drop_duplicates('user_uid').head(30) # 변경

        # result_dict[(seller, brand)] = recommended
        seller_recommendation_pool[seller].append(recommended)
        # print(seller_recommendation_pool)

    for seller, rec_lists in seller_recommendation_pool.items():
        combined = pd.concat(rec_lists, ignore_index=True)
        combined = combined.drop_duplicates(subset=['user_uid'])
        combined = combined.sort_values(by='final_score', ascending=False).head(30)
        result_dict[seller] = combined

    return result_dict

## brand-influencer recommendation metric 
def calculate_brand_to_user_category_match(user_data, recommended_dict, product_info):
    brand_match_results = {}

    for seller_uid, recommended in recommended_dict.items():
        # 브랜드의 전체 상품 카테고리 (중복 제거)
        product_info['seller_uid'] = product_info['seller_uid'].astype(int)
        brand_products = product_info[product_info['seller_uid'] == seller_uid]
        brand_categories = set(brand_products['product_category'].dropna().astype(str).str.strip().unique())

        # 추천된 유저 리스트
        user_uids = recommended['user_uid'].tolist()

        # 매칭 체크
        match_count = 0
        total_count = len(user_uids)

        for uid in user_uids:
            user_row = user_data[user_data['user_uid'] == uid].drop_duplicates('user_uid')
            if user_row.empty:
                continue

            interests = set(user_row.iloc[0][['interestcategory_1', 'interestcategory_2', 'interestcategory_3']].dropna())
            interests = {str(cat).strip() for cat in interests}

            # 교집합이 하나라도 있으면 매칭
            if brand_categories & interests:
                match_count += 1

        match_rate = match_count / total_count if total_count > 0 else 0
        brand_match_results[seller_uid] = round(match_rate * 100, 2)

    # 전체 평균 계산
    all_rates = list(brand_match_results.values())
    overall_avg = round(np.mean(all_rates), 2) if all_rates else 0.0

    return brand_match_results, overall_avg