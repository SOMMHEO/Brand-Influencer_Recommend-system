import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F

def get_top_k_recommendations(model, user_encoder, item_encoder, num_users, num_items, top_k=10):
    model.eval()
    device = next(model.parameters()).device  # 모델이 사용 중인 디바이스 추적
    recommendations = {}
    
    with torch.no_grad():
        for user_id in range(num_users):
            user_tensor = torch.tensor([user_id] * num_items).to(device)
            item_tensor = torch.tensor(range(num_items)).to(device)
            
            scores = model(user_tensor, item_tensor)
            _, indices = torch.topk(scores, top_k)
            
            recommended_items = item_encoder.inverse_transform(indices.cpu().numpy())
            recommendations[user_encoder.inverse_transform([user_id])[0]] = recommended_items.tolist()
    
    return recommendations

def precision_at_k(recommended_items, relevant_items, k):
    """
    Precision at k: 추천된 상위 k개의 아이템 중 실제 관심 있는 아이템의 비율
    """
    if len(recommended_items) == 0:
        return 0.0
    return len(set(recommended_items[:k]) & set(relevant_items)) / k

def recall_at_k(recommended_items, relevant_items, k):
    """
    Recall at k: 실제 관심 있는 아이템 중 상위 k개 추천 아이템의 비율
    """
    if len(relevant_items) == 0:
        return 0.0
    return len(set(recommended_items[:k]) & set(relevant_items)) / len(relevant_items)

def hit_rate_at_k(recommended_items, relevant_items, k):
    """
    Hit rate at k: 상위 k개 추천 아이템 중 하나라도 실제 관심 아이템이 포함되어 있으면 1
    """
    return int(any(item in relevant_items for item in recommended_items[:k]))

def mrr_at_k(recommended_items, relevant_items):
    """
    Mean Reciprocal Rank (MRR): 추천된 아이템에서 첫 번째 관심 아이템의 순위에 대한 역수
    """
    for i, item in enumerate(recommended_items):
        if item in relevant_items:
            return 1 / (i + 1)  # 순위는 1부터 시작
    return 0

def evaluate_recommendations(user_df, df_original, recommendations, top_k=10):
    """
    추천 리스트에 대해 Precision, Recall, Hit Rate, MRR 계산
    """
    precision_list = []
    recall_list = []
    hit_rate_list = []
    mrr_list = []

    total_users = user_df['user_uid'].unique()  # 전체 유저 리스트
    
    for user_uid in total_users:
        # 추천된 아이템 리스트
        recommended_items = recommendations.get(user_uid, [])
        
        # 실제 조회한 아이템 (유저가 실제로 조회한 아이템들)
        actual_items = df_original[df_original['user_uid'] == user_uid]['item_uid'].values.tolist()
        
        # 추천 아이템이 없거나 실제 조회 아이템이 없다면 건너뜀
        if not recommended_items or not actual_items:
            continue

        # Precision, Recall 계산
        precision = precision_at_k(recommended_items, actual_items, top_k)
        recall = recall_at_k(recommended_items, actual_items, top_k)
        precision_list.append(precision)
        recall_list.append(recall)
        
        # Hit Rate 계산
        hit_rate = hit_rate_at_k(recommended_items, actual_items, top_k)
        hit_rate_list.append(hit_rate)

        # MRR 계산
        mrr = mrr_at_k(recommended_items, actual_items)
        mrr_list.append(mrr)

    # 평균 Precision, Recall, Hit Rate, MRR 계산
    avg_precision = np.mean(precision_list)
    avg_recall = np.mean(recall_list)
    avg_hit_rate = np.mean(hit_rate_list)
    avg_mrr = np.mean(mrr_list)

    return avg_precision, avg_recall, avg_hit_rate, avg_mrr