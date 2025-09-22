from training_modules.DB_connection import *

import faiss
import pickle
import numpy as np
from datetime import datetime

import pymysql
from sshtunnel import SSHTunnelForwarder
import json

class RecommenderService:
    def __init__(self, index_path, features_path, db_connector):
        self.db_connector = db_connector
        print("RecommenderService initalized with DB connector.")
        """
        Initialize RecommenderService and load Faiss index & mapping files
        """
        print("Loading Faiss index from",  index_path)
        try:
            self.index = faiss.read_index(index_path)
            with open(features_path, "rb") as f:
                self.features = pickle.load(f)
            print("Faiss index and features loaded sucessfully.")
        except Exception as e:
            print(f"Error loading resources : {e}")
            self.index = None
            self.features = None
    
    ## DB Insert
    def Insert_Realtime_Item(self, member_uid, string_recommended_uids):
        # JSON 문자열로 변환
        recommended_uids_json = json.dumps(string_recommended_uids, ensure_ascii=False)
        regtime = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        try:
            self.db_connector.connection.ping(reconnect=True) # db 연결 끊겼을 시 재접속
            with self.db_connector.connection.cursor() as cursor: # db_connector
                sql = '''
                INSERT INTO int_real_time_product_recommendation (member_uid, item_uids, regtime, is_clicked)
                VALUES (%s, %s, %s, %s)
                '''
                cursor.execute(sql, (member_uid, recommended_uids_json, regtime, 0))
                self.db_connector.connection.commit()
                print(f"INSERT 완료: {member_uid}, {regtime}")
        except Exception as e:
            print(f"INSERT 실패: {e}")
    
    
    def recommend(self, member_uid: str, item_uid: str, k: int = 5):
        try:
            query_item_index = self.features['id_map'].get(int(item_uid))
        except (ValueError, KeyError) as e:
            print(f"Error: Invalid item_uid or key not found in id_map. {e}")
            return []
        
        if query_item_index is None:
            return []

        query_embedding = self.features['embeddings'][query_item_index].reshape(1, -1)

        distance, indices = self.index.search(query_embedding, k + 1)
        recommended_uids = [self.features['id_reverse_map'][i] for i in indices[0] if self.features['id_reverse_map'][i] != item_uid]
        
        # item_uid -> int to str
        string_recommended_uids = [str(uid) for uid in recommended_uids]
        
        self.Insert_Realtime_Item(member_uid, string_recommended_uids[:k])
        
        return string_recommended_uids[:k]
    