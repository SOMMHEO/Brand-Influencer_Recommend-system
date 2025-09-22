from training_modules.DB_connection import *
from training_modules.Real_time_Recommender import *
from pydantic import BaseModel
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import os

db_connector = None
recommender_service = None

class RealTimeItemRequest(BaseModel):
    item_uid: str
    member_uid: str

# @app.on_event("startup")
# def startup_event():
#     """
#     Initializes the recommender service on server startup
#     """
#     global db_connector, recommender_service

#     with open('config/accounts.json', 'r', encoding='utf-8') as f:
#         acc = json.load(f)
    
#     ssh_ip = acc['SSH_IP']
#     ssh_id = acc['SSH_ID']
#     ssh_pw = acc['SSH_PW']
#     mysql_id = acc['DB_ID']
#     mysql_pw = acc['DB_PW']
#     db_name = acc['DB_NAME']

#     db_connector = SSHMySQLConnector(ssh_ip, ssh_id, ssh_pw, mysql_id, mysql_pw, db_name)
#     db_connector.connect()

#     recommender_service = RecommenderService(
#         index_path = "bin/faiss_index.bin",
#         features_path = "bin/features.pkl",
#         db_connector = db_connector
#     )

@asynccontextmanager
async def lifespan(app: FastAPI):
    global db_connector, recommender_service
    
    # --- 서버 시작(startup) 시 실행될 코드 ---
    # accounts.json 파일 로딩
    with open('config/accounts.json', 'r', encoding='utf-8') as f:
        acc = json.load(f)
    
    # DB 연결 객체 생성
    db_connector = SSHMySQLConnector(
        acc['SSH_IP'], acc['SSH_ID'], acc['SSH_PW'], 
        acc['DB_ID'], acc['DB_PW'], acc['DB_NAME']
    )
    db_connector.connect()
    
    # RecommenderService 객체 생성
    # os.chdir("C:/Users/ehddl/Desktop/업무/code/recommendation_code/NCF/")

    recommender_service = RecommenderService(
        index_path="bin/faiss_item_index.bin",
        features_path="bin/features.pkl",
        db_connector=db_connector
    )

    print("--- Server startup complete. ---")
    yield
    # --- 서버 종료(shutdown) 시 실행될 코드 ---
    if db_connector:
        db_connector.close()
    print("--- Server shutdown complete. ---")

# FastAPI 앱을 생성할 때 lifespan을 인자로 전달
app = FastAPI(lifespan=lifespan)

# @app.post("/recommend_items")
# def recommend_items(request: RealTimeItemRequest):
#     """
#     Receives a product ID and returns a list of recommended items.
#     """
#     string_recommended_uids = recommender_service.recommend(request.member_uid, request.item_uid)

#     if not string_recommended_uids:
#         return JSONResponse(
#             content={"error": "Recommendations not found"}, 
#             status_code=404
#         )
    
#     return {"recommended_uids" : string_recommended_uids}

@app.post('/recommend_items')
def recommend_items(request: RealTimeItemRequest):
    print(f':받은_편지함_트레이: 요청 받은 UID → member_uid: {request.member_uid}, item_uid: {request.item_uid}')
    string_recommended_uids = recommender_service.recommend(request.member_uid, request.item_uid)
    print(f':다트: 추천 결과: {string_recommended_uids}')
    if not string_recommended_uids:
        print(':경고: 추천 결과 없음 → 404 리턴')
        return JSONResponse(
            content={'error': 'Recommendations not found'},
            status_code=404
        )
    return {'recommended_uids': string_recommended_uids}