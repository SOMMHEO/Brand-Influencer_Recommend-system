from training_modules.DB_connection import *
from training_modules.Data_preprocessing import *
from training_modules.NCF_model import *
from training_modules.Training_Recommender_Evaluation import *
from libraries.slack_hook import SlackHook

from dotenv import load_dotenv
import os
import boto3
import io
from datetime import datetime
import time

load_dotenv()
aws_access_key = os.getenv("aws_accessKey")
aws_secret_key = os.getenv("aws_secretKey")
region_name='ap-northeast-2'
 
s3 = boto3.client('s3',
                    aws_access_key_id=aws_access_key,
                    aws_secret_access_key=aws_secret_key,
                    region_name=region_name) # ec2 사용 시 변경

def main():
    # slack alarm - start time
    slack = SlackHook()
    start_time = time.time()
    start_dt = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # inference start message
    slack.send_message(f":rocket: 추천 inference 시작! (local test) \n\n*Start Time (시작 시간):* {start_dt}")

    user_info, product_info, user_view = get_all_infos()

    print(len(user_info), len(product_info), len(user_view))
    
    # save the encoder
    bucket_name = 'flexmatch-data'
    user_encoder_key = 'recommendation_system/encoder/user_encoder.pkl'
    item_encoder_key = 'recommendation_system/encoder/item_encoder.pkl'

    # Data preprocessing
    preprocessor = Preprocessor(user_info, product_info, user_view, bucket_name, user_encoder_key, item_encoder_key)
    # user_info, user_info_ori = preprocessor.preprocess_user_info()
    # product_info, product_info_ori = preprocessor.preprocess_product_info()
    # user_view, user_view_ori = preprocessor.preprocess_user_view()
    user_info = preprocessor.preprocess_user_info()
    product_info = preprocessor.preprocess_product_info()
    user_view = preprocessor.preprocess_user_view()

    splitter = DataSplitter(user_view, user_info, product_info)
    # train_data, test_data, user_encoder, item_encoder = splitter.encode_and_split()
    train_data, test_data, user_encoder, item_encoder, user_info_ori, product_info_ori, user_view_ori = splitter.encode_and_split()
    train, test = splitter.negative_sampling(train_data, test_data)

    train_dataset = NCFDataset(train) # train_with_neg
    test_dataset = NCFDataset(test) # test_with_neg

    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)

    # save the prepocessed data
    current_date = datetime.now().strftime('%y-%m-%d')

    user_buffer = io.StringIO()
    user_info_ori.to_csv(user_buffer, index=False)
    user_ino_data_path = f'recommendation_system/data/{current_date}/user_info_ori.csv' # 날짜별로 저장
    
    product_buffer = io.StringIO()
    product_info_ori.to_csv(product_buffer, index=False)
    product_data_path = f'recommendation_system/data/{current_date}/product_info_ori.csv'
    
    user_view_buffer = io.StringIO()
    user_view_ori.to_csv(user_view_buffer, index=False)
    user_view_path = f'recommendation_system/data/{current_date}/user_view_ori.csv'

    s3.put_object(Bucket=bucket_name, Key=user_ino_data_path, Body=user_buffer.getvalue())
    s3.put_object(Bucket=bucket_name, Key=product_data_path, Body=product_buffer.getvalue())
    s3.put_object(Bucket=bucket_name, Key=user_view_path, Body=user_view_buffer.getvalue())

    # training
    num_users = len(user_encoder.classes_)
    num_items = len(item_encoder.classes_)

    gmf_model = train_gmf_model(train_loader, num_users, num_items, latent_dim=32)
    mlp_model = train_mlp_model(train_loader, num_users, num_items, layers=[64,32,16,8])

    # 사전 학습 weight로 NeuMF 구성
    neumf_model = build_neumf_model(num_users, num_items, latent_dim=32, layers=[64,32,16,8])

    # NeuMF fine-tuning
    train_neumf_model(train_loader, neumf_model, epochs=20) 

    recommendations = get_top_k_recommendations(neumf_model, user_encoder, item_encoder, num_users, num_items, top_k=10)

    avg_precision, avg_recall, avg_hit_rate, avg_mrr = evaluate_recommendations(user_info_ori, user_view_ori, recommendations, top_k=10)

    print(f"Precision@10: {avg_precision:.4f}")
    print(f"Recall@10: {avg_recall:.4f}")
    print(f"Hit Rate@10: {avg_hit_rate:.4f}")
    print(f"MRR@10: {avg_mrr:.4f}")

    end_time = time.time()
    end_dt = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    elapsed_time = end_time - start_time

    minutes = int(elapsed_time // 60)
    seconds = int(elapsed_time % 60)
    elapsed_str = f"{minutes}분 {seconds}초"

    message = (
    f"\U0001F7E2 NCF 추천 모델 평가 완료! (local test) \n\n"
    f"*Start Time (학습 시작 시간):* {start_dt}\n"
    f"*End Time (학습 완료 시간):* {end_dt}\n"
    # f"*Elapsed Time:* {elapsed_time:.2f} minutes\n\n"
    f"*Elapsed Time (학습에 걸린 시간):* {elapsed_str}\n\n"
    f"*Metrics:*\n"
    f"> Precision@10: {avg_precision:.4f}\n"
    f"> Recall@10: {avg_recall:.4f}\n"
    f"> Hit Rate@10: {avg_hit_rate:.4f}\n"
    f"> MRR@10: {avg_mrr:.4f}"
    )

    slack.send_message(message)


if __name__ =='__main__':
    print("GPU 사용 가능 여부:", torch.cuda.is_available())
    # print("사용 중인 GPU:", torch.cuda.current_device())
    # print("GPU 이름:", torch.cuda.get_device_name(torch.cuda.current_device()))

    main()