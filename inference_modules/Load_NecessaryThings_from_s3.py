import os
from dotenv import load_dotenv
import boto3
import pandas as pd
import pickle
import io
from datetime import datetime
import torch
import sys


load_dotenv(dotenv_path="config/.env")

class LoadNecessaryThings:
    def __init__(self, aws_access_key, aws_secret_key, bucket_name, region='ap-northeast-2'):
        self.bucket_name = bucket_name
        self.s3 = boto3.client(
            's3',
            aws_access_key_id=aws_access_key,
            aws_secret_access_key=aws_secret_key,
            region_name=region
        )
        self.csv_dfs = {}
        # self.load_data() # main에서 실행
        self.user_encoder = None
        self.item_encoder = None
        self.neumf_model = None
        self.device = None

    def load_data(self):
        current_date = datetime.now().strftime('%y-%m-%d')
        data_prefix = f'recommendation_system/data/{current_date}/'
        response = self.s3.list_objects_v2(Bucket=self.bucket_name, Prefix=data_prefix)

        for obj in response.get('Contents', []):
            key = obj['Key']
            if key.endswith('.csv'):
                file_obj = self.s3.get_object(Bucket=self.bucket_name, Key=key)
                df = pd.read_csv(file_obj['Body'])
                self.csv_dfs[key.split('/')[-1]] = df

        self.user_info_ori = self.csv_dfs['user_info_ori.csv']
        self.product_info_ori = self.csv_dfs['product_info_ori.csv']
        self.user_view_ori = self.csv_dfs['user_view_ori.csv']
    
    def load_encoders(self):
        user_encoder_key = 'recommendation_system/encoder/user_encoder.pkl'
        item_encoder_key = 'recommendation_system/encoder/item_encoder.pkl'
        user_encoder_obj = self.s3.get_object(Bucket=self.bucket_name, Key=user_encoder_key)
        item_encoder_obj = self.s3.get_object(Bucket=self.bucket_name, Key=item_encoder_key)
        self.user_encoder = pickle.loads(user_encoder_obj['Body'].read())
        self.item_encoder = pickle.loads(item_encoder_obj['Body'].read())
    
    def load_model(self):
        model_key = 'recommendation_system/model/neumf_model.pth'
        model_obj = self.s3.get_object(Bucket=self.bucket_name, Key=model_key)
        model_data = model_obj['Body'].read()
        sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))
        from training_modules.NCF_model import NeuMF
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.neumf_model = torch.load(io.BytesIO(model_data), map_location=self.device, weights_only=False)