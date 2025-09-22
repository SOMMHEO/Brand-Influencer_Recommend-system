import os
from dotenv import load_dotenv
import pandas as pd
import numpy as np
from tqdm import tqdm
import pickle
import re, emoji
import boto3
from io import BytesIO
from sklearn.preprocessing import LabelEncoder

load_dotenv(dotenv_path="config/.env")
aws_access_key = os.getenv("aws_accessKey")
aws_secret_key = os.getenv("aws_secretKey") 
region_name='ap-northeast-2'
 
s3 = boto3.client('s3',
                    aws_access_key_id=aws_access_key,
                    aws_secret_access_key=aws_secret_key,
                    region_name=region_name) # ec2 사용 시 변경


class Preprocessor:
    def __init__(self, user_info: pd.DataFrame, product_info: pd.DataFrame, user_view: pd.DataFrame, bucket_name: str, user_encoder_key: str, item_encoder_key: str):
        self.user_info = user_info
        self.product_info = product_info
        self.user_view = user_view
        self.bucket_name = bucket_name
        self.user_encoder_key = user_encoder_key
        self.item_encoder_key = item_encoder_key
        self.s3 = boto3.client('s3') 

    def preprocess_user_info(self) -> pd.DataFrame:
        self.user_info.rename(columns={'uid':'user_uid'}, inplace=True)
        self.user_info['user_id'] = self.user_info['user_id'].astype(str)  # 변경 -> brand_matching에서 user_id를 기준으로 merge 진행
        self.user_info['user_uid'] = self.user_info['user_uid'].astype(str)
        self.user_info['add1'] = self.user_info['add1'].astype(str)
        self.user_info['sex'] = self.user_info['sex'].astype(str)
        self.user_info['interestcategory'] = self.user_info['interestcategory'].astype(str)
        self.user_info['reg_datetime'] = pd.to_datetime(self.user_info['reg_datetime']) # 변경 -> brand_matching에서 회원가입 날짜 확인

        # fillna가 안되는 경우를 대비해서 str type으로 변환 후 replace도 함께 활용
        self.user_info['sex'].replace('', 'w', inplace=True)
        self.user_info['sex'].replace(' ', 'w', inplace=True)
        self.user_info['sex'].fillna('w', inplace=True)

        self.user_info['interestcategory'].replace('', '뷰티', inplace=True)
        self.user_info['interestcategory'].replace(' ', '뷰티', inplace=True)
        self.user_info['interestcategory'].fillna('뷰티', inplace=True)

        # 카테고리명 정제
        category_map = {
            'BABY/KIDS': '베이비/키즈',
            'BEAUTY': '뷰티',
            'FASHION': '패션',
            'FOOD': '푸드',
            'HEALTHY': '헬시',
            'HOME/LIVING': '홈/리빙',
            'SERVICE': '서비스',
            'SPORT': '스포츠',
            'TEST 카테고리.. TEST': '뷰티'
        }

        for k, v in category_map.items():
            self.user_info['interestcategory'] = self.user_info['interestcategory'].str.replace(k, v)

        self.user_info.loc[(self.user_info['group_key'] == '2') & (self.user_info['SNS'].isnull()), 'SNS'] = '인스타그램'

        # 카테고리 및 SNS 분할
        self.user_info['interestcategory_1'] = self.user_info['interestcategory'].str.split('@').str[0]
        self.user_info['interestcategory_2'] = self.user_info['interestcategory'].str.split('@').str[1]
        self.user_info['interestcategory_3'] = self.user_info['interestcategory'].str.split('@').str[2]

        self.user_info['SNS_1'] = self.user_info['SNS'].str.split('@').str[0]
        self.user_info['SNS_2'] = self.user_info['SNS'].str.split('@').str[1]
        self.user_info['SNS_3'] = self.user_info['SNS'].str.split('@').str[2]

        # user_info_ori = self.user_info.copy()

        return self.user_info

    def preprocess_product_info(self) -> pd.DataFrame:
        self.product_info['item_uid'] = self.product_info['item_uid'].astype(str)
        self.product_info['seller_uid'] = self.product_info['seller_uid'].astype(str)
        self.product_info.drop_duplicates(subset=['item_uid', 'item_name', 'brand'], inplace=True)

        # product_info_ori = self.product_info.copy()
        
        return self.product_info

    def preprocess_user_view(self) -> pd.DataFrame:
        self.user_view.rename(columns={'user_id': 'user_uid', 'item_id': 'item_uid'}, inplace=True)
        self.user_view = self.user_view[['user_uid', 'item_uid', 'view_cnt', 'timestamp']]
        self.user_view['user_uid'] = self.user_view['user_uid'].astype(str)
        self.user_view['item_uid'] = self.user_view['item_uid'].astype(str)
        self.user_view['timestamp'] = pd.to_datetime(self.user_view['timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')

        # filtering
        # item_lst = list(self.product_info['item_uid'].unique())
        # user_lst = list(self.user_info['user_uid'].unique())
        user_lst = self.user_info['user_uid'].values.flatten().tolist()
        item_lst = self.product_info['item_uid'].values.flatten().tolist()
        self.user_view = self.user_view[self.user_view['item_uid'].isin(item_lst)]
        self.user_view = self.user_view[self.user_view['user_uid'].isin(user_lst)]
        self.user_view = self.user_view[self.user_view['item_uid'].str.isdigit()]

        # user_view_ori = self.user_view.copy()
        return self.user_view
class DataSplitter:
    def __init__(self, user_view: pd.DataFrame, user_info: pd.DataFrame, product_info: pd.DataFrame):
        self.user_view = user_view
        self.user_info = user_info
        self.product_info = product_info

    def encode_and_split(self):
        user_encoder = LabelEncoder()
        item_encoder = LabelEncoder()

        self.user_info = self.user_info.loc[:, ~self.user_info.columns.duplicated()]
        self.product_info = self.product_info.loc[:, ~self.product_info.columns.duplicated()]
        self.user_view = self.user_view.loc[:, ~self.user_view.columns.duplicated()]

        self.user_info = self.user_info.drop_duplicates(subset='user_uid')
        self.product_info = self.product_info.drop_duplicates(subset='item_uid')

        user_info_ori = self.user_info.copy()
        product_info_ori = self.product_info.copy()
        user_view_ori = self.user_view.copy()

        self.user_info['user_uid'] = user_encoder.fit_transform(self.user_info['user_uid'])
        self.product_info['item_uid'] = item_encoder.fit_transform(self.product_info['item_uid'])

        self.user_view['user_uid'] = user_encoder.transform(self.user_view['user_uid'])
        self.user_view['item_uid'] = item_encoder.transform(self.user_view['item_uid'])
        self.user_view['label'] = 1

        # save encoder to
        user_buf = BytesIO()
        pickle.dump(user_encoder, user_buf)
        user_buf.seek(0)

        bucket_name = 'flexmatch-data'
        user_encoder_key = 'recommendation_system/encoder/user_encoder.pkl'
        item_encoder_key = 'recommendation_system/encoder/item_encoder.pkl'
        
        s3.upload_fileobj(user_buf, bucket_name, user_encoder_key)

        item_buf = BytesIO()
        pickle.dump(item_encoder, item_buf)
        item_buf.seek(0)
        s3.upload_fileobj(item_buf, bucket_name, item_encoder_key)

        # Leave-One-Out 방식 분할
        self.user_view = self.user_view.sort_values(by=['user_uid', 'timestamp'])
        user_view_counts = self.user_view.groupby('user_uid').size()
        single_view_users = user_view_counts[user_view_counts == 1].index

        test_data = self.user_view[~self.user_view['user_uid'].isin(single_view_users)].groupby('user_uid').head(1)
        train_data = self.user_view.drop(test_data.index)

        return train_data, test_data, user_encoder, item_encoder, user_info_ori, product_info_ori, user_view_ori

    def negative_sampling(self, train_data, test_data, neg_ratio=20, test_neg_num=99):
        train_users = set(train_data['user_uid'].unique())
        test_item_set = set(zip(test_data['user_uid'], test_data['item_uid']))
        total_items = set(self.product_info['item_uid'].unique())

        negative_samples = []
        for user in tqdm(train_users):
            seen_items = set(self.user_view[self.user_view['user_uid'] == user]['item_uid'])
            candidate_items = list(total_items - seen_items - {item for (u, item) in test_item_set if u == user})
            pos_count = len(seen_items)
            neg_count = min(pos_count * neg_ratio, len(candidate_items))
            negative_items = np.random.choice(candidate_items, size=neg_count, replace=False)

            for neg_item in negative_items:
                negative_samples.append([user, neg_item, 0])

        negative_df = pd.DataFrame(negative_samples, columns=['user_uid', 'item_uid', 'label'])
        train = pd.concat([negative_df, train_data.drop(['timestamp', 'view_cnt'], axis=1)], axis=0)

        # 테스트 셋 네거티브 샘플링
        test_negative_samples = []
        test_users = test_data['user_uid'].unique()

        for user in tqdm(test_users):
            seen_items = set(self.user_view[self.user_view['user_uid'] == user]['item_uid'])
            test_item = test_data[test_data['user_uid'] == user]['item_uid'].iloc[0]
            candidate_items = list(total_items - seen_items - {test_item})
            neg_count = min(test_neg_num, len(candidate_items))
            negative_items = np.random.choice(candidate_items, size=neg_count, replace=False)

            for neg_item in negative_items:
                test_negative_samples.append([user, neg_item, 0])

        test_negative_df = pd.DataFrame(test_negative_samples, columns=['user_uid', 'item_uid', 'label'])
        test = pd.concat([test_data.drop(['timestamp', 'view_cnt'], axis=1), test_negative_df], axis=0)

        return train, test
    
def clean_text(text):
    if not isinstance(text, str):
        return ''
    
    text = emoji.replace_emoji(text, replace='')
    text = re.sub(r'[^가-힣a-zA-Z0-9\s]', '', text)
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)

    return text.strip()

def preprocess_item_name(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    brand_regex = '|'.join(re.escape(brand) for brand in df['brand'].unique() if pd.notna(brand))
    if brand_regex:
        df['item_name'] = df['item_name'].str.replace(brand_regex, '', regex=True, case=False)

    df['item_name'] = df['item_name'].str.replace(r'(\d+차|n차|N차)', '', regex=True, case=False)
    df['item_name'] = df['item_name'].str.replace(r'\b\d+\s*(?:ml|g|kg|l|근|포|개|인분)\b|\b한근\b', '', regex=True, case=False)
    df['item_name'] = df['item_name'].str.replace(r'\b(?:x|X)\s*\d+(?:\s*\S+)?', '', regex=True, case=False)

    df['item_name'] = df['item_name'].str.replace(r'\b\d+[가-힣]+\b', '', regex=True)
    df['item_name'] = df['item_name'].str.replace(r'\b[a-zA-Z]+\d+\b', '', regex=True)
    df['item_name'] = df['item_name'].str.replace(r'\b[a-zA-Z]+[가-힣]+\b', '', regex=True)

    df['item_name_preprocessed'] = df['item_name_preprocessed'].str.replace(r'[a-zA-Z0-9]+', '', regex=True)
    df['item_name_preprocessed'] = df['item_name_preprocessed'].str.replace(r'[^\w\s]', '', regex=True)
    df['item_name'] = df['item_name'].str.strip().str.replace(r'\s+', ' ', regex=True)

    return df