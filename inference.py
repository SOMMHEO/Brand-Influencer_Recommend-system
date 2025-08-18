from inference_modules.DB_connection_for_inference import *
from inference_modules.Load_NecessaryThings_from_s3 import *
from inference_modules.Recommender import *
from libraries.slack_hook import SlackHook

from datetime import datetime
import time
import sys

def main(sp_user_uid):
    # AWS authentication
    aws_access_key = os.getenv("aws_accessKey")
    aws_secret_key = os.getenv("aws_secretKey")
    bucket_name = 'flexmatch-data'
    region = 'ap-northeast-2'

    # load current_product_info (current archive regist item)
    # 여기서 instagram_user_info는 사용 X
    product_info, user_sales_info, instagram_user_category_info, not_conn_instagram_user_info, conn_instagram_user_info = get_additional_DB_info()

    # load data, encoder, model
    Load_from_s3 = LoadNecessaryThings(aws_access_key, aws_secret_key, bucket_name, region)

    Load_from_s3.load_data()
    user_info_ori = Load_from_s3.user_info_ori
    product_info_ori = Load_from_s3.product_info_ori
    user_view_ori = Load_from_s3.user_view_ori

    Load_from_s3.load_encoders()
    user_encoder = Load_from_s3.user_encoder
    item_encoder = Load_from_s3.item_encoder

    Load_from_s3.load_model()
    device = Load_from_s3.device
    neumf_model = Load_from_s3.neumf_model

    # slack alarm
    slack = SlackHook()
    start_time = time.time()
    start_dt = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # inference start message
    slack.send_message(f":rocket: 추천 inference 시작! (local test)\n\n*Start Time (시작 시간):* {start_dt}")

    final_recommendations, average_matching_rate = generate_recommendations_and_matching_rate(
    user_encoder, item_encoder, neumf_model, user_view_ori, product_info_ori, user_info_ori, product_info, device, sp_user_uid
    )

    print(average_matching_rate)

    end_time = time.time()
    end_dt = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    elapsed_time = end_time - start_time
    minutes = int(elapsed_time // 60)
    seconds = int(elapsed_time % 60)
    elapsed_str = f"{minutes}분 {seconds}초"


    message = (
        f":bulb: 모든 상품 추천 Inference 완료! (local test) \n\n"
        f"*Start Time (시작 시간):* {start_dt}\n"
        f"*End Time (완료 시간):* {end_dt}\n"
        f"*Elapsed Time (걸린 시간):* {elapsed_str}\n"
    )
    slack.send_message(message)
    print(message)

if __name__ =='__main__':

    if len(sys.argv) <= 1:
        uid = None
    else:
        uid = sys.argv[1]

    main(uid)
