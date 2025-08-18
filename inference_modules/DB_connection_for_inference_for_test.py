import pymysql
from sshtunnel import SSHTunnelForwarder
import json
import pandas as pd
from datetime import datetime


########################################################################################################
import os
import csv



def get_uid_to_category_map(type_,filepath):
    try:
        df = pd.read_excel(filepath)
        # 컬럼명이 있을 경우
        if type_=='inference':
            return {
                str(int(row['user_uid'])): str(row['interestcategory'])
                for _, row in df.iterrows()
                if not pd.isna(row['user_uid']) and not pd.isna(row['interestcategory'])
            }
        elif type_ == 'brand_matching':
            return {
                str(int(row['seller_uid'])): str(row['category_name'])
                for _, row in df.iterrows()
                if not pd.isna(row['seller_uid']) and not pd.isna(row['category_name'])
            }
    except Exception as e:
        print(f"❌ category map 생성 실패: {e}")
        return {}
    
def read_excel_to_list(filename):
    try:
        df = pd.read_excel(filename, header=None)  # 헤더 무시하고 읽기
        df = df.iloc[1:, :]  # 첫 번째 행 제거
        return df.values.tolist()  # 2차원 배열로 변환
    except Exception as e:
        print(f"엑셀 파일 읽기 실패: {e}")
        return []


def read_txt_to_list(filename):
    result = []
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:  # 빈 줄 제거
                    result.append(line)
    except FileNotFoundError:
        print(f"파일을 찾을 수 없습니다: {filename}")
    return result


def convert_csv_to_excel(csv_path, excel_path=None):
    try:
        df = pd.read_csv(csv_path, delimiter=';')

        # 저장할 xlsx 경로 자동 생성 (확장자만 바꿔줌)
        if not excel_path:
            excel_path = os.path.splitext(csv_path)[0] + '.xlsx'

        df.to_excel(excel_path, index=False)
        #print(f"✅ 엑셀 파일 저장 완료: {excel_path}")
        return excel_path
    except Exception as e:
        print(f"❌ 변환 실패: {e}")
        #return None

def get_user_category_map(filepath):
    import pandas as pd
    df = pd.read_excel(filepath)
    return {str(int(row['user_uid'])): str(row['interestcategory']) 
            for _, row in df.iterrows() if not pd.isna(row['user_uid'])}

def extract_result(times, folder, arr_data, first_row=None, category_map=None, avg_match_rate=None, user_category_map=None):
    if not os.path.exists("result"):
        os.makedirs("result")
    target_dir = os.path.join("result", folder)
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    timestamp = times
    filename = os.path.join(target_dir, f"{folder}_{timestamp}.csv")

    file_exists = os.path.exists(filename)

    with open(filename, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter=';')

        if folder == "inference":
            if not file_exists:
                writer.writerow(["uid", "category", "average_matching_rate"] + [f"result_id_{i+1}" for i in range(10)])

            for key, val in arr_data.items():
                uid_str = str(key)
                category = category_map.get(uid_str, "") if category_map else ""
                only_ids = [item[0] for item in val[:10] if isinstance(item, (list, tuple)) and len(item) > 0]
                writer.writerow([key, category, round(avg_match_rate or 0, 4) * 100] + only_ids)

        elif isinstance(arr_data, list) and len(arr_data) == 2 and isinstance(arr_data[1], list):
            seller_uid = arr_data[0]
            user_data_list = arr_data[1][:10]  # 최대 10명

            # user_uid 추출
            user_uids = [str(d.get("user_uid", "")) for d in user_data_list]
            # infl_list.xlsx에서 category 매핑
            user_categories = [user_category_map.get(uid, "") for uid in user_uids]

            category = category_map.get(str(seller_uid), "") if category_map else ""

            # 헤더 작성
            if not file_exists:
                header = ["seller_uid", "category", "avg_match_rate"]
                for i in range(len(user_uids)):
                    header += [f"user_uid_{i+1}", f"user_category_{i+1}"]
                writer.writerow(header)

            # 데이터 작성
            row = [seller_uid, category, round(avg_match_rate or 0, 4)]
            for uid, cat in zip(user_uids, user_categories):
                row += [uid, cat]

            writer.writerow(row)

########################################################################################################



class SSHMySQLConnector:
    def __init__(self, ssh_host, ssh_username, ssh_password, db_username, db_password, db_name):
        self.ssh_host = ssh_host
        self.ssh_username = ssh_username
        self.ssh_password = ssh_password
        self.db_username = db_username
        self.db_password = db_password
        self.db_name = db_name
        self.tunnel = None
        self.connection = None

    def connect(self):
        # SSH 터널을 설정하여 데이터베이스 서버에 안전하게 접속
        self.tunnel = SSHTunnelForwarder(
            (self.ssh_host, 22),  # SSH 서버 주소와 포트
            ssh_username=self.ssh_username,
            ssh_password=self.ssh_password,
            remote_bind_address=('127.0.0.1', 3306)  # 데이터베이스의 주소와 포트
        )
        self.tunnel.start()
        
        # 터널을 통해 MySQL 데이터베이스에 연결
        self.connection = pymysql.connect(
            host='127.0.0.1',  # 로컬호스트에 터널링
            user=self.db_username,
            passwd=self.db_password,
            db=self.db_name,
            port=self.tunnel.local_bind_port  # 터널의 로컬 포트 사용
        )

    def execute_query(self, query):
        # 쿼리 실행 후 데이터를 DataFrame으로 반환
        return pd.read_sql_query(query, self.connection)

    def close(self):
        if self.connection:
            self.connection.close()
        if self.tunnel:
            self.tunnel.stop()


    #### INSERT 함수 새로 생성 ###############################################
    def insert_query(self, table_name, value_arr):
        try:
            with self.connection.cursor() as cursor:
                for row in value_arr:
                    columns = ', '.join(row.keys())
                    placeholders = ', '.join(['%s'] * len(row))
                    values = list(row.values())

                    sql = f"INSERT INTO {table_name} ({columns}) VALUES ({placeholders})"
                    
                    # 디버깅 출력
                    #print("[SQL]", sql)
                    #print("[VALUES]", values)

                    cursor.execute(sql, values)

            self.connection.commit()
            #print("[✅ INSERT COMMIT 완료]")
        except Exception as e:
            self.connection.rollback()
            #print("[❌ INSERT 실패]", e)
    ########################################################################


def sendQuery(query):

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
    results = connector.execute_query(query)
    # print(results)
    connector.close()

    return results

def get_product_info(): 

    query_product_info = '''
    SELECT 
        p.brand,
        p.seller_uid,
        p.name AS item_name,
        p.uid AS item_uid,
        p.sale_price AS price,
        c.name AS product_category,
        COALESCE(COUNT(i.item_id), 0) AS viewcount,
        d.sell_type
    FROM 
        int_product p
    LEFT JOIN 
        int_category c ON p.category_1 = c.code
    LEFT JOIN 
        int_statistics_member_interaction i ON p.uid = i.item_id
    LEFT JOIN 
        int_product_detail d ON p.uid = d.t_uid
    WHERE 
        p.regist_status = 0
        AND (
            (d.sell_type = 1 AND NOW() BETWEEN FROM_UNIXTIME(p.starttime) AND FROM_UNIXTIME(p.term_expiretime))
            OR
            (d.sell_type = 2 AND CURDATE() BETWEEN d.addinfo2 AND d.addinfo3)
        )
    GROUP BY 
        p.brand, p.seller_uid, p.name, p.uid, p.sale_price, c.name
    '''
    product_info = sendQuery(query_product_info)
    
    return product_info

def InsertQuery(member_uid, products_dict, is_clicked=0):
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
    regtime = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # JSON 문자열로 변환
    products_json = json.dumps(products_dict, ensure_ascii=False)
    try:
        with connector.connection.cursor() as cursor:
            sql = '''
            INSERT INTO int_product_recommendation (member_uid, products, regtime, is_clicked)
            VALUES (%s, %s, %s, %s)
            '''
            cursor.execute(sql, (member_uid, products_json, regtime, is_clicked))
            connector.connection.commit()
            print(f"INSERT 완료: {member_uid}, {regtime}")
    except Exception as e:
        print(f"INSERT 실패: {e}")
    finally:
        connector.close()