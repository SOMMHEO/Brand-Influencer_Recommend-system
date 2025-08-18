import os
from dotenv import load_dotenv
import pymysql
import pandas as pd
from sshtunnel import SSHTunnelForwarder
import boto3
import json

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

def sendQuery(query):

    # f = open('C:/Users/ehddl/Downloads/get_DB/etc/accounts.txt', 'r', encoding='utf-8') # boto3로 변경 or json으로 변경
    # acc= dict()
    # for a in f:
    #     b = a.split(':')
    #     acc[b[0]] = b[1].replace('\n','')

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
    print(results)
    connector.close()

    return results

def get_all_infos(): 

    query_user_info = '''
    SELECT 
        m.group_key,
        m.uid,
        m.user_id,
        m.add1,
        FROM_UNIXTIME(m.regtime) AS reg_datetime,
        COALESCE(NULLIF(m.age, 0), 30) AS age,  -- m.age가 NULL이거나 0이면 30으로 처리
        COALESCE(m.sex, 'F') AS sex,   -- m.sex가 NULL이면 'F'로 처리
        CONCAT_WS('@',
            CASE WHEN m.add1 IS NOT NULL THEN '인스타그램' ELSE NULL END,
            CASE WHEN m.add2 IS NOT NULL THEN '네이버블로그' ELSE NULL END,
            CASE WHEN m.add3 IS NOT NULL THEN '유튜브' ELSE NULL END,
            CASE WHEN m.add4 IS NOT NULL THEN '틱톡' ELSE NULL END
        ) AS SNS,
        COALESCE(s.interestcategory, '뷰티') AS interestcategory,
        
        COALESCE(SUM(CASE WHEN p.m_status = 'done' THEN 1 ELSE 0 END), 0) AS success_match_count,
        COUNT(p.member_uid) AS total_match_count
    FROM 
        op_member m
    LEFT JOIN 
        op_mem_seller s ON m.user_id = s.user_id
    LEFT JOIN 
        int_product_match_list p ON m.uid = p.member_uid
    WHERE m.group_key = 2  -- 추가
    GROUP BY 
        m.group_key, m.uid, m.user_id, s.interestcategory, SNS, m.age, m.sex
    '''
    user_info = sendQuery(query_user_info)

    # 학습할 때는 DB에 있는 전체 상품 데이터를 전부 학습(현재 판매중이지 않은 상품에 대한 필터링 X)
    query_product_info = '''
    SELECT 
        p.brand,
        p.seller_uid,
        p.name AS item_name,
        p.uid AS item_uid,
        p.sale_price AS price,
        c.name AS product_category,
        COALESCE(COUNT(i.item_id), 0) AS viewcount
    FROM 
        int_product p
    LEFT JOIN 
        int_category c ON p.category_1 = c.code
    LEFT JOIN 
        int_statistics_member_interaction i ON p.uid = i.item_id
    GROUP BY 
        p.brand, p.seller_uid, p.name, p.uid, p.sale_price, c.name
    '''
    product_info = sendQuery(query_product_info)
    
    query_user_view = '''
    SELECT 
        user_id,
        item_id,
        COUNT(*) AS view_cnt,
        MAX(timestamp) AS timestamp
    FROM 
        int_statistics_member_interaction
    GROUP BY 
        user_id, item_id
    '''
    user_view = sendQuery(query_user_view)
    
    return user_info, product_info, user_view