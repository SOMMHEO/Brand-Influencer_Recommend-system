import pymysql
from sshtunnel import SSHTunnelForwarder
import json
import pandas as pd
from datetime import datetime

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

# inference 시에 현재 판매중이지 않은 상품에 대한 필터링 O
def get_additional_DB_info(): 

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

    query_user_sales_info = '''
        SELECT o.uid as user_uid, s.total_visit, s.total_order, s.match_total_price
        FROM op_mem_seller_statistics s
        JOIN (
            SELECT member_uid, MAX(regdate) AS max_regdate
            FROM op_mem_seller_statistics
            GROUP BY member_uid
        ) latest ON s.member_uid = latest.member_uid AND s.regdate = latest.max_regdate
        JOIN op_member o ON o.uid = s.member_uid
        JOIN S3_RECENT_USER_INFO_MTR u ON o.add1 = u.acnt_nm
        ORDER BY s.uid DESC;
    '''
    user_sales_info = sendQuery(query_user_sales_info)

    query_instagram_user_category_info = '''
        SELECT member_uid, user_id, acnt_id, acnt_nm, main_category, top_3_category
        FROM op_mem_seller_score
    '''

    instagram_user_category_info = sendQuery(query_instagram_user_category_info)

    query_not_conn_instagram_user_info = '''
        SELECT acnt_id, acnt_nm, follower_cnt, follow_cnt, media_cnt
        FROM S3_RECENT_USER_INFO_MTR
    '''

    not_conn_instagram_user_info = sendQuery(query_not_conn_instagram_user_info)

    query_conn_instagram_user_info = '''
        SELECT acnt_id, acnt_nm, follower_cnt, follow_cnt, media_cnt
        FROM S3_CONN_v2_RECENT_USER_INFO_MTR
    '''

    conn_instagram_user_info = sendQuery(query_conn_instagram_user_info)

    return product_info, user_sales_info, instagram_user_category_info, not_conn_instagram_user_info, conn_instagram_user_info



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