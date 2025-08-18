import pymysql
import os
import pandas as pd
import json 
from sshtunnel import SSHTunnelForwarder

class SSHMySQLConnector:
    def __init__(self):
        self.tunnel = None
        self.connection = None

        # 계정정보 세팅 및 DB 연결까지 자동 수행
        self.get_account()
        self.connect()

    def get_account(self):
        # JSON 파일 열기
        base_dir = os.path.dirname(os.path.abspath(__file__))  # __lib__/
        config_path = os.path.join(base_dir, '..', 'config', 'accounts.json')

        with open(config_path, 'r', encoding='utf-8') as f:
            account = json.load(f)

        # JSON 값들을 클래스 인스턴스 변수에 할당
        self.ssh_host = account["SSH_IP"]
        self.ssh_username = account["SSH_ID"]
        self.ssh_password = account["SSH_PW"]
        self.db_username = account["DB_ID"]
        self.db_password = account["DB_PW"]
        self.db_name = account["DB_NAME"]

    def connect(self):
        # SSH 터널을 설정하여 데이터베이스 서버에 안전하게 접속
        self.tunnel = SSHTunnelForwarder(
            (self.ssh_host, 22),
            ssh_username=self.ssh_username,
            ssh_password=self.ssh_password,
            remote_bind_address=('127.0.0.1', 3306)
        )
        self.tunnel.start()

        # MySQL 연결
        self.connection = pymysql.connect(
            host='127.0.0.1',
            user=self.db_username,
            passwd=self.db_password,
            db=self.db_name,
            port=self.tunnel.local_bind_port
        )

    def select_query(self, query):
        return pd.read_sql_query(query, self.connection)

    def insert_query(self, query, args=None):
        with self.connection.cursor() as cursor:
            cursor.execute(query, args)
        self.connection.commit()

    def close(self):
        if self.connection:
            self.connection.close()
        if self.tunnel:
            self.tunnel.stop()
