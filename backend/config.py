import argparse
from datetime import timedelta

HOSTNAME = "127.0.0.1"
PORT = "3306"
USERNAME = "root"
PASSWORD = "cuizhihao"
DATABASE = "tsp"
DB_URI = "mysql+pymysql://{}:{}@{}:{}/{}?charset=utf8".format(USERNAME, PASSWORD, HOSTNAME, PORT, DATABASE)
SQLALCHEMY_DATABASE_URI = DB_URI
UPLOAD_FOLDER = "upload"


# 邮箱配置
MAIL_SERVER = "smtp.qq.com"
MAIL_PORT = 465
MAIL_USE_SSL = True
MAIL_USERNAME = "276522002@qq.com"
MAIL_PASSWORD = "bdwlvwiwcgvibgef"
MAIL_DEFAULT_SENDER = "276522002@qq.com"


# redis配置
REDIS_HOST = '127.0.0.1'
REDIS_PORT = 6379
REDIS_DB = 0
REDIS_PASSWORD = None
REDIS_ENCODING = 'utf-8'

# jwt配置
JWT_SECRET_KEY = "#ddad34dff/sad23#jsda3234h34adf"  # Change this!
JWT_ACCESS_TOKEN_EXPIRES = 6000
JWT_REFRESH_TOKEN_EXPIRES = timedelta(days=30)


# 模型参数设置
