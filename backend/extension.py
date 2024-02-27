from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
from flask_mail import Mail
from flask_jwt_extended import JWTManager

db = SQLAlchemy()
cors = CORS(origins="http://127.0.0.1:5173", supports_credentials=True)
mail = Mail()
jwt = JWTManager()

