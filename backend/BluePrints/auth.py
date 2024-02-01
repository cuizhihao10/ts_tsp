from flask import Blueprint
from flask_mail import Message
from extension import mail
from flask import request, jsonify
import string
import random
import config
from utils import redis_captcha
from redis import StrictRedis
from .forms import RegisterForm, LoginForm
from models import User
from extension import db
from flask_jwt_extended import jwt_required, create_access_token, create_refresh_token, get_jwt_identity


redis_store = StrictRedis(host=config.REDIS_HOST, port=config.REDIS_PORT, decode_responses=True)


bp = Blueprint("auth", __name__, url_prefix="/auth")


@bp.route("/login/", methods=["POST"])
def login():
    form = LoginForm(request.form)
    if form.validate():
        account = form.account.data
        password = form.password.data
        if (User.query.filter_by(username=account).first() and User.query.filter_by(username=account).first()
                .verify_password(password)):
            user = User.query.filter_by(username=account).first()
            access_token = create_access_token(identity=user.id)
            refresh_token = create_refresh_token(identity=user.id)
            return jsonify({"code": 200, "msg": "login success", "data": {"access_token": access_token,
                                                                          "refresh_token": refresh_token}})
        elif (User.query.filter_by(email=account).first() and User.query.filter_by(email=account).first()
                .verify_password(password)):
            user = User.query.filter_by(email=account).first()
            access_token = create_access_token(identity=user.id)
            refresh_token = create_refresh_token(identity=user.id)
            return jsonify({"code": 200, "msg": "login success", "data": {"access_token": access_token,
                                                                          "refresh_token": refresh_token}})
        else:
            return jsonify({"code": 300, "msg": "login failed", "error_message": "Account or password error!"})
    else:
        return jsonify({"code": 500, "msg": "login failed", "error_message": form.errors})


@bp.route("/register/", methods=["POST"])
def register():
    form = RegisterForm(request.form)
    if form.validate():
        password = form.password.data
        user = User(username=form.username.data, email=form.email.data)
        user.hash_password(password)
        db.session.add(user)
        db.session.commit()
        return jsonify({"code": 200, "msg": "send success", "data": {"form": form.data}})
    else:
        return jsonify({"code": 500, "msg": "send success", "error_message": form.errors})


@bp.route("/captcha/email/")
def get_email_captcha():
    email = request.args.get("email")
    source = string.digits*4
    captcha = random.sample(source, 4)
    captcha = "".join(captcha)
    message = Message(subject="Registration verification code", recipients=[email], body=f"Your verification code is:{captcha}, Please input within 1 minute")
    mail.send(message)
    redis_captcha.redis_set(key=email, value=captcha)
    return jsonify({"code": 200, "msg": "send success", "data": {"email": email}})


@bp.route("/mail/test")
def mail_test():
    message = Message(subject="邮箱测试", recipients=["1870295615@qq.com"], body="测试")
    mail.send(message)
    return "send success"
