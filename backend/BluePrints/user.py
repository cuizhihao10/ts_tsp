from flask import Blueprint
from flask import request, jsonify
from flask_jwt_extended import jwt_required, create_access_token, get_jwt_identity
from models import User

bp = Blueprint("user", __name__, url_prefix="/user")


# 使用刷新JWT来获取普通JWT
@bp.route("/refresh/", methods=["POST"])
@jwt_required(refresh=True)
def refresh_token():
    identity = get_jwt_identity()
    access_token = create_access_token(identity=identity)
    return jsonify({"code": 200, "msg": "refresh success", "data": {"access_token": access_token}})


@bp.route("/info/")
@jwt_required()
def get_user_info():
    uid = get_jwt_identity()
    username = User.query.filter_by(id=uid).first().username
    return jsonify({"code": 200, "msg": "send success", "data": {"id": uid, "username": username}})


@bp.route("/logout/", methods=["POST"])
@jwt_required()
def logout():
    return jsonify({"code": 200, "msg": "logout success"})
