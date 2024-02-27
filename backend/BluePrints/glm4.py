import os
import base64
import shutil
from extension import db
from io import BytesIO
from PIL import Image, ImageFilter
from zhipuai import ZhipuAI
from flask import Blueprint, request, jsonify
from flask_jwt_extended import jwt_required, get_jwt_identity
from models import UploadFile, SegmentationFile, ClassificationFile
from sqlalchemy import cast, String

bp = Blueprint("glm4", __name__, url_prefix="/glm4")


@bp.route("/", methods=["POST"])
@jwt_required()
def get_response():
    post_data = request.get_json()
    message = post_data.get('message')
    client = ZhipuAI(api_key="40f59567767cf2cfdd827f132ab1bf3a.5RXCZwG8fxww3pir")  # 请填写您自己的APIKey
    response = client.chat.completions.create(
        model="glm-4",  # 填写需要调用的模型名称
        messages=[
            {"role": "user", "content": message},
        ],
        stream=True,
        do_sample=True,
        temperature=0.7,
    )
    for chunk in response:
        print(chunk.choices[0].delta)
