from flask import Flask
from extension import db, cors, mail, jwt
from flask_migrate import Migrate
import config
from models import User, UploadFile, SegmentationFile, ClassificationFile
from BluePrints.segmentation import bp as segmentation_bp
from BluePrints.classification import bp as classification_bp
from BluePrints.auth import bp as auth_bp
from BluePrints.user import bp as user_bp
from BluePrints.getHistory import bp as history_bp
from flask import request, jsonify, make_response


app = Flask(__name__)
app.config.from_object(config)

db.init_app(app)
cors.init_app(app)
mail.init_app(app)
jwt.init_app(app)


migrate = Migrate(app, db)

app.register_blueprint(segmentation_bp)
app.register_blueprint(classification_bp)
app.register_blueprint(auth_bp)
app.register_blueprint(user_bp)
app.register_blueprint(history_bp)


# with app.app_context():
#     with db.engine.connect() as con:
#         rs = con.execute(text("select 1"))
#         # for row in rs:
#         print(rs.fetchone())


# with app.app_context():
#     db.create_all()

@app.route('/api')
def hello_world():
    return 'Hello, World!'


@jwt.expired_token_loader
def my_expired_token_callback(jwt_header, jwt_payload):
    """返回 flask Response 格式"""
    return jsonify(code="401", err="token 已过期"), 401


@app.route('/OriginalImg_output/<path:file>', methods=['GET'])
def show_ori_photo(file):
    if request.method == 'GET':
        if not file is None:
            image_data = open(f'OriginalImg_output/{file}', "rb").read()
            response = make_response(image_data)
            response.headers['Content-Type'] = 'image/png'
            return response


@app.route('/inference_model/output/visualization/<path:file>', methods=['GET'])
def show_seg_photo(file):
    if request.method == 'GET':
        if not file is None:
            image_data = open(f'inference_model/output/visualization/{file}', "rb").read()
            response = make_response(image_data)
            response.headers['Content-Type'] = 'image/png'
            return response


@app.route("/user/add")
def add_user():
    user = User(username="cuizhihao", password="123456")
    db.session.add(user)
    db.session.commit()
    return "add user success"


@app.route("/user/query")
def query_user():
    user = User.query.filter_by(username="cuizhihao").first()
    return f"username: {user.username}, password: {user.password}"


if __name__ == '__main__':
    app.run(debug=True, threaded=True)
