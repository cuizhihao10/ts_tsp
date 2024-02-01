from extension import db
from datetime import datetime
from passlib.hash import sha256_crypt


class User(db.Model):
    __tablename__ = "user"
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    username = db.Column(db.String(100), nullable=False, unique=True)
    password = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), nullable=False, unique=True)

    def hash_password(self, password):
        """密码加密"""
        self.password = sha256_crypt.encrypt(password)

    def verify_password(self, password):
        """校验密码"""
        return sha256_crypt.verify(password, self.password)


class UploadFile(db.Model):
    __tablename__ = "uploadfile"
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    filename = db.Column(db.String(200), nullable=False, unique=True)
    # type = db.Column(db.String(100), nullable=False)
    sub_files = db.Column(db.String(1000), nullable=False)
    size = db.Column(db.String(100), nullable=False)
    path = db.Column(db.String(200), nullable=False)
    upload_time = db.Column(db.DateTime, default=datetime.now())
    author_id = db.Column(db.Integer, db.ForeignKey("user.id"))
    author = db.relationship("User", backref="uploadfiles")


class SegmentationFile(db.Model):
    __tablename__ = "segmentationfile"
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    filename = db.Column(db.String(200), nullable=False)
    type = db.Column(db.String(100), nullable=False)
    size = db.Column(db.String(100), nullable=False)
    path = db.Column(db.String(200), nullable=False)
    original_images_sequence = db.Column(db.String(200), nullable=False)
    original_images_path = db.Column(db.String(200), nullable=False)
    segmentation_result = db.Column(db.String(1000), nullable=False)
    segmentation_time = db.Column(db.DateTime, default=datetime.now())
    author_id = db.Column(db.Integer, db.ForeignKey("user.id"))
    author = db.relationship("User", backref="segmentationfiles")


class ClassificationFile(db.Model):
    __tablename__ = "classificationfile"
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    filename = db.Column(db.String(200), nullable=False)
    type = db.Column(db.String(100), nullable=False)
    size = db.Column(db.String(100), nullable=False)
    path = db.Column(db.String(200), nullable=False)
    original_images_sequence = db.Column(db.String(200), nullable=False)
    original_images_path = db.Column(db.String(200), nullable=False)
    classification_time = db.Column(db.DateTime, default=datetime.now())
    classification_result = db.Column(db.String(1000), nullable=False)
    author_id = db.Column(db.Integer, db.ForeignKey("user.id"))
    author = db.relationship("User", backref="classificationfiles")
